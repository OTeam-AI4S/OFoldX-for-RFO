from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch

from ofoldx.conversion import SourceCheckpoint, save_repository
from ofoldx.model import Model
from ofoldx.models.registry import MODELS
from ofoldx.pipelines import PIPELINES, PairedComponent, PairedPipeline
from ofoldx.processor import Processor


class RepositoryPairedPipeline(PairedPipeline[Model, Processor]):
    task: ClassVar[str] = "fold"

    model_architecture = "test_repository"

    def prepare(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.processor.config.scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.model.weight + self.model.bias

    def postprocess(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return output

    def fold(self, input: torch.Tensor) -> torch.Tensor:
        prepared = self.prepare(input)
        return self.postprocess(prepared, self.forward(prepared))

    __call__ = fold


PIPELINES.register(
    "test_paired_pipeline",
    pipeline=RepositoryPairedPipeline,
    model_architecture="test_repository",
)


class RepositoryScorePipeline(PairedPipeline[Model, Processor]):
    task: ClassVar[str] = "score"

    model_architecture = "test_repository"

    def prepare(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.processor.config.scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.model.weight + self.model.bias

    def postprocess(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return output.mean()

    def score(self, input: torch.Tensor) -> torch.Tensor:
        prepared = self.prepare(input)
        return self.postprocess(prepared, self.forward(prepared))

    __call__ = score


PIPELINES.register("test_repository_score", pipeline=RepositoryScorePipeline, model_architecture="test_repository")


class RepositoryCustomPipeline(RepositoryScorePipeline):
    task: ClassVar[str] = "custom"

    def postprocess(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return output.sum()


PIPELINES.register("test_repository_custom", pipeline=RepositoryCustomPipeline, model_architecture="test_repository")


def test_conversion_preserves_user_documents_and_precedes_upload(
    tmp_path, repository_model, repository_processor, monkeypatch
):
    destination = tmp_path / "converted"
    destination.mkdir()
    original = {"README.md": "# Caller model card\n", "license.md": "Caller AGPL notice\n"}
    for name, content in original.items():
        (destination / name).write_text(content)
    checkpoint = tmp_path / "source.ckpt"
    checkpoint.write_bytes(b"weights")
    uploads = []

    class Api:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kwargs):
            pass

        def upload_folder(self, *, folder_path, repo_id, **kwargs):
            folder = Path(folder_path)
            assert all(
                (folder / name).is_file() for name in ("README.md", "license.md", "license-faq.md", "conversion.json")
            )
            assert json.loads((folder / "conversion.json").read_text())["sources"]["checkpoint"]["files"]
            uploads.append(repo_id)

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    save_repository(
        RepositoryPairedPipeline(model=repository_model, processor=repository_processor),
        destination,
        sources={"checkpoint": checkpoint},
        converter="example.convert",
        repo_id="example/converted",
    )
    assert uploads == ["example/converted"]
    assert {name: (destination / name).read_text() for name in original} == original


@pytest.mark.parametrize("source_kind", ["string", "path", "explicit"])
def test_conversion_refuses_missing_local_source_before_writing(
    tmp_path, repository_model, repository_processor, monkeypatch, source_kind
):
    monkeypatch.chdir(tmp_path)
    reference = "weights/missing.ckpt"
    source = {
        "string": reference,
        "path": Path(reference),
        "explicit": SourceCheckpoint(reference),
    }[source_kind]
    destination = tmp_path / "converted"
    with pytest.raises(FileNotFoundError):
        save_repository(
            RepositoryPairedPipeline(model=repository_model, processor=repository_processor),
            destination,
            sources={"checkpoint": source},
            converter="example.convert",
        )
    assert not destination.exists()


def test_reconversion_updates_generated_variant(tmp_path, repository_model, repository_processor):
    checkpoint = tmp_path / "source.ckpt"
    checkpoint.write_bytes(b"original")
    pipeline = RepositoryPairedPipeline(model=repository_model, processor=repository_processor)
    destination = tmp_path / "converted"
    for variant in ("first-variant", "second-variant"):
        save_repository(
            pipeline,
            destination,
            sources={"checkpoint": checkpoint},
            converter="example.convert",
            options={"variant": variant},
        )
    card = (destination / "README.md").read_text()
    assert "second-variant" in card and "first-variant" not in card
    assert json.loads((destination / "conversion.json").read_text())["options"]["variant"] == "second-variant"


class RepositoryPairedComponent(PairedComponent[Model, Processor]):
    model_architecture = "test_repository"


class TestPairedComponent:
    def test_round_trip(
        self,
        tmp_path: Path,
        repository_model: Any,
        repository_processor: Any,
    ) -> None:
        component = RepositoryPairedComponent(model=repository_model, processor=repository_processor)
        assert component.source is None
        (tmp_path / "pipeline.json").write_text("{}", encoding="utf-8")

        component.save(tmp_path)
        loaded = RepositoryPairedComponent.from_pretrained(tmp_path, backend="eager")

        assert (tmp_path / "model.json").is_file()
        assert (tmp_path / "processor.json").is_file()
        assert not (tmp_path / "pipeline.json").exists()
        assert type(loaded.model) is type(repository_model)
        assert type(loaded.processor) is type(repository_processor)
        assert loaded.source is not None
        assert loaded.source.path == tmp_path
        assert loaded.source.subdirectory == ""
        assert loaded.model.backend == "eager"


class TestPairedPipeline:
    def test_construction_validates_pair_and_accepts_subclasses(
        self, tmp_path: Path, repository_model: Any, repository_processor: Any
    ) -> None:
        class CustomModel(type(repository_model)):
            pass

        class CustomProcessor(type(repository_processor)):
            pass

        workflow = RepositoryPairedPipeline(model=CustomModel(), processor=CustomProcessor())
        assert isinstance(workflow.model, CustomModel)
        assert isinstance(workflow.processor, CustomProcessor)
        assert workflow(torch.ones(3)).shape == (3,)
        destination = tmp_path / "unregistered"
        with pytest.raises(TypeError, match="register the exact classes"):
            workflow.save(destination)
        assert not destination.exists()
        with pytest.raises(TypeError, match="test_repository"):
            RepositoryPairedPipeline(model=repository_model, processor=Processor())

        MODELS.register("test_custom_pair_identity", model=CustomModel, processor=CustomProcessor)

        class CustomPipeline(RepositoryPairedPipeline):
            model_architecture = "test_custom_pair_identity"

        PIPELINES.register(
            "test_custom_pair_pipeline", pipeline=CustomPipeline, model_architecture="test_custom_pair_identity"
        )
        workflow = CustomPipeline(model=CustomModel(), processor=CustomProcessor())
        workflow.save(tmp_path / "registered")
        loaded = CustomPipeline.from_pretrained(tmp_path / "registered")
        assert type(loaded.model) is CustomModel
        assert type(loaded.processor) is CustomProcessor
        torch.testing.assert_close(loaded(torch.ones(3)), workflow(torch.ones(3)))

    def test_construction_uses_eval_without_disabling_gradients(
        self, repository_model: Any, repository_processor: Any
    ) -> None:
        assert repository_model.training
        pipeline = RepositoryPairedPipeline(model=repository_model, processor=repository_processor)
        assert not pipeline.model.training
        output = pipeline(torch.ones(3)).sum()
        output.backward()
        assert pipeline.model.weight.grad is not None
        pipeline.model.train()
        assert pipeline.model.training

    def test_round_trip(
        self,
        tmp_path: Path,
        repository_model: Any,
        repository_processor: Any,
    ) -> None:
        pipeline = RepositoryPairedPipeline(model=repository_model, processor=repository_processor)
        assert pipeline.source is None

        pipeline.save(tmp_path)
        loaded = RepositoryPairedPipeline.from_pretrained(tmp_path, backend="eager")

        assert json.loads((tmp_path / "pipeline.json").read_text(encoding="utf-8")) == {
            "pipeline": "test_paired_pipeline"
        }
        assert type(loaded.model) is type(repository_model)
        assert type(loaded.processor) is type(repository_processor)
        assert loaded.source is not None
        assert loaded.source.path == tmp_path
        assert loaded.source.subdirectory == ""
        assert loaded.processor.config is not None
        assert repository_processor.config is not None
        assert dict(loaded.processor.config) == dict(repository_processor.config)
        assert loaded.processor.dtype is torch.float16
        assert loaded.model.backend == "eager"
        for key, expected in repository_model.state_dict().items():
            torch.testing.assert_close(loaded.model.state_dict()[key], expected)

    def test_stages_and_model_repository(
        self, tmp_path: Path, repository_model: Any, repository_processor: Any
    ) -> None:
        from ofoldx import AutoPipelineForFold, pipeline, save_model

        save_model(tmp_path, model=repository_model, processor=repository_processor)
        loaded = AutoPipelineForFold(tmp_path)
        input = torch.tensor([1.0, 2.0, 3.0])
        prepared = loaded.prepare(input)
        staged = loaded.postprocess(prepared, loaded.forward(prepared))
        expected = input * 2 * torch.arange(3) + torch.arange(3) + 3
        torch.testing.assert_close(staged, expected)
        torch.testing.assert_close(loaded(input), expected)
        torch.testing.assert_close(
            AutoPipelineForFold.from_pretrained(pretrained_model_name_or_path=tmp_path)(input), expected
        )
        torch.testing.assert_close(
            RepositoryPairedPipeline.from_pretrained(pretrained_model_name_or_path=tmp_path)(input), expected
        )
        torch.testing.assert_close(pipeline("fold", model=tmp_path)(input), expected)
        assert loaded.id == "test_paired_pipeline"
        assert loaded.task == "fold"

    def test_task_counterparts_share_repository(
        self, tmp_path: Path, repository_model: Any, repository_processor: Any
    ) -> None:
        from ofoldx import AutoPipelineForFold, AutoPipelineForScore

        designer = RepositoryPairedPipeline(model=repository_model, processor=repository_processor)
        designer.save(tmp_path)
        scorer = AutoPipelineForScore(tmp_path)
        input = torch.tensor([1.0, 2.0, 3.0])
        expected = (input * 2 * torch.arange(3) + torch.arange(3) + 3).mean()
        torch.testing.assert_close(scorer(input), expected)
        assert scorer.task == "score"
        assert scorer.source.path == tmp_path
        scorer.save(tmp_path / "score")
        folded = AutoPipelineForFold(tmp_path / "score")
        torch.testing.assert_close(folded(input).mean(), expected)

    def test_custom_task(self, tmp_path: Path, repository_model: Any, repository_processor: Any) -> None:
        from ofoldx import pipeline, save_model

        save_model(tmp_path, model=repository_model, processor=repository_processor)
        loaded = pipeline("custom", model=tmp_path)
        input = torch.tensor([1.0, 2.0, 3.0])
        expected = (input * 2 * torch.arange(3) + torch.arange(3) + 3).sum()
        torch.testing.assert_close(loaded(input), expected)
        assert loaded.task == "custom"

    def test_unregistered_subclass_inherits_task_and_executes(
        self, tmp_path: Path, repository_model: Any, repository_processor: Any
    ) -> None:
        class SumPipeline(RepositoryScorePipeline):
            def postprocess(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
                return output.sum()

        workflow = SumPipeline(model=repository_model, processor=repository_processor)
        input = torch.tensor([1.0, 2.0, 3.0])
        expected = (input * 2 * torch.arange(3) + torch.arange(3) + 3).sum()
        assert workflow.task == "score"
        torch.testing.assert_close(workflow(input), expected)
        with pytest.raises(KeyError):
            workflow.save(tmp_path)
