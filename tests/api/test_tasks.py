from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from ofoldx import pipeline
from ofoldx.api.tasks import read_input
from ofoldx.io import save_structure
from tests.pipelines.test_paired import RepositoryPairedPipeline


class TestReadInput:
    def test_structure(self, peptide) -> None:
        structure = peptide("ALA")

        assert read_input(structure) is structure

    def test_native_artifact(self, tmp_path: Path, peptide) -> None:
        structure = peptide("ALA", "GLY")
        save_structure(structure, tmp_path / "sample.safetensors")

        assert read_input(tmp_path / "sample.safetensors").digest() == structure.digest()


class TestPipeline:
    def test_local_repository(self, tmp_path: Path, repository_model: Any, repository_processor: Any) -> None:
        RepositoryPairedPipeline(model=repository_model, processor=repository_processor).save(tmp_path)
        loaded = pipeline(
            "fold",
            model=tmp_path,
            token="token",
            device="cpu",
            dtype=torch.float64,
            backend="eager",
        )

        input = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        expected = input * 2 * torch.arange(3, dtype=torch.float64) + torch.arange(3, dtype=torch.float64) + 3
        torch.testing.assert_close(loaded(input), expected)
        assert loaded.source is not None
        assert loaded.source.path == tmp_path

    def test_hub_revision(
        self,
        tmp_path: Path,
        repository_model: Any,
        repository_processor: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        first = tmp_path / ("a" * 40)
        second = tmp_path / ("b" * 40)
        RepositoryPairedPipeline(model=repository_model, processor=repository_processor).save(first)
        changed_model = type(repository_model)(type(repository_model.config)(dict(repository_model.config)))
        with torch.no_grad():
            changed_model.bias.add_(100)
        RepositoryPairedPipeline(model=changed_model, processor=repository_processor).save(second)
        main_downloads = 0

        def download(*, revision: str | None = None, **kwargs: Any) -> str:
            nonlocal main_downloads
            if revision == first.name:
                return str(first)
            if revision == second.name:
                return str(second)
            main_downloads += 1
            return str(first if main_downloads == 1 else second)

        monkeypatch.setattr("huggingface_hub.snapshot_download", download)
        loaded = pipeline("fold", model="owner/model", revision="main")

        input = torch.tensor([1.0, 2.0, 3.0])
        expected = input * 2 * torch.arange(3) + torch.arange(3) + 3
        torch.testing.assert_close(loaded(input), expected)
        assert loaded.source is not None
        assert loaded.source.commit == first.name

    def test_capability(self, tmp_path: Path, repository_model: Any, repository_processor: Any) -> None:
        RepositoryPairedPipeline(model=repository_model, processor=repository_processor).save(tmp_path)

        with pytest.raises(ValueError, match="does not support"):
            pipeline("design", model=tmp_path)

    def test_task(self, tmp_path: Path, repository_model: Any, repository_processor: Any) -> None:
        RepositoryPairedPipeline(model=repository_model, processor=repository_processor).save(tmp_path)

        with pytest.raises(ValueError, match="does not support"):
            pipeline(cast(Any, "not_a_task"), model=tmp_path)
