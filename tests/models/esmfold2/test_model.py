"""Behavior of ofoldx.models.esmfold2.model."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from ofoldx.auto import AutoModel, AutoPipelineForFold
from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.errors import InvalidRequestError, UnsupportedError
from ofoldx.models.esmfold2 import (
    EsmFold2ExperimentalModel,
    EsmFold2ExperimentalProcessor,
    EsmFold2ExperimentalProcessorConfig,
    EsmFold2Model,
    EsmFold2Processor,
    EsmFold2ProcessorConfig,
)
from ofoldx.models.esmfold2.model import EsmFold2AtomEncoder, EsmFold2DiffusionStructureHead


class TestReferencedRepository:
    def test_roundtrip(self, referenced_repository, floating_dtype, peptide):
        directory, reference = referenced_repository
        complete = reference.model.state_dict()
        stored = load_file(directory / "model.safetensors")
        external = {name for name in complete if name.startswith("esmc.")}
        assert external and set(stored) == complete.keys() - external
        loaded = AutoPipelineForFold.from_pretrained(directory, dtype=floating_dtype, backend="eager")
        assert all(parameter.dtype == floating_dtype for parameter in loaded.model.parameters())
        assert not any(parameter.requires_grad for parameter in loaded.model.esmc.parameters())
        reference.model.to(dtype=floating_dtype)
        torch.testing.assert_close(loaded.model.state_dict(), reference.model.state_dict(), rtol=0, atol=0)
        prepared = loaded.processor.prepare_input(peptide("GLY", "GLY"))
        options = {"num_samples": 1, "num_sampling_steps": 1, "num_recycles": 1}
        with torch.no_grad():
            expected = reference.model.sample_with_confidence(
                prepared.features, **options, generator=torch.Generator().manual_seed(771)
            )
            actual = loaded.model.sample_with_confidence(
                prepared.features, **options, generator=torch.Generator().manual_seed(771)
            )
        torch.testing.assert_close(actual.coordinates, expected.coordinates, rtol=2e-5, atol=2e-5)

    @pytest.mark.parametrize("damage", ["missing", "unexpected", "external"])
    def test_invalid_head(self, referenced_repository, damage):
        directory, reference = referenced_repository
        path = directory / "model.safetensors"
        state = {name: value.clone() for name, value in load_file(path).items()}
        if damage == "missing":
            state.pop(next(iter(state)))
        elif damage == "unexpected":
            state["unrecognized.weight"] = torch.zeros(1)
        else:
            name = next(name for name in reference.model.state_dict() if name.startswith("esmc."))
            state[name] = reference.model.state_dict()[name].clone()
        save_file(state, path)
        with pytest.raises(RuntimeError, match="folding-head parameters"):
            AutoModel.from_pretrained(directory)

    @pytest.mark.parametrize("damage", ["missing", "architecture"])
    def test_invalid_backbone(self, referenced_repository, damage):
        from multimolecule import EsmCModel

        directory, reference = referenced_repository
        source = Path(reference.model.config.esmc_repository)
        if damage == "missing":
            path = source / "model.safetensors"
            state = {name: value.clone() for name, value in load_file(path).items()}
            state.pop("encoder.layer.0.attention.self.query.weight")
            save_file(state, path)
        else:
            config = deepcopy(reference.model.esmc.config)
            config.num_hidden_layers += 1
            EsmCModel(config, add_pooling_layer=False).save_pretrained(source)
        with pytest.raises(RuntimeError):
            AutoModel.from_pretrained(directory)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_compiled_sampling(device, esm_config):
    from torch._dynamo.testing import CompileCounterWithBackend

    torch.manual_seed(23)
    config = esm_config(False)
    head = EsmFold2DiffusionStructureHead(config, backend="eager").to(device).eval()
    reference = deepcopy(head)
    counter = CompileCounterWithBackend("aot_eager")
    if device == "cuda":
        head.denoiser.compile(fullgraph=True, mode="reduce-overhead")
    else:
        head.denoiser.compile(backend=counter, fullgraph=True)
    inputs = {
        "pair": torch.randn(1, 2, 2, config.pair_size, device=device),
        "single_inputs": torch.randn(1, 2, config.input_embeddings.input_size, device=device),
        "relative_position_encoding": torch.randn(1, 2, 2, config.pair_size, device=device),
        "reference_positions": torch.randn(1, 4, 3, device=device),
        "reference_charges": torch.zeros(1, 4, device=device),
        "reference_element_features": torch.zeros(1, 4, 128, device=device),
        "reference_atom_name_character_features": torch.zeros(1, 4, 4, 64, device=device),
        "reference_space_id": torch.zeros(1, 4, dtype=torch.long, device=device),
        "atom_to_token": torch.tensor([[0, 0, 1, 1]], device=device),
    }
    options = {"num_sampling_steps": 3, "return_atom_representations": True}
    atol = 1e-4 if device == "cuda" else 2e-5
    eager_generator = torch.Generator(device=device).manual_seed(771)
    compiled_generator = torch.Generator(device=device).manual_seed(771)
    with torch.no_grad():
        expected = reference.sample(**inputs, **options, num_samples=2, generator=eager_generator)
        actual = head.sample(**inputs, **options, num_samples=2, generator=compiled_generator)
        retained = {name: value.clone() for name, value in actual.items()}
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=2e-5, atol=atol)
        changed = {
            **inputs,
            "pair": inputs["pair"].flip(-1),
            "reference_positions": inputs["reference_positions"] + 0.5,
        }
        for num_samples in (2, 1):
            expected_next = reference.sample(**changed, **options, num_samples=num_samples, generator=eager_generator)
            actual_next = head.sample(**changed, **options, num_samples=num_samples, generator=compiled_generator)
            for name in expected_next:
                torch.testing.assert_close(actual_next[name], expected_next[name], rtol=2e-5, atol=atol)
            for name in actual:
                torch.testing.assert_close(actual[name], retained[name], rtol=0, atol=0)
    assert torch.equal(compiled_generator.get_state(), eager_generator.get_state())
    if device == "cpu":
        assert counter.frame_count > 0


@pytest.mark.parametrize("value,error", [(float("nan"), InvalidRequestError), (1e100, UnsupportedError)])
def test_raw_reference_diagnostics(value, error):
    model = EsmFold2AtomEncoder(
        atom_hidden_size=8,
        token_hidden_size=8,
        num_blocks=0,
        num_attention_heads=1,
        num_spatial_rope_pairs_per_axis=0,
        num_reference_space_rope_pairs=0,
    )
    with pytest.raises(error):
        model.encode_reference(
            torch.full((1, 1, 3), value, dtype=torch.float64),
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1),
            torch.zeros(1, 1, 128),
            torch.zeros(1, 1, 4, 64),
            torch.zeros(1, 1, dtype=torch.long),
        )


@pytest.mark.parametrize("experimental", [False, True])
def test_repository_roundtrip(experimental, tmp_path, peptide, esm_config, native_repository_roundtrip):
    torch.manual_seed(361)
    model_type = EsmFold2ExperimentalModel if experimental else EsmFold2Model
    config_type = EsmFold2ExperimentalProcessorConfig if experimental else EsmFold2ProcessorConfig
    processor_type = EsmFold2ExperimentalProcessor if experimental else EsmFold2Processor
    model = model_type(esm_config(experimental)).double().eval().set_backend("eager")
    processor = processor_type(config_type(msa_conditioning=False), dtype=torch.float64)
    structure = peptide("GLY", "GLY")
    prepared = processor.prepare_input(structure)
    loaded, loaded_processor = native_repository_roundtrip(model, processor, tmp_path / "native")
    restored = loaded_processor.prepare_input(structure)
    assert prepared.features.residue_type_ids.tolist() == [[TOKEN_INDEX["G"]] * 2]
    with torch.no_grad():
        actual = model.sample_with_confidence(
            prepared.features,
            num_samples=3,
            num_sampling_steps=1,
            num_recycles=1,
            generator=torch.Generator().manual_seed(771),
        )
        reloaded = loaded.sample_with_confidence(
            restored.features,
            num_samples=3,
            num_sampling_steps=1,
            num_recycles=1,
            generator=torch.Generator().manual_seed(771),
        )
    actual_sample = actual.sample if experimental else actual
    reloaded_sample = reloaded.sample if experimental else reloaded
    torch.testing.assert_close(actual_sample.coordinates, reloaded_sample.coordinates, rtol=2e-10, atol=2e-10)
    assert torch.isfinite(actual_sample.coordinates).all()
    confidence = actual.confidence if experimental else actual
    restored_confidence = reloaded.confidence if experimental else reloaded
    torch.testing.assert_close(confidence.chain_pair_mean_tm, restored_confidence.chain_pair_mean_tm)
    bins = confidence.pae_logits.shape[-1]
    centers = torch.arange(bins, dtype=torch.float64).add(0.5) * (32 / bins)
    scale = 1.24 * (19 - 15) ** (1 / 3) - 1.8
    expected = (confidence.pae_logits.softmax(-1) / (1 + (centers / scale).square())).sum(-1).mean((-2, -1))
    torch.testing.assert_close(confidence.chain_pair_mean_tm[..., 0, 0], expected)
