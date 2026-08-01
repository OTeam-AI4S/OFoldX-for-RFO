"""Behavior of ofoldx.models.alphafold3.pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data.molecular import ComponentRef
from ofoldx.data.structure import CoordinateEvidence, replace_coordinates
from ofoldx.models.alphafold3 import (
    AlphaFold3Config,
    AlphaFold3Model,
    AlphaFold3Pipeline,
    AlphaFold3Processor,
    alphafold3_has_clash,
)
from ofoldx.models.alphafold3.model import AlphaFold3SamplingOutput


def test_decoded_predictions(peptide):
    processor = AlphaFold3Processor(dtype=torch.float64)
    prepared = processor.prepare_input(peptide("ALA", "GLY"))
    atoms, tokens, samples = len(prepared.context.slots.atom), len(prepared.context.slots.layout.token_component), 2
    coordinates = torch.arange(samples * atoms * 3, dtype=torch.float64).reshape(1, samples, atoms, 3)
    output = AlphaFold3SamplingOutput(  # type: ignore[call-arg]
        single=torch.zeros(1, tokens, 384, dtype=torch.float64),
        pair=torch.zeros(1, tokens, tokens, 128, dtype=torch.float64),
        coordinates=coordinates,
        distogram_logits=torch.zeros(1, tokens, tokens, 64, dtype=torch.float64),
        plddt_logits=torch.zeros(1, samples, atoms, 50, dtype=torch.float64),
        pae_logits=torch.zeros(1, samples, tokens, tokens, 64, dtype=torch.float64),
        pde_logits=torch.zeros(1, samples, tokens, tokens, 64, dtype=torch.float64),
        resolved_logits=torch.zeros(1, samples, atoms, 2, dtype=torch.float64),
        batch_size=[1],
    )
    with torch.device("meta"):
        model = AlphaFold3Model(AlphaFold3Config(trunk={"num_pairformer_blocks": 0}, template={"num_blocks": 0}))
    result = AlphaFold3Pipeline(model=model, processor=processor).postprocess(prepared, output)
    structure = result.predictions[1].structure
    np.testing.assert_array_equal(structure.coordinates[prepared.context.slots.atom], coordinates[0, 1].numpy())
    assert (structure.coordinate_evidence[prepared.context.slots.atom] == CoordinateEvidence.PREDICTED).all()
    assert result.predictions[0].pair_metrics["pae"].row_axis == (ComponentRef("A", 0), ComponentRef("A", 1))


class TestHasClash:
    @pytest.mark.parametrize("collapsed", [False, True])
    def test_collapsed_chain(self, peptide, collapsed):
        structure = peptide("ALA", "GLY", "ALA")
        if collapsed:
            every = np.ones(len(structure.coordinates), dtype=bool)
            structure, _ = replace_coordinates(
                structure,
                np.flatnonzero(every),
                np.zeros((len(every), 3)),
                scope=every,
                evidence=CoordinateEvidence.PREDICTED,
            )
        assert alphafold3_has_clash(structure) is collapsed
