"""Behavior of ofoldx.models.grnade.pipeline."""

from __future__ import annotations

import pytest
import torch

from ofoldx.data.molecular import ComponentRef
from ofoldx.data.tokenization import TOKEN_INDEX, TOKENS
from ofoldx.models.design import PositionSelection
from ofoldx.models.grnade import (
    GrnadePipeline,
    GrnadeProcessor,
    GrnadeV032Pipeline,
    GrnadeV032Processor,
)


def adenine_bias(rows: int) -> torch.Tensor:
    """A bias that makes every sampled class ``RNA_A`` (the fixture library holds no other RNA component)."""
    bias = torch.zeros(rows, len(TOKENS), dtype=torch.float64)
    bias[:, TOKEN_INDEX["RNA_A"]] = 1000.0
    return bias


@pytest.mark.parametrize(
    "pipeline_type,processor_type", [(GrnadePipeline, GrnadeProcessor), (GrnadeV032Pipeline, GrnadeV032Processor)]
)
def test_design_retains_source(rna, make_model, library, pipeline_type, processor_type):
    structure = rna(3)
    processor = processor_type(dtype=torch.float64)
    edges = processor.prepare_input(structure).features.edge_scalar_features.shape[-1]
    pipeline = pipeline_type(model=make_model(edges), processor=processor)
    selection = PositionSelection(components=(ComponentRef("A", 1),))
    result = pipeline.design(structure, design=selection, residue_logit_bias=adenine_bias(3), seed=5, library=library)
    record = result.records[0]
    assert [position.basis for position in record.positions] == ["retained_unchanged"]
    assert result.candidates[0].scores["sequence_recovery"].value == 1.0
    assert result.candidates[0].scores["perplexity"].scope == frozenset({ComponentRef("A", 1)})
