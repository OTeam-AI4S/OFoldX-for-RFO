"""Behavior of ofoldx.models.membranempnn.model."""

from __future__ import annotations

import pytest
import torch

from ofoldx.data.molecular import ComponentRef
from ofoldx.errors import InvalidRequestError
from ofoldx.models.conditions import TiedResidueIdentities

RANKS = torch.arange(3).unsqueeze(0)


@pytest.fixture
def case(designer, peptide):
    input = designer.processor.prepare_input(peptide("ALA", "GLY", "LYS"), membrane_labels=1).features
    input.residue_mask[0, -1] = False
    return designer.model, input


@pytest.mark.parametrize("payload", [-1, 2**40])
def test_membrane_categories(case, payload):
    model, input = case
    expected = model(input, decoding_ranks=RANKS).logits[input.residue_mask]
    input.membrane_labels[~input.residue_mask] = payload
    actual = model(input, decoding_ranks=RANKS).logits[input.residue_mask]
    torch.testing.assert_close(actual, expected)
    input.membrane_labels[0, 0] = payload
    with pytest.raises(InvalidRequestError, match="membrane labels"):
        model(input, decoding_ranks=RANKS)


def test_labels_required(case):
    model, input = case
    input.membrane_labels = None
    with pytest.raises(InvalidRequestError, match="membrane_labels"):
        model(input, decoding_ranks=RANKS)


@pytest.mark.parametrize("weight", [0.0, -1.0])
def test_degenerate_tied_weight(designer, peptide, weight):
    pair = (ComponentRef("A", 0), ComponentRef("A", 1))
    input = designer.processor.prepare_input(
        peptide("ALA", "GLY"), conditions=(TiedResidueIdentities((pair,)),), tied_weights={pair[0]: weight}
    ).features
    output = designer.model.sample(input, decoding_ranks=torch.zeros(1, 2, dtype=torch.long))
    assert torch.isfinite(output.sampling_probabilities).all()
