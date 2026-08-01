"""Behavior of ofoldx.models.oinvfold.objective."""

from __future__ import annotations

import pytest
import torch

from ofoldx.data.tokenization import TOKENS
from ofoldx.errors import UnsupportedError
from ofoldx.models.oinvfold import OInvFoldLoss, OInvFoldOutput, OInvFoldPolymerProcessor


@pytest.fixture
def target(peptide):
    structure = peptide("ALA", "GLY", "ALA")
    prepared, target = OInvFoldPolymerProcessor().prepare(structure, target=structure)
    mask = target.mask.clone()
    mask[0] = False
    return type(target)(target.tokens, mask)


def test_designed_rows_scored(target, polymer_model):
    trained = polymer_model.trained_classes
    logits = torch.linspace(-1.0, 1.0, len(TOKENS)).expand(3, -1).masked_fill(~trained, -torch.inf)
    loss = OInvFoldLoss()(OInvFoldOutput(logits=logits), target)
    picked = logits.log_softmax(-1)[[1, 2], target.tokens[1:]]
    torch.testing.assert_close(loss, -picked.mean())


def test_unpredicted_class_refused(target):
    logits = torch.zeros(3, len(TOKENS))
    logits[1, target.tokens[1]] = -torch.inf
    with pytest.raises(UnsupportedError, match="model_capacity"):
        OInvFoldLoss()(OInvFoldOutput(logits=logits), target)
