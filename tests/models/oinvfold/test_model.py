"""Behavior of ofoldx.models.oinvfold.model."""

from __future__ import annotations

import pytest
import torch

from ofoldx.data.tokenization import TOKEN_INDEX, TOKENS
from ofoldx.models.oinvfold import OInvFoldPolymerConfig, OInvFoldPolymerModel, OInvFoldPolymerProcessor


def test_logits_on_released_columns(polymer_model, peptide):
    input = OInvFoldPolymerProcessor(dtype=torch.float64).prepare_input(peptide("ALA", "GLY", "ALA")).features
    with torch.no_grad():
        logits = polymer_model(input, mask_probability=0.0).logits
    assert logits.shape == (3, len(TOKENS))
    assert torch.equal(torch.isinf(logits), ~polymer_model.trained_classes.expand(3, -1))


@pytest.mark.parametrize(("modality", "shared"), [("protein", "RNA"), ("dna", "DNA"), ("rna", "RNA")])
def test_trained_classes_per_modality(modality, shared):
    config = OInvFoldPolymerConfig(modality=modality, hidden_size=8, decoder_hidden_size=8, num_encoder_blocks=1)
    trained = OInvFoldPolymerModel(config).trained_classes
    assert int(trained.sum()) == 25
    assert trained[[TOKEN_INDEX[f"{shared}_{letter}"] for letter in "ACG"]].all()
    assert trained[[TOKEN_INDEX["DNA_T"], TOKEN_INDEX["RNA_U"]]].all()
