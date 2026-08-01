"""Behavior of ofoldx.models.odesign.objective."""

from __future__ import annotations

import pytest
import torch

from ofoldx.data.molecular import AtomSite, ComponentRef
from ofoldx.models.design import PositionSelection
from ofoldx.models.odesign import ODesignProcessor
from ofoldx.models.odesign.model import ODesignOutput
from ofoldx.models.odesign.objective import odesign_loss, weighted_denoising_loss


@pytest.fixture
def task(peptide):
    processor = ODesignProcessor()
    prepared, truth = processor.prepare(
        peptide("ALA", "GLY"),
        target=peptide("ALA", "GLY"),
        design=PositionSelection(components=(ComponentRef("A", 0),)),
        fixed_atoms=(AtomSite("A", 1, "CA"),),
    )
    return prepared.features, truth


def displaced(input, target) -> ODesignOutput:
    r"""One denoising output half an ångström off the truth, with flat pair logits."""
    tokens = input.token_mask.shape[-1]
    return ODesignOutput(
        coordinates=(target.coordinates + 0.5)[:, None].clone().requires_grad_(),
        distogram_logits=torch.zeros(1, tokens, tokens, 4),
        token_bond_logits=torch.zeros(1, tokens, tokens, 2),
        noise_level=torch.ones(1, 1),
        batch_size=[1],
    )


class TestWeightedDenoisingLoss:
    def test_fixed_atoms_unscored(self, task):
        input, target = task
        output = displaced(input, target)
        loss = weighted_denoising_loss(output, input, target)
        (gradient,) = torch.autograd.grad(loss, output.coordinates)
        fixed = input.fixed_coordinate_mask[0]
        assert bool(torch.isfinite(loss))
        assert not bool(gradient[0, 0][fixed].any())
        assert bool(gradient[0, 0][~fixed].any())


class TestOdesignLoss:
    def test_inapplicable_terms_nan(self, task):
        input, target = task
        total, losses = odesign_loss(displaced(input, target), input, target, num_distogram_bins=4)
        assert bool(torch.isfinite(total))
        assert bool(torch.isnan(losses["covalent_bond_distance_loss"]))
        assert bool(torch.isnan(losses["token_bond_loss"]))
