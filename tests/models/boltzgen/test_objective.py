"""Public BoltzGen objectives ignore unavailable numerical payloads."""

from types import SimpleNamespace

import torch

from ofoldx.models.boltzgen.objective import b_factor_loss, distogram_loss, pde_loss


def test_pde_masked_payload():
    input = SimpleNamespace(
        atom_mask=torch.ones(1, 3, dtype=torch.bool),
        token_mask=torch.ones(1, 3, dtype=torch.bool),
        token_center_atom_index=torch.tensor([[0, 1, 2]]),
    )
    mask = torch.tensor([[[True, True, False]]])
    truth = torch.tensor([[[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]]])
    prediction = torch.tensor([[[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]]], requires_grad=True)
    logits = torch.zeros(1, 1, 3, 3, 4)
    logits[:, :, 2, :] = float("nan")
    logits[:, :, :, 2] = float("nan")
    logits.requires_grad_()
    loss = pde_loss(logits, prediction, truth, mask, input)
    (gradient,) = torch.autograd.grad(loss.sum(), logits)
    expected = pde_loss(torch.zeros_like(logits), prediction.detach().nan_to_num(), truth.nan_to_num(), mask, input)
    torch.testing.assert_close(loss, expected)
    assert bool(torch.isfinite(gradient).all())
    assert not bool(gradient[:, :, 2].any())
    assert not bool(gradient[:, :, :, 2].any())


def test_unknown_b_factor():
    logits = torch.zeros(1, 2, 4)
    logits[0, 1] = float("nan")
    logits.requires_grad_()
    measured = torch.tensor([[True, False]])
    loss = b_factor_loss(logits, torch.tensor([[10.0, float("nan")]]), measured, torch.ones_like(measured))
    (gradient,) = torch.autograd.grad(loss.sum(), logits)
    torch.testing.assert_close(
        loss, b_factor_loss(torch.zeros_like(logits), torch.tensor([[10.0, 0.0]]), measured, torch.ones_like(measured))
    )
    assert bool(torch.isfinite(gradient).all())
    assert not bool(gradient[0, 1].any())


def test_distogram_masked_payload():
    coordinates = torch.tensor([[[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [float("inf"), 0.0, 0.0]]]])
    mask = torch.tensor([[True, True, False]])
    logits = torch.zeros(1, 3, 3, 1, 4)
    logits[:, 2] = float("nan")
    logits[:, :, 2] = float("nan")
    logits.requires_grad_()
    loss = distogram_loss(logits, coordinates, mask)
    (gradient,) = torch.autograd.grad(loss.sum(), logits)
    torch.testing.assert_close(loss, distogram_loss(torch.zeros_like(logits), coordinates.nan_to_num(), mask))
    assert bool(torch.isfinite(gradient).all())
    assert not bool(gradient[:, 2].any())
    assert not bool(gradient[:, :, 2].any())
