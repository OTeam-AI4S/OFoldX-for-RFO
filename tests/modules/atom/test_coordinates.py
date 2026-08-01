from __future__ import annotations

import pytest
import torch
from danling.tensors import NestedTensor

from ofoldx.modules.atom.coordinates import gather_observed_coordinates, project_coordinates


def test_unavailable_payload_is_ignored():
    values = torch.tensor([[[1.0, 2.0, 3.0], [float("nan"), float("inf"), -float("inf")]]], requires_grad=True)
    observed = torch.tensor([[True, False]])
    result = project_coordinates(values, observed, dtype=torch.float64)
    torch.testing.assert_close(result, torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]], dtype=torch.float64))
    gradient = torch.autograd.grad(result.sum(), values)[0]
    torch.testing.assert_close(gradient, observed.unsqueeze(-1).expand_as(values).float())


def test_coordinate_failures_are_distinct():
    with pytest.raises(ValueError, match="finite"):
        project_coordinates(torch.tensor([[float("nan"), 0.0, 0.0]]), torch.tensor([True]))
    with pytest.raises(OverflowError, match="dtype range"):
        project_coordinates(
            torch.tensor([[1e100, 0.0, 0.0]], dtype=torch.float64), torch.tensor([True]), dtype=torch.float32
        )


def test_nested_projection_preserves_gradients():
    parts = [
        torch.tensor([[1.0, 2.0, 3.0], [float("nan"), 0.0, 0.0]], requires_grad=True),
        torch.empty(0, 3, requires_grad=True),
    ]
    values = NestedTensor(parts, ragged_dims=(0,))
    observed = NestedTensor([torch.tensor([True, False]), torch.empty(0, dtype=torch.bool)], ragged_dims=(0,))
    result = project_coordinates(values, observed)
    gradients = torch.autograd.grad(result.concat.sum(), parts)
    torch.testing.assert_close(result.element_sizes(), values.element_sizes())
    torch.testing.assert_close(gradients[0], torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
    torch.testing.assert_close(gradients[1], torch.empty(0, 3))


def test_empty_gather_validates_correspondence():
    coordinates = torch.empty(1, 0, 3, requires_grad=True)
    mask = torch.empty(1, 0, dtype=torch.bool)
    indices = torch.tensor([[-1, -1]])
    result, observed = gather_observed_coordinates(coordinates, mask, indices, torch.ones(1, 2, dtype=torch.bool))
    assert not observed.any()
    torch.testing.assert_close(result, torch.zeros(1, 2, 3))
    assert torch.autograd.grad(result.sum(), coordinates)[0].shape == coordinates.shape
    with pytest.raises(ValueError, match="source rows"):
        gather_observed_coordinates(coordinates, mask, torch.tensor([[0]]), torch.zeros(1, 1, dtype=torch.bool))
