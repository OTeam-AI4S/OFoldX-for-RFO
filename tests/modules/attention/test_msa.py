"""Public MSA column availability before normalization."""

import torch

from ofoldx.modules.attention.msa import MsaColumnGlobalAttention


def test_global_column_unavailable_gradients():
    module = MsaColumnGlobalAttention(msa_size=4, num_attention_heads=1, backend="eager").double()
    values = torch.randn(1, 2, 2, 4, dtype=torch.float64)
    mask = torch.tensor([[[True, True], [False, True]]])
    values[~mask] = torch.nan
    values.requires_grad_()
    output = module(values, msa_mask=mask)
    assert torch.isfinite(output).all()
    gradients = torch.autograd.grad(output.square().sum(), (values, *module.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.equal(gradients[0][~mask], torch.zeros_like(gradients[0][~mask]))
