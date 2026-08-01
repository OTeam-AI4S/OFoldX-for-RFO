"""Outer-product dispatch selects unavailable inputs before provider math."""

import torch

from ofoldx.modules.backend import outer_product_mean


def test_provider_input_availability(monkeypatch):
    import importlib

    owner = importlib.import_module("ofoldx.modules.backend.outer_product_mean")

    def provider(msa, mask, **kwargs):
        assert torch.isfinite(msa).all()
        assert torch.equal(msa[~mask], torch.zeros_like(msa[~mask]))
        return msa.sum(dim=-3)

    monkeypatch.setattr(owner, "_oops_outer_product_mean", provider)
    monkeypatch.setattr(owner, "resolve_tensor_backend", lambda *args, **kwargs: "oops")
    msa = torch.tensor([[[[1.0, 2.0], [torch.nan, torch.inf]]]], requires_grad=True)
    mask = torch.tensor([[[True, False]]])
    result = outer_product_mean(
        msa,
        mask,
        msa_norm_weight=torch.ones(2),
        msa_norm_bias=torch.zeros(2),
        left_weight=torch.ones(2, 2),
        right_weight=torch.ones(2, 2),
        output_weight=torch.ones(2, 4),
        output_bias=torch.zeros(2),
        msa_norm_eps=1e-5,
        denominator_eps=0.0,
        divide_before_projection=True,
        clamp_denominator=False,
        backend="oops",
    )
    gradient = torch.autograd.grad(result.sum(), msa)[0]
    assert torch.isfinite(gradient).all()
    assert torch.equal(gradient[~mask], torch.zeros_like(gradient[~mask]))
