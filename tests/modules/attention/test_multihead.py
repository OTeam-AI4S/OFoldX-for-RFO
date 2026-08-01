from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from danling.tensors import NestedTensor

from ofoldx.modules.attention import (
    GatedMultiheadAttentionProjection,
    MultiheadAttention,
    MultiheadAttentionProjection,
    ProjectedAttention,
)
from ofoldx.modules.attention.multihead import GlobalAttention
from ofoldx.modules.backend import attention as attention_backend


class TestProjectedAttention:
    @pytest.mark.parametrize("backend", ["eager", "sdpa"])
    def test_mask_layouts(self, backend: str) -> None:
        attention = ProjectedAttention(backend=backend)
        query = torch.zeros(2, 1, 2, 1)
        key = torch.zeros(2, 1, 3, 1)
        value = torch.tensor(
            [
                [[[-1.0], [2.0], [3.0]]],
                [[[-10.0], [20.0], [30.0]]],
            ]
        )

        pair_output = attention(
            query,
            key,
            value,
            attention_mask=torch.tensor([[True, False, False], [False, True, False]]),
        )
        key_output = attention(
            query,
            key,
            value,
            attention_mask=torch.tensor(
                [
                    [[True, False, False]],
                    [[False, False, True]],
                ]
            ),
        )

        torch.testing.assert_close(
            pair_output,
            torch.tensor(
                [
                    [[[-1.0], [2.0]]],
                    [[[-10.0], [20.0]]],
                ]
            ),
        )
        torch.testing.assert_close(
            key_output,
            torch.tensor(
                [
                    [[[-1.0], [-1.0]]],
                    [[[30.0], [30.0]]],
                ]
            ),
        )

    @pytest.mark.parametrize("backend", ["eager", "sdpa"])
    def test_empty_row(self, backend: str) -> None:
        attention = ProjectedAttention(backend=backend)
        query = torch.zeros(1, 1, 2, 1)
        key = torch.zeros(1, 1, 3, 1)
        value = torch.tensor([[[[1.0], [2.0], [3.0]]]])
        mask = torch.tensor([[False, False, False], [True, False, False]])

        output = attention(query, key, value, attention_mask=mask)

        torch.testing.assert_close(output[..., 0, :], torch.zeros_like(output[..., 0, :]))
        torch.testing.assert_close(output[..., 1, :], torch.ones_like(output[..., 1, :]))

    @pytest.mark.parametrize("backend", ["eager", "sdpa"])
    def test_head_mask(self, backend: str) -> None:
        attention = ProjectedAttention(backend=backend)
        query = torch.zeros(1, 2, 3, 1)
        key = torch.zeros(1, 2, 2, 1)
        value = torch.tensor([[[[1.0], [2.0]], [[10.0], [20.0]]]])
        mask = torch.tensor([[[True, False], [False, True]]])

        output = attention(query, key, value, attention_mask=mask)

        expected = torch.tensor([[[[1.0], [1.0], [1.0]], [[20.0], [20.0], [20.0]]]])
        torch.testing.assert_close(output, expected)

    @pytest.mark.parametrize("backend", ["eager", "sdpa"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_mask_precedes_bias(self, backend: str, dtype: torch.dtype) -> None:
        attention = ProjectedAttention(backend=backend)
        value = torch.tensor([[[[1.0], [9.0]]]], dtype=dtype, requires_grad=True)
        query = torch.zeros(1, 1, 2, 1, dtype=dtype, requires_grad=True)
        key = torch.zeros(1, 1, 2, 1, dtype=dtype, requires_grad=True)
        bias = torch.tensor([[[[0.0, torch.finfo(dtype).max]]]], dtype=dtype, requires_grad=True)
        output = attention(
            query,
            key,
            value,
            biases=(bias,),
            attention_mask=torch.tensor([[True, False], [False, False]]),
        )

        torch.testing.assert_close(output, torch.tensor([[[[1.0], [0.0]]]], dtype=dtype))
        output.sum().backward()
        torch.testing.assert_close(value.grad, torch.tensor([[[[1.0], [0.0]]]], dtype=dtype))
        for operand in (query, key, bias):
            torch.testing.assert_close(operand.grad, torch.zeros_like(operand), atol=0, rtol=0)

    @pytest.mark.parametrize("backend", ["eager", "sdpa"])
    def test_shared_head_bias_batch_collision(self, backend: str) -> None:
        torch.manual_seed(93)
        operands = [
            torch.randn(*shape, dtype=torch.float64, requires_grad=True)
            for shape in ((2, 2, 2, 4), (2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3))
        ]
        reference_operands = [operand.detach().clone().requires_grad_() for operand in operands]
        query, key, value, bias = operands
        reference_query, reference_key, reference_value, reference_bias = reference_operands

        actual = ProjectedAttention(backend=backend)(query, key, value, biases=(bias,))
        expected = torch.nn.functional.scaled_dot_product_attention(
            reference_query, reference_key, reference_value, attn_mask=reference_bias
        )

        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        actual_grads = torch.autograd.grad(actual.square().sum(), operands)
        reference_grads = torch.autograd.grad(expected.square().sum(), reference_operands)
        for actual_grad, reference_grad in zip(actual_grads, reference_grads):
            torch.testing.assert_close(actual_grad, reference_grad, atol=1e-12, rtol=1e-12)

    def test_oops_adapter_suffix_bias_axes(self, monkeypatch) -> None:
        def native_attention(query, key, value, *, mask, pair_bias, scale):
            assert query.dim() == 4
            assert mask is None
            return torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=pair_bias, scale=scale)

        monkeypatch.setattr(attention_backend, "_oops_pair_bias_attention", native_attention)
        query = torch.randn(2, 3, 4, 2, 2, 4, dtype=torch.float64, requires_grad=True)
        key = torch.randn(2, 3, 4, 2, 3, 4, dtype=torch.float64, requires_grad=True)
        value = torch.randn_like(key, requires_grad=True)
        bias = torch.randn(3, 4, 2, 2, 3, dtype=torch.float64, requires_grad=True)
        operands = (query, key, value, bias)
        reference_operands = tuple(operand.detach().clone().requires_grad_() for operand in operands)

        actual = ProjectedAttention(backend="oops")(query, key, value, biases=(bias,))
        expected = torch.nn.functional.scaled_dot_product_attention(
            *reference_operands[:3], attn_mask=reference_operands[3]
        )

        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        actual_grads = torch.autograd.grad(actual.square().sum(), operands)
        reference_grads = torch.autograd.grad(expected.square().sum(), reference_operands)
        for actual_grad, reference_grad in zip(actual_grads, reference_grads):
            torch.testing.assert_close(actual_grad, reference_grad, atol=1e-12, rtol=1e-12)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    def test_eager_autocast_finite(self, dtype: torch.dtype) -> None:
        query = torch.full((1, 1, 2, 2), 300.0, dtype=dtype, requires_grad=True)
        key = torch.tensor([[[[300.0, 300.0], [0.0, 0.0]]]], dtype=dtype, requires_grad=True)
        value = torch.tensor([[[[1.0, 1.0], [9.0, 9.0]]]], dtype=dtype, requires_grad=True)

        with torch.autocast("cpu", dtype=torch.float16):
            output = ProjectedAttention(backend="eager")(query, key, value)

        torch.testing.assert_close(output, torch.ones_like(output), atol=0, rtol=0)
        assert output.dtype == dtype
        output.sum().backward()
        assert all(torch.isfinite(operand.grad).all() for operand in (query, key, value))

    def test_eager_float64_values_gradients(self) -> None:
        query = torch.zeros(1, 1, 1, 2, dtype=torch.float64)
        key = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
        value = torch.tensor([[[[1.0 + 1e-8], [1.0 + 3e-8]]]], dtype=torch.float64, requires_grad=True)

        output = ProjectedAttention(backend="eager")(query, key, value)

        torch.testing.assert_close(output, torch.full_like(output, 1.0 + 2e-8), atol=1e-15, rtol=0)
        output.sum().backward()
        torch.testing.assert_close(value.grad, torch.full_like(value, 0.5), atol=0, rtol=0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sdpa_float32_bias_low_precision(self, dtype: torch.dtype) -> None:
        query = torch.zeros(1, 1, 2, 2, dtype=dtype, requires_grad=True)
        key = torch.zeros(1, 1, 2, 2, dtype=dtype, requires_grad=True)
        value = torch.tensor([[[[1.0, 1.0], [9.0, 9.0]]]], dtype=dtype, requires_grad=True)
        bias = torch.tensor([[[[1e5, 0.0], [1e5, 0.0]]]], requires_grad=True)
        operands = (query, key, value, bias)
        reference_operands = tuple(operand.detach().clone().requires_grad_() for operand in operands)

        actual = ProjectedAttention(backend="sdpa")(query, key, value, biases=(bias,))
        expected = torch.nn.functional.scaled_dot_product_attention(
            *reference_operands[:3], attn_mask=reference_operands[3]
        )

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual_grads = torch.autograd.grad(actual.sum(), operands)
        reference_grads = torch.autograd.grad(expected.sum(), reference_operands)
        for actual_grad, reference_grad in zip(actual_grads, reference_grads):
            torch.testing.assert_close(actual_grad, reference_grad, atol=0, rtol=0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_native_nested_sdpa_cuda(self, dtype: torch.dtype) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
        from torch.utils._python_dispatch import TorchDispatchMode

        native_calls = []

        class RecordNativeAttention(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func == torch.ops.aten._efficient_attention_forward.default:
                    native_calls.append(
                        (args[0].shape, args[1].shape, args[4].dtype, args[6], args[7], kwargs["scale"])
                    )
                return func(*args, **(kwargs or {}))

        query = NestedTensor([torch.randn(2, length, 16, dtype=dtype) for length in (3, 5)], ragged_dims=(1,))
        key = NestedTensor([torch.randn(2, length, 16, dtype=dtype) for length in (2, 4)], ragged_dims=(1,))
        value = NestedTensor([torch.randn(2, length, 16, dtype=dtype) for length in (2, 4)], ragged_dims=(1,))

        with FakeTensorMode() as mode, RecordNativeAttention():
            fake_query, fake_key, fake_value = (mode.from_tensor(tensor).to("cuda:0") for tensor in (query, key, value))
            output = attention_backend._sdpa_attention(
                fake_query, fake_key, fake_value, biases=None, attention_mask=None, dropout_p=0.1, scale=1.0
            )

        assert native_calls == [(torch.Size([1, 8, 2, 16]), torch.Size([1, 6, 2, 16]), torch.int32, 5, 4, 1.0)]
        assert is_fake(output.concat)
        assert output.device == torch.device("cuda:0")
        assert output.dtype == dtype
        assert output.shape == fake_query.shape
        assert output.concat.shape == (8, 2, 16)
        assert output.ragged_dims == fake_query.ragged_dims == (1,)
        assert output.packed_dim_order == fake_query.packed_dim_order == (1, 0, 2)
        assert output.packed_offsets() is fake_query.packed_offsets()
        assert output.element_sizes() is fake_query.element_sizes()

    def test_nested_cross_attention_gradients(self) -> None:
        generator = torch.Generator().manual_seed(127)
        query_parts = tuple(
            torch.randn(2, length, 3, dtype=torch.float64, generator=generator, requires_grad=True) for length in (2, 4)
        )
        key_parts = tuple(
            torch.randn(2, length, 3, dtype=torch.float64, generator=generator, requires_grad=True) for length in (3, 5)
        )
        value_parts = tuple(
            torch.randn(2, length, 4, dtype=torch.float64, generator=generator, requires_grad=True) for length in (3, 5)
        )
        bias_parts = tuple(
            torch.randn(2, queries, keys, dtype=torch.float64, generator=generator, requires_grad=True)
            for queries, keys in ((2, 3), (4, 5))
        )
        mask_parts = (torch.ones(2, 3, dtype=torch.bool), torch.ones(4, 5, dtype=torch.bool))
        mask_parts[0][0] = False
        mask_parts[1][-1, -1] = False
        query = NestedTensor(query_parts, ragged_dims=(1,))
        key = NestedTensor(key_parts, ragged_dims=(1,))
        value = NestedTensor(value_parts, ragged_dims=(1,))
        bias = NestedTensor(bias_parts, ragged_dims=(1, 2))
        mask = NestedTensor(mask_parts, ragged_dims=(0, 1))
        actual = ProjectedAttention(backend="eager")(query, key, value, biases=(bias,), attention_mask=mask)
        reference_parts = tuple(
            tuple(part.detach().clone().requires_grad_() for part in parts)
            for parts in (query_parts, key_parts, value_parts, bias_parts)
        )
        expected = tuple(
            torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=b.masked_fill(~valid, float("-inf")))
            for q, k, v, b, valid in zip(*reference_parts, mask_parts, strict=True)
        )
        assert actual.ragged_dims == query.ragged_dims
        torch.testing.assert_close(actual.packed_offsets(), query.packed_offsets())
        for part, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(part, reference, rtol=1e-12, atol=1e-12)
        actual_loss = sum(part.square().sum() for part in actual)
        expected_loss = sum(part.square().sum() for part in expected)
        actual_operands = (*query_parts, *key_parts, *value_parts, *bias_parts)
        reference_operands = tuple(part for parts in reference_parts for part in parts)
        gradients = torch.autograd.grad(actual_loss, actual_operands)
        reference_gradients = torch.autograd.grad(expected_loss, reference_operands)
        for gradient, reference in zip(gradients, reference_gradients, strict=True):
            torch.testing.assert_close(gradient, reference, rtol=1e-12, atol=1e-12)


class TestMultiheadAttentionProjection:
    def test_invalid_heads(self) -> None:
        with pytest.raises(ValueError):
            MultiheadAttentionProjection(hidden_size=8, num_attention_heads=3)

    def test_biases(self) -> None:
        projection = GatedMultiheadAttentionProjection(
            hidden_size=8,
            num_attention_heads=2,
            query_bias=False,
            output_bias=True,
            gate_bias=True,
            gate_bias_init=1.0,
        )

        assert projection.query_projection.bias is None
        assert projection.key_projection.bias is None
        assert projection.value_projection.bias is None
        assert projection.output_projection.bias is not None
        assert projection.gate_projection.bias is not None
        torch.testing.assert_close(projection.gate_projection.bias, torch.ones(8))

    def test_head_size(self) -> None:
        projection = MultiheadAttentionProjection(hidden_size=8, num_attention_heads=3, head_size=4)
        hidden = torch.randn(1, 3, 8)

        query, key, value = projection.project(hidden, hidden, hidden)
        output = projection.merge(torch.randn(1, 3, 3, 4), hidden)

        assert query.shape == key.shape == value.shape == (1, 3, 3, 4)
        assert output.shape == hidden.shape


class TestMultiheadAttention:
    def test_invalid_backend(self) -> None:
        with pytest.raises(ValueError):
            MultiheadAttention(MultiheadAttentionProjection(hidden_size=8, num_attention_heads=2), backend="bogus")

    @pytest.mark.parametrize("backend", ["eager", "sdpa"])
    def test_value_bias_attention_dropout(self, backend: str) -> None:
        projection = MultiheadAttentionProjection(
            hidden_size=1,
            num_attention_heads=1,
            query_bias=False,
            output_bias=True,
            value_bias=True,
        )
        with torch.no_grad():
            projection.query_projection.weight.zero_()
            projection.key_projection.weight.zero_()
            projection.value_projection.weight.zero_()
            projection.value_projection.bias.fill_(2)
            projection.output_projection.weight.fill_(3)
            projection.output_projection.bias.fill_(5)
        attention = MultiheadAttention(projection, backend=backend, dropout=0.5)
        hidden = torch.zeros(1, 4, 1)

        torch.manual_seed(91)
        expected_mass = torch.nn.functional.dropout(torch.full((1, 1, 4, 4), 0.25), p=0.5).sum(dim=-1)
        torch.manual_seed(91)
        actual = attention(hidden)
        expected = (6 * expected_mass + 5).transpose(-1, -2)

        torch.testing.assert_close(actual, expected)
        assert not torch.equal(actual, torch.full_like(actual, 11))
        actual.sum().backward()
        torch.testing.assert_close(projection.value_projection.bias.grad, 3 * expected_mass.sum().reshape(1))

        attention.eval()
        torch.testing.assert_close(attention(hidden), torch.full_like(actual, 11))

    def test_eager_projection_autocast(self) -> None:
        attention = MultiheadAttention(
            MultiheadAttentionProjection(hidden_size=4, num_attention_heads=2), backend="eager"
        )
        projected_dtypes = []
        handle = attention.projection.query_projection.register_forward_hook(
            lambda module, args, output: projected_dtypes.append(output.dtype)
        )
        with torch.autocast("cpu", dtype=torch.float16):
            output = attention(torch.randn(2, 3, 4))
            assert torch.is_autocast_enabled("cpu")
        handle.remove()

        assert projected_dtypes == [torch.float16]
        assert output.dtype == torch.float16

    @pytest.mark.parametrize("backend", ["auto", "eager", "oops"])
    def test_nested_dropout_bias_topology(self, backend: str) -> None:
        projection = MultiheadAttentionProjection(
            hidden_size=1, num_attention_heads=1, query_bias=False, value_bias=True, output_bias=True
        )
        with torch.no_grad():
            projection.query_projection.weight.zero_()
            projection.key_projection.weight.zero_()
            projection.value_projection.weight.zero_()
            projection.value_projection.bias.fill_(2)
            projection.output_projection.weight.fill_(3)
            projection.output_projection.bias.fill_(5)
        attention = MultiheadAttention(projection, backend=backend, dropout=0.5)
        hidden = NestedTensor([torch.zeros(length, 1) for length in (2, 4)], ragged_dims=(0,))
        probabilities = torch.cat((torch.full((4, 1), 0.5), torch.full((16, 1), 0.25)))
        torch.manual_seed(91)
        dropped = torch.nn.functional.dropout(probabilities, p=0.5)
        mass = torch.cat((dropped[:4].reshape(2, 2).sum(-1), dropped[4:].reshape(4, 4).sum(-1)))

        torch.manual_seed(91)
        output = attention(hidden)
        output.concat.sum().backward()

        assert output.ragged_dims == (0,)
        assert torch.equal(output.packed_offsets(), hidden.packed_offsets())
        torch.testing.assert_close(output.concat, (6 * mass + 5).unsqueeze(-1))
        torch.testing.assert_close(projection.value_projection.bias.grad, 3 * mass.sum().reshape(1))

    def test_nested_attention_values_gradients(self) -> None:
        attention = (
            MultiheadAttention(
                MultiheadAttentionProjection(hidden_size=8, num_attention_heads=2, value_bias=True, output_bias=True),
                backend="eager",
            )
            .double()
            .eval()
        )
        reference = deepcopy(attention.projection)
        template = NestedTensor([torch.empty(length, 8, dtype=torch.float64) for length in (3, 5)], ragged_dims=(0,))
        values = torch.randn_like(template.concat, requires_grad=True)
        reference_values = values.detach().clone().requires_grad_()
        hidden = template.packed_like(values)
        expected_parts = []
        for start, end in ((0, 3), (3, 8)):
            current = reference_values[start:end]
            query = reference.query_projection(current).unflatten(-1, (2, 4)).transpose(0, 1)
            key = reference.key_projection(current).unflatten(-1, (2, 4)).transpose(0, 1)
            value = reference.value_projection(current).unflatten(-1, (2, 4)).transpose(0, 1)
            attended = torch.nn.functional.scaled_dot_product_attention(query, key, value)
            expected_parts.append(reference.output_projection(attended.transpose(0, 1).flatten(-2)))
        expected = torch.cat(expected_parts)

        actual = attention(hidden).concat
        actual_grads = torch.autograd.grad(actual.square().sum(), (values, *attention.projection.parameters()))
        expected_grads = torch.autograd.grad(expected.square().sum(), (reference_values, *reference.parameters()))

        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)


def test_global_attention_excludes_masked_keys() -> None:
    attention = GlobalAttention(hidden_size=1, num_attention_heads=1, head_size=1, backend="eager").double()
    with torch.no_grad():
        attention.query_projection.weight.fill_(1000.0)
        attention.key_projection.weight.fill_(-1000.0)
        attention.value_projection.weight.fill_(1.0)
        attention.gate_projection.weight.zero_()
        attention.gate_projection.bias.zero_()
        attention.output_projection.weight.fill_(1.0)
        attention.output_projection.bias.zero_()
    query = torch.tensor([[[1.0], [100.0]]], dtype=torch.float64, requires_grad=True)
    output = attention(query, torch.tensor([[[True], [False]]]))
    torch.testing.assert_close(output[0, 0], query.new_tensor([0.5]))
    gradient = torch.autograd.grad(output[0, 0].sum(), query)[0]
    torch.testing.assert_close(gradient, query.new_tensor([[[0.5], [0.0]]]))


def test_global_attention_broadcast_mask() -> None:
    attention = GlobalAttention(hidden_size=6, num_attention_heads=2, head_size=3, backend="eager").double()
    query = torch.randn(2, 7, 6, dtype=torch.float64)
    broadcast_mask = torch.ones(2, 1, 1, dtype=torch.bool)
    full_mask = torch.ones(2, 7, 1, dtype=torch.bool)
    torch.testing.assert_close(attention(query, broadcast_mask), attention(query, full_mask))
    torch.testing.assert_close(attention(query, broadcast_mask), attention(query))


@pytest.mark.parametrize("masked", [False, True])
def test_global_attention_ragged_axes(masked: bool) -> None:
    attention = GlobalAttention(hidden_size=6, num_attention_heads=2, head_size=3, backend="eager").double()
    query = NestedTensor([torch.randn(3, 2, 6), torch.randn(5, 4, 6)], ragged_dims=(0, 1)).double()
    values = query.concat.detach().requires_grad_()
    query = query.packed_like(values)
    mask_parts = [torch.tensor([[[True], [False]]]).expand(3, -1, -1), torch.ones(5, 4, 1, dtype=torch.bool)]
    mask = NestedTensor(mask_parts, ragged_dims=(0, 1)) if masked else None
    actual = attention(query, mask)
    expected = [
        attention(part, validity if masked else None)
        for part, validity in zip(torch.unbind(query), mask_parts, strict=True)
    ]
    for part, reference in zip(torch.unbind(actual), expected, strict=True):
        torch.testing.assert_close(part, reference)
    actual_loss = sum(part.square().sum() for part in torch.unbind(actual))
    expected_loss = sum(part.square().sum() for part in expected)
    actual_gradients = torch.autograd.grad(actual_loss, (values, *attention.parameters()), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected_loss, (values, *attention.parameters()))
    for gradient, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, reference)


@pytest.mark.parametrize("numeric", [False, True])
def test_global_unavailable_payload(numeric):
    module = GlobalAttention(hidden_size=4, num_attention_heads=1, head_size=4, backend="eager").double()
    mask = torch.tensor([[[True], [False]]])
    weights = mask.double().requires_grad_() if numeric else mask
    values = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [torch.nan] * 4]], dtype=torch.float64, requires_grad=True)
    result = module(values, weights)
    assert torch.isfinite(result).all()
    gradient = torch.autograd.grad(result[:, :1].square().sum(), values)[0]
    assert torch.isfinite(gradient).all()
    assert torch.equal(gradient[:, 1], torch.zeros_like(gradient[:, 1]))


def test_global_coefficient_gradient():
    module = GlobalAttention(hidden_size=1, num_attention_heads=1, head_size=1, backend="eager").double()
    with torch.no_grad():
        for projection in (
            module.query_projection,
            module.key_projection,
            module.value_projection,
            module.output_projection,
        ):
            projection.weight.fill_(1)
        module.gate_projection.weight.zero_()
        module.gate_projection.bias.zero_()
        module.output_projection.bias.zero_()
    values = torch.tensor([[[2.0], [7.0], [3.0]]], dtype=torch.float64)
    weights = torch.tensor([[[1.0], [0.0], [1.0]]], dtype=torch.float64, requires_grad=True)
    # Finite zero-coefficient values affect the global query's coefficient VJP.
    result = module(values, weights)
    changed = values.clone()
    changed[:, 1] += 5
    changed_result = module(changed, weights)
    torch.testing.assert_close(result[:, (0, 2)], changed_result[:, (0, 2)])
    first = torch.autograd.grad(result[:, 0].sum(), weights, retain_graph=True)[0]
    second = torch.autograd.grad(changed_result[:, 0].sum(), weights)[0]
    assert not torch.equal(first[:, 1], second[:, 1])
