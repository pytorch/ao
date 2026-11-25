# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch
from torch.autograd import Function

from torchao.prototype.activation_compression import (
    activation_sparsity,
    compress_activation,
    compress_saved_activations,
    decompress_activation,
    measure_compression,
    synthesize_sparse_squared_relu,
)
from torchao.prototype.activation_compression.codec import _BITPACK_CHUNK

_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _generator() -> torch.Generator:
    return torch.Generator().manual_seed(0)


def _sparse(shape, sparsity=0.85, dtype=torch.float32):
    return synthesize_sparse_squared_relu(
        shape, sparsity, dtype=dtype, generator=_generator()
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("method", ["bitpack", "zlib"])
def test_roundtrip_matches_exactly(dtype, method):
    x = _sparse((32, 64), sparsity=0.85, dtype=dtype)
    restored = decompress_activation(compress_activation(x, method))
    assert restored.shape == x.shape
    assert restored.dtype == x.dtype
    assert restored.device == x.device
    assert torch.equal(restored, x)


@pytest.mark.parametrize("method", ["bitpack", "zlib"])
def test_roundtrip_noncontiguous_and_storage_offset(method):
    base = _sparse((40, 40), sparsity=0.9)
    x = base[1:, ::2]
    assert not x.is_contiguous()
    offset = base[3:]
    assert offset.is_contiguous() and offset.storage_offset() != 0
    for view in (x, offset):
        assert torch.equal(decompress_activation(compress_activation(view, method)), view)


@pytest.mark.parametrize("method", ["bitpack", "zlib"])
def test_roundtrip_edges(method):
    samples = [
        torch.zeros(8, 4),
        torch.ones(5),
        torch.tensor(3.5),
        torch.empty(0, 4),
        torch.tensor([float("inf"), float("-inf"), 0.0, -2.0]),
    ]
    for x in samples:
        restored = decompress_activation(compress_activation(x, method))
        assert restored.shape == x.shape
        assert torch.equal(restored, x)


@pytest.mark.parametrize("method", ["bitpack", "zlib"])
def test_nan_payload_roundtrips(method):
    x = torch.tensor([float("nan"), 0.0, 1.5])
    restored = decompress_activation(compress_activation(x, method))
    assert torch.isnan(restored[0])
    assert torch.equal(restored[1:], x[1:])


def test_bitpack_crosses_chunk_boundary():
    n = _BITPACK_CHUNK + 3
    x = _sparse((n,), sparsity=0.9)
    restored = decompress_activation(compress_activation(x, "bitpack"))
    assert torch.equal(restored, x)


@pytest.mark.parametrize("method", ["bitpack", "zlib"])
def test_packed_values_do_not_alias_the_input(method):
    x = _sparse((128, 128), sparsity=0.9)
    original = x.clone()
    packed = compress_activation(x, method)
    x.fill_(7)
    assert torch.equal(decompress_activation(packed), original)
    values = packed.payload[-1]
    assert values.untyped_storage().nbytes() < original.untyped_storage().nbytes()


def test_bitpack_ratio_at_squared_relu_sparsity():
    # bytes/element = 1/8 + density * element_size.
    # fp32 at 85% sparsity is about 5.5x; fp16/bf16 about 4.7x.
    fp32 = _sparse((256, 256), sparsity=0.85, dtype=torch.float32)
    fp16 = _sparse((256, 256), sparsity=0.85, dtype=torch.float16)
    assert activation_sparsity(fp32) > 0.8
    fp32_packed = compress_activation(fp32, "bitpack")
    fp16_packed = compress_activation(fp16, "bitpack")
    assert fp32.numel() * fp32.element_size() / fp32_packed.nbytes > 4.0
    assert fp16.numel() * fp16.element_size() / fp16_packed.nbytes > 3.0


def test_half_sparse_bitpack_is_still_smaller():
    x = _sparse((256, 256), sparsity=0.5)
    packed = compress_activation(x, "bitpack")
    assert packed.nbytes < x.numel() * x.element_size()


def test_measure_compression_reports_savings_and_zero_error():
    x = _sparse((128, 128), sparsity=0.9)
    measured = measure_compression(x, "bitpack", warmup=1, iters=2)
    assert measured.ratio > 4.0
    assert measured.max_abs_error == 0.0
    assert measured.compress_ms >= 0.0


def test_synthesize_rejects_bad_sparsity():
    with pytest.raises(ValueError, match="sparsity"):
        synthesize_sparse_squared_relu((4,), 1.5)


def test_unknown_method_and_bad_level():
    x = torch.zeros(4)
    with pytest.raises(ValueError, match="method"):
        compress_activation(x, method="nvcomp")
    with pytest.raises(ValueError, match="zlib_level"):
        compress_activation(x, method="zlib", zlib_level=12)
    with pytest.raises(TypeError, match="floating"):
        compress_activation(torch.zeros(4, dtype=torch.int32))


class _SaveActivation(Function):
    @staticmethod
    def forward(ctx, activation):
        ctx.save_for_backward(activation)
        return activation

    @staticmethod
    def backward(ctx, grad):
        (saved,) = ctx.saved_tensors
        return grad * (1 + saved)


@pytest.mark.parametrize("method", ["bitpack", "zlib"])
def test_hook_matches_grads_and_stores_less(method):
    base = _sparse((64, 256), sparsity=0.9)
    weight = torch.randn(64, 256, generator=_generator())

    def run(compress: bool):
        x = base.detach().clone().requires_grad_(True)
        w = weight.detach().clone().requires_grad_(True)
        stats = None
        if compress:
            with compress_saved_activations(
                method=method, min_numel=1, min_sparsity=0.5
            ) as stats:
                y = torch.nn.functional.linear(x, w)
            y.sum().backward()
        else:
            y = torch.nn.functional.linear(x, w)
            y.sum().backward()
        return y.detach(), x.grad, w.grad, stats

    y_c, gx_c, gw_c, stats = run(True)
    y, gx, gw, _ = run(False)
    assert torch.equal(y_c, y)
    assert torch.equal(gx_c, gx)
    assert torch.equal(gw_c, gw)
    assert stats.n_compressed >= 1
    assert stats.stored_bytes < stats.dense_bytes

    saved = base.detach().clone().requires_grad_(True)
    with compress_saved_activations(
        method=method, min_numel=1, min_sparsity=0.5
    ) as only_activation:
        out = _SaveActivation.apply(saved)
    out.sum().backward()
    reference = base.detach().clone().requires_grad_(True)
    _SaveActivation.apply(reference).sum().backward()
    assert torch.equal(saved.grad, reference.grad)
    assert only_activation.n_compressed == 1
    assert only_activation.dense_bytes / only_activation.stored_bytes > 3.0


def test_dense_and_integer_tensors_are_not_packed():
    class _SavePair(Function):
        @staticmethod
        def forward(ctx, x, idx):
            ctx.save_for_backward(x, idx)
            return x.sum()

        @staticmethod
        def backward(ctx, grad):
            x, _idx = ctx.saved_tensors
            return torch.ones_like(x) * grad, None

    x = torch.randn(4096, requires_grad=True)
    idx = torch.zeros(8192, dtype=torch.long)
    with compress_saved_activations() as stats:
        y = _SavePair.apply(x, idx)
    y.backward()
    assert stats.n_saved >= 1
    assert stats.n_compressed == 0
    assert stats.stored_bytes == stats.dense_bytes


def test_hook_does_not_apply_outside_the_block():
    x = _sparse((64, 128), sparsity=0.95).requires_grad_(True)
    with compress_saved_activations(min_numel=1, min_sparsity=0.5) as stats:
        x.sum()
    n_saved = stats.n_saved
    z = x.detach().requires_grad_(True)
    (z * 2).sum().backward()
    assert stats.n_saved == n_saved


def test_squared_relu_ffn_grads_match_under_the_hook():
    torch.manual_seed(0)
    up = torch.nn.Linear(32, 128)
    down = torch.nn.Linear(128, 32, bias=False)
    up.weight.data.normal_(std=32**-0.5)
    up.bias.data.fill_(-1.04)
    x0 = torch.randn(16, 32)

    def run(compress: bool):
        x = x0.detach().clone().requires_grad_(True)
        w_up = up.weight.detach().clone().requires_grad_(True)
        b_up = up.bias.detach().clone().requires_grad_(True)
        w_down = down.weight.detach().clone().requires_grad_(True)

        def forward():
            hidden = torch.relu(torch.nn.functional.linear(x, w_up, b_up)).square()
            return torch.nn.functional.linear(hidden, w_down)

        if compress:
            with compress_saved_activations(min_numel=1, min_sparsity=0.5) as stats:
                y = forward()
            y.sum().backward()
            return y.detach(), x.grad, w_up.grad, w_down.grad, stats
        y = forward()
        y.sum().backward()
        return y.detach(), x.grad, w_up.grad, w_down.grad, None

    y_c, gx_c, gu_c, gd_c, stats = run(True)
    y, gx, gu, gd, _ = run(False)
    assert torch.equal(y_c, y)
    assert torch.equal(gx_c, gx)
    assert torch.equal(gu_c, gu)
    assert torch.equal(gd_c, gd)
    assert stats.n_compressed >= 1
    assert stats.stored_bytes < stats.dense_bytes


def test_hook_rejects_bad_arguments():
    with pytest.raises(ValueError, match="min_sparsity"):
        with compress_saved_activations(min_sparsity=1.5):
            pass
    with pytest.raises(ValueError, match="min_numel"):
        with compress_saved_activations(min_numel=-1):
            pass
