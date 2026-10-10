import os
import subprocess
import sys
import textwrap

import pytest
import torch

import torchao
from torchao.utils import is_cuda_version_at_least, is_sm_at_least_100

if not (
    torch.cuda.is_available()
    and is_sm_at_least_100()
    and is_cuda_version_at_least(12, 8)
):
    pytest.skip("Test requires CUDA 12.8+ with SM >= 100", allow_module_level=True)

from torchao.prototype.moe_training.ep import permute_mxfp8_fwd_hp_bwd
from torchao.prototype.moe_training.ep.permute import (
    _triton_permute_fwd,
    permute_and_pad,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.quantization.utils import compute_error


def test_mxfp8_permute_forward():
    device = "cuda"
    tokens = 64
    dim = 128
    num_experts = 8
    ep_degree = 1
    block_size = 32

    input_tensor = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16)

    mx_input = MXTensor.to_mx(
        input_tensor, elem_dtype=torch.float8_e4m3fn, block_size=block_size
    )

    # Create num_tokens_per_expert tensor
    tokens_per_expert = tokens // num_experts
    num_tokens_per_expert = torch.full(
        (num_experts,), tokens_per_expert, dtype=torch.int32, device=device
    )

    (
        padded_shape,
        mx_output,
        permuted_indices,
        num_tokens_per_expert_padded,
        offsets,
    ) = permute_mxfp8_fwd_hp_bwd(
        mx_input,
        num_tokens_per_expert,
        ep_degree,
        num_experts,
        block_size,
    )

    # BF16 reference
    (
        _,
        ref_output,
        _,
        _,
        _,
    ) = permute_and_pad(
        input_tensor,
        num_tokens_per_expert,
        ep_degree,
        num_experts,
        block_size,
    )

    # Compare outputs
    output = mx_output.dequantize()
    sqnr = compute_error(output, ref_output)
    assert sqnr >= 30.0, f"SQNR too low: {sqnr} dB"

    # Note: backward is tested in an e2e integration test with other mxfp8 EP pipeline components


def _reference_permute(x: torch.Tensor, permuted_indices: torch.Tensor):
    """The original implementation: index into x with a zero row appended for -1."""
    return torch.vstack((x, x.new_zeros((1, x.shape[-1]))))[permuted_indices, :]


@pytest.mark.parametrize(
    "routing",
    [
        "balanced",
        "ragged",
        "one_hot_expert",
        "single_token",
        "empty",
        "align_plus_one",
    ],
)
@pytest.mark.parametrize("cols", [256, 1000])
@pytest.mark.parametrize("non_contiguous", [False, True])
def test_permute_and_pad_matches_reference(routing, cols, non_contiguous):
    device = "cuda"
    ep_degree, num_local_experts, alignment = 4, 8, 32
    n = ep_degree * num_local_experts
    gen = torch.Generator().manual_seed(0)
    counts = {
        "balanced": [alignment] * n,
        "ragged": torch.randint(0, 100, (n,), generator=gen).tolist(),
        "one_hot_expert": [500] + [0] * (n - 1),
        "single_token": [1] + [0] * (n - 1),
        "empty": [0] * n,
        "align_plus_one": [alignment + 1] * n,
    }[routing]
    num_tokens_per_expert = torch.tensor(counts, dtype=torch.int32, device=device)
    rows = sum(counts)

    if non_contiguous:
        x = torch.randn(cols, rows, device=device, dtype=torch.bfloat16).t()
    else:
        x = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    x.requires_grad_(True)

    input_shape, out, permuted_indices, _, _ = permute_and_pad(
        x, num_tokens_per_expert, ep_degree, num_local_experts, alignment
    )

    x_ref = x.detach().clone().requires_grad_(True)
    out_ref = _reference_permute(x_ref, permuted_indices)

    assert input_shape == (rows + 1, cols)
    assert torch.equal(out, out_ref)
    assert (out[permuted_indices == -1] == 0).all()

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    assert torch.equal(x.grad, x_ref.grad)


def test_permute_and_pad_non_contiguous_grad():
    """Backward through permute_and_pad with a non-contiguous upstream gradient."""
    device = "cuda"
    ep_degree, num_local_experts, alignment = 4, 8, 32
    counts = torch.randint(
        0,
        80,
        (ep_degree * num_local_experts,),
        generator=torch.Generator().manual_seed(0),
    )
    num_tokens_per_expert = counts.to(device=device, dtype=torch.int32)
    x = torch.randn(int(counts.sum()), 384, device=device, dtype=torch.bfloat16)
    x.requires_grad_(True)

    _, out, permuted_indices, _, _ = permute_and_pad(
        x, num_tokens_per_expert, ep_degree, num_local_experts, alignment
    )
    x_ref = x.detach().clone().requires_grad_(True)
    out_ref = _reference_permute(x_ref, permuted_indices)

    grad = torch.randn(out.shape[1], out.shape[0], device=device, dtype=out.dtype).t()
    assert not grad.is_contiguous()
    out.backward(grad)
    out_ref.backward(grad)
    assert torch.equal(x.grad, x_ref.grad)


def test_triton_permute_fwd_out_of_range_indices():
    """Indices outside [0, rows) gather zeros, like the -1 padding value."""
    device = "cuda"
    rows, cols = 64, 256
    x = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    idx = [0, 5, rows, rows + 1_000_000, -1, -7, 63]
    idx = torch.tensor(idx, device=device, dtype=torch.int32)
    valid = (idx >= 0) & (idx < rows)

    out = _triton_permute_fwd(x, idx)
    assert torch.equal(out[valid], x[idx[valid].long()])
    assert not out[~valid].any()


@pytest.mark.parametrize("dynamic", [False, True])
def test_permute_and_pad_compile(dynamic):
    device = "cuda"
    ep_degree, num_local_experts, alignment = 4, 8, 32
    compiled = torch.compile(permute_and_pad, fullgraph=True, dynamic=dynamic)
    gen = torch.Generator().manual_seed(0)
    for _ in range(2):  # second call exercises a new shape
        counts = torch.randint(0, 80, (ep_degree * num_local_experts,), generator=gen)
        num_tokens_per_expert = counts.to(device=device, dtype=torch.int32)
        rows = int(counts.sum())
        x = torch.randn(rows, 512, device=device, dtype=torch.bfloat16)
        x.requires_grad_(True)

        input_shape, out, permuted_indices, _, _ = compiled(
            x, num_tokens_per_expert, ep_degree, num_local_experts, alignment
        )
        x_ref = x.detach().clone().requires_grad_(True)
        out_ref = _reference_permute(x_ref, permuted_indices)

        assert input_shape == (rows + 1, 512)
        assert torch.equal(out, out_ref)
        grad = torch.randn_like(out)
        out.backward(grad)
        out_ref.backward(grad)
        assert torch.equal(x.grad, x_ref.grad)


def _run_in_child(code: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run `code` in a fresh Python process that imports this same torchao.

    Kernels with bad addressing can fault or hang instead of returning wrong
    values, and either would take down the pytest process.
    """
    torchao_root = os.path.dirname(os.path.dirname(torchao.__file__))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (torchao_root, env.get("PYTHONPATH")) if p
    )
    try:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"child process hung for more than {timeout} s")


_FWD_ADDRESS_PAST_INT32_CHILD = textwrap.dedent(
    """
    import torch
    from torchao.prototype.moe_training.ep.permute import _triton_permute_fwd

    cols = 1024
    big = 2**31 // cols + 1024  # big * cols > 2**31
    n = 4096
    # Gather a small x into the last n rows of a big output. These rows straddle
    # the 2**31-element boundary; the last 1024 start past it.
    x = torch.randn(n, cols, device="cuda", dtype=torch.bfloat16)
    idx = torch.full((big,), -1, device="cuda", dtype=torch.int32)
    idx[big - n :] = torch.arange(n, device="cuda", dtype=torch.int32)
    out = _triton_permute_fwd(x, idx)
    torch.cuda.synchronize()
    assert torch.equal(out[big - n :], x), "store past 2**31 elements is wrong"
    assert not out[: big - n].any(), "padding rows are not zero"
    """
)


def test_triton_permute_fwd_addresses_past_int32():
    """The gather's store must not overflow int32 once rows * cols >= 2**31."""
    free, _ = torch.cuda.mem_get_info()
    if free < 6 * 2**30:
        pytest.skip("needs ~6 GiB of free GPU memory")
    proc = _run_in_child(_FWD_ADDRESS_PAST_INT32_CHILD)
    assert proc.returncode == 0, proc.stderr[-2000:]
