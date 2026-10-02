import pytest
import torch

from torchao.utils import is_cuda_version_at_least, is_sm_at_least_100

if not (
    torch.cuda.is_available()
    and is_sm_at_least_100()
    and is_cuda_version_at_least(12, 8)
):
    pytest.skip("Test requires CUDA 12.8+ with SM >= 100", allow_module_level=True)

from torchao.prototype.moe_training.ep import permute_mxfp8_fwd_hp_bwd
from torchao.prototype.moe_training.ep.permute import permute_and_pad
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


def _free_gib() -> float:
    free, _ = torch.cuda.mem_get_info()
    return free / 1024**3


def test_permute_kernels_address_past_int32():
    """Row offsets must be widened to int64 before scaling by the column count.

    The permute kernels index with ``row_offsets * cols``, and ``row_offsets`` is
    ``program_id * BLOCK + tl.arange(...)``, i.e. int32. Once ``rows * cols``
    reaches 2**31 that product wraps and the access lands at a negative/aliased
    address. The backward's is a *scatter*, so it corrupts memory silently rather
    than faulting.

    This is reachable in practice: a DeepSeek-V3-16B EP=8 step with imbalanced
    routing put 1,141,504 rows x 2048 cols = 2,337,800,192 elements through these
    kernels, which manifested as NaN gradients across the whole model a dozen
    steps into training.

    Identity permutation, so every output row must equal its input row -- a
    wrapped address sends the write somewhere else. Only the first column is
    fingerprinted, so the check itself stays cheap.
    """
    cols = 1024
    rows = 2**31 // cols + 1024  # 2,098,176 rows -> 2,148,532,224 elements
    need_gib = rows * cols * 2 * 2 / 1024**3  # bf16 in + bf16 out
    if _free_gib() < need_gib + 4:
        pytest.skip(f"needs ~{need_gib + 4:.0f} GiB free, have {_free_gib():.0f}")

    from torchao.prototype.moe_training.ep.permute import (
        _triton_permute_bwd,
        _triton_permute_fwd,
    )

    device = "cuda"
    marks = (torch.arange(rows, device=device, dtype=torch.float32) % 1000 + 1).to(
        torch.bfloat16
    )
    idx = torch.arange(rows, device=device, dtype=torch.int32)

    x = torch.zeros(rows, cols, device=device, dtype=torch.bfloat16)
    x[:, 0] = marks
    out = _triton_permute_fwd(x, idx)
    assert torch.equal(out[:, 0], marks), "forward store wrapped past int32"
    assert out[:, 1:].abs().sum() == 0, "forward wrote outside its rows"
    del x, out
    torch.cuda.empty_cache()

    grad = torch.zeros(rows, cols, device=device, dtype=torch.bfloat16)
    grad[:, 0] = marks
    back = _triton_permute_bwd(grad, idx, rows, cols)
    assert torch.equal(back[:, 0], marks), "backward gather/scatter wrapped past int32"
    assert back[:, 1:].abs().sum() == 0, "backward wrote outside its rows"
