import pytest
import torch

# Import requant kernel
from torchao.prototype.moe_training.ep.syncless.mxfp8_requant_kernel import (
    mxfp8_dequant_requant_col_major,
    triton_scale_blocked_layout_with_offset,
)

# Import silu_mul kernels
from torchao.prototype.moe_training.ep.syncless.silu_mul_kernel import (
    silu_mul_bw,
    silu_mul_fw,
)

# Import quantization kernels for reference
from torchao.prototype.mx_formats.kernels import (
    triton_mx_block_rearrange,
    triton_to_mxfp8_dim0,
    triton_to_mxfp8_dim1,
)
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.testing.utils import skip_if_rocm


def _torch_silu_mul_fw(h13_input: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Reference PyTorch implementation of silu_mul forward (matching kernel precision)."""
    h1, h3 = h13_input[:num_tokens].chunk(2, dim=-1)

    # Better precision: keep everything in float32 until final cast
    h1_f32 = h1.to(torch.float32)
    h3_f32 = h3.to(torch.float32)
    silu_h1_f32 = h1_f32 * torch.sigmoid(h1_f32)
    result = (silu_h1_f32 * h3_f32).to(h1.dtype)

    return result


def _torch_silu_mul_bw(h13_input: torch.Tensor, grad_h: torch.Tensor, num_tokens: int):
    """Reference PyTorch implementation of silu_mul backward (matching kernel precision)."""
    h1, h3 = h13_input[:num_tokens].chunk(2, dim=-1)

    # Recompute forward (must match forward kernel exactly)
    h1_f32 = h1.to(torch.float32)
    h3_f32 = h3.to(torch.float32)
    sig_h1 = torch.sigmoid(h1_f32)
    silu_h1_f32 = h1_f32 * sig_h1
    # Keep everything in float32 until final cast for precision
    h = (silu_h1_f32 * h3_f32).to(h1.dtype)

    # Backward - keep in float32 for gradient precision
    grad_h_valid = grad_h[:num_tokens]
    grad_h_f32 = grad_h_valid.to(torch.float32)
    dsilu = sig_h1 + h1_f32 * sig_h1 * (1.0 - sig_h1)
    grad_h1 = (grad_h_f32 * h3_f32 * dsilu).to(h1.dtype)
    grad_h3 = (grad_h_f32 * silu_h1_f32).to(h1.dtype)

    return h, torch.cat([grad_h1, grad_h3], dim=-1)


@pytest.mark.parametrize("num_tokens", [32, 128, 256, 512])
@pytest.mark.parametrize("hidden_dim", [512, 1024, 2048])
@pytest.mark.parametrize("sym_mem_buffer_rows", [64, 256, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_silu_mul_fw_kernel(
    num_tokens: int, hidden_dim: int, sym_mem_buffer_rows: int, dtype: torch.dtype
):
    """Test silu_mul forward kernel against PyTorch reference."""
    if sym_mem_buffer_rows < num_tokens:
        pytest.skip("sym_mem_buffer_rows must be >= num_tokens")

    device = "cuda"
    saved_activations_buffer_rows = sym_mem_buffer_rows + 100  # Larger buffer

    # Create test inputs
    input_buffer = torch.randn(
        saved_activations_buffer_rows, 2 * hidden_dim, dtype=dtype, device=device
    )
    saved_activation_buffer_offset = torch.tensor(50, dtype=torch.int64, device=device)
    num_tokens_tensor = torch.tensor(num_tokens, dtype=torch.int64, device=device)

    # Get slice for reference computation
    offset_val = saved_activation_buffer_offset.item()
    h13_slice = input_buffer[offset_val : offset_val + sym_mem_buffer_rows]

    # Reference computation
    ref_output = _torch_silu_mul_fw(h13_slice, num_tokens)

    # Zero-pad to sym_mem_buffer_rows
    ref_padded = torch.zeros(
        sym_mem_buffer_rows, hidden_dim, dtype=dtype, device=device
    )
    ref_padded[:num_tokens] = ref_output

    # Kernel computation
    kernel_output = silu_mul_fw(
        input_buffer,
        saved_activation_buffer_offset,
        num_tokens_tensor,
        sym_mem_buffer_rows,
    )

    # Verify shapes
    assert kernel_output.shape == (
        sym_mem_buffer_rows,
        hidden_dim,
    ), "Output shape mismatch"
    assert kernel_output.dtype == dtype, "Output dtype mismatch"

    # Verify numerics (relaxed tolerances for BF16 + float32 intermediate computations)
    torch.testing.assert_close(kernel_output, ref_padded, rtol=1e-2, atol=1e-3)

    # Verify zero-padding beyond num_tokens
    if num_tokens < sym_mem_buffer_rows:
        assert torch.all(kernel_output[num_tokens:] == 0), (
            "Rows beyond num_tokens should be zero"
        )


@pytest.mark.parametrize("num_tokens", [32, 128, 256, 512])
@pytest.mark.parametrize("hidden_dim", [512, 1024, 2048])
@pytest.mark.parametrize("sym_mem_buffer_rows", [64, 256, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_silu_mul_bw_kernel(
    num_tokens: int, hidden_dim: int, sym_mem_buffer_rows: int, dtype: torch.dtype
):
    """Test silu_mul backward kernel against PyTorch reference."""
    if sym_mem_buffer_rows < num_tokens:
        pytest.skip("sym_mem_buffer_rows must be >= num_tokens")

    device = "cuda"
    saved_activations_buffer_rows = sym_mem_buffer_rows + 100  # Larger buffer

    # Create test inputs
    h13_buffer = torch.randn(
        saved_activations_buffer_rows, 2 * hidden_dim, dtype=dtype, device=device
    )
    grad_h = torch.randn(sym_mem_buffer_rows, hidden_dim, dtype=dtype, device=device)
    saved_activation_buffer_offset = torch.tensor(50, dtype=torch.int64, device=device)
    num_tokens_tensor = torch.tensor(num_tokens, dtype=torch.int64, device=device)

    # Pre-allocate output buffers
    h_out = torch.zeros(sym_mem_buffer_rows, hidden_dim, dtype=dtype, device=device)
    grad_h13_out = torch.zeros(
        sym_mem_buffer_rows, 2 * hidden_dim, dtype=dtype, device=device
    )

    # Get slice for reference computation
    offset_val = saved_activation_buffer_offset.item()
    h13_slice = h13_buffer[offset_val : offset_val + sym_mem_buffer_rows]

    # Reference computation
    ref_h, ref_grad_h13 = _torch_silu_mul_bw(h13_slice, grad_h, num_tokens)

    # Zero-pad reference outputs to sym_mem_buffer_rows
    ref_h_padded = torch.zeros(
        sym_mem_buffer_rows, hidden_dim, dtype=dtype, device=device
    )
    ref_grad_h13_padded = torch.zeros(
        sym_mem_buffer_rows, 2 * hidden_dim, dtype=dtype, device=device
    )
    ref_h_padded[:num_tokens] = ref_h
    ref_grad_h13_padded[:num_tokens] = ref_grad_h13

    # Kernel computation
    silu_mul_bw(
        h13_buffer,
        grad_h,
        saved_activation_buffer_offset,
        num_tokens_tensor,
        h_out,
        grad_h13_out,
    )

    # Verify shapes
    assert h_out.shape == (sym_mem_buffer_rows, hidden_dim), "h_out shape mismatch"
    assert grad_h13_out.shape == (
        sym_mem_buffer_rows,
        2 * hidden_dim,
    ), "grad_h13_out shape mismatch"
    assert h_out.dtype == dtype, "h_out dtype mismatch"
    assert grad_h13_out.dtype == dtype, "grad_h13_out dtype mismatch"

    # Verify numerics (relaxed tolerances for BF16 + float32 intermediate computations)
    torch.testing.assert_close(h_out, ref_h_padded, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(grad_h13_out, ref_grad_h13_padded, rtol=1e-2, atol=1e-3)

    # Verify zero-padding beyond num_tokens
    if num_tokens < sym_mem_buffer_rows:
        assert torch.all(h_out[num_tokens:] == 0), (
            "h_out rows beyond num_tokens should be zero"
        )
        assert torch.all(grad_h13_out[num_tokens:] == 0), (
            "grad_h13_out rows beyond num_tokens should be zero"
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_silu_mul_kernels_gradient_correctness(dtype: torch.dtype):
    """Test that silu_mul kernels produce correct gradients via autograd."""
    device = "cuda"
    num_tokens = 64
    hidden_dim = 512
    sym_mem_buffer_rows = 128
    saved_activations_buffer_rows = 256

    # Create test inputs with gradient tracking
    input_buffer = torch.randn(
        saved_activations_buffer_rows,
        2 * hidden_dim,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    saved_activation_buffer_offset = torch.tensor(50, dtype=torch.int64, device=device)
    num_tokens_tensor = torch.tensor(num_tokens, dtype=torch.int64, device=device)

    # Forward pass with kernel
    h_fwd = silu_mul_fw(
        input_buffer,
        saved_activation_buffer_offset,
        num_tokens_tensor,
        sym_mem_buffer_rows,
    )

    # Create gradient for backward
    grad_h = torch.randn_like(h_fwd)

    # Pre-allocate backward outputs
    h_out = torch.zeros_like(h_fwd)
    grad_h13_out = torch.zeros(
        sym_mem_buffer_rows, 2 * hidden_dim, dtype=dtype, device=device
    )

    # Backward pass with kernel
    silu_mul_bw(
        input_buffer,
        grad_h,
        saved_activation_buffer_offset,
        num_tokens_tensor,
        h_out,
        grad_h13_out,
    )

    # Verify forward recomputation matches (relaxed tolerances for BF16)
    torch.testing.assert_close(h_fwd, h_out, rtol=1e-2, atol=1e-3)

    # Reference gradient computation using autograd (matching kernel precision)
    input_buffer_ref = input_buffer.clone().detach().requires_grad_(True)
    offset_val = saved_activation_buffer_offset.item()
    h13_slice = input_buffer_ref[offset_val : offset_val + num_tokens]
    h1, h3 = h13_slice.chunk(2, dim=-1)
    # Match kernel precision: keep everything in float32 until final cast
    h1_f32 = h1.to(torch.float32)
    h3_f32 = h3.to(torch.float32)
    silu_h1_f32 = h1_f32 * torch.sigmoid(h1_f32)
    ref_output = (silu_h1_f32 * h3_f32).to(h1.dtype)
    ref_output.backward(grad_h[:num_tokens])

    # Compare gradients
    expected_grad = torch.zeros_like(input_buffer_ref.grad)
    expected_grad[offset_val : offset_val + num_tokens] = input_buffer_ref.grad[
        offset_val : offset_val + num_tokens
    ]

    # Extract kernel gradients at the correct offset
    kernel_grad_slice = grad_h13_out[:num_tokens]

    torch.testing.assert_close(
        kernel_grad_slice,
        input_buffer_ref.grad[offset_val : offset_val + num_tokens],
        rtol=1e-2,
        atol=1e-3,
    )


@skip_if_rocm("ROCm enablement in progress")
@pytest.mark.parametrize(
    "num_tokens,dim,sym_mem_buffer_rows",
    [
        (128, 128, 128),
        (128, 256, 256),
        (256, 512, 256),
        (512, 1024, 512),
        (128, 128, 256),
        (256, 256, 512),
    ],
)
def test_mxfp8_dequant_requant_col_major(
    num_tokens: int, dim: int, sym_mem_buffer_rows: int
):
    """Test fused dequant-requant against two-step reference: dequant→fp32→requant_dim1.

    Also verifies zero-padding beyond num_tokens when num_tokens < sym_mem_buffer_rows.
    Tests the preallocated-buffer + offset API.
    """
    device = "cuda"
    block_size = 32
    buffer_extra = 64
    out_buffer_extra = 96  # extra columns in the output buffer

    torch.manual_seed(42)
    src_bf16 = torch.randn(
        num_tokens + buffer_extra, dim, dtype=torch.bfloat16, device=device
    )
    src_e4m3, src_scales = triton_to_mxfp8_dim0(
        src_bf16, inner_block_size=block_size, scaling_mode="rceil"
    )
    src_scales_u8 = src_scales.view(torch.uint8)

    offset = torch.tensor(buffer_extra // 2, dtype=torch.int64, device=device)
    num_tokens_t = torch.tensor(num_tokens, dtype=torch.int64, device=device)

    # Reference: dequant to FP32 with PyTorch, then requant with dim1 (32×1) scaling
    ref_dequant = _torch_mxfp8_dequant_buffer(
        src_e4m3,
        src_scales_u8,
        buffer_offset=offset.item(),
        num_tokens=num_tokens,
        sym_mem_buffer_rows=sym_mem_buffer_rows,
        out_dtype=torch.float32,
        scale_block_size=block_size,
    )
    ref_data, ref_scales = triton_to_mxfp8_dim1(
        ref_dequant, inner_block_size=block_size, scaling_mode="rceil"
    )

    # Preallocate output buffers with extra space to test offset writes
    out_total_cols = sym_mem_buffer_rows + out_buffer_extra
    out_data = torch.zeros(
        dim, out_total_cols, dtype=torch.float8_e4m3fn, device=device
    )
    out_scales = torch.zeros(
        dim, out_total_cols // block_size, dtype=torch.uint8, device=device
    )
    out_offset = torch.tensor(out_buffer_extra // 2, dtype=torch.int64, device=device)

    # Fused kernel with preallocated output buffer + offset
    mxfp8_dequant_requant_col_major(
        src_e4m3,
        src_scales_u8,
        buffer_offset=offset,
        num_tokens=num_tokens_t,
        sym_mem_buffer_rows=sym_mem_buffer_rows,
        out_data=out_data,
        out_scales=out_scales,
        out_offset=out_offset,
        scale_block_size=block_size,
    )

    # Extract the written region from the output buffer (with sync, for torch native impl)
    out_offset_val = out_offset.item()
    fused_data = out_data[:, out_offset_val : out_offset_val + sym_mem_buffer_rows].t()
    fused_scales = out_scales[
        :,
        out_offset_val // block_size : out_offset_val // block_size
        + sym_mem_buffer_rows // block_size,
    ].view(torch.float8_e8m0fnu)

    # Both ref_data and fused_data are (M, dim) col-major
    torch.testing.assert_close(
        fused_data.to(torch.float32),
        ref_data.to(torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        fused_scales.view(torch.uint8),
        ref_scales.view(torch.uint8),
        rtol=0,
        atol=0,
    )

    # Verify zero-padding beyond num_tokens
    if num_tokens < sym_mem_buffer_rows:
        fused_data_f32 = fused_data.to(torch.float32)
        assert torch.all(fused_data_f32[num_tokens:, :] == 0), (
            "Rows beyond num_tokens should be zero"
        )

    # Verify output buffer is untouched outside the written region
    before_region = out_data[:, :out_offset_val]
    assert torch.all(before_region.to(torch.float32) == 0), (
        "Output buffer before out_offset should be untouched"
    )


# for test reference
def _torch_mxfp8_dequant_buffer(
    e4m3_data_buffer: torch.Tensor,
    e8m0_scales_buffer: torch.Tensor,
    buffer_offset: int,
    num_tokens: int,
    sym_mem_buffer_rows: int,
    out_dtype: torch.dtype,
    scale_block_size: int = 32,
) -> torch.Tensor:
    """Pure PyTorch reference for MXFP8 dequantization from a buffer at offset."""
    dim = e4m3_data_buffer.shape[1]
    scale_dim = e8m0_scales_buffer.shape[1]

    data_slice = e4m3_data_buffer[buffer_offset : buffer_offset + num_tokens]
    scales_slice = e8m0_scales_buffer[buffer_offset : buffer_offset + num_tokens]

    # Dequant: e4m3 * 2^(e8m0 - 127)
    scales_u8 = scales_slice.view(torch.uint8)
    scales_fp32 = torch.exp2((scales_u8.to(torch.float32) - 127.0))

    data_f32 = data_slice.to(torch.float32).reshape(
        num_tokens, scale_dim, scale_block_size
    )
    scales_f32 = scales_fp32.unsqueeze(-1)
    dequanted = (data_f32 * scales_f32).reshape(num_tokens, dim)

    out = torch.zeros(
        sym_mem_buffer_rows, dim, dtype=out_dtype, device=e4m3_data_buffer.device
    )
    out[:num_tokens] = dequanted.to(out_dtype)
    return out


@skip_if_rocm("ROCm enablement in progress")
@pytest.mark.parametrize("m", [256, 512, 1024])
@pytest.mark.parametrize("total_k", [1024, 2048, 4096])
@pytest.mark.parametrize("buffer_extra_cols", [0, 128, 256])
def test_triton_scale_blocked_layout_with_offset(
    m: int,
    total_k: int,
    buffer_extra_cols: int,
):
    """Test offset-aware scale rearrangement against reference without offset.

    Creates a larger buffer, places scale data at an offset, and verifies
    the rearranged output matches calling the original kernel on a
    contiguous slice.
    """
    device = "cuda"
    block_size = 32
    scale_cols = total_k // block_size
    col_offset = buffer_extra_cols // block_size
    total_buffer_scale_cols = scale_cols + col_offset + 16  # extra padding

    # Create scale data for the region
    input_data = torch.randn(m, total_k, device=device)
    e8m0_scales, _ = to_mx(
        input_data, elem_dtype=torch.float8_e4m3fn, block_size=block_size
    )

    # Place scales into a larger buffer at the offset
    scale_buffer = torch.zeros(
        m, total_buffer_scale_cols, dtype=torch.uint8, device=device
    )
    scale_buffer[:, col_offset : col_offset + scale_cols] = e8m0_scales.view(
        torch.uint8
    )

    # Reference: call triton_mx_block_rearrange on contiguous slice
    ref_blocked = triton_mx_block_rearrange(e8m0_scales)

    # Offset-aware kernel: reads from buffer at col_offset, writes at output offset 0
    col_offset_tensor = torch.tensor(
        col_offset * block_size, dtype=torch.int64, device=device
    )
    output_buffer = torch.zeros(ref_blocked.numel(), dtype=torch.uint8, device=device)

    triton_scale_blocked_layout_with_offset(
        scale_buffer.view(torch.float8_e8m0fnu),
        input_col_offset=col_offset_tensor,
        output_buffer=output_buffer,
        output_write_offset=torch.zeros(1, dtype=torch.int64, device=device),
    )

    offset_blocked = output_buffer[: ref_blocked.numel()].view(ref_blocked.shape)

    # Use assert_close with rtol=0,atol=0 instead of torch.equal to avoid
    # PyTorch reduce_kernel which can hit illegal instruction on SM 10.x.
    torch.testing.assert_close(
        ref_blocked.view(torch.uint8).to(torch.int32),
        offset_blocked.to(torch.int32),
        rtol=0,
        atol=0,
    )
