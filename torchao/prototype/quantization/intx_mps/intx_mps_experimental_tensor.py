# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Generalized low-bit tensor subclass for the experimental MPS kernel path.

Supports int1 through int8 weight-only quantization.  Uses
``torchao._pack_weight_{n}bit`` for packing and
``torchao._linear_fp_act_{n}bit_weight`` for matmul — the experimental
MPS operators that include the simdgroup-tiled GEMM kernel for M > 1
and the qmv kernel for M == 1 (decode).

Packing format:
    - packed_weight: [N, nbit*K/8] uint8 (nbit bits per weight, packed along K)
    - scales:        [N, K/group_size] (activation dtype)
    - zeros:         [N, K/group_size] (activation dtype, = -scale * zero_point)

Dequantization: weight = scale * val + zero

This is a lighter-weight alternative to ``Int4TilePackedTo4dTensor``:
    - No K padding to 1024 (only K % 32 required for the GEMM kernel)
    - No N padding to 8/16 (only N % 4 required for M > 1)
    - Uses the simdgroup-tiled GEMM kernel for prefill (M > 1) and the
      qmv kernel for decode (M == 1)
"""

from typing import List, Optional

import torch

from torchao.quantization.quant_primitives import (
    MappingType,
    _choose_qparams_affine,
    _quantize_affine,
)
from torchao.utils import TorchAOBaseTensor, fill_defaults

__all__ = [
    "IntxMPSExperimentalTensor",
]


def _ensure_mps_experimental_lib_loaded(nbit: int = 4) -> None:
    """Load the experimental MPS ops library if not already loaded."""
    try:
        getattr(torch.ops.torchao, f"_linear_fp_act_{nbit}bit_weight")
        getattr(torch.ops.torchao, f"_pack_weight_{nbit}bit")
    except AttributeError:
        from torchao.experimental.ops.mps.utils import _load_torchao_mps_lib

        _load_torchao_mps_lib()


class IntxMPSExperimentalTensor(TorchAOBaseTensor):
    """Low-bit quantized tensor using the experimental MPS kernel path.

    Supports int1 through int8 weight-only quantization.

    Tensor Attributes:
        packed_weight: [N, nbit*K/8] uint8 packed low-bit weights
        scales: [N, K/group_size] scale tensor (activation dtype)
        zeros: [N, K/group_size] zero tensor (activation dtype, = -scale * zero_point)

    Non-Tensor Attributes:
        nbit: number of bits per weight (1-8)
        block_size: quantization block size, e.g. [1, group_size]
        shape: original weight shape (out_features, in_features)
    """

    tensor_data_names = ["packed_weight", "scales", "zeros"]
    tensor_attribute_names = [
        "nbit",
        "block_size",
        "shape",
        "use_native_int8",
    ]

    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        nbit: int,
        block_size: List[int],
        shape: torch.Size,
        use_native_int8: bool = False,
    ):
        kwargs = {}
        kwargs["device"] = packed_weight.device
        kwargs["dtype"] = scales.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        nbit: int,
        block_size: List[int],
        shape: torch.Size,
        use_native_int8: bool = False,
    ):
        self.packed_weight = packed_weight
        self.scales = scales
        self.zeros = zeros
        self.nbit = nbit
        self.block_size = block_size
        self.use_native_int8 = use_native_int8
        # Note: shape is set by _make_wrapper_subclass in __new__, not here.

    def _quantization_type(self):
        return f"nbit={self.nbit}, shape={self.shape}, block_size={self.block_size}, device={self.device}"

    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        block_size: List[int],
        nbit: Optional[int] = None,
        choose_qparams_algorithm: Optional[str] = "min_max",
    ) -> "IntxMPSExperimentalTensor":
        """Quantize a high-precision weight tensor to low-bit experimental MPS format.

        Quantization is performed on CPU to avoid MPS allocator
        fragmentation.  The MPS allocator (aten/src/ATen/mps/
        MPSAllocator.mm) places tensors into Metal heaps (MTLHeap).
        torch.mps.empty_cache() only releases heaps where every block
        is free. The source comment says: "a heap returns its memory
        to the system only as a whole, so freeing single blocks
        reclaims nothing; only heaps no allocation is left in can be
        released."  When the intermediate quantization steps (padding,
        choose_qparams, quantize_affine) run on MPS, they create
        temporary tensors that share heaps with the final packed
        weight.  After deleting the intermediates and calling
        empty_cache(), the heaps still hold the packed weight and
        cannot be released; the freed intermediate blocks within them
        are not reusable for new allocations (confirmed empirically:
        a fresh allocation after cleanup jumps current_allocated_memory
        by its full size rather than reusing the freed space).  The net
        effect is > 3x memory overhead on PyTorch <= 2.13, which uses
        MTLHeapTypeAutomatic and cannot coalesce freed blocks within a
        partially-occupied heap.  PyTorch 2.14 (PR #190438) switches to
        MTLHeapTypePlacement with offset-ordered coalescing, which
        resolves this issue: the bad path measures 1.00x (zero stuck
        memory), matching the CPU path.  The CPU workaround is kept for
        compatibility with PyTorch <= 2.13.  By moving the weight to
        CPU, doing all quantization there, and only moving the final
        packed result to MPS, the MPS allocator only ever holds the
        packed weight, giving zero overhead on all versions.
        See: https://github.com/pytorch/pytorch/pull/190438

        Args:
            hp_tensor: [N, K] weight tensor (bfloat16 or float32)
            block_size: quantization block size, e.g. [1, group_size]
            nbit: number of bits per weight (1-8).
            choose_qparams_algorithm: "min_max" (default).
                Uses simple min-max scaling for quantization parameters.

        Returns:
            IntxMPSExperimentalTensor with packed weights, scales, and zeros
        """
        if nbit is None:
            nbit = getattr(cls, "default_nbit", 4)
        _ensure_mps_experimental_lib_loaded(nbit)

        assert 1 <= nbit <= 8, f"nbit must be 1-8, got {nbit}"
        assert len(block_size) == hp_tensor.ndim
        assert all(x == 1 for x in block_size[:-1]), (
            f"Only per-group quantization supported, got block_size: {block_size}"
        )

        if choose_qparams_algorithm != "min_max":
            raise ValueError(
                f"Unsupported choose_qparams_algorithm: {choose_qparams_algorithm}. "
                f"Only 'min_max' is supported."
            )

        group_size = block_size[-1]
        orig_out_features, orig_in_features = hp_tensor.shape[-2:]
        orig_device = hp_tensor.device
        orig_dtype = hp_tensor.dtype

        # The experimental kernel requires K % 8 == 0 and N % 4 == 0 (for M > 1).
        # The simdgroup-tiled GEMM kernel (selected for M > 1) additionally
        # requires K % 32 == 0 and group_size % 32 == 0.  Pad K to a multiple
        # of 32 so the GEMM kernel is always selected for prefill rather than
        # falling back to the slower pack_mm path.  The extra K padding is at
        # most 31 columns and the extra N padding is at most 3 rows.  Both are
        # filled with zeros, which dequantize back to exactly 0: quantize(0.0)
        # yields zero_point, and dequantize(zero_point) = scale*zp + (-scale*zp)
        # = 0 by cancellation (scale is clamped to eps, never 0).  So the padded
        # region contributes nothing to the matmul.
        in_features = (orig_in_features + 31) // 32 * 32
        out_features = (orig_out_features + 3) // 4 * 4

        # --- Native int8 per-channel delegation ---
        # For int8 per-channel (group_size == K), delegate to PyTorch's
        # native _weight_int8pack_mm, which uses a purpose-built Metal
        # kernel achieving ~0.98x bf16 prefill speed.  The generalized
        # kernel achieves ~0.95x because it stages dequantized weights
        # through shared memory for compatibility with int1-int7.  At int8,
        # symmetric per-channel (native pytorch) is slightly faster and
        # almost as accurate as the generalized kernel (no zero-point float
        # noise).
        is_per_channel = group_size == orig_in_features
        if nbit == 8 and is_per_channel:
            return cls._from_hp_native_int8(
                hp_tensor, block_size, orig_device, orig_dtype
            )

        # The generalized kernel only supports group sizes {32, 64, 128, 256}.
        if group_size not in (32, 64, 128, 256):
            raise ValueError(
                f"group_size must be 32, 64, 128, or 256 (got {group_size}). "
                f"For int8 per-channel, use group_size=K (={orig_in_features})."
            )

        if in_features % group_size != 0:
            raise ValueError(
                f"Padded K ({in_features}) must be divisible by group_size "
                f"({group_size}). Original K={orig_in_features} was padded to a "
                f"multiple of 32. Try a different group_size or a larger K."
            )

        # Do quantization on CPU to avoid MPS allocator fragmentation.
        hp_tensor_cpu = hp_tensor.cpu()
        hp_tensor_padded = torch.nn.functional.pad(
            hp_tensor_cpu,
            (0, in_features - orig_in_features, 0, out_features - orig_out_features),
        )

        # Quantization: asymmetric min-max (with zero point)
        target_dtype = torch.uint8
        quant_min = 0
        quant_max = (1 << nbit) - 1

        scale, zero_point = _choose_qparams_affine(
            hp_tensor_padded,
            mapping_type=MappingType.ASYMMETRIC.name,
            block_size=block_size,
            target_dtype=target_dtype,
            quant_min=quant_min,
            quant_max=quant_max,
            scale_dtype=hp_tensor_cpu.dtype,
            zero_point_dtype=hp_tensor_cpu.dtype,
        )

        int_data = _quantize_affine(
            hp_tensor_padded,
            block_size,
            scale,
            zero_point,
            target_dtype,
            quant_min=quant_min,
            quant_max=quant_max,
        )

        # Pack using the experimental _pack_weight_{nbit}bit op (CPU)
        pack_op = getattr(torch.ops.torchao, f"_pack_weight_{nbit}bit")
        packed_weight = pack_op(int_data)

        # Reshape scales and zeros to [N, K/group_size]
        num_groups = in_features // group_size
        scale = scale.reshape(out_features, num_groups)
        zero_point = zero_point.reshape(out_features, num_groups)

        # The experimental kernel uses: weight = scale * val + zero
        # where zero = -scale * zero_point
        zeros = (-zero_point * scale).to(orig_dtype)

        # Move only the final packed results to the original device
        packed_weight = packed_weight.to(orig_device)
        scale = scale.to(orig_device)
        zeros = zeros.to(orig_device)

        # Free CPU intermediates
        del hp_tensor_cpu, hp_tensor_padded, zero_point, int_data

        return cls(
            packed_weight=packed_weight,
            scales=scale,
            zeros=zeros,
            nbit=nbit,
            block_size=block_size,
            shape=hp_tensor.shape,  # original shape, not padded
        )

    @classmethod
    def _from_hp_native_int8(
        cls,
        hp_tensor: torch.Tensor,
        block_size: List[int],
        orig_device: torch.device,
        orig_dtype: torch.dtype,
    ) -> "IntxMPSExperimentalTensor":
        """Quantize to native PyTorch int8 per-channel format.

        Stores int8 weights and 1D scales in the format expected by
        _weight_int8pack_mm.  The linear dispatch detects use_native_int8
        and routes to the native operator instead of the Metal kernel.
        """
        # Quantize on CPU to avoid MPS allocator fragmentation.
        w_cpu = hp_tensor.cpu().float()
        w_max_abs = w_cpu.abs().amax(dim=1, keepdim=True)
        scales = w_max_abs / 127.0
        # Clamp scales to avoid division by zero
        scales = scales.clamp(min=1e-8)
        w_q = torch.round(w_cpu / scales).clamp(-128, 127).to(torch.int8)
        scales_1d = scales.squeeze(1).to(orig_dtype)

        # Move to target device
        w_q = w_q.to(orig_device)
        scales_1d = scales_1d.to(orig_device)

        # Store a dummy zeros tensor for format consistency (unused)
        zeros = torch.zeros(scales_1d.shape[0], 1, dtype=orig_dtype, device=orig_device)

        return cls(
            packed_weight=w_q,  # [N, K] int8 (not packed — native format)
            scales=scales_1d,  # [N] 1D scales
            zeros=zeros,
            nbit=8,
            block_size=block_size,
            shape=hp_tensor.shape,
            use_native_int8=True,
        )


implements = IntxMPSExperimentalTensor.implements
implements_torch_function = IntxMPSExperimentalTensor.implements_torch_function


@implements(torch.ops.aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    """Linear dispatch: route to _linear_fp_act_{nbit}bit_weight."""
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )

    assert weight_tensor.block_size[0] == 1, (
        f"Requires groupwise quantization, got block_size: {weight_tensor.block_size}"
    )

    nbit = weight_tensor.nbit
    group_size = weight_tensor.block_size[-1]
    packed_weight = weight_tensor.packed_weight
    scales = weight_tensor.scales
    zeros = weight_tensor.zeros
    original_shape = weight_tensor.shape
    use_native_int8 = getattr(weight_tensor, "use_native_int8", False)

    orig_act_size = input_tensor.size()
    orig_dtype = input_tensor.dtype
    orig_k = input_tensor.shape[-1]

    # Fold batch dimensions into the first dimension
    act_mat = input_tensor.reshape(-1, input_tensor.shape[-1])

    # The kernel requires activation dtype to match scale/zero dtype.
    act_mat = act_mat.to(scales.dtype)

    # --- Native int8 per-channel delegation ---
    # When use_native_int8 is set, the weight is stored in native
    # _weight_int8pack_mm format ([N, K] int8 + [N] 1D scales).
    # Route directly to the native operator (see from_hp comment for why
    # the generalized kernel cannot handle per-channel int8).
    if use_native_int8:
        k_padded = packed_weight.size(-1)
        if k_padded != orig_k:
            act_mat = torch.nn.functional.pad(act_mat, (0, k_padded - orig_k))
        y = torch.ops.aten._weight_int8pack_mm(act_mat, packed_weight, scales)
        orig_out_features = original_shape[-2]
        y = y[:, :orig_out_features]
        y = y.reshape(*orig_act_size[:-1], orig_out_features)
        if bias is not None:
            y = y + bias.to(y.dtype)
        return y.to(orig_dtype)

    # Pad the activation's K dimension to match the packed weight's K.
    # from_hp pads K to a multiple of 32.
    # packed_weight is [N_pad, nbit*K_pad/8]
    # so K_pad = packed_weight.size(-1) * 8 / nbit >= orig_k.
    # The padded K columns of the weight are quantized to zero, so they
    # contribute nothing.
    k_padded = packed_weight.size(-1) * 8 // nbit
    if k_padded != orig_k:
        act_mat = torch.nn.functional.pad(act_mat, (0, k_padded - orig_k))

    # Run the experimental MPS low-bit linear
    linear_op = getattr(torch.ops.torchao, f"_linear_fp_act_{nbit}bit_weight")
    y = linear_op(act_mat, packed_weight, group_size, scales, zeros)

    # Trim output features (remove N padding)
    orig_out_features = original_shape[-2]
    y = y[:, :orig_out_features]

    # Unfold batch dimensions
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    if bias is not None:
        y = y + bias.to(y.dtype)

    return y.to(orig_dtype)


@implements(torch.ops.aten.t.default)
def _(func, _types, args, _kwargs):
    """.t() accessor — dequantize, transpose, return plain tensor."""
    (self,) = args
    dequant = self.dequantize()
    return torch.ops.aten.t.default(dequant)


@implements(torch.ops.aten.permute.default)
def _(func, _types, args, _kwargs):
    """Permute — dequantize, permute, return plain tensor."""
    self, dims = args
    dequant = self.dequantize()
    return torch.ops.aten.permute.default(dequant, dims)


@implements(torch.ops.aten.transpose.int)
def _(func, _types, args, _kwargs):
    """Transpose — dequantize, transpose, return plain tensor."""
    self, dim0, dim1 = args
    dequant = self.dequantize()
    return torch.ops.aten.transpose.int(dequant, dim0, dim1)


@implements(torch.ops.aten.slice.Tensor)
def _(func, _types, args, _kwargs):
    """Slice — dequantize, slice, return plain tensor.  Simple but correct."""
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    dequant = self.dequantize()
    sliced = torch.ops.aten.slice.Tensor(dequant, dim, start, end, step)
    return sliced


def _dequantize_int_n_mps_experimental(
    weight: "IntxMPSExperimentalTensor",
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Dequantize to a plain tensor by unpacking and applying scale/zero.

    Uses the experimental _linear_fp_act_{nbit}bit_weight with an identity
    matrix to unpack the low-bit weights, similar to how
    Int4TilePackedTo4dTensor dequantizes.
    """
    nbit = weight.nbit
    _ensure_mps_experimental_lib_loaded(nbit)

    orig_shape = weight.shape  # (out_features, in_features)
    out_features, in_features = orig_shape[-2], orig_shape[-1]
    packed_weight = weight.packed_weight
    scales = weight.scales
    zeros = weight.zeros
    group_size = weight.block_size[-1]
    device = packed_weight.device
    dtype = scales.dtype

    # Build identity matrix to extract weight rows
    # packed_weight is [N_padded, nbit*K_padded/8]
    k_padded = packed_weight.size(-1) * 8 // nbit
    identity = torch.eye(in_features, k_padded, device=device, dtype=dtype)

    # Run the linear op: identity @ packed_weight → [in_features, N_padded]
    if getattr(weight, "use_native_int8", False):
        # Native int8: packed_weight is [N, K] int8, scales is [N] 1D
        y = torch.ops.aten._weight_int8pack_mm(identity, packed_weight, scales)
    else:
        linear_op = getattr(torch.ops.torchao, f"_linear_fp_act_{nbit}bit_weight")
        y = linear_op(identity, packed_weight, group_size, scales, zeros)

    # Trim to original shape and transpose
    y = y[:in_features, :out_features].t().contiguous()

    if output_dtype is not None:
        y = y.to(output_dtype)
    return y


# Attach dequantize as a method on the tensor subclass
IntxMPSExperimentalTensor.dequantize = _dequantize_int_n_mps_experimental

IntxMPSExperimentalTensor.__module__ = "torchao.prototype.quantization.intx_mps"

# Allow a model with IntxMPSExperimentalTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([IntxMPSExperimentalTensor])

