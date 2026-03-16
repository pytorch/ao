# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""
MXFP8 Quantization Utilities for CuTe DSL

Self-contained PTX utility functions for MXFP8/MXFP4 quantization kernels.
Adapted from fbcode/ads_mkl/ops/cute_dsl/quack/mxfp8/utils.py.

Key operations:
- abs_max_bf16x2: Find absolute maximum of 2 bf16 values (SIMD)
- fused_amax_to_e8m0_scale: Convert amax to E8M0 scale factor
- mul_cvt_8x: Scale and convert to FP8 (fused multiply-convert for 8 elements)

References:
- TransformerEngine: transformer_engine/common/util/ptx.cuh
- TransformerEngine: transformer_engine/common/cast/mxfp8/specialized/quantize_mxfp8.cuh
"""

from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Uint16, Uint32, Uint8
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T


class CompileCache:
    """Cache for compiled CuTe kernels."""

    def __init__(self) -> None:
        self._cache: dict[tuple, object] = {}

    def __contains__(self, key: tuple) -> bool:
        return key in self._cache

    def __getitem__(self, key: tuple) -> object:
        return self._cache[key]

    def __setitem__(self, key: tuple, value: object) -> None:
        self._cache[key] = value


TORCH_TO_CUTE_DTYPE: dict[torch.dtype, Type[cutlass.Numeric]] = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
}


def get_cuda_stream() -> cuda.CUstream:
    """Get the current CUDA stream for kernel launch."""
    return cuda.CUstream(torch.cuda.current_stream().stream_id)


# =============================================================================
# MMA Layout Constants for SM100 Blockscaled GEMM
# =============================================================================
MMA_ATOM_M0: int = 32
MMA_ATOM_M1: int = 4
MMA_ATOM_K: int = 4
MMA_ATOM_M_TOTAL: int = MMA_ATOM_M0 * MMA_ATOM_M1  # 128

E8M0_NEUTRAL_SCALE: int = 127

E4M3_MAX_NORM_RCP: float = 1.0 / 448.0
E5M2_MAX_NORM_RCP: float = 1.0 / 57344.0
E2M1_MAX_NORM_RCP: float = 1.0 / 6.0

MXFP8_BLOCK_SIZE: int = 32
MXFP4_BLOCK_SIZE: int = 32


# =============================================================================
# MMA Scale Offset Computation
# =============================================================================


@cute.jit
def compute_mma_scale_offset(
    row: cutlass.Int32,
    scale_col: cutlass.Int32,
    rest_m: cutlass.Int32,
    rest_k: cutlass.Int32,
) -> cutlass.Int64:
    """Compute linear offset in MMA atom-tiled layout tensor for scale factors."""
    atom_m0: cutlass.Int32 = row & cutlass.Int32(31)
    atom_m1: cutlass.Int32 = (row >> cutlass.Int32(5)) & cutlass.Int32(3)
    rest_m_idx: cutlass.Int32 = row >> cutlass.Int32(7)
    atom_k: cutlass.Int32 = scale_col & cutlass.Int32(3)
    rest_k_idx: cutlass.Int32 = scale_col >> cutlass.Int32(2)

    ATOM_COSIZE: int = 512
    stride_m0: cutlass.Int32 = cutlass.Int32(16)
    stride_m1: cutlass.Int32 = cutlass.Int32(4)
    stride_k: cutlass.Int32 = cutlass.Int32(1)
    stride_rest_k: cutlass.Int32 = cutlass.Int32(ATOM_COSIZE)
    stride_rest_m: cutlass.Int32 = cutlass.Int32(ATOM_COSIZE) * rest_k

    offset: cutlass.Int64 = (
        cutlass.Int64(atom_m0) * cutlass.Int64(stride_m0)
        + cutlass.Int64(atom_m1) * cutlass.Int64(stride_m1)
        + cutlass.Int64(rest_m_idx) * cutlass.Int64(stride_rest_m)
        + cutlass.Int64(atom_k) * cutlass.Int64(stride_k)
        + cutlass.Int64(rest_k_idx) * cutlass.Int64(stride_rest_k)
    )

    return offset


# =============================================================================
# Low-level PTX Operations via Inline Assembly
# =============================================================================


@dsl_user_op
def store_u8_global(ptr: cutlass.Int64, val: Uint8, *, loc=None, ip=None) -> None:
    """Store a single byte to global memory."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(ptr).ir_value(loc=loc, ip=ip),
            Uint8(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.b8 [$0], $1;",
        "l,c",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def abs_max_bf16x2(
    p1: Uint32,
    p2: Uint32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """SIMD absolute max of two bf16x2 packed values."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(p1).ir_value(loc=loc, ip=ip),
                Uint32(p2).ir_value(loc=loc, ip=ip),
            ],
            "max.xorsign.abs.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def mul_cvt_8x_e4m3(
    in_0: Float32,
    in_1: Float32,
    in_2: Float32,
    in_3: Float32,
    in_4: Float32,
    in_5: Float32,
    in_6: Float32,
    in_7: Float32,
    scale: Float32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int64:
    """Fused multiply and convert 8x f32 values to e4m3x8."""
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [
                Float32(in_0).ir_value(loc=loc, ip=ip),
                Float32(in_1).ir_value(loc=loc, ip=ip),
                Float32(in_2).ir_value(loc=loc, ip=ip),
                Float32(in_3).ir_value(loc=loc, ip=ip),
                Float32(in_4).ir_value(loc=loc, ip=ip),
                Float32(in_5).ir_value(loc=loc, ip=ip),
                Float32(in_6).ir_value(loc=loc, ip=ip),
                Float32(in_7).ir_value(loc=loc, ip=ip),
                Float32(scale).ir_value(loc=loc, ip=ip),
            ],
            "{\n"
            ".reg .f32 mc8e4_s0, mc8e4_s1, mc8e4_s2, mc8e4_s3, mc8e4_s4, mc8e4_s5, mc8e4_s6, mc8e4_s7;\n"
            ".reg .b16 mc8e4_fp8_01, mc8e4_fp8_23, mc8e4_fp8_45, mc8e4_fp8_67;\n"
            ".reg .b32 mc8e4_lo32, mc8e4_hi32;\n"
            "mul.f32 mc8e4_s0, $1, $9;\n"
            "mul.f32 mc8e4_s1, $2, $9;\n"
            "mul.f32 mc8e4_s2, $3, $9;\n"
            "mul.f32 mc8e4_s3, $4, $9;\n"
            "mul.f32 mc8e4_s4, $5, $9;\n"
            "mul.f32 mc8e4_s5, $6, $9;\n"
            "mul.f32 mc8e4_s6, $7, $9;\n"
            "mul.f32 mc8e4_s7, $8, $9;\n"
            "cvt.rn.satfinite.e4m3x2.f32 mc8e4_fp8_01, mc8e4_s1, mc8e4_s0;\n"
            "cvt.rn.satfinite.e4m3x2.f32 mc8e4_fp8_23, mc8e4_s3, mc8e4_s2;\n"
            "cvt.rn.satfinite.e4m3x2.f32 mc8e4_fp8_45, mc8e4_s5, mc8e4_s4;\n"
            "cvt.rn.satfinite.e4m3x2.f32 mc8e4_fp8_67, mc8e4_s7, mc8e4_s6;\n"
            "mov.b32 mc8e4_lo32, {mc8e4_fp8_01, mc8e4_fp8_23};\n"
            "mov.b32 mc8e4_hi32, {mc8e4_fp8_45, mc8e4_fp8_67};\n"
            "mov.b64 $0, {mc8e4_lo32, mc8e4_hi32};\n"
            "}\n",
            "=l,f,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def mul_cvt_8x_e5m2(
    in_0: Float32,
    in_1: Float32,
    in_2: Float32,
    in_3: Float32,
    in_4: Float32,
    in_5: Float32,
    in_6: Float32,
    in_7: Float32,
    scale: Float32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int64:
    """Fused multiply and convert 8x f32 values to e5m2x8."""
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [
                Float32(in_0).ir_value(loc=loc, ip=ip),
                Float32(in_1).ir_value(loc=loc, ip=ip),
                Float32(in_2).ir_value(loc=loc, ip=ip),
                Float32(in_3).ir_value(loc=loc, ip=ip),
                Float32(in_4).ir_value(loc=loc, ip=ip),
                Float32(in_5).ir_value(loc=loc, ip=ip),
                Float32(in_6).ir_value(loc=loc, ip=ip),
                Float32(in_7).ir_value(loc=loc, ip=ip),
                Float32(scale).ir_value(loc=loc, ip=ip),
            ],
            "{\n"
            ".reg .f32 mc8e5_s0, mc8e5_s1, mc8e5_s2, mc8e5_s3, mc8e5_s4, mc8e5_s5, mc8e5_s6, mc8e5_s7;\n"
            ".reg .b16 mc8e5_fp8_01, mc8e5_fp8_23, mc8e5_fp8_45, mc8e5_fp8_67;\n"
            ".reg .b32 mc8e5_lo32, mc8e5_hi32;\n"
            "mul.f32 mc8e5_s0, $1, $9;\n"
            "mul.f32 mc8e5_s1, $2, $9;\n"
            "mul.f32 mc8e5_s2, $3, $9;\n"
            "mul.f32 mc8e5_s3, $4, $9;\n"
            "mul.f32 mc8e5_s4, $5, $9;\n"
            "mul.f32 mc8e5_s5, $6, $9;\n"
            "mul.f32 mc8e5_s6, $7, $9;\n"
            "mul.f32 mc8e5_s7, $8, $9;\n"
            "cvt.rn.satfinite.e5m2x2.f32 mc8e5_fp8_01, mc8e5_s1, mc8e5_s0;\n"
            "cvt.rn.satfinite.e5m2x2.f32 mc8e5_fp8_23, mc8e5_s3, mc8e5_s2;\n"
            "cvt.rn.satfinite.e5m2x2.f32 mc8e5_fp8_45, mc8e5_s5, mc8e5_s4;\n"
            "cvt.rn.satfinite.e5m2x2.f32 mc8e5_fp8_67, mc8e5_s7, mc8e5_s6;\n"
            "mov.b32 mc8e5_lo32, {mc8e5_fp8_01, mc8e5_fp8_23};\n"
            "mov.b32 mc8e5_hi32, {mc8e5_fp8_45, mc8e5_fp8_67};\n"
            "mov.b64 $0, {mc8e5_lo32, mc8e5_hi32};\n"
            "}\n",
            "=l,f,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_u128_global(
    ptr: cutlass.Int64,
    val_lo: cutlass.Int64,
    val_hi: cutlass.Int64,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store a 128-bit value to global memory (vectorized store using st.global.v2.b64)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(ptr).ir_value(loc=loc, ip=ip),
            cutlass.Int64(val_lo).ir_value(loc=loc, ip=ip),
            cutlass.Int64(val_hi).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.b64 [$0], {$1, $2};",
        "l,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# =============================================================================
# BF16 Conversion and Packing Utilities
# =============================================================================


@dsl_user_op
def bitcast_bf16_to_u16(val: cutlass.BFloat16, *, loc=None, ip=None) -> Uint16:
    """Bitcast BFloat16 to Uint16 for packing into bf16x2."""
    bf16_val = cutlass.BFloat16(val).ir_value(loc=loc, ip=ip)
    return Uint16(llvm.bitcast(Uint16.mlir_type, bf16_val, loc=loc, ip=ip))


@dsl_user_op
def pack_bf16x2(lo: Uint16, hi: Uint16, *, loc=None, ip=None) -> Uint32:
    """Pack two BF16 values into bf16x2 format for SIMD operations."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint16(lo).ir_value(loc=loc, ip=ip),
                Uint16(hi).ir_value(loc=loc, ip=ip),
            ],
            "mov.b32 $0, {$1, $2};",
            "=r,h,h",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def pack_2xbf16_to_u32(
    bf16_0: cutlass.BFloat16,
    bf16_1: cutlass.BFloat16,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Pack two BFloat16 values into a single Uint32 (bf16x2 format)."""
    bf16_0_val = cutlass.BFloat16(bf16_0).ir_value(loc=loc, ip=ip)
    bf16_1_val = cutlass.BFloat16(bf16_1).ir_value(loc=loc, ip=ip)
    bf16_0_u16 = llvm.bitcast(Uint16.mlir_type, bf16_0_val, loc=loc, ip=ip)
    bf16_1_u16 = llvm.bitcast(Uint16.mlir_type, bf16_1_val, loc=loc, ip=ip)
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [bf16_0_u16, bf16_1_u16],
            "mov.b32 $0, {$1, $2};\n",
            "=r,h,h",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cvt_bf16x8_to_f32x8(
    bf16_pair_01: Uint32,
    bf16_pair_23: Uint32,
    bf16_pair_45: Uint32,
    bf16_pair_67: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32]:
    """Convert 8 packed BF16 values (4 x bf16x2) to 8 Float32 values."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 8),
        [
            Uint32(bf16_pair_01).ir_value(loc=loc, ip=ip),
            Uint32(bf16_pair_23).ir_value(loc=loc, ip=ip),
            Uint32(bf16_pair_45).ir_value(loc=loc, ip=ip),
            Uint32(bf16_pair_67).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .b16 cvt8_b0, cvt8_b1, cvt8_b2, cvt8_b3, cvt8_b4, cvt8_b5, cvt8_b6, cvt8_b7;\n"
        "mov.b32 {cvt8_b0, cvt8_b1}, $8;\n"
        "mov.b32 {cvt8_b2, cvt8_b3}, $9;\n"
        "mov.b32 {cvt8_b4, cvt8_b5}, $10;\n"
        "mov.b32 {cvt8_b6, cvt8_b7}, $11;\n"
        "cvt.f32.bf16 $0, cvt8_b0;\n"
        "cvt.f32.bf16 $1, cvt8_b1;\n"
        "cvt.f32.bf16 $2, cvt8_b2;\n"
        "cvt.f32.bf16 $3, cvt8_b3;\n"
        "cvt.f32.bf16 $4, cvt8_b4;\n"
        "cvt.f32.bf16 $5, cvt8_b5;\n"
        "cvt.f32.bf16 $6, cvt8_b6;\n"
        "cvt.f32.bf16 $7, cvt8_b7;\n"
        "}\n",
        "=f,=f,=f,=f,=f,=f,=f,=f,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    v0 = Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip))
    v1 = Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip))
    v2 = Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip))
    v3 = Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip))
    v4 = Float32(llvm.extractvalue(T.f32(), result, [4], loc=loc, ip=ip))
    v5 = Float32(llvm.extractvalue(T.f32(), result, [5], loc=loc, ip=ip))
    v6 = Float32(llvm.extractvalue(T.f32(), result, [6], loc=loc, ip=ip))
    v7 = Float32(llvm.extractvalue(T.f32(), result, [7], loc=loc, ip=ip))
    return v0, v1, v2, v3, v4, v5, v6, v7


# =============================================================================
# Fused Amax-to-E8M0 Scale Computation
# =============================================================================


@dsl_user_op
def fused_amax_to_e8m0_scale(
    amax_packed: Uint32,
    max_norm_rcp: Float32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Uint8, Float32]:
    """Fused operation to convert bf16x2 packed amax to E8M0 scale and inverse scale."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [
            Uint32(amax_packed).ir_value(loc=loc, ip=ip),
            Float32(max_norm_rcp).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .b16 fae_bf16_lo, fae_bf16_hi, fae_amax_bf16;\n"
        "mov.b32 {fae_bf16_lo, fae_bf16_hi}, $2;\n"
        "max.xorsign.abs.bf16 fae_amax_bf16, fae_bf16_lo, fae_bf16_hi;\n"
        ".reg .f32 fae_amax_f32;\n"
        "cvt.f32.bf16 fae_amax_f32, fae_amax_bf16;\n"
        ".reg .f32 fae_scaled, fae_zero;\n"
        "mul.f32 fae_scaled, fae_amax_f32, $3;\n"
        "mov.f32 fae_zero, 0f00000000;\n"
        ".reg .b16 fae_e8m0_packed;\n"
        "cvt.rp.satfinite.ue8m0x2.f32 fae_e8m0_packed, fae_zero, fae_scaled;\n"
        ".reg .b32 fae_exp;\n"
        "cvt.u32.u16 fae_exp, fae_e8m0_packed;\n"
        "and.b32 fae_exp, fae_exp, 0xFF;\n"
        "mov.b32 $0, fae_exp;\n"
        ".reg .u32 fae_inv_exp, fae_inv_bits;\n"
        "sub.u32 fae_inv_exp, 254, fae_exp;\n"
        "shl.b32 fae_inv_bits, fae_inv_exp, 23;\n"
        "mov.b32 $1, fae_inv_bits;\n"
        "}\n",
        "=r,=f,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    e8m0_u32 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    inv_scale_val = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)

    e8m0_scale = Uint8(
        llvm.inline_asm(
            T.i8(),
            [e8m0_u32],
            "cvt.u8.u32 $0, $1;\n",
            "=c,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )

    return e8m0_scale, Float32(inv_scale_val)


# =============================================================================
# FP8 Conversion Functions
# =============================================================================


@dsl_user_op
def float_to_fp8_e4m3(val: Float32, *, loc=None, ip=None) -> Uint8:
    """Convert a pre-scaled float32 value to FP8 E4M3."""
    out_u16 = Uint16(
        llvm.inline_asm(
            T.i16(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $1;",
            "=h,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    return Uint8(out_u16 & 0xFF)


@dsl_user_op
def float_to_fp8_e5m2(val: Float32, *, loc=None, ip=None) -> Uint8:
    """Convert a pre-scaled float32 value to FP8 E5M2."""
    out_u16 = Uint16(
        llvm.inline_asm(
            T.i16(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            "cvt.rn.satfinite.e5m2x2.f32 $0, $1, $1;",
            "=h,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    return Uint8(out_u16 & 0xFF)


# =============================================================================
# MXFP4 E2M1 Conversion Functions (SM100+ required)
# =============================================================================


@dsl_user_op
def mul_cvt_8x_e2m1(
    in_0: Float32,
    in_1: Float32,
    in_2: Float32,
    in_3: Float32,
    in_4: Float32,
    in_5: Float32,
    in_6: Float32,
    in_7: Float32,
    scale: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Fused multiply and convert 8x f32 values to e2m1x8 (packed into 32 bits)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(in_0).ir_value(loc=loc, ip=ip),
                Float32(in_1).ir_value(loc=loc, ip=ip),
                Float32(in_2).ir_value(loc=loc, ip=ip),
                Float32(in_3).ir_value(loc=loc, ip=ip),
                Float32(in_4).ir_value(loc=loc, ip=ip),
                Float32(in_5).ir_value(loc=loc, ip=ip),
                Float32(in_6).ir_value(loc=loc, ip=ip),
                Float32(in_7).ir_value(loc=loc, ip=ip),
                Float32(scale).ir_value(loc=loc, ip=ip),
            ],
            "{\n"
            ".reg .f32 me2_s0, me2_s1, me2_s2, me2_s3, me2_s4, me2_s5, me2_s6, me2_s7;\n"
            ".reg .b8 me2_fp4_01, me2_fp4_23, me2_fp4_45, me2_fp4_67;\n"
            "mul.f32 me2_s0, $1, $9;\n"
            "mul.f32 me2_s1, $2, $9;\n"
            "mul.f32 me2_s2, $3, $9;\n"
            "mul.f32 me2_s3, $4, $9;\n"
            "mul.f32 me2_s4, $5, $9;\n"
            "mul.f32 me2_s5, $6, $9;\n"
            "mul.f32 me2_s6, $7, $9;\n"
            "mul.f32 me2_s7, $8, $9;\n"
            "cvt.rn.satfinite.e2m1x2.f32 me2_fp4_01, me2_s1, me2_s0;\n"
            "cvt.rn.satfinite.e2m1x2.f32 me2_fp4_23, me2_s3, me2_s2;\n"
            "cvt.rn.satfinite.e2m1x2.f32 me2_fp4_45, me2_s5, me2_s4;\n"
            "cvt.rn.satfinite.e2m1x2.f32 me2_fp4_67, me2_s7, me2_s6;\n"
            "mov.b32 $0, {me2_fp4_01, me2_fp4_23, me2_fp4_45, me2_fp4_67};\n"
            "}\n",
            "=r,f,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def float_to_fp4_e2m1(val: Float32, *, loc=None, ip=None) -> Uint8:
    """Convert a pre-scaled float32 value to FP4 E2M1 (scalar version)."""
    result = llvm.inline_asm(
        T.i8(),
        [Float32(val).ir_value(loc=loc, ip=ip)],
        "{\n"
        ".reg .b8 fp4_packed;\n"
        ".reg .b32 fp4_u32;\n"
        "cvt.rn.satfinite.e2m1x2.f32 fp4_packed, $1, $1;\n"
        "cvt.u32.u8 fp4_u32, fp4_packed;\n"
        "and.b32 fp4_u32, fp4_u32, 0x0F;\n"
        "cvt.u8.u32 $0, fp4_u32;\n"
        "}\n",
        "=c,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Uint8(result)


@dsl_user_op
def store_u64_global(
    ptr: cutlass.Int64,
    val: cutlass.Int64,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store a 64-bit value to global memory."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(ptr).ir_value(loc=loc, ip=ip),
            cutlass.Int64(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.b64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def pack_2xu32_to_u64(
    lo: Uint32,
    hi: Uint32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int64:
    """Pack two 32-bit values into a single 64-bit value."""
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [
                Uint32(lo).ir_value(loc=loc, ip=ip),
                Uint32(hi).ir_value(loc=loc, ip=ip),
            ],
            "mov.b64 $0, {$1, $2};",
            "=l,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# Stochastic Rounding for Scale Computation
# =============================================================================


@dsl_user_op
def fused_amax_to_e8m0_scale_stochastic(
    amax_packed: Uint32,
    max_norm_rcp: Float32,
    rand_bits: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Uint8, Float32]:
    """Fused amax to E8M0 scale with stochastic rounding."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [
            Uint32(amax_packed).ir_value(loc=loc, ip=ip),
            Float32(max_norm_rcp).ir_value(loc=loc, ip=ip),
            Uint32(rand_bits).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .b16 faes_bf16_lo, faes_bf16_hi, faes_amax_bf16;\n"
        "mov.b32 {faes_bf16_lo, faes_bf16_hi}, $2;\n"
        "max.xorsign.abs.bf16 faes_amax_bf16, faes_bf16_lo, faes_bf16_hi;\n"
        ".reg .f32 faes_amax_f32;\n"
        "cvt.f32.bf16 faes_amax_f32, faes_amax_bf16;\n"
        ".reg .f32 faes_scaled;\n"
        "mul.f32 faes_scaled, faes_amax_f32, $3;\n"
        ".reg .b32 faes_bits, faes_exp, faes_mantissa;\n"
        ".reg .pred faes_has_mantissa, faes_round_up;\n"
        "mov.b32 faes_bits, faes_scaled;\n"
        "bfe.u32 faes_exp, faes_bits, 23, 8;\n"
        "and.b32 faes_mantissa, faes_bits, 0x7FFFFF;\n"
        "setp.ne.u32 faes_has_mantissa, faes_mantissa, 0;\n"
        ".reg .b32 faes_rand_thresh;\n"
        "shr.u32 faes_rand_thresh, $4, 9;\n"
        "and.b32 faes_rand_thresh, faes_rand_thresh, 0x7FFFFF;\n"
        "setp.gt.and.u32 faes_round_up, faes_mantissa, faes_rand_thresh, faes_has_mantissa;\n"
        "@faes_round_up add.u32 faes_exp, faes_exp, 1;\n"
        "mov.b32 $0, faes_exp;\n"
        ".reg .u32 faes_inv_exp, faes_inv_bits;\n"
        "sub.u32 faes_inv_exp, 254, faes_exp;\n"
        "shl.b32 faes_inv_bits, faes_inv_exp, 23;\n"
        "mov.b32 $1, faes_inv_bits;\n"
        "}\n",
        "=r,=f,r,f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    e8m0_u32 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    inv_scale_val = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)

    e8m0_scale = Uint8(
        llvm.inline_asm(
            T.i8(),
            [e8m0_u32],
            "cvt.u8.u32 $0, $1;\n",
            "=c,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )

    return e8m0_scale, Float32(inv_scale_val)


@dsl_user_op
def load_u32_global(ptr: cutlass.Int64, *, loc=None, ip=None) -> Uint32:
    """Load a 32-bit value from global memory."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int64(ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
