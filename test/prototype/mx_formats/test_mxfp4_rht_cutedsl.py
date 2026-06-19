# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.prototype.mx_formats.cutedsl import _mxfp4_rht_cutedsl_kernels_available

pytestmark = pytest.mark.skipif(
    not _mxfp4_rht_cutedsl_kernels_available,
    reason="mxfp4 rht cutedsl unavailable (needs SM10.x, CUDA>=12.8, nvidia-cutlass-dsl)",
)


class TestE2M1Packing:
    def test_pack_bitexact_vs_reference(self):
        from torchao.prototype.mx_formats.cutedsl.cute_utils import (
            pack32_e2m1_to_bytes,
        )
        from torchao.prototype.mx_formats.kernels import (
            f32_to_f4_unpacked,
            pack_uint4,
        )

        torch.manual_seed(0)
        # len 32, spans the E2M1 grid + saturation + both parities/signs
        x = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.5,
                -1.0,
                -6.0,
                7.0,
                -7.0,
                0.25,
                5.0,
                -2.5,
            ]
            * 2,
            dtype=torch.float32,
            device="cuda",
        )
        ours = pack32_e2m1_to_bytes(x)  # (16,) uint8
        ref = (
            pack_uint4(f32_to_f4_unpacked(x.reshape(1, 32))).view(torch.uint8).flatten()
        )  # (16,)
        torch.testing.assert_close(ours, ref, rtol=0, atol=0)


class TestFp4Scale:
    @pytest.mark.parametrize("mode", ["floor", "rceil"])
    def test_scale_bytes_match_eager(self, mode):
        from torchao.prototype.mx_formats.cutedsl.cute_utils import (
            compute_block_scale_e8m0_fp4,
        )
        from torchao.prototype.mx_formats.mx_tensor import ScaleCalculationMode, to_mx

        torch.manual_seed(0)
        x = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda") * 7.0
        m = {
            "floor": ScaleCalculationMode.FLOOR,
            "rceil": ScaleCalculationMode.RCEIL,
        }[mode]
        # plain (unswizzled) e8m0 scales, shape (64, 1)
        s_ref, _ = to_mx(x, torch.float4_e2m1fn_x2, block_size=32, scaling_mode=m)
        s_ours = compute_block_scale_e8m0_fp4(x, mode)  # (64, 1) float8_e8m0fnu
        torch.testing.assert_close(
            s_ours.view(torch.uint8).flatten(),
            s_ref.view(torch.uint8).flatten(),
            rtol=0,
            atol=0,
        )


class TestFwht32:
    def test_fwht32_sign_matches_dense(self):
        from torchao.prototype.mx_formats.cutedsl.fwht import fwht32_sign_host
        from torchao.prototype.spinquant.hadamard_utils import hadamard_matrix

        torch.manual_seed(0)
        x = torch.randn(128, 32, dtype=torch.float32, device="cuda")
        sign = (torch.randint(0, 2, (32,), device="cuda") * 2 - 1).to(torch.int32)
        # normalized (1/sqrt(32)), symmetric Sylvester/Walsh-Hadamard matrix
        H = hadamard_matrix(32, device="cuda").to(torch.float32)
        ref = (x @ H) * sign.to(torch.float32)
        ours = fwht32_sign_host(x, sign)  # (128, 32) f32
        torch.testing.assert_close(ours, ref, rtol=1e-3, atol=1e-3)


class TestMxfp4RhtSmoke:
    def test_runs_and_shapes(self):
        from torchao.prototype.mx_formats.cutedsl import (
            mxfp4_rht_quantize_cutedsl_2d,
        )

        torch.manual_seed(0)
        M, K = 128, 256
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        sign = [1, -1] * 16
        q, s = mxfp4_rht_quantize_cutedsl_2d(x, sign, 32, "floor", True)
        assert q.shape == (M, K // 2)
        assert q.dtype == torch.uint8
        assert q.stride() == (K // 2, 1)
        assert s.dtype == torch.float8_e8m0fnu
        assert int((s.view(torch.uint8) != 0).sum()) > 0


class TestMxfp4RhtE2E:
    @pytest.mark.parametrize("mode", ["floor", "rceil"])
    @pytest.mark.parametrize("shape", [(128, 256), (256, 512), (512, 128)])
    def test_bitexact_vs_emulated_same_rht(self, mode, shape):
        # (A) Feed the SAME RHT values to both sides via the validated
        # ``fwht32_sign_host`` helper (the EXACT transform the kernel applies
        # internally). This isolates quant/pack/scale/swizzle: since the FWHT is
        # identical on both sides and Tasks 1-2 validated quant/pack/scale
        # bit-exactly, this must be bit-exact -- any diff is a real kernel bug.
        from torchao.prototype.mx_formats.cutedsl import (
            mxfp4_rht_quantize_cutedsl_2d,
        )
        from torchao.prototype.mx_formats.cutedsl.fwht import fwht32_sign_host
        from torchao.prototype.mx_formats.mx_tensor import ScaleCalculationMode, to_mx
        from torchao.prototype.mx_formats.utils import to_blocked

        torch.manual_seed(0)
        M, K = shape
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        sign = (torch.randint(0, 2, (32,), device="cuda") * 2 - 1).to(torch.int32)

        # The kernel TMA-loads the bf16 input and widens each element to fp32
        # before the FWHT, so the reference must apply the FWHT to the SAME
        # bf16-rounded-then-widened values (``x.float()``). ``fwht32_sign_host``
        # is bit-identical to the device ``fwht32_sign`` the kernel inlines.
        rht = fwht32_sign_host(x.float().reshape(-1, 32), sign).reshape(M, K)

        sm = {
            "floor": ScaleCalculationMode.FLOOR,
            "rceil": ScaleCalculationMode.RCEIL,
        }[mode]
        # to_mx returns (scale, data) in that order.
        s_ref, q_ref = to_mx(
            rht, torch.float4_e2m1fn_x2, block_size=32, scaling_mode=sm
        )
        s_ref_sw = to_blocked(s_ref.view(M, K // 32))

        q, s = mxfp4_rht_quantize_cutedsl_2d(x, sign.tolist(), 32, mode, True)

        torch.testing.assert_close(
            q.view(torch.uint8), q_ref.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(
            s.view(torch.uint8).flatten()[: s_ref_sw.numel()],
            s_ref_sw.view(torch.uint8).flatten(),
            rtol=0,
            atol=0,
        )
        assert q.stride() == (K // 2, 1)

    @pytest.mark.parametrize("mode", ["floor", "rceil"])
    def test_sqnr_vs_dense_reference(self, mode):
        # (B) Faithfulness of the WHOLE fused pipeline (incl. the FWHT) vs the
        # true high-precision dense RHT ``(x @ H) * sign``.
        from torchao.prototype.mx_formats.cutedsl import (
            mxfp4_rht_quantize_cutedsl_2d,
        )
        from torchao.prototype.mx_formats.kernels import (
            f4_unpacked_to_f32,
            unpack_uint4,
        )
        from torchao.prototype.spinquant.hadamard_utils import hadamard_matrix
        from torchao.quantization.utils import compute_error

        torch.manual_seed(0)
        M, K = 256, 512
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        sign = (torch.randint(0, 2, (32,), device="cuda") * 2 - 1).to(torch.int32)

        # True high-precision dense RHT on the (bf16-rounded) input the kernel
        # sees: (x @ H) * sign. hadamard_matrix(32) is normalized (1/sqrt(32)).
        H = hadamard_matrix(32, device="cuda").to(torch.float32)
        rht_true = ((x.float().reshape(-1, 32) @ H) * sign.float()).reshape(M, K)

        # Plain (unswizzled) scales of shape (M, K // 32) for easy dequant.
        q, s = mxfp4_rht_quantize_cutedsl_2d(x, sign.tolist(), 32, mode, False)

        # Dequant: unpack two nibbles/byte -> e2m1 codes (low nibble first),
        # decode to fp32 values, then multiply by 2^(e8m0_byte - 127) per block.
        codes = unpack_uint4(q)  # (M, K) uint8 fp4 codes in bits 0-3
        vals = f4_unpacked_to_f32(codes).reshape(M, K)
        e8 = s.view(torch.uint8).to(torch.int32).reshape(M, K // 32)
        scale = torch.pow(
            torch.tensor(2.0, device="cuda"), (e8 - 127).float()
        ).repeat_interleave(32, dim=1)
        deq = vals * scale

        sqnr = compute_error(rht_true, deq).item()
        assert sqnr >= 13.0, f"SQNR {sqnr} dB below 13 dB for mode={mode}"


class TestMxTensorThreading:
    @pytest.mark.parametrize("mode", ["floor", "rceil"])
    def test_mxtensor_cutedsl_matches_standalone(self, mode):
        # The opt-in CUTEDSL path through MXTensor.to_mx must produce qdata/scale
        # bit-identical to the standalone op called with the same arguments
        # (same is_swizzled_scales=True -> apples-to-apples).
        from torchao.prototype.mx_formats.config import MXFP4CastKernelChoice
        from torchao.prototype.mx_formats.cutedsl import (
            mxfp4_rht_quantize_cutedsl_2d,
        )
        from torchao.prototype.mx_formats.mx_tensor import (
            MXTensor,
            ScaleCalculationMode,
        )

        torch.manual_seed(0)
        M, K = 256, 512
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        sign = (
            (torch.randint(0, 2, (32,), device="cuda") * 2 - 1).to(torch.int32).tolist()
        )
        sm = {
            "floor": ScaleCalculationMode.FLOOR,
            "rceil": ScaleCalculationMode.RCEIL,
        }[mode]
        mxt = MXTensor.to_mx(
            x,
            torch.float4_e2m1fn_x2,
            block_size=32,
            scaling_mode=sm,
            is_swizzled_scales=True,
            mxfp4_cast_kernel_choice=MXFP4CastKernelChoice.CUTEDSL,
            rht_sign_vector=sign,
        )
        q_ref, s_ref = mxfp4_rht_quantize_cutedsl_2d(x, sign, 32, mode, True)
        torch.testing.assert_close(
            mxt.qdata.view(torch.uint8), q_ref.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(
            mxt.scale.view(torch.uint8).flatten(),
            s_ref.view(torch.uint8).flatten(),
            rtol=0,
            atol=0,
        )

    def test_default_path_unchanged(self):
        # The default (TORCH) fp4 cast still works and does NOT require a sign
        # vector -- the new trailing params are opt-in only.
        from torchao.prototype.mx_formats.mx_tensor import MXTensor

        torch.manual_seed(0)
        x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        mxt = MXTensor.to_mx(x, torch.float4_e2m1fn_x2, block_size=32)
        assert mxt.qdata.shape == (128, 128)
