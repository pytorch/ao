# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import io
import itertools
import random
import re
import unittest
import warnings
from typing import List, Tuple

import pytest

import torch
import torch.nn as nn

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

if not TORCH_VERSION_AT_LEAST_2_5:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


from torchao.float8.config import (
    CastConfig,
    Float8LinearConfig,
    Float8LinearRecipeName,
    recipe_name_to_linear_config,
    ScalingGranularity,
    ScalingType,
)
from torchao.float8.float8_linear import Float8Linear
from torchao.float8.float8_linear_utils import (
    convert_to_float8_training,
    linear_requires_sync,
    sync_float8_amax_and_scale_history,
)
from torchao.float8.float8_python_api import addmm_float8_unwrapped
from torchao.float8.float8_scaling_utils import (
    get_maybe_axiswise_dim,
    hp_tensor_to_float8_dynamic,
)
from torchao.float8.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    hp_tensor_and_scale_to_float8,
    LinearMMConfig,
    ScaledMMConfig,
)
from torchao.float8.float8_utils import (
    compute_error,
    e4m3_dtype,
    e5m2_dtype,
    fp8_tensor_statistics,
    FP8_TYPES,
    tensor_to_scale,
)
from torchao.testing.float8.test_utils import get_test_float8_linear_config

random.seed(0)
torch.manual_seed(0)


is_cuda_8_9 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
is_cuda_9_0 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)


def bitwise_identical(a: Float8Tensor, b: Float8Tensor) -> bool:
    assert torch.all(a._scale == b._scale).item(), "scales are not identical"
    assert torch.all(a._data == b._data).item(), "data is not identical"
    return True


class TestFloat8Tensor:
    def test_preserves_dtype(self) -> None:
        # hp means high precision, lp means low precision
        hp_dtypes = (torch.float32, torch.float16, torch.bfloat16)
        lp_dtypes = FP8_TYPES
        for hp_dtype, lp_dtype in itertools.product(hp_dtypes, lp_dtypes):
            x1_hp = torch.randn(4, 4, dtype=hp_dtype)
            x1_s = tensor_to_scale(x1_hp, lp_dtype)
            x2_lp = hp_tensor_and_scale_to_float8(x1_hp, x1_s, lp_dtype)
            x3_hp = x2_lp.to_original_precision()
            assert x3_hp.dtype == hp_dtype

    def test_differentiable_casts(self) -> None:
        lp_dtypes = (e4m3_dtype, e5m2_dtype)
        for f8_dtype in lp_dtypes:
            x = torch.randn(1).requires_grad_()
            grad = torch.randn(1)
            x_s = tensor_to_scale(x, f8_dtype)
            x_f8 = hp_tensor_and_scale_to_float8(x, x_s, f8_dtype)
            x_f8_hp = x_f8.to_original_precision()
            x_f8_hp.backward(grad)
            # the gradient should be unchanged through both casts
            torch.testing.assert_close(grad, x.grad, rtol=0, atol=0)

    def test_split_cat(self):
        a = torch.rand(16, 16, dtype=torch.bfloat16)
        scale = tensor_to_scale(a, e4m3_dtype)
        fp8_a = hp_tensor_and_scale_to_float8(a, scale, e4m3_dtype)

        splits = torch.split(fp8_a, 16)
        catted = torch.cat(splits, dim=0)
        assert bitwise_identical(fp8_a, catted)

    def test_index_put(self):
        a = torch.rand(16, dtype=torch.bfloat16)
        scale_a = tensor_to_scale(a, e4m3_dtype)
        fp8_a = hp_tensor_and_scale_to_float8(a, scale_a, e4m3_dtype)

        index = torch.randint(0, 15, (16,), dtype=torch.long)

        b = torch.rand(16, 16, dtype=torch.bfloat16)
        scale_b = tensor_to_scale(b, e4m3_dtype)
        fp8_b = hp_tensor_and_scale_to_float8(b, scale_a, e4m3_dtype)
        fp8_b_bad = hp_tensor_and_scale_to_float8(b, scale_b, e4m3_dtype)

        with pytest.raises(AssertionError):
            b[index] = fp8_a
            fp8_b[index] = a
            fp8_b_bad[index] = fp8_a
        fp8_b[index] = fp8_a

    def test_copy_(self):
        a = torch.rand(16, dtype=torch.bfloat16)
        scale_a = tensor_to_scale(a, e4m3_dtype)
        fp8_a = hp_tensor_and_scale_to_float8(a, scale_a, e4m3_dtype)

        b = torch.empty(16, dtype=torch.bfloat16)
        b.copy_(fp8_a)  # Should work
        torch.testing.assert_close(b, fp8_a.to_original_precision())
        with pytest.raises(RuntimeError):
            fp8_a.copy_(b)  # Should fail

        fp8_b = Float8Tensor(
            torch.empty(16, dtype=e4m3_dtype),
            scale_a,
            torch.bfloat16,
            fp8_a._linear_mm_config,
        )
        fp8_b.copy_(fp8_a)
        torch.testing.assert_close(fp8_a._data, fp8_b._data)

    @pytest.mark.parametrize("shape", [(8, 16), (4, 8, 16), (2, 4, 8, 16)])
    @pytest.mark.parametrize("axiswise_dim", [0, -1])
    def test_axiswise_dynamic_cast(self, shape, axiswise_dim):
        a = torch.randn(*shape, dtype=torch.bfloat16)
        linear_mm_config = LinearMMConfig()
        a_fp8 = hp_tensor_to_float8_dynamic(
            a,
            e4m3_dtype,
            linear_mm_config,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=axiswise_dim,
        )
        a_dq = a_fp8.to_original_precision()
        sqnr = compute_error(a, a_dq)
        assert sqnr >= 25.0

    def test_axiswise_reshape(self):
        a = torch.randn(3, 5, 7, dtype=torch.bfloat16)
        linear_mm_config = LinearMMConfig()

        # if we scale across dim0, we can only reshape to [3, -1]
        a_fp8_d0 = hp_tensor_to_float8_dynamic(
            a,
            e4m3_dtype,
            linear_mm_config,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=0,
        )
        assert list(a_fp8_d0._data.shape) == [3, 5, 7]
        assert list(a_fp8_d0._scale.shape) == [1, 5, 7]

        a_fp8_d0_r = a_fp8_d0.reshape(3, -1)
        assert list(a_fp8_d0_r.shape) == [3, 5 * 7]
        assert list(a_fp8_d0_r._scale.shape) == [1, 5 * 7]
        # verify numerics did not change
        assert torch.allclose(
            a_fp8_d0.to_original_precision(),
            a_fp8_d0_r.to_original_precision().reshape(3, 5, 7),
            atol=0,
            rtol=0,
        )
        with pytest.raises(RuntimeError):
            a_fp8_d0_r2 = a_fp8_d0.reshape(-1, 7)

        # if we scale across dim2, we can only reshape to [-1, 7]
        a_fp8_d2 = hp_tensor_to_float8_dynamic(
            a,
            e4m3_dtype,
            linear_mm_config,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
        )
        assert list(a_fp8_d2._data.shape) == [3, 5, 7]
        assert list(a_fp8_d2._scale.shape) == [3, 5, 1]

        a_fp8_d2_r = a_fp8_d2.reshape(-1, 7)
        assert list(a_fp8_d2_r.shape) == [3 * 5, 7]
        assert list(a_fp8_d2_r._scale.shape) == [3 * 5, 1]
        # verify numerics did not change
        assert torch.allclose(
            a_fp8_d2.to_original_precision(),
            a_fp8_d2_r.to_original_precision().reshape(3, 5, 7),
            atol=0,
            rtol=0,
        )
        with pytest.raises(RuntimeError):
            a_fp8_d2_r2 = a_fp8_d2.reshape(3, -1)

    @pytest.mark.parametrize("a_shape", [(16, 32), (2, 16, 32), (1, 2, 16, 32)])
    @pytest.mark.parametrize(
        "a_granularity,b_granularity",
        [
            (ScalingGranularity.AXISWISE, ScalingGranularity.AXISWISE),
            (ScalingGranularity.AXISWISE, ScalingGranularity.TENSORWISE),
            (ScalingGranularity.TENSORWISE, ScalingGranularity.AXISWISE),
        ],
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(not is_cuda_9_0, "Requires CUDA capability >= 9.0")
    def test_axiswise_gemm(self, a_shape, a_granularity, b_granularity):
        a = torch.randn(*a_shape, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")

        linear_mm_config = LinearMMConfig()

        a_fp8 = hp_tensor_to_float8_dynamic(
            a,
            e4m3_dtype,
            linear_mm_config,
            gemm_input_role=GemmInputRole.INPUT,
            scaling_granularity=a_granularity,
            axiswise_dim=get_maybe_axiswise_dim(-1, a_granularity),
        )
        a_fp8 = a_fp8.reshape(-1, a_shape[-1])

        b_fp8 = hp_tensor_to_float8_dynamic(
            b,
            e4m3_dtype,
            linear_mm_config,
            gemm_input_role=GemmInputRole.WEIGHT,
            scaling_granularity=b_granularity,
            axiswise_dim=get_maybe_axiswise_dim(-1, b_granularity),
        )

        c_fp8_compute = torch.mm(a_fp8, b_fp8.t())
        a = a.reshape(-1, a_shape[-1])
        c_ref = torch.mm(a, b.t())
        sqnr = compute_error(c_ref, c_fp8_compute)
        assert sqnr >= 25.0


class TestFloat8Linear:
    def _test_linear_impl(
        self,
        x,
        m_ref,
        config: Float8LinearConfig,
    ):
        m_fp8 = Float8Linear.from_float(
            copy.deepcopy(m_ref),
            config,
        )
        for _ in range(2):
            if linear_requires_sync(config):
                sync_float8_amax_and_scale_history(m_fp8)
            y_fp8 = m_fp8(x)
            y_fp8.sum().backward()
            y_ref = m_ref(x)
            y_ref.sum().backward()

        assert y_ref.shape == y_fp8.shape

        y_sqnr = compute_error(y_ref, y_fp8)
        g_sqnr = compute_error(m_ref.weight.grad, m_fp8.weight.grad)
        # verify sqnr is reasonable
        assert y_sqnr >= 18.0, f"{y_sqnr} is too low"
        assert g_sqnr >= 17.0, f"{g_sqnr} is too low"
        if m_ref.bias is not None:
            torch.testing.assert_close(m_ref.bias.grad, m_fp8.bias.grad)

        # verify all of the amax buffers got updated
        if linear_requires_sync(config):
            # only check buffers that are actually used, based on per-tensor
            # scaling settings
            amax_buffer_names = []
            amax_history_buffer_names = []
            scale_buffer_names = []
            if config.cast_config_input.scaling_type is ScalingType.DELAYED:
                amax_buffer_names.append("fp8_amax_input")
                amax_history_buffer_names.append("fp8_amax_history_input")
                scale_buffer_names.append("fp8_scale_input")
            if config.cast_config_weight.scaling_type is ScalingType.DELAYED:
                amax_buffer_names.append("fp8_amax_weight")
                amax_history_buffer_names.append("fp8_amax_history_weight")
                scale_buffer_names.append("fp8_scale_weight")
            if config.cast_config_grad_output.scaling_type is ScalingType.DELAYED:
                amax_buffer_names.append("fp8_amax_grad_output")
                amax_history_buffer_names.append("fp8_amax_history_grad_output")
                scale_buffer_names.append("fp8_scale_grad_output")

            # verify all of the amax buffers got updated
            max_float8_pos = {torch.finfo(dtype).max for dtype in FP8_TYPES}
            for buffer_name in amax_buffer_names:
                buffer_value = getattr(m_fp8, buffer_name)
                for init_val in max_float8_pos:
                    assert torch.ne(
                        buffer_value, torch.tensor(init_val)
                    ), f"{buffer_name} not filled, current value {buffer_value}"

            # verify all of the amax history buffers got updated
            for buffer_name in amax_history_buffer_names:
                buffer_value = getattr(m_fp8, buffer_name)
                assert torch.max(buffer_value) > 0.0, f"{buffer_name} not filled"

            # verify all of the scale buffers got updated
            for buffer_name in scale_buffer_names:
                buffer_value = getattr(m_fp8, buffer_name)
                assert torch.ne(
                    buffer_value, torch.tensor(1.0)
                ), f"{buffer_name} not filled, current value {buffer_value}"

            # verify initialization flags got updated
            assert m_fp8.is_amax_initialized, "Amax was not properly initialized"

    @pytest.mark.parametrize("emulate", [True, False] if is_cuda_8_9 else [True])
    @pytest.mark.parametrize("x_shape", [(16, 16), (2, 16, 16), (3, 2, 16, 16)])
    @pytest.mark.parametrize(
        "scaling_type_input",
        [ScalingType.DELAYED, ScalingType.DYNAMIC, ScalingType.STATIC],
    )
    @pytest.mark.parametrize(
        "scaling_type_weight",
        [ScalingType.DELAYED, ScalingType.DYNAMIC, ScalingType.STATIC],
    )
    @pytest.mark.parametrize(
        "scaling_type_grad_output",
        [ScalingType.DELAYED, ScalingType.DYNAMIC],
    )
    @pytest.mark.parametrize("linear_dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("linear_bias", [False, True])
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_linear_from_config_params(
        self,
        x_shape,
        emulate: bool,
        scaling_type_input: ScalingType,
        scaling_type_weight: ScalingType,
        scaling_type_grad_output: ScalingType,
        linear_dtype: torch.dtype,
        linear_bias: bool,
    ):
        x = torch.randn(*x_shape, device="cuda", dtype=linear_dtype)
        m_ref = nn.Linear(16, 32, bias=linear_bias, device="cuda", dtype=linear_dtype)

        config = get_test_float8_linear_config(
            scaling_type_input,
            scaling_type_weight,
            scaling_type_grad_output,
            emulate,
        )

        self._test_linear_impl(
            x,
            m_ref,
            config,
        )

    # Note: there are now too many config combinations to test all of
    # them, so this function factors out some of the recipes which are annoying
    # to combine with the main testing function.
    # TODO(future PR): make this cleaner.
    @pytest.mark.parametrize(
        "recipe_name",
        [
            Float8LinearRecipeName.ALL_AXISWISE,
            Float8LinearRecipeName.LW_AXISWISE_WITH_GW_HP,
        ],
    )
    @pytest.mark.parametrize("x_shape", [(16, 16), (2, 16, 16), (3, 2, 16, 16)])
    @pytest.mark.parametrize("linear_bias", [True, False])
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_linear_from_recipe(
        self,
        recipe_name,
        x_shape,
        linear_bias: bool,
    ):
        if torch.cuda.get_device_capability() < (9, 0):
            warnings.warn(
                f"CUDA capability {torch.cuda.get_device_capability()} < (9.0)"
            )
            pytest.skip()

        linear_dtype = torch.bfloat16
        x = torch.randn(*x_shape, device="cuda", dtype=linear_dtype)
        m_ref = nn.Linear(16, 32, bias=linear_bias, device="cuda", dtype=linear_dtype)
        config = recipe_name_to_linear_config(recipe_name)
        self._test_linear_impl(
            x,
            m_ref,
            config,
        )

    @pytest.mark.parametrize("emulate", [True, False] if is_cuda_8_9 else [True])
    @pytest.mark.parametrize(
        "linear_dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_autocast_outputs(
        self,
        emulate: bool,
        linear_dtype: torch.dtype,
    ):
        m_ref = nn.Linear(32, 16, device="cuda", dtype=linear_dtype)
        config = Float8LinearConfig(
            cast_config_input=CastConfig(scaling_type=ScalingType.DELAYED),
            cast_config_weight=CastConfig(scaling_type=ScalingType.DELAYED),
            cast_config_grad_output=CastConfig(scaling_type=ScalingType.DELAYED),
            emulate=emulate,
        )
        m = Float8Linear.from_float(copy.deepcopy(m_ref), config)

        # autocast off
        x = torch.randn(16, 32, device="cuda", dtype=linear_dtype)
        if linear_requires_sync(config):
            sync_float8_amax_and_scale_history(m)
        y = m(x)
        assert y.dtype == linear_dtype, f"y.dtype is {y.dtype}, expected {linear_dtype}"

        # autocast on
        with torch.autocast("cuda"):
            if linear_requires_sync(config):
                sync_float8_amax_and_scale_history(m)
            y = m(x)
        assert y.dtype == torch.half, f"y.dtype is {y.dtype}, expected {torch.half}"

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if linear_requires_sync(config):
                sync_float8_amax_and_scale_history(m)
            y = m(x)
        assert (
            y.dtype == torch.bfloat16
        ), f"y.dtype is {y.dtype}, expected {torch.bfloat16}"

    @pytest.mark.parametrize(
        "linear_dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    @pytest.mark.parametrize("emulate", [True, False] if is_cuda_8_9 else [True])
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_type_cast(self, linear_dtype: torch.dtype, emulate: bool):
        m = nn.Linear(32, 16, device="cuda", dtype=linear_dtype)
        config = Float8LinearConfig(emulate=emulate)
        m = Float8Linear.from_float(copy.deepcopy(m), config)

        # Cast the module to dtype
        m = m.to(dtype=linear_dtype)
        if linear_requires_sync(config):
            # Check amax buffer types
            for key in [
                "fp8_amax_input",
                "fp8_amax_history_input",
                "fp8_scale_input",
                "fp8_amax_weight",
                "fp8_amax_history_weight",
                "fp8_scale_weight",
                "fp8_amax_grad_output",
                "fp8_amax_history_grad_output",
                "fp8_scale_grad_output",
            ]:
                assert (
                    m._buffers[key].dtype == torch.float32
                ), f"{key}.dtype is {m._buffers[key].dtype}, expected torch.float32"

        # autocast off
        x = torch.randn(16, 32, device="cuda", dtype=linear_dtype)
        if linear_requires_sync(config):
            sync_float8_amax_and_scale_history(m)
        y = m(x)
        assert y.dtype == linear_dtype, f"y.dtype is {y.dtype}, expected {linear_dtype}"

        # autocast on
        with torch.autocast("cuda"):
            if linear_requires_sync(config):
                sync_float8_amax_and_scale_history(m)
            y = m(x)
        assert y.dtype == torch.half, f"y.dtype is {y.dtype}, expected {torch.half}"

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if linear_requires_sync(config):
                sync_float8_amax_and_scale_history(m)
            y = m(x)
        assert (
            y.dtype == torch.bfloat16
        ), f"y.dtype is {y.dtype}, expected {torch.bfloat16}"

    def test_repr(self):
        m = nn.Linear(32, 16)
        config = Float8LinearConfig(
            cast_config_weight=CastConfig(scaling_type=ScalingType.DELAYED),
            emulate=True,
        )
        m = Float8Linear.from_float(
            copy.deepcopy(m),
            config=config,
        )
        s = m.__repr__()
        assert "i:dyn_ten,w:del_ten,go:dyn_ten" in s

    @unittest.skipIf(not is_cuda_8_9, "CUDA 8.9 not available")
    def test_inference_mode(self):
        x = torch.randn(32, 32, device="cuda")
        m = nn.Sequential(nn.Linear(32, 32)).cuda()
        m = convert_to_float8_training(m)
        with torch.inference_mode(mode=True):
            y = m(x)


class TestScaledMM:
    @unittest.skipIf(
        not is_cuda_8_9,
        "CUDA not available",
    )
    @pytest.mark.parametrize(
        "base_dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    @pytest.mark.parametrize("use_fast_accum", [True, False])
    def test_scaled_mm_vs_emulated(self, base_dtype, use_fast_accum):
        torch.manual_seed(42)
        input_dtype = e4m3_dtype
        output_dtype = base_dtype
        compare_type = torch.float32

        a = torch.randn(16, 16, device="cuda", dtype=base_dtype)
        b = torch.randn(32, 16, device="cuda", dtype=base_dtype).t()

        a_scale = tensor_to_scale(a, input_dtype).float()
        b_scale = tensor_to_scale(b, input_dtype).float()

        a_fp8 = hp_tensor_and_scale_to_float8(a, a_scale, input_dtype)
        b_fp8 = hp_tensor_and_scale_to_float8(b, b_scale, input_dtype)

        out_scaled_mm = addmm_float8_unwrapped(
            a_fp8._data,
            a_fp8._scale,
            b_fp8._data,
            b_fp8._scale,
            output_dtype=output_dtype,
            use_fast_accum=use_fast_accum,
        )
        out_emulated = torch.mm(
            a_fp8._data.float() / a_fp8._scale,
            b_fp8._data.float() / b_fp8._scale,
        ).to(output_dtype)

        if output_dtype != base_dtype:
            out_scaled_mm = out_scaled_mm.to(compare_type)
            out_emulated = out_emulated.to(compare_type)

        if base_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 7e-2, 7e-2
        else:
            atol, rtol = 2e-3, 2e-3
        torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)

    @unittest.skipIf(not is_cuda_8_9, "CUDA not available")
    def test_different_configs_error(self):
        x_fp32 = torch.randn(16, 16, device="cuda")
        x_scale = torch.tensor(1.0, device="cuda")
        fp8_dtype = e4m3_dtype
        linear_config_a = LinearMMConfig(
            ScaledMMConfig(False, True, False, False),
            ScaledMMConfig(False, False, False, False),
            ScaledMMConfig(False, False, False, False),
        )
        linear_config_b = LinearMMConfig(
            ScaledMMConfig(True, True, False, False),
            ScaledMMConfig(True, False, False, False),
            ScaledMMConfig(True, False, False, False),
        )
        a = hp_tensor_and_scale_to_float8(
            x_fp32,
            x_scale,
            fp8_dtype,
            linear_config_a,
            GemmInputRole.INPUT,
        )
        b = hp_tensor_and_scale_to_float8(
            x_fp32,
            x_scale,
            fp8_dtype,
            linear_config_b,
            GemmInputRole.WEIGHT,
        )
        with pytest.raises(
            AssertionError,
            match="linear_mm_config.output mismatch",
        ):
            a @ b

    @unittest.skipIf(
        not is_cuda_8_9,
        "CUDA not available",
    )
    @pytest.mark.parametrize(
        "base_dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    @pytest.mark.parametrize("use_fast_accum", [True, False])
    def test_pad_inner_dim(self, base_dtype, use_fast_accum):
        torch.manual_seed(42)
        input_dtype = e4m3_dtype
        compare_type = torch.float32

        a = torch.randn(16, 41, device="cuda", dtype=base_dtype)
        b = torch.randn(41, 128, device="cuda", dtype=base_dtype)

        a_scale = tensor_to_scale(a, input_dtype).float()
        b_scale = tensor_to_scale(b, input_dtype).float()

        a_fp8 = hp_tensor_and_scale_to_float8(
            a, a_scale, input_dtype, None, GemmInputRole.INPUT
        )
        b_fp8 = hp_tensor_and_scale_to_float8(
            b, b_scale, input_dtype, None, GemmInputRole.WEIGHT
        )

        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "Expected trailing dimension of mat1 to be divisible by 16 but got mat1 shape: (16x41."
            ),
        ):
            a_fp8 @ b_fp8

        scaled_mm_config = ScaledMMConfig(False, use_fast_accum, False, True)
        pad_config = LinearMMConfig(
            scaled_mm_config, scaled_mm_config, scaled_mm_config
        )

        a_fp8 = hp_tensor_and_scale_to_float8(
            a,
            a_scale,
            input_dtype,
            pad_config,
            GemmInputRole.INPUT,
        )
        b_fp8 = hp_tensor_and_scale_to_float8(
            b,
            b_scale,
            input_dtype,
            pad_config,
            GemmInputRole.WEIGHT,
        )
        out_padded = a_fp8 @ b_fp8
        out_padded.to(compare_type)

        emulated_scaled_mm_config = ScaledMMConfig(True, use_fast_accum, False, False)
        emulated_config = LinearMMConfig(
            emulated_scaled_mm_config,
            emulated_scaled_mm_config,
            emulated_scaled_mm_config,
        )
        a_fp8 = hp_tensor_and_scale_to_float8(
            a,
            a_scale,
            input_dtype,
            emulated_config,
            GemmInputRole.INPUT,
        )
        b_fp8 = hp_tensor_and_scale_to_float8(
            b,
            b_scale,
            input_dtype,
            emulated_config,
            GemmInputRole.WEIGHT,
        )
        out_emualted = a_fp8 @ b_fp8
        out_emualted.to(compare_type)

        if base_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 7e-2, 7e-2
        else:
            atol, rtol = 2e-3, 2e-3
        torch.testing.assert_close(out_padded, out_emualted, atol=atol, rtol=rtol)


class TestNumerics:
    @pytest.mark.parametrize(
        "float8_dtype",
        [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        ],
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_small_amax_float16(self, float8_dtype):
        # If we calculate scale naively with FP8_MAX_POS / amax,
        # the result may not be representable in fp16. Verify that
        # the way we calculate scales actually works for tensors with
        # small values.
        #
        #   naive_s = fp8_max_pos / (amax + eps)
        #
        # failing case:
        #
        #   fp8_max_pos / (amax + eps) >= fp16_max_pos, or
        #
        #   amax + eps >= fp8_max_pos / fp16_max_pos

        float8_max_pos = torch.finfo(float8_dtype).max
        FP16_MAX_POS = torch.finfo(torch.float16).max

        target_amax = float8_max_pos / (FP16_MAX_POS + 1e-12)
        x = torch.tensor([target_amax], dtype=torch.float16, device="cuda")
        scale = tensor_to_scale(x, float8_dtype)
        assert not torch.any(torch.isinf(scale))


class TestFloat8LinearUtils(unittest.TestCase):
    def test_swap_root_linear(self):
        for emulate in [True, False]:
            module = nn.Linear(3, 3)
            config = Float8LinearConfig(emulate=emulate)
            module = convert_to_float8_training(module, config=config)
            self.assertIsInstance(module, Float8Linear)
            self.assertEqual(module.linear_mm_config.output.emulate, emulate)
            self.assertEqual(module.linear_mm_config.output.emulate, emulate)

    def test_swap_root_linear_with_children_raises(self):
        for emulate in [True, False]:
            module = nn.Linear(3, 3)
            module.child = nn.Sequential(nn.Linear(3, 3))
            config = Float8LinearConfig(emulate=emulate)
            with self.assertRaisesRegex(
                AssertionError,
                "Does not support a root nn.Linear with children",
            ):
                convert_to_float8_training(module, config=config)

    def test_swap_submodule_linears(self):
        class MLP(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.lin1 = nn.Linear(dim, 4 * dim)
                self.lin2 = nn.Linear(4 * dim, dim)

        for emulate in [True, False]:
            model = nn.Sequential(MLP(3), nn.Linear(3, 3), MLP(3))
            config = Float8LinearConfig(emulate=emulate)
            model = convert_to_float8_training(model, config=config)
            self.assertIsInstance(model[0].lin1, Float8Linear)
            self.assertIsInstance(model[0].lin2, Float8Linear)
            self.assertIsInstance(model[1], Float8Linear)
            self.assertIsInstance(model[2].lin1, Float8Linear)
            self.assertIsInstance(model[2].lin2, Float8Linear)

    def test_swap_linears_with_filters(self):
        class MLP(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.lin1 = nn.Linear(dim, 4 * dim)
                self.lin2 = nn.Linear(4 * dim, 4 * dim)

        model = nn.Sequential(MLP(8), nn.Linear(32, 32), MLP(40))
        # filter out the linear layers whose shape is smaller than 32 or non-divisible by 16.

        size_limit = 32

        def module_filter_fn(mod, fqn):
            return (
                mod.in_features >= size_limit
                and mod.out_features >= size_limit
                and mod.in_features % 16 == 0
                and mod.out_features % 16 == 0
            )

        config = Float8LinearConfig(emulate=True)
        model = convert_to_float8_training(
            model,
            config=config,
            module_filter_fn=module_filter_fn,
        )
        # in_features=8, out_features=32, 8 is less than 32.
        self.assertNotIsInstance(model[0].lin1, Float8Linear)
        # in_features=32, out_features=32,
        self.assertIsInstance(model[0].lin2, Float8Linear)
        # in_features=32, out_features=32,
        self.assertIsInstance(model[1], Float8Linear)
        # in_features=40, out_features=160, 40 is not divisible by 16.
        self.assertNotIsInstance(model[2].lin1, Float8Linear)
        # in_features=160, out_features=160,
        self.assertIsInstance(model[2].lin2, Float8Linear)

    def test_swap_submodule_linears_with_skip(self):
        class MLP(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.lin1 = nn.Linear(dim, 4 * dim)
                self.lin2 = nn.Linear(4 * dim, dim)

        model = nn.Sequential(MLP(3), nn.Linear(3, 3), MLP(3))
        module_filter_fn = lambda mod, fqn: fqn not in [
            "0.lin2",
            "2.lin1",
        ]
        config = Float8LinearConfig(emulate=True)
        model = convert_to_float8_training(
            model,
            config=config,
            module_filter_fn=module_filter_fn,
        )
        self.assertTrue(type(model[0].lin1) is Float8Linear)
        self.assertTrue(type(model[0].lin2) is nn.Linear)
        self.assertTrue(type(model[1]) is Float8Linear)
        self.assertTrue(type(model[2].lin1) is nn.Linear)
        self.assertTrue(type(model[2].lin2) is Float8Linear)

    def test_fp8_tensor_statistics(self):
        hp_dtypes = (torch.float32, torch.float16, torch.bfloat16)
        lp_dtypes = (e4m3_dtype, e5m2_dtype)
        for hp_dtype, lp_dtype in itertools.product(hp_dtypes, lp_dtypes):
            x1_hp = torch.ones(4, 4, dtype=hp_dtype)
            tensor_len = x1_hp.numel()

            # Overflow caused by a too large scaling factor
            s_overflow = torch.tensor(1e9)
            fp8_overflow = hp_tensor_and_scale_to_float8(x1_hp, s_overflow, lp_dtype)
            (zero_cnt, max_cnt) = fp8_tensor_statistics(fp8_overflow, lp_dtype)
            self.assertEqual((zero_cnt, max_cnt), (0, tensor_len))

            # Underflow caused by a too small scaling factor
            s_underflow = torch.tensor(1e-9)
            fp8_underflow = hp_tensor_and_scale_to_float8(x1_hp, s_underflow, lp_dtype)
            (zero_cnt, max_cnt) = fp8_tensor_statistics(fp8_underflow, lp_dtype)
            self.assertEqual((zero_cnt, max_cnt), (tensor_len, 0))

            # Both overflow and underflow
            x2_hp = torch.cat((x1_hp * 1e9, x1_hp * 1.0, x1_hp * 1e-9), 0)
            fp8_over_underflow = hp_tensor_and_scale_to_float8(
                x2_hp, torch.tensor(1.0), lp_dtype
            )
            (zero_cnt, max_cnt) = fp8_tensor_statistics(fp8_over_underflow, lp_dtype)
            self.assertEqual((zero_cnt, max_cnt), (tensor_len, tensor_len))


if __name__ == "__main__":
    pytest.main([__file__])
