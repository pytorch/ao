# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
import copy
import itertools
import logging
import os
import unittest
from functools import partial

import torch
import torch.nn as nn
from parameterized import parameterized
from torch._dynamo import config
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck

import torchao
from torchao.dtypes import Int4CPULayout, Int4XPULayout, TensorCoreTiledLayout
from torchao.quantization import safe_int_mm
from torchao.quantization.autoquant import (
    AQFloat8PerRowScalingDynamicallyQuantizedLinearWeight,
    AQFloat8PerTensorScalingDynamicallyQuantizedLinearWeight,
    AQFloat8WeightOnlyQuantizedLinearWeight,
    AQGemliteInt4G64WeightOnlyQuantizedLinearWeight,
    AQInt4G32WeightOnlyQuantizedLinearWeight,
    AQInt4G128WeightOnlyQuantizedMarlinSparseLinearWeight,
    AQInt8DynamicallyQuantizedLinearWeight,
    AQInt8WeightOnlyQuantizedLinearWeight,
    AQInt8WeightOnlyQuantizedLinearWeight2,
    AQInt8WeightOnlyQuantizedLinearWeight3,
    AutoQuantizableLinearWeight,
)

# APIs to be deprecated (used for torch 2.2.2 and 2.3)
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    _replace_with_custom_fn_if_matches_filter,
    int4_weight_only,
    int8_dynamic_activation_int4_weight,
    int8_dynamic_activation_int8_weight,
    int8_weight_only,
    quantize_,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    dequantize_affine,
)
from torchao.quantization.smoothquant import (
    SmoothFakeDynamicallyQuantizedLinear,
    get_scale,
    smooth_fq_linear_to_inference,
    swap_linear_with_smooth_fq_linear,
)
from torchao.quantization.subclass import (
    Int4WeightOnlyQuantizedLinearWeight,
    Int8DynamicallyQuantizedLinearWeight,
    Int8WeightOnlyQuantizedLinearWeight,
)
from torchao.quantization.utils import (
    LoggingTensorMode,
    _apply_logging_hook,
    _fqn_to_op_to_shape_to_count,
    _quant_int8_dynamic_per_token_linear,
    _quantize_activation_per_token_absmax,
    compute_error,
    dequantize_per_channel,
    dynamically_quantize_per_channel,
)
from torchao.quantization.utils import (
    compute_error as SQNR,
)
from torchao.testing.utils import skip_if_rocm
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_7,
    benchmark_model,
    check_cpu_version,
    check_xpu_version,
    is_fbcode,
    is_sm_at_least_89,
    is_sm_at_least_90,
    unwrap_tensor_subclass,
)

try:
    import gemlite  # noqa: F401

    has_gemlite = True
except ModuleNotFoundError:
    has_gemlite = False


logger = logging.getLogger("INFO")

torch.manual_seed(0)
config.cache_size_limit = 100

COMMON_DEVICES = ["cpu", "cuda"]

COMMON_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

ACT_MAPPING_TYPES = [MappingType.ASYMMETRIC, MappingType.SYMMETRIC]

COMMON_DEVICE_DTYPE = list(itertools.product(COMMON_DEVICES, COMMON_DTYPES)).copy()


def _int8wo_api(mod):
    quantize_(mod, int8_weight_only(set_inductor_config=False))


def _int8wo_groupwise_api(mod):
    group_size = 32
    quantize_(mod, int8_weight_only(group_size=group_size, set_inductor_config=False))


def _int8da_int8w_api(
    mod,
    act_mapping_type=MappingType.SYMMETRIC,
):
    quantize_(
        mod,
        int8_dynamic_activation_int8_weight(
            act_mapping_type=act_mapping_type,
            set_inductor_config=False,
        ),
    )


def _int4wo_api(mod, use_hqq=False):
    if check_cpu_version(next(mod.parameters()).device):
        quantize_(
            mod,
            int4_weight_only(
                layout=Int4CPULayout(), use_hqq=use_hqq, set_inductor_config=False
            ),
        )
        unwrap_tensor_subclass(mod)
    elif check_xpu_version(next(mod.parameters()).device):
        quantize_(
            mod, int4_weight_only(layout=Int4XPULayout()), set_inductor_config=False
        )
        unwrap_tensor_subclass(mod)
    else:
        quantize_(mod, int4_weight_only(set_inductor_config=False))


def _int8da_int4w_api(mod):
    quantize_(mod, int8_dynamic_activation_int4_weight(set_inductor_config=False))


# TODO: use this to reduce the number of tests
TENSOR_SUBCLASS_APIS = [
    _int8wo_api,
    _int8da_int8w_api,
    _int4wo_api,
]


def undo_recommended_configs():
    torch._inductor.config.coordinate_descent_tuning = False
    torch._inductor.config.coordinate_descent_check_all_directions = False
    torch._inductor.config.force_fuse_int_mm_with_mul = False
    torch._inductor.config.fx_graph_cache = False
    torch._inductor.config.triton.unique_kernel_names = False
    torch.set_float32_matmul_precision("highest")


def combine_parameters(a, b):
    new_tuples = []
    for tuple1, tuple2 in itertools.product(a, b):
        new_tuples.append(tuple1 + tuple2)
    return new_tuples


def run_supported_device_dtype(test_method):
    """Assumes that the 3rd arg (args[2]) of the decorated method is device and
    there is a `test_dtype` kwarg or the 4th arg (args[3]) that indicates the dtype for testing
    """

    def wrapper(*args, **kwargs):
        if len(args) < 3:
            raise unittest.SkipTest(
                f"Not enough args. Expected more than or equal to 3, but got {len(args)}"
            )
        device = args[2]
        dtype = kwargs["test_dtype"] if "test_dtype" in kwargs else args[3]
        if device == "cuda" and not torch.cuda.is_available():
            raise unittest.SkipTest("Need CUDA available.")
        if (
            device == "cuda"
            and torch.cuda.is_available()
            and dtype == torch.bfloat16
            and torch.cuda.get_device_capability() < (8, 0)
        ):
            raise unittest.SkipTest("Need CUDA and SM80+ available.")
        return test_method(*args, **kwargs)

    return wrapper


class SmoothquantUnitTest(unittest.TestCase):
    # first, let's reproduce the graphic from the paper, Figure 4, to ensure
    # we are calculating the scales correctly
    def test_figure_4(self):
        X = torch.FloatTensor([1, -16, 2, 6, -2, 8, -1, -9]).reshape(1, 2, 4)
        W = torch.FloatTensor([2, 1, -2, 1, -1, -1, 2, -1, -2, -1, -1, 1]).reshape(4, 3)
        X_mul_W = torch.matmul(X, W)

        smoothquant_scale = get_scale(
            torch.amax(torch.abs(X), dim=(0, 1)),
            torch.amax(torch.abs(W), dim=1),
            alpha=0.5,
        )

        # reproduce scaled calculation
        X_scaled = X / smoothquant_scale.reshape(1, 1, -1)
        W_scaled = torch.matmul(torch.diag(smoothquant_scale), W)
        X_scaled_mul_scaled_W = torch.matmul(X_scaled, W_scaled)
        assert torch.allclose(X_mul_W, X_scaled_mul_scaled_W), "not close!"
        assert X_mul_W.shape == X_scaled_mul_scaled_W.shape

    # next, run the above test on a sample of representative inputs
    def test_tensors(self):
        x_shape = (1, 5, 7)
        w_shape = (7, 9)
        for i in range(3):
            X = torch.randn(x_shape) * 10
            W = torch.randn(w_shape)
            s = get_scale(
                torch.amax(torch.abs(X), dim=(0, 1)),
                torch.amax(torch.abs(W), dim=1),
                alpha=0.5,
            )

            Y = torch.matmul(X, W)
            Y_ref = torch.matmul(
                X / s.reshape(1, 1, -1),
                torch.matmul(torch.diag(s), W),
            )
            assert torch.allclose(Y, Y_ref, atol=1e-3, rtol=1e-3), "not close!"

    def _test_smooth_linear_impl(self, x_shape, lin_shape, device):
        orig_backend = torch.backends.quantized.engine
        # so we can use the full range
        torch.backends.quantized.engine = "qnnpack"

        x = torch.randn(*x_shape, device=device) * 9 + 10

        lin_fp32 = nn.Linear(*lin_shape, device=device)  # misc: ignore
        lin_smooth = SmoothFakeDynamicallyQuantizedLinear.from_float(
            copy.deepcopy(lin_fp32), alpha=0.25
        )
        lin_smooth_skip_scaling = SmoothFakeDynamicallyQuantizedLinear.from_float(
            copy.deepcopy(lin_fp32), alpha=0.25
        )

        lin_fp32_copy = copy.deepcopy(lin_fp32)  # assignment: ignore
        lin_fp32_copy.qconfig = torch.ao.quantization.QConfig(  # assignment: ignore
            activation=None,
            weight=torch.ao.quantization.default_per_channel_weight_observer,
        )
        lin_dynamic_q = torch.ao.nn.quantized.dynamic.Linear.from_float(
            lin_fp32_copy.cpu()
        )

        y_ref = lin_fp32(x)

        # calibrate the smoothquant versions
        y_smooth_nocalib = lin_smooth(x)
        _ = lin_smooth_skip_scaling(x)
        lin_smooth.to_inference()
        lin_smooth_skip_scaling.debug_skip_scaling = True
        lin_smooth_skip_scaling.to_inference()

        # verify that with scaling turned off, numerics match quantized version
        y_smooth_fq_only = lin_smooth_skip_scaling(x)
        y_smooth_fq = lin_smooth(x)
        y_dynamic_q = lin_dynamic_q(x.cpu()).to(device)

        # print('y_ref', y_ref)
        # print('y_smooth_nocalib', y_smooth_nocalib)
        # print('y_smooth_fq', y_smooth_fq)
        # print('y_smooth_fq_only', y_smooth_fq_only)
        # print('y_dynamic_q', y_dynamic_q)

        sqnr_smooth_fq = compute_error(y_ref, y_smooth_fq)
        sqnr_dynamic_q = compute_error(y_ref, y_dynamic_q)
        sqnr_fq = compute_error(y_smooth_fq_only, y_dynamic_q)
        # print('sqnr_smooth', sqnr_smooth_fq, 'sqnr_dynamic', sqnr_dynamic_q, 'sqnr_fq', sqnr_fq)

        assert torch.allclose(y_ref, y_smooth_nocalib), (
            "y_ref not close to y_smooth_nocalib"
        )
        # after https://github.com/pytorch-labs/ao_benchmarks/pull/32,
        # numerics do not match exactly between production c++ code
        # and this Python code
        # assert torch.allclose(
        #     y_smooth_fq_only, y_dynamic_q,
        #     atol=torch.max(y_smooth_fq_only).item()*0.01,
        #     rtol=0.00001), \
        #     'y_smooth_fq_only not close to y_dynamic_q'

        self.assertTrue(sqnr_smooth_fq.item() >= 40.0, f"got: {sqnr_smooth_fq.item()}")
        self.assertTrue(sqnr_dynamic_q.item() >= 40.0, f"got: {sqnr_dynamic_q.item()}")
        self.assertTrue(sqnr_fq.item() >= 40.0, f"got: {sqnr_fq.item()}")

        # Restore backend
        torch.backends.quantized.engine = orig_backend

    def test_smooth_linear_cpu(self):
        self._test_smooth_linear_impl((1, 5, 3), (3, 4), "cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_smooth_linear_cuda(self):
        self._test_smooth_linear_impl((1, 32, 32), (32, 16), "cuda")

    def test_smooth_linear_edge_cases(self):
        orig_backend = torch.backends.quantized.engine
        # so we can use the full range
        torch.backends.quantized.engine = "qnnpack"
        lin_fp32 = nn.Linear(3, 4)
        lin_smooth = SmoothFakeDynamicallyQuantizedLinear.from_float(
            lin_fp32, alpha=0.25
        )

        # test different ranks
        x0 = torch.randn(4, 5, 3)
        x1 = torch.randn(1, 8, 5, 3)
        x2 = torch.randn(2, 3, 7, 5, 3)

        # calibrate
        _ = lin_smooth(x0)
        _ = lin_smooth(x1)
        _ = lin_smooth(x2)

        # inference
        lin_smooth.to_inference()
        _ = lin_smooth(x0)
        _ = lin_smooth(x1)
        _ = lin_smooth(x2)

        # Restore backend
        torch.backends.quantized.engine = orig_backend

    def test_swap(self):
        m = nn.Sequential(
            nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4)),
            nn.Linear(4, 4),
        )
        m_copy = copy.deepcopy(m)
        swap_linear_with_smooth_fq_linear(m_copy, skip_fqn_list=["0.2"])

        # verify all linears are swapped
        assert isinstance(m_copy[0][0], SmoothFakeDynamicallyQuantizedLinear)
        assert isinstance(m_copy[0][1], nn.ReLU)
        # this one was skipped
        assert isinstance(m_copy[0][2], nn.Linear)
        assert isinstance(m_copy[1], SmoothFakeDynamicallyQuantizedLinear)

        # verify results do not change without smoothing
        x = torch.randn(4, 4)
        y_ref = m(x)
        y = m_copy(x)
        assert torch.allclose(y_ref, y)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_weight_t_and_non_t_numerics_match(self):
        # verify that numerics match whether weight is stored
        # in transposed format (for cuBLAS) vs non-transposed format
        # (for torch.compile)
        dtype = torch.half
        device = "cuda"
        lin_ref = nn.Linear(32, 16, dtype=dtype, device=device)
        lin_eager_t = copy.deepcopy(lin_ref)
        lin_opt_t = copy.deepcopy(lin_eager_t)
        lin_opt = copy.deepcopy(lin_eager_t)
        lin_eager_t = SmoothFakeDynamicallyQuantizedLinear.from_float(lin_eager_t)
        lin_opt_t = SmoothFakeDynamicallyQuantizedLinear.from_float(lin_opt_t)
        lin_opt = SmoothFakeDynamicallyQuantizedLinear.from_float(lin_opt)
        lin_opt.store_w_int_repr_t = False

        x = torch.randn(32, 32, dtype=dtype, device=device)

        y_calib_eager_t = lin_eager_t(x)
        y_calib_opt_t = lin_opt_t(x)
        y_calib_opt = lin_opt(x)
        torch.testing.assert_close(y_calib_eager_t, y_calib_opt_t)
        torch.testing.assert_close(y_calib_eager_t, y_calib_opt)

        lin_eager_t.to_inference()
        lin_opt_t.to_inference()
        lin_opt.to_inference()

        torch.testing.assert_close(lin_eager_t.W_int_repr, lin_opt_t.W_int_repr)
        torch.testing.assert_close(lin_eager_t.W_int_repr, lin_opt.W_int_repr)

        lin_opt_t = torch.compile(lin_opt_t, mode="max-autotune")
        lin_opt = torch.compile(lin_opt, mode="max-autotune")

        y_ref = lin_ref(x)
        y_eager = lin_eager_t(x)
        y_opt_t = lin_opt_t(x)
        y_opt = lin_opt(x)

        if not torch.any(torch.isinf(y_ref)) and torch.any(torch.isinf(y_eager)):
            # eager mode torch._int_mm is sometimes buggy, when this happens
            # we can't really compare the compiled version against it properly
            print("eager mode torch._int_mm known bad, test is inconclusive")
            return

        sqnr_eager_opt_t = compute_error(y_eager, y_opt_t)
        sqnr_eager_opt = compute_error(y_eager, y_opt)
        # since torch.compile for a torch.half model can
        # change numerics significantly, we can only test for a high SQNR here
        # and not for closeness
        self.assertTrue(sqnr_eager_opt_t >= 45.0)
        self.assertTrue(sqnr_eager_opt >= 45.0)
        # y_opt_t and y_opt should be equivalent
        torch.testing.assert_close(y_opt_t, y_opt)

    def test_selective_torch_compile(self):
        m = nn.Sequential(
            nn.Linear(4, 4),
            nn.Sequential(
                nn.Linear(4, 4),
                nn.Linear(4, 4),
            ),
            nn.Linear(4, 4),
        )
        x = torch.randn(4, 4)
        y_ref = m(x)

        _replace_with_custom_fn_if_matches_filter(
            m,
            lambda mod: torch.compile(mod),
            lambda mod, fqn: isinstance(mod, nn.Linear) and fqn != "1.0",
        )

        self.assertTrue(isinstance(m[0], torch._dynamo.eval_frame.OptimizedModule))
        self.assertTrue(isinstance(m[1][0], nn.Linear))
        self.assertTrue(isinstance(m[1][1], torch._dynamo.eval_frame.OptimizedModule))
        self.assertTrue(isinstance(m[2], torch._dynamo.eval_frame.OptimizedModule))

        y = m(x)
        torch.testing.assert_close(y, y_ref)

    def test_debug_x_absmax(self):
        m = nn.Sequential(nn.Linear(3, 4))
        x0 = torch.randn(4, 5, 3)
        m(x0)
        swap_linear_with_smooth_fq_linear(m)
        # no calibration, straight to inference, should not crash
        smooth_fq_linear_to_inference(m, debug_skip_calibration=True)
        m(x0)


class PythonQuantUtilOpUnitTest(unittest.TestCase):
    def _test_dynamic_quant_per_channel_numerics_impl(
        self, qmin, qmax, int_dtype, qint_dtype, float_dtype, device
    ):
        # verifies that dynamic quant per channel in plain pytorch matches
        # numerics of production AO code
        # TODO(future): test this on cpu-half, need to first make
        # torch.aminmax support half on cpu

        x = torch.randn(16, 32, device=device, dtype=float_dtype)
        y_vals, y_scale, y_zero_point = dynamically_quantize_per_channel(
            x, qmin, qmax, int_dtype
        )

        min_val, max_val = torch.aminmax(x, dim=1)

        # reference
        weight_obs = torch.ao.quantization.MovingAveragePerChannelMinMaxObserver(
            dtype=qint_dtype,
            quant_min=qmin,
            quant_max=qmax,
            qscheme=torch.per_channel_symmetric,
            averaging_constant=1.0,  # make it ignore previous iterations
        )
        weight_obs(x)
        y_ref_scale, y_ref_zp = weight_obs.calculate_qparams()
        y_ref_scale = y_ref_scale.to(device)
        y_ref_zp = y_ref_zp.to(device)
        # quantize_per_channel doesn't work for half, so we cast there and back
        x_for_ref = x.half().float() if float_dtype == torch.float16 else x
        y_ref = torch.quantize_per_channel(
            x_for_ref, y_ref_scale, y_ref_zp, 0, qint_dtype
        )

        torch.testing.assert_close(
            y_scale, y_ref.q_per_channel_scales().to(float_dtype)
        )
        assert torch.equal(y_zero_point, y_ref.q_per_channel_zero_points())
        # this test case has one element where the rounding is off by one
        # from Python-only code vs the c++ code, it's easy to repro with
        # various shapes.
        # Discussion here is relevant: https://github.com/pytorch/pytorch/issues/16498
        # TODO(future): figure out what to do about this
        # assert torch.equal(int_vals, q_reference.int_repr())
        assert torch.max(torch.abs(y_vals - y_ref.int_repr())) <= 1

        # dequantize
        x_dq = dequantize_per_channel(
            y_vals, y_scale, y_zero_point, out_dtype=float_dtype
        )
        x_ref_dq = y_ref.dequantize().to(float_dtype)
        # off-by-one for scale is okay
        torch.testing.assert_close(
            x_dq, x_ref_dq, atol=torch.max(y_scale).item() * 1.01, rtol=0.0001
        )

    def test_dynamic_quant_per_channel_numerics_cpu(self):
        test_cases = ((-128, 127, torch.int8, torch.qint8, torch.float32, "cpu"),)
        for row in test_cases:
            self._test_dynamic_quant_per_channel_numerics_impl(*row)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skip("AssertionError: Tensor-likes are not close!")
    def test_dynamic_quant_per_channel_numerics_cuda(self):
        test_cases = (
            (-128, 127, torch.int8, torch.qint8, torch.float32, "cuda"),
            (-128, 127, torch.int8, torch.qint8, torch.float16, "cuda"),
        )
        for row in test_cases:
            self._test_dynamic_quant_per_channel_numerics_impl(*row)

    def _test_quantize_per_token_impl(self, device, dtype):
        x = torch.randn(3, 3, 3, device=device, dtype=dtype)
        xq, scales = _quantize_activation_per_token_absmax(x)
        block_size = (1, 1, 3)
        x_dq = dequantize_affine(
            xq, block_size, scales, None, torch.int8, output_dtype=x.dtype
        )
        sqnr = compute_error(x, x_dq)
        self.assertTrue(sqnr >= 45.0)

    def test_quantize_per_token_cpu(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._test_quantize_per_token_impl("cpu", dtype)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_quantize_per_token_cuda(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._test_quantize_per_token_impl("cuda", dtype)

    def _test_per_token_linear_impl(self, device, dtype):
        x = torch.randn(2, 16, 8, device=device, dtype=dtype)
        w = torch.randn(16, 8, device=device, dtype=dtype)
        wq, w_scales, _w_zp = dynamically_quantize_per_channel(w, -127, 127, torch.int8)
        # Note: need to make the weight contiguous because we are
        # testing in eager mode and cuBlas will not give correct results
        # for a transposed weight
        y = _quant_int8_dynamic_per_token_linear(
            x, wq.t().contiguous(), w_scales, None, dtype
        )
        y_ref = torch.matmul(x, w.t())
        sqnr = compute_error(y_ref, y)
        self.assertTrue(sqnr >= 42.0)

    def test_per_token_linear_cpu(self):
        for dtype in (torch.float32,):
            self._test_per_token_linear_impl("cpu", dtype)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @skip_if_rocm("ROCm enablement in progress")
    def test_per_token_linear_cuda(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._test_per_token_linear_impl("cuda", dtype)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test__int_mm(self):
        # TODO(future): figure out what here needs to move to PT core,
        # if it's not already tested there

        m, k, n = 32, 32, 16
        x = torch.randint(-128, 127, (m, k), dtype=torch.int8, device="cuda")
        w = torch.randint(-128, 127, (k, n), dtype=torch.int8, device="cuda")

        y_ref = torch.matmul(x.float(), w.float()).to(torch.int32)
        y_raw = safe_int_mm(x, w)

        wrap_in_mm_opt = torch.compile(safe_int_mm, mode="max-autotune")
        # note: triton chokes on the line below on k == 8 and n == 8 with
        # https://www.internalfb.com/phabricator/paste/view/P683467944
        # TODO(future): file an issue
        y_opt = wrap_in_mm_opt(x, w)

        torch.testing.assert_close(y_ref, y_raw, atol=0, rtol=0)
        torch.testing.assert_close(y_ref, y_opt, atol=0, rtol=0)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test__int_mm_eager_and_torch_compile_numerics(self):
        def __int_mm_ref(x, w):
            x = x.cpu().to(torch.int32)
            w = w.cpu().to(torch.int32)
            y = torch.matmul(x, w)
            return y.cuda()

        shapes = (
            # minimal test shape
            ((1, 32, 32), (32, 16)),
            # paste of real linear shapes from LLaMa 1.5b
            ((17, 1, 1536), (1536, 1536)),
            ((17, 8, 4096), (4096, 1536)),
            ((17, 1, 1536), (1536, 4096)),
            ((17, 8, 1536), (1536, 1536)),
            ((17, 1, 4096), (4096, 1536)),
            ((17, 8, 1536), (1536, 4096)),
        )

        for x_shape, w_shape in shapes:

            def wrap_torch_int_mm(x, w):
                b, n, k = x.shape
                k, m = w.shape
                x = x.reshape(b * n, k)
                res = safe_int_mm(x, w)
                res = res.reshape(b, n, m)
                return res

            wrap_torch_int_mm_opt = torch.compile(
                wrap_torch_int_mm, mode="max-autotune"
            )

            x = torch.randint(-128, 127, x_shape, dtype=torch.int8, device="cuda")
            w = torch.randint(-128, 127, w_shape, dtype=torch.int8, device="cuda")

            z_ref = __int_mm_ref(x, w)
            z_eager = wrap_torch_int_mm(x, w)
            z_torch_compile = wrap_torch_int_mm_opt(x, w)
            # print(z_ref)
            # print(z_eager)
            # print(z_torch_compile)

            torch.testing.assert_close(z_ref, z_eager, atol=0, rtol=0)
            torch.testing.assert_close(z_ref, z_torch_compile, atol=0, rtol=0)


class TestSubclass(unittest.TestCase):
    @run_supported_device_dtype
    def _test_dequantize_impl(
        self,
        test_subclass_from_float,
        test_device,
        min_sqnr=35,
        test_dtype=torch.bfloat16,
        test_shape=(32, 64, 64),
    ):
        m, k, n = test_shape
        lin = torch.nn.Linear(k, n, device=test_device).to(test_dtype)
        w = lin.weight.detach()
        lin.weight = torch.nn.Parameter(
            test_subclass_from_float(lin.weight), requires_grad=False
        )
        self.assertGreater(
            SQNR(w, lin.weight.dequantize()),
            min_sqnr,
            f"{lin.weight.__class__.__name__} failed dtype={test_dtype}",
        )
        self.assertGreater(
            SQNR(w.t(), lin.weight.t().dequantize()),
            min_sqnr,
            f"{lin.weight.__class__.__name__} failed transpose on dtype={test_dtype}",
        )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    def test_dequantize_int8_dynamic_quant_subclass(self, device, dtype):
        self._test_dequantize_impl(
            Int8DynamicallyQuantizedLinearWeight.from_float,
            device,
            35,
            test_dtype=dtype,
        )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    def test_dequantize_int8_weight_only_quant_subclass(self, device, dtype):
        self._test_dequantize_impl(
            Int8WeightOnlyQuantizedLinearWeight.from_float, device, 35, test_dtype=dtype
        )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @skip_if_rocm("ROCm enablement in progress")
    def test_dequantize_int4_weight_only_quant_subclass(self, device, dtype):
        if device == "cpu":
            self.skipTest(f"Temporarily skipping for {device}")
        if dtype != torch.bfloat16:
            self.skipTest("Currently only supports bfloat16.")
        for test_shape in [(16, 1024, 16)] + (
            [(1, 1024, 8)] if device == "cuda" else []
        ):
            self._test_dequantize_impl(
                Int4WeightOnlyQuantizedLinearWeight.from_float,
                device,
                15,
                test_shape=test_shape,
                test_dtype=dtype,
            )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @skip_if_rocm("ROCm enablement in progress")
    def test_dequantize_int4_weight_only_quant_subclass_grouped(self, device, dtype):
        if device == "cpu":
            self.skipTest(f"Temporarily skipping for {device}")
        if dtype != torch.bfloat16:
            self.skipTest("Currently only supports bfloat16.")
        m_shapes = [16, 256] + ([1] if device == "cuda" else [])
        n_shapes = [16] + ([8, 13] if device == "cuda" else [])
        for groupsize in [256, 128]:
            for inner_k_tiles in [8, 4, 2]:
                for m in m_shapes:
                    for n in n_shapes:
                        self._test_dequantize_impl(
                            lambda w: Int4WeightOnlyQuantizedLinearWeight.from_float(
                                w, groupsize, inner_k_tiles
                            ),
                            device,
                            15,
                            test_shape=[m, 256, n],
                            test_dtype=dtype,
                        )

    @run_supported_device_dtype
    def _test_lin_weight_subclass_impl(
        self,
        test_subclass_from_float,
        test_device,
        min_sqnr=35,
        test_dtype=torch.bfloat16,
        test_shape=(32, 64, 32),
    ):
        if not "cuda" in test_device:
            self.skipTest("test requires cuda")
        with torch.no_grad():
            m, k, n = test_shape
            x = torch.randn(m, k, device=test_device, dtype=test_dtype)
            lin = torch.nn.Linear(k, n, device=test_device).to(test_dtype)
            ref_f = lin(x)

            lin.weight = torch.nn.Parameter(
                test_subclass_from_float(lin.weight), requires_grad=False
            )
            test = lin(x)
            self.assertGreater(
                SQNR(ref_f, test),
                min_sqnr,
                f"{lin.weight.__class__.__name__} failed, no compile, dtype={test_dtype}, (m, k, n)={test_shape}",
            )
            lin_comp = torch.compile(lin, mode="max-autotune")
            test_comp = lin_comp(x)
            self.assertGreater(
                SQNR(ref_f, test_comp),
                min_sqnr,
                f"{lin.weight.__class__.__name__} failed at compile with dtype={test_dtype}, (m, k, n)={test_shape}",
            )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    def test_int8_dynamic_quant_subclass(self, device, dtype):
        self._test_lin_weight_subclass_impl(
            Int8DynamicallyQuantizedLinearWeight.from_float,
            device,
            35,
            test_dtype=dtype,
        )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    def test_int8_weight_only_quant_subclass(self, device, dtype):
        undo_recommended_configs()
        self._test_lin_weight_subclass_impl(
            Int8WeightOnlyQuantizedLinearWeight.from_float, device, 40, test_dtype=dtype
        )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    def test_aq_int8_dynamic_quant_subclass(self, device, dtype):
        self._test_lin_weight_subclass_impl(
            AQInt8DynamicallyQuantizedLinearWeight.from_float,
            device,
            35,
            test_dtype=dtype,
        )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @unittest.skip(
        "This segfaults in CI cuda only, disable to unblock PR, we can investigate "
        "later if needed"
    )
    def test_aq_int8_weight_only_quant_subclass(self, device, dtype):
        self._test_lin_weight_subclass_impl(
            AQInt8WeightOnlyQuantizedLinearWeight.from_float,
            device,
            35,
            test_dtype=dtype,
        )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    def test_aq_int8_weight_only_quant_2_subclass(self, device, dtype):
        self._test_lin_weight_subclass_impl(
            AQInt8WeightOnlyQuantizedLinearWeight2.from_float,
            device,
            35,
            test_dtype=dtype,
        )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    def test_aq_int8_weight_only_quant_3_subclass(self, device, dtype):
        self._test_lin_weight_subclass_impl(
            AQInt8WeightOnlyQuantizedLinearWeight3.from_float,
            device,
            35,
            test_dtype=dtype,
        )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @unittest.skipIf(not is_sm_at_least_90(), "Need H100 to run")
    def test_aq_float8_weight_only_quant_subclass(self, device, dtype):
        self._test_lin_weight_subclass_impl(
            AQFloat8WeightOnlyQuantizedLinearWeight.from_float,
            device,
            30,
            test_dtype=dtype,
        )

    def test_autoquantizable_flatten_unflatten(self):
        from torchao.quantization import DEFAULT_AUTOQUANT_CLASS_LIST

        weight = torch.randn(16, 32)
        qtensor_class_list = DEFAULT_AUTOQUANT_CLASS_LIST
        aqw = AutoQuantizableLinearWeight.from_float(weight, qtensor_class_list)
        tensor_data_name_dict, tensor_attributes = aqw.__tensor_flatten__()
        tensor_data_dict = {name: getattr(aqw, name) for name in tensor_data_name_dict}
        outer_size = aqw.size()
        outer_stride = aqw.stride()
        type(aqw).__tensor_unflatten__(
            tensor_data_dict, tensor_attributes, outer_size, outer_stride
        )

    @parameterized.expand(
        [
            (device, dtype, f"device_{device}_dtype_{str(dtype).split('.')[-1]}")
            for device, dtype in COMMON_DEVICE_DTYPE
        ]
    )
    @unittest.skipIf(not is_sm_at_least_90(), "Need H100 to run")
    @unittest.skip("TODO this is not working correctly")
    def test_aq_float8_dynamic_quant_rowwise_scaling_subclass(
        self, device, dtype, name
    ):
        if dtype != torch.bfloat16:
            with self.assertRaisesRegex(
                AssertionError, "PerRow quantization only works for bfloat16 precision"
            ):
                self._test_lin_weight_subclass_impl(
                    AQFloat8PerRowScalingDynamicallyQuantizedLinearWeight.from_float,
                    device,
                    25,
                    test_dtype=dtype,
                )
        else:
            self._test_lin_weight_subclass_impl(
                AQFloat8PerRowScalingDynamicallyQuantizedLinearWeight.from_float,
                device,
                25,
                test_dtype=dtype,
            )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @unittest.skipIf(not is_sm_at_least_90(), "Need H100 to run")
    @unittest.skip("TODO this is not working correctly")
    def test_aq_float8_dynamic_quant_tensorwise_scaling_subclass(self, device, dtype):
        self._test_lin_weight_subclass_impl(
            AQFloat8PerTensorScalingDynamicallyQuantizedLinearWeight.from_float,
            device,
            25,
            test_dtype=dtype,
        )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @skip_if_rocm("ROCm enablement in progress")
    def test_int4_weight_only_quant_subclass(self, device, dtype):
        if device == "cpu":
            self.skipTest(f"Temporarily skipping for {device}")
        if dtype != torch.bfloat16:
            self.skipTest(f"Fails for {dtype}")
        for test_shape in [(16, 1024, 16)] + (
            [(1, 1024, 8)] if device == "cuda" else []
        ):
            self._test_lin_weight_subclass_impl(
                Int4WeightOnlyQuantizedLinearWeight.from_float,
                device,
                10,
                test_shape=test_shape,
                test_dtype=dtype,
            )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @skip_if_rocm("ROCm enablement in progress")
    @unittest.skip("Skip to fix CI until we deprecate these APIs long term")
    def test_int4_weight_only_quant_subclass_grouped(self, device, dtype):
        if dtype != torch.bfloat16:
            self.skipTest(f"Fails for {dtype}")
        m_shapes = [16, 256] + ([1] if device == "cuda" else [])
        n_shapes = [16] + ([8, 13] if device == "cuda" else [])
        for groupsize in [128, 64]:
            for inner_k_tiles in [8, 4, 2]:
                for m in m_shapes:
                    for n in n_shapes:
                        self._test_lin_weight_subclass_impl(
                            lambda w: Int4WeightOnlyQuantizedLinearWeight.from_float(
                                w, groupsize, inner_k_tiles
                            ),
                            device,
                            10,
                            test_shape=[m, 256, n],
                            test_dtype=dtype,
                        )

    @torch.no_grad()
    @run_supported_device_dtype
    def _test_lin_weight_subclass_api_impl(
        self,
        api,
        test_device,
        min_sqnr=35,
        test_dtype=torch.bfloat16,
        test_shape=(32, 64, 32),
    ):
        m, k, n = test_shape
        x = torch.randn(m, k, device=test_device, dtype=test_dtype)
        mod = nn.Sequential(
            nn.Linear(k, n, device=test_device),
            nn.ReLU(),
            nn.Linear(n, n, device=test_device),
        ).to(test_dtype)
        ref_f = mod(x)
        api(mod)

        # test get_plain()
        if hasattr(mod[0].weight, "tensor_impl"):
            mod[0].weight.tensor_impl.get_plain()

        test = mod(x)

        self.assertGreater(
            SQNR(ref_f, test),
            min_sqnr,
            f"API failed, no compile dtype={test_dtype}, (m, k, n)={test_shape}",
        )

        mod_qc = torch.compile(mod, mode="max-autotune")
        test_comp = mod_qc(x)
        self.assertGreater(
            SQNR(ref_f, test_comp),
            min_sqnr,
            f"API failed when compiled with dtype={test_dtype}, (m, k, n)={test_shape}",
        )

    @parameterized.expand(
        list(
            itertools.product(
                COMMON_DEVICES,
                COMMON_DTYPES,
                ACT_MAPPING_TYPES,
            )
        )
    )
    @unittest.skip("skip because there is some bug in inductor codegen")
    def test_int8_dynamic_quant_subclass_api(self, device, dtype, act_mapping):
        api = partial(
            _int8da_int8w_api,
            act_mapping_type=act_mapping,
        )
        self._test_lin_weight_subclass_api_impl(api, device, 35, test_dtype=dtype)

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @unittest.skipIf(is_fbcode(), "broken in fbcode")
    def test_int8_weight_only_quant_subclass_api(self, device, dtype):
        undo_recommended_configs()
        self._test_lin_weight_subclass_api_impl(
            _int8wo_api, device, 40, test_dtype=dtype
        )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @torch._inductor.config.patch({"freezing": True})
    @skip_if_rocm("Test flaky on ROCm, under investigation")
    def test_int8_weight_only_quant_with_freeze(self, device, dtype):
        torch._dynamo.reset()
        self._test_lin_weight_subclass_api_impl(
            _int8wo_api, device, 40, test_dtype=dtype
        )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    def test_int4_weight_only_quant_subclass_api(self, device, dtype):
        if dtype != torch.bfloat16:
            self.skipTest(f"Fails for {dtype}")
        for test_shape in [(16, 1024, 16)] + (
            [(1, 1024, 256)] if device == "cuda" else []
        ):
            self._test_lin_weight_subclass_api_impl(
                _int4wo_api, device, 15, test_shape=test_shape, test_dtype=dtype
            )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    def test_int4_weight_only_hqq_quant_subclass_api(self, device, dtype):
        if dtype != torch.bfloat16:
            self.skipTest(f"Fails for {dtype}")
        for test_shape in [(16, 1024, 16), (1, 1024, 256)]:
            api = partial(
                _int4wo_api,
                use_hqq=True,
            )
            self._test_lin_weight_subclass_api_impl(
                api, device, 15, test_shape=test_shape, test_dtype=dtype
            )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @unittest.skipIf(not has_gemlite, "gemlite not available")
    def test_gemlite_layout(self, device, dtype):
        if dtype != torch.float16:
            self.skipTest("gemlite only works for fp16 dtype")
        from torchao.quantization import gemlite_uintx_weight_only

        if device == "cpu":
            self.skipTest(f"gemlite is for cuda, not {device}")
        for packing_bitwidth in [32, 8]:
            for bit_width in [4, 8]:
                for group_size in [64, 32, None] if bit_width == 4 else [None]:
                    api = lambda mod: quantize_(
                        mod,
                        gemlite_uintx_weight_only(
                            group_size, bit_width, packing_bitwidth
                        ),
                    )
                    for test_shape in [
                        [1, 1024, 512],
                        [16, 256, 1024],
                        [128, 256, 1024],
                    ]:
                        print(
                            packing_bitwidth, bit_width, group_size, test_shape, dtype
                        )
                        self._test_lin_weight_subclass_api_impl(
                            api,
                            device,
                            15,
                            test_shape=test_shape,
                            test_dtype=dtype,
                        )

        # test that shapes with non divisible by 128 shapes aren't causing errors
        self._test_lin_weight_subclass_api_impl(
            lambda mod: quantize_(mod, gemlite_uintx_weight_only(None, 4, 32)),
            device,
            15,
            test_shape=[1, 1025, 513],
            test_dtype=dtype,
        )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @skip_if_rocm("ROCm enablement in progress")
    def test_int4_weight_only_quant_subclass_api_grouped(self, device, dtype):
        if dtype != torch.bfloat16:
            self.skipTest(f"Fails for {dtype}")
        layout_list = []
        if check_cpu_version(device):
            layout_list.append(Int4CPULayout())
        elif check_xpu_version(device):
            layout_list.append(Int4XPULayout())
        else:
            for inner_k_tiles in [4, 2]:
                layout_list.append(TensorCoreTiledLayout(inner_k_tiles=inner_k_tiles))
        for test_shape in [(256, 256, 16)] + (
            [(256, 256, 8)] if device == "cuda" else []
        ):
            for groupsize in [64, 32]:
                for layout in layout_list:
                    kwargs = {"groupsize": groupsize, "layout": layout}

                    def api(mod):
                        kwargs_copy = kwargs.copy()
                        kwargs_copy["group_size"] = groupsize
                        del kwargs_copy["groupsize"]
                        quantize_(mod, int4_weight_only(**kwargs_copy))

                    self._test_lin_weight_subclass_api_impl(
                        api,
                        device,
                        15,
                        test_shape=test_shape,
                        test_dtype=dtype,
                    )


class TestDynamicQuant(unittest.TestCase):
    def test_dynamic_quant(self):
        M, K, N = 8, 16, 8
        x = torch.randn(M, K)
        m = nn.Sequential(nn.Linear(K, N))

        y_ref = m(x)
        quantize_(m, int8_dynamic_activation_int8_weight())
        y_test = m(x)

        sqnr = compute_error(y_ref, y_test)
        self.assertGreater(sqnr, 40.0)
        # self.assertTrue(isinstance(m[0], DynamicallyPerAxisQuantizedLinear))


class TestWeightOnlyInt8Quant(unittest.TestCase):
    def test_weight_only_quant(self):
        for x_shape in [[2, 4], [5, 5, 5, 4], [1, 4, 4]]:
            x = torch.randn(*x_shape)
            m = nn.Sequential(nn.Linear(4, 5))
            y_ref = m(x)
            _int8wo_api(m)
            y_wo = m(x)
            sqnr = compute_error(y_ref, y_wo)
            self.assertGreater(sqnr, 43.0)

    def test_weight_only_groupwise_quant(self):
        for x_shape in [[128, 512]]:
            x = torch.randn(*x_shape)
            m = nn.Sequential(nn.Linear(512, 32))
            y_ref = m(x)
            _int8wo_groupwise_api(m)
            self.assertEqual(
                m[0].weight.tensor_impl.int_data.shape, torch.Size([32, 512])
            )
            self.assertEqual(m[0].weight.tensor_impl.scale.shape, torch.Size([32, 16]))
            y_wo = m(x)
            sqnr = compute_error(y_ref, y_wo)
            self.assertGreater(sqnr, 45.0)

    def test_weight_only_groupwise_embedding_quant(self):
        group_size = 64
        m = nn.Embedding(4096, 128)
        input = torch.randint(0, 4096, (1, 6))

        quantize_(
            m,
            int8_weight_only(group_size=group_size),
            filter_fn=lambda x, *args: isinstance(x, nn.Embedding),
        )
        y_q = m(input)
        y_ref = m.weight.dequantize()[input]

        sqnr = compute_error(y_ref, y_q)

        self.assertGreater(sqnr, 45.0)

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @torch.no_grad()
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_weight_only_quant_force_mixed_mm(self, device, dtype):
        undo_recommended_configs()
        if device != "cuda":
            self.skipTest(
                f"weight_only_quant_force_mixed_mm can't be constructed on {device}"
            )
        if dtype == torch.bfloat16 and torch.cuda.get_device_capability() < (8, 0):
            self.skipTest("test requires SM capability of at least (8, 0).")
        from torch._inductor import config

        mixed_mm_key, mixed_mm_val = ("mixed_mm_choice", "triton")

        with config.patch(
            {
                "epilogue_fusion": True,
                mixed_mm_key: mixed_mm_val,
            }
        ):
            for x_shape in [[2, 4], [5, 5, 5, 4], [1, 4, 4]]:
                torch._dynamo.reset()
                x = torch.randn(*x_shape).to(device).to(dtype)
                m = nn.Sequential(nn.Linear(4, 5)).to(device).to(dtype)
                y_ref = m(x)
                _int8wo_api(m)
                m(x)
                m_c = torch.compile(m, mode="max-autotune")
                y_wo, (code,) = run_and_get_code(m_c, x)
                sqnr = compute_error(y_ref, y_wo)
                self.assertGreaterEqual(sqnr, 38)

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_weight_only_quant_use_mixed_mm(self, device, dtype):
        undo_recommended_configs()
        if device != "cuda":
            self.skipTest(
                f"weight_only_quant_force_mixed_mm can't be constructed on {device}"
            )
        if dtype == torch.bfloat16 and torch.cuda.get_device_capability() < (8, 0):
            self.skipTest("test requires SM capability of at least (8, 0).")
        torch.manual_seed(0)
        from torch._inductor import config

        mixed_mm_key, mixed_mm_val = ("mixed_mm_choice", "triton")

        with config.patch(
            {
                "epilogue_fusion": False,
                mixed_mm_key: mixed_mm_val,
            }
        ):
            for x_shape in [[2, 4], [5, 5, 5, 4], [1, 4, 4]]:
                torch._dynamo.reset()
                x = torch.randn(*x_shape).to(device).to(dtype)
                m = nn.Sequential(nn.Linear(4, 5)).to(device).to(dtype)
                y_ref = m(x)
                _int8wo_api(m)
                m_c = torch.compile(m, mode="max-autotune")
                y_wo, (code,) = run_and_get_code(m_c, x)
                sqnr = compute_error(y_ref, y_wo)
                self.assertGreater(sqnr, 42.75)


class TestSaveLoadMeta(unittest.TestCase):
    @torch.no_grad()
    @run_supported_device_dtype
    def _test_handle_save_load_meta_impl(
        self, api, test_device, min_sqnr=35, test_dtype=torch.bfloat16
    ):
        logger.info(f"TestSaveLoad: {api}, {test_device}, {test_dtype}")
        m, k, n = 32, 64, 32

        class test_model(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(k, n)
                self.relu = nn.ReLU()
                self.lin2 = nn.Linear(n, n)

            def forward(self, x):
                x = self.lin1(x)
                x = self.relu(x)
                x = self.lin2(x)
                return x

        x = torch.randn(m, k, dtype=test_dtype, device=test_device)

        # get float reference
        model = test_model().to(dtype=test_dtype, device=test_device).eval()
        ref_f = model(x)

        # save quantized state_dict
        api(model)

        # make sure the model is still runnable
        model(x)

        torch.save(model.state_dict(), "test.pth")
        # get quantized reference
        model_qc = torch.compile(model, mode="max-autotune")
        ref_q = model_qc(x).detach()

        assert SQNR(ref_f, ref_q) > min_sqnr, (
            f"got sqnr: {SQNR(ref_f, ref_q)}, expected: {min_sqnr}"
        )

        # load model structure
        with torch.device("meta"):
            model = test_model().to(dtype=test_dtype)
        api(model)

        # load quantized state_dict
        state_dict = torch.load("test.pth", mmap=True)
        os.remove("test.pth")

        model.load_state_dict(state_dict, assign=True)
        model = model.to(device=test_device, dtype=test_dtype).eval()

        # make sure the model is still runnable
        model(x)

        model_qc = torch.compile(model, mode="max-autotune")
        test = model_qc(x).detach()

        assert SQNR(ref_f, test) > min_sqnr, (
            f"got sqnr: {SQNR(ref_f, ref_q)}, expected: {min_sqnr}"
        )
        self.assertTrue(torch.equal(ref_q, test))

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @unittest.skipIf(
        is_fbcode(), "'PlainAQTTensorImpl' object has no attribute 'int_data'"
    )
    @torch.no_grad()
    def test_save_load_dqtensors(self, device, dtype):
        if device == "cpu":
            self.skipTest("indcutor failed for cpu right now")
        self._test_handle_save_load_meta_impl(
            _int8da_int8w_api, device, test_dtype=dtype
        )

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @torch.no_grad()
    @unittest.skipIf(is_fbcode(), "broken in fbcode")
    def test_save_load_int8woqtensors(self, device, dtype):
        undo_recommended_configs()
        self._test_handle_save_load_meta_impl(_int8wo_api, device, test_dtype=dtype)

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @torch.no_grad()
    def test_save_load_int4woqtensors(self, device, dtype):
        if dtype != torch.bfloat16:
            self.skipTest(f"Fails for {dtype}")
        self._test_handle_save_load_meta_impl(_int4wo_api, device, 20, test_dtype=dtype)


class TorchCompileUnitTest(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_fullgraph(self):
        lin_fp16 = nn.Linear(32, 16, device="cuda", dtype=torch.float16)
        lin_smooth = SmoothFakeDynamicallyQuantizedLinear.from_float(
            lin_fp16, alpha=0.25
        )

        x0 = torch.randn(17, 1, 32, device="cuda", dtype=torch.float16)

        # calibrate
        _ = lin_smooth(x0)

        # inference
        lin_smooth.to_inference()

        # torch.compile
        lin_smooth_opt = torch.compile(lin_smooth, fullgraph=True)
        # print(lin_smooth_opt)

        lin_smooth_opt(x0)
        # print(y)


class UtilsUnitTest(unittest.TestCase):
    def test_shape_logger(self):
        x = torch.randn(4, 4)

        m = nn.Sequential(
            nn.Linear(4, 4),
            nn.Sequential(
                nn.Linear(4, 4),
            ),
        )

        _apply_logging_hook(m)
        with LoggingTensorMode():
            m(x)
            m(x)

        for fqn, d1 in _fqn_to_op_to_shape_to_count.items():  # noqa: PERF102
            for op, d2 in d1.items():  # noqa: PERF102
                for shape, count in d2.items():  # noqa: PERF102
                    # print(fqn, op, shape, count)
                    pass


class SmoothquantIntegrationTest(unittest.TestCase):
    @torch.no_grad()
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skip("Seg fault?")
    def test_non_dynamically_quantizable_linear(self):
        if torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0):
            self.skipTest("test requires SM capability of at least (8, 0).")
        model = (
            torch.nn.Sequential(
                torch.nn.modules.linear.NonDynamicallyQuantizableLinear(32, 32),
                torch.nn.ReLU(),
            )
            .to("cuda")
            .to(torch.bfloat16)
        )
        example_input = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
        ref = model(example_input)
        swap_linear_with_smooth_fq_linear(model)
        model(ref)
        smooth_fq_linear_to_inference(model)
        model_c = torch.compile(model, mode="max-autotune")
        out = model_c(example_input)
        sqnr = SQNR(ref, out)
        self.assertTrue(sqnr >= 25)
        self.assertTrue(isinstance(model[0], SmoothFakeDynamicallyQuantizedLinear))

    @torch.inference_mode()
    @unittest.skipIf(is_fbcode(), "can't load tokenizer")
    def test_on_dummy_distilbert(self):
        # https://huggingface.co/distilbert-base-uncased#how-to-use
        from transformers import (  # type: ignore[import-untyped]
            DistilBertModel,
            DistilBertTokenizer,
        )

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # print(model)
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors="pt")
        output_ref = model(**encoded_input)
        # print(output_ref)

        #
        # smooth_quant
        #
        model_copy = copy.deepcopy(model)
        swap_linear_with_smooth_fq_linear(model_copy, alpha=0.75)
        # calibrate
        model_copy(**encoded_input)
        # inference
        smooth_fq_linear_to_inference(model_copy)
        output_1_2 = model_copy(**encoded_input)
        # print(output_1_1)
        # print(output_1_2)
        sqnr_sq = compute_error(
            output_ref.last_hidden_state, output_1_2.last_hidden_state
        )
        print("sqnr_sq", sqnr_sq)
        self.assertTrue(sqnr_sq >= 20.0)

        #
        # reference - dynamic linear quant
        #
        model_copy2 = copy.deepcopy(model)
        qconfig = torch.ao.quantization.QConfig(
            activation=None,
            weight=torch.ao.quantization.default_per_channel_weight_observer,
        )
        model_copy2 = torch.ao.quantization.quantize_dynamic(
            model_copy2,
            {torch.nn.Linear: qconfig},
        )
        output_2_2 = model_copy2(**encoded_input)
        # print(output_2_2)
        sqnr_pt_quant = compute_error(
            output_ref.last_hidden_state, output_2_2.last_hidden_state
        )
        print("sqnr_pt_quant", sqnr_pt_quant)
        self.assertTrue(sqnr_sq >= 8.0)


class TestAutoQuant(unittest.TestCase):
    @parameterized.expand(
        combine_parameters(
            COMMON_DEVICE_DTYPE,
            [
                (16, 128, 128),
                (64, 128, 128),
                # (2**15, 128, 128), TODO: Runs out of shared memory on T4
                (16, 128, 256),
                # (64, 128, 256), # TODO: Runs out of shared memory on T4
                (16, 256, 128),
                (64, 256, 128),
                # (256, 256, 128), TODO: Runs out of shared memory on T4
            ],
        )
    )
    def test_autoquant_one_input(self, device, dtype, m, k, n):
        undo_recommended_configs()
        print("(m, k, n): ", (m, k, n))
        if device != "cuda" or not torch.cuda.is_available():
            self.skipTest(f"autoquant currently does not support {device}")
        if torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0):
            if dtype == torch.bfloat16:
                self.skipTest("bfloat16 requires sm80+")
            if m == 1:
                self.skipTest(f"Shape {(m, k, n)} requires sm80+")
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.use_mixed_mm = True
        torch._inductor.config.force_fuse_int_mm_with_mul = True
        torch._dynamo.config.automatic_dynamic_shapes = False

        example_input = torch.randn(m, k, device=device, dtype=dtype)
        model = (
            torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(k, n),
                torch.nn.ReLU(),
            )
            .to(device)
            .to(dtype)
        )
        out = model(example_input)
        torchao.autoquant(model, set_inductor_config=False)
        out2 = model(example_input)
        sqnr = SQNR(out, out2)
        self.assertTrue(sqnr >= 30)

    @parameterized.expand(
        combine_parameters(
            COMMON_DEVICE_DTYPE,
            [
                (1, 1, 128, 128),
                (1, 32, 128, 128),
                (32, 32, 128, 128),
            ],
        )
    )
    def test_autoquant_compile(self, device, dtype, m1, m2, k, n):
        undo_recommended_configs()

        is_supported_device = device == "cuda" and (
            torch.cuda.is_available() or torch.version.hip is not None
        )

        if not is_supported_device:
            self.skipTest(f"autoquant currently does not support {device}")

        # Check CUDA-specific requirements if running on CUDA
        if (
            is_supported_device and torch.version.hip is None
        ):  # Only apply to CUDA, not ROCm
            device_capability = torch.cuda.get_device_capability()
            if device_capability < (8, 0):
                if dtype == torch.bfloat16:
                    self.skipTest("bfloat16 requires sm80+")
                if m1 == 1 or m2 == 1:
                    self.skipTest(f"Shape {(m1, m2, k, n)} requires sm80+")

        # TODO remove this once https://github.com/pytorch/pytorch/issues/155838 is resolved
        if m1 == 1 or m2 == 1:
            self.skipTest(f"Shape {(m1, m2, k, n)} is flaky, skipping")
        model = (
            torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(k, n),
                torch.nn.ReLU(),
            )
            .to(device)
            .to(dtype)
        )
        example_input = torch.randn(m1, k, device=device, dtype=dtype)
        example_input2 = torch.randn(m2, k, device=device, dtype=dtype)
        out = model(example_input)

        mod = torchao.autoquant(
            torch.compile(model), manual=True, set_inductor_config=False
        )
        mod(example_input)
        mod(example_input2)
        mod.finalize_autoquant()

        out2 = mod(example_input)
        sqnr = SQNR(out, out2)
        self.assertTrue(sqnr >= 30)

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    def test_autoquant_mha(self, device, dtype):
        if device != "cuda" or not torch.cuda.is_available():
            self.skipTest(f"autoquant currently does not support {device}")

        class MHAModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mha = torch.nn.MultiheadAttention(4096, 32)
                self.lin = torch.nn.Linear(4096, 4096)

            def forward(self, x):
                y = self.mha(x, x, x)[0]
                return self.lin(y)

        mod = MHAModel().to(device).to(dtype)
        input = torch.randn(1, 1, 4096).to(device).to(dtype)
        mod(*input)

        torchao.autoquant(mod, set_inductor_config=False)
        assert not isinstance(mod.mha.out_proj.weight, AutoQuantizableLinearWeight)
        assert isinstance(mod.lin.weight, AutoQuantizableLinearWeight)
        mod(*input)
        from torchao.quantization.autoquant import _AUTOQUANT_CACHE

        assert len(_AUTOQUANT_CACHE) > 0

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    def test_autoquant_manual(self, device, dtype):
        undo_recommended_configs()
        if device != "cuda" or not torch.cuda.is_available():
            self.skipTest(f"autoquant currently does not support {device}")
        if torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0):
            if dtype == torch.bfloat16:
                self.skipTest("bfloat16 requires sm80+")
        m1, m2, k, n = 32, 32, 128, 128
        model = (
            torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(k, n),
                torch.nn.ReLU(),
            )
            .to(device)
            .to(dtype)
        )
        example_input = torch.randn(m1, k, device=device, dtype=dtype)
        example_input2 = torch.randn(m2, k, device=device, dtype=dtype)
        out = model(example_input)

        mod = torchao.autoquant(
            torch.compile(model), manual=True, set_inductor_config=False
        )
        mod(example_input)
        mod(example_input2)
        mod.finalize_autoquant()
        out2 = mod(example_input)
        sqnr = SQNR(out, out2)
        self.assertTrue(sqnr >= 30)

        mod2 = torchao.autoquant(model, manual=True, set_inductor_config=False)
        mod2(example_input)
        mod2(example_input2)
        mod2.finalize_autoquant()
        out3 = mod(example_input)
        sqnr2 = SQNR(out, out3)
        self.assertTrue(sqnr2 >= 30)

    @parameterized.expand(
        combine_parameters(
            COMMON_DEVICE_DTYPE,
            [
                (1, 1, 128, 128),
                (1, 32, 128, 128),
                (32, 32, 128, 128),
            ],
        )
    )
    def test_autoquant_kwargs(self, device, dtype, m1, m2, k, n):
        undo_recommended_configs()
        if device != "cuda" or not torch.cuda.is_available():
            self.skipTest(f"autoquant currently does not support {device}")
        if torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0):
            if dtype == torch.bfloat16:
                self.skipTest("bfloat16 requires sm80+")
            if m1 == 1 or m2 == 1:
                self.skipTest(f"Shape {(m1, m2, k, n)} requires sm80+")

        # Note: This test was incorrectly written before with this skip condition:
        #
        #     m1 == 1 or m2 == 1 and not TORCH_VERSION_AT_LEAST_2_5:
        #
        # This is actually equivalent to:
        #
        #     m1 == 1 or (m2 == 1 and not TORCH_VERSION_AT_LEAST_2_5)
        #
        # which means we always skips the test as long as `m1 == 1` regardless of
        # the pytorch version, which was not the intended behavior. Unfortunately,
        # unskipping this test now leads to the following error when calling
        # `aten._int_mm`:
        #
        #     RuntimeError: self.size(0) needs to be greater than 16, but got 1
        #
        # Therefore, we keep around this skip condition for now since it doesn't
        # change the test behavior from before. For more details, please see
        # https://github.com/pytorch/ao/pull/2720.
        if m1 == 1:
            self.skipTest(f"Shape {(m1, m2, k, n)} is not supported")

        class NeedsKwargs(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rel = torch.nn.ReLU()
                self.lin = torch.nn.Linear(k, n)

            def forward(self, x, y):
                x = self.rel(x)
                z = self.lin(x + y)
                return z

        model = NeedsKwargs().to(device).to(dtype)
        example_input = {
            "x": torch.randn(m1, k, device=device, dtype=dtype),
            "y": torch.randn(m1, k, device=device, dtype=dtype),
        }
        out = model(**example_input)

        mod = torchao.autoquant(torch.compile(model), set_inductor_config=False)
        mod(**example_input)

        out2 = mod(**example_input)
        sqnr = SQNR(out, out2)
        self.assertTrue(sqnr >= 30)

    @parameterized.expand(
        combine_parameters(
            COMMON_DEVICE_DTYPE,
            [
                (16, 128, 128),
            ],
        )
    )
    def test_autoquant_double_access(self, device, dtype, m, k, n):
        undo_recommended_configs()
        if device != "cuda" or not torch.cuda.is_available():
            self.skipTest(f"autoquant currently does not support {device}")
        if torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0):
            if dtype == torch.bfloat16:
                self.skipTest("bfloat16 requires sm80+")

        class DoubleAccess(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = torch.nn.Linear(k, n)
                self.lin2 = torch.nn.Linear(n, k)
                self.lin3 = torch.nn.Linear(k, n)
                self.lin3.weight = self.lin1.weight

            def forward(self, x):
                x = self.lin1(x)
                x = self.lin2(x)
                x = self.lin3(x)
                return x

        x_in = torch.randn(m, k, device=device, dtype=dtype)
        model = DoubleAccess().to(device).to(dtype)
        model(x_in)
        torchao.autoquant(model, set_inductor_config=False)
        assert not isinstance(model.lin1.weight.weight, AutoQuantizableLinearWeight)
        model(x_in)

    @parameterized.expand(list(itertools.product(["cuda"], COMMON_DTYPES)))
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_autoquant_min_sqnr(self, device, dtype):
        m, k, n = 128, 128, 128
        example_input = torch.randn(m, k, device=device, dtype=dtype)
        model = (
            torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(k, n),
                torch.nn.ReLU(),
            )
            .to(device)
            .to(dtype)
        )
        out = model(example_input)
        torchao.autoquant(model, min_sqnr=60)
        out2 = model(example_input)
        sqnr = SQNR(out, out2)
        # without setting min_sqnr to 60, we get around 45-50 final sqnr
        # setting min_sqnr for individual linear to be 60 allows us to achieve >= 50 final sqnr
        self.assertTrue(sqnr >= 50, f"sqnr: {sqnr}")

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_autoquant_hp_float(self):
        device = "cuda"
        dtype = torch.float32
        m, k, n = 128, 128, 128
        example_input = torch.randn(m, k, device=device, dtype=dtype)
        for qclass in torchao.quantization.DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST:
            model = (
                torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Linear(k, n, bias=True),
                    torch.nn.ReLU(),
                )
                .to(device)
                .to(dtype)
            )
            ref = model(example_input)
            qtensor_class_list = [qclass]
            torchao.autoquant(
                model,
                qtensor_class_list=qtensor_class_list,
            )
            out = model(example_input)
            self.assertIn(
                type(model[1].weight),
                qtensor_class_list,
            )
            self.assertGreater(compute_error(out, ref), 40)

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(not has_gemlite, "gemlite not available")
    def test_autoquant_int4wo(self, device, dtype):
        if device == "cpu":
            self.skipTest(f"int4wo is for cuda, not {device}")

        m, k, n = 128, 128, 128
        example_input = torch.randn(m, k, device=device, dtype=dtype)

        for qclass in [
            AQGemliteInt4G64WeightOnlyQuantizedLinearWeight,
            AQInt4G32WeightOnlyQuantizedLinearWeight,
            AQInt4G128WeightOnlyQuantizedMarlinSparseLinearWeight,
        ]:
            model = (
                torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Linear(k, n, bias=True),
                    torch.nn.ReLU(),
                )
                .to(device)
                .to(dtype)
            )
            ref = model(example_input)
            qtensor_class_list = [qclass]
            torchao.autoquant(
                model,
                qtensor_class_list=qtensor_class_list,
            )
            out = model(example_input)

            self.assertIn(type(model[1].weight), qtensor_class_list)
            self.assertGreater(compute_error(ref, out), 20)

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @unittest.skipIf(not is_sm_at_least_90(), "Need cuda arch greater than SM90")
    @unittest.skipIf(
        True, "Skipping for now, do to lowering bug in inductor"
    )  # TODO unblock when fixed
    def test_autoquant_float8(self, device, dtype):
        if device == "cpu":
            self.skipTest(f"float8 is for cuda, not {device}")

        # note: marlin sparse layout failed when scale_t has a dimension of 1d
        m, k, n = 128, 128, 128
        example_input = torch.randn(m, k, device=device, dtype=dtype)

        for qclass in [
            AQFloat8PerRowScalingDynamicallyQuantizedLinearWeight,
            AQFloat8PerTensorScalingDynamicallyQuantizedLinearWeight,
            AQFloat8WeightOnlyQuantizedLinearWeight,
        ]:
            if (
                dtype in (torch.float32, torch.float16)
                and qclass is AQFloat8PerRowScalingDynamicallyQuantizedLinearWeight
            ):
                continue
            model = (
                torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Linear(k, n, bias=True),
                    torch.nn.ReLU(),
                )
                .to(device)
                .to(dtype)
            )
            ref = model(example_input)
            qtensor_class_list = [qclass]
            torchao.autoquant(model, qtensor_class_list=qtensor_class_list)
            out = model(example_input)

            self.assertIn(type(model[1].weight), qtensor_class_list)
            self.assertGreater(compute_error(ref, out), 20)


@unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
@unittest.skip(
    "AOTI tests are failing right now, repro by commenting out the skip and run:"
    "python test/integration/test_integration.py -k TestAOTI.test_aoti_06"
)
class TestAOTI(unittest.TestCase):
    @parameterized.expand(
        list(itertools.product(TENSOR_SUBCLASS_APIS, COMMON_DEVICES, COMMON_DTYPES)),
    )
    def test_aoti(self, api, test_device, test_dtype):
        if (
            test_device == "cuda"
            and torch.cuda.is_available()
            and test_dtype == torch.bfloat16
            and torch.cuda.get_device_capability() < (8, 0)
        ):
            self.skipTest("Need CUDA and SM80+ available.")

        logger.info(f"TestAOTI: {api}, {test_device}, {test_dtype}")

        m, k, n = 32, 64, 32

        class test_model(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(k, n)
                self.relu = nn.ReLU()
                self.lin2 = nn.Linear(n, n)

            def forward(self, x):
                x = self.lin1(x)
                x = self.relu(x)
                x = self.lin2(x)
                return x

        x = torch.randn(m, k, dtype=test_dtype, device=test_device)

        # get float reference
        model = test_model().to(dtype=test_dtype, device=test_device).eval()
        model(x)

        api(model)
        if not TORCH_VERSION_AT_LEAST_2_7:
            unwrap_tensor_subclass(model)

        # running model
        model(x)

        # make sure it compiles
        torch._inductor.config.mixed_mm_choice = "triton"

        example_inputs = (x,)
        torch._inductor.aoti_compile_and_package(
            torch.export.export(model, example_inputs, strict=True)
        )


@unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
class TestExport(unittest.TestCase):
    @parameterized.expand(
        list(
            itertools.product(
                TENSOR_SUBCLASS_APIS + [_int8da_int4w_api],
                COMMON_DEVICES,
                COMMON_DTYPES,
            )
        ),
    )
    def test_export(self, api, test_device, test_dtype):
        if (
            test_device == "cuda"
            and torch.cuda.is_available()
            and test_dtype == torch.bfloat16
            and torch.cuda.get_device_capability() < (8, 0)
        ):
            self.skipTest("Need CUDA and SM80+ available.")

        logger.info(f"TestExport: {api}, {test_device}, {test_dtype}")

        m, k, n = 32, 64, 32

        class test_model(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(k, n)
                self.relu = nn.ReLU()
                self.lin2 = nn.Linear(n, n)

            def forward(self, x):
                x = self.lin1(x)
                x = self.relu(x)
                x = self.lin2(x)
                return x

        x = torch.randn(m, k, dtype=test_dtype, device=test_device)

        # get float reference
        model = test_model().to(dtype=test_dtype, device=test_device).eval()
        model(x)

        api(model)
        if not TORCH_VERSION_AT_LEAST_2_7:
            unwrap_tensor_subclass(model)

        # running model
        ref = model(x)

        # make sure it compiles
        example_inputs = (x,)
        # TODO: export changes numerics right now, this is because of functionalization according to Zhengxu
        # we can re-enable this after non-functional IR is enabled in export
        # model = torch.export.export(model, example_inputs).module()
        model = torch.export.export_for_training(
            model, example_inputs, strict=True
        ).module()
        after_export = model(x)
        self.assertTrue(torch.equal(after_export, ref))
        if api is _int8da_int4w_api:
            targets = [n.target for n in model.graph.nodes]
            self.assertTrue(torch.ops.torchao.choose_qparams_affine.default in targets)
            self.assertTrue(torch.ops.torchao.quantize_affine.default in targets)
            self.assertFalse(torch.ops.aten.narrow.default in targets)

    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_export_float8(self):
        class SimpleNetwork(torch.nn.Module):
            def __init__(self):
                super(SimpleNetwork, self).__init__()
                self.linear = torch.nn.Linear(
                    in_features=32, out_features=16, bias=False
                )

            def forward(self, x):
                return self.linear(x)

        model = SimpleNetwork().eval().cuda()
        inp = torch.randn(2, 32).cuda()
        config = Float8DynamicActivationFloat8WeightConfig()
        quantize_(model, config)

        ep = torch.export.export(model, (inp,))
        print(ep)
        FileCheck().check_count(
            "torch.ops.torchao.choose_scale_float8.default", 1, exactly=True
        ).run(str(ep.graph))


class TestUtils(unittest.TestCase):
    @parameterized.expand(
        list(itertools.product(TENSOR_SUBCLASS_APIS, COMMON_DEVICES, COMMON_DTYPES)),
    )
    def test_get_model_size_aqt(self, api, test_device, test_dtype):
        if test_dtype != torch.bfloat16:
            self.skipTest(f"{api} in {test_dtype} is not supported yet")
        if test_device != "cuda" or not torch.cuda.is_available():
            self.skipTest(f"{api} currently does not support {test_device}")
        k, n = 1024, 1024
        model = (
            torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(k, n),
                torch.nn.ReLU(),
            )
            .to(test_device)
            .to(test_dtype)
        )
        size = torchao.utils.get_model_size_in_bytes(model)
        api(model)
        size2 = torchao.utils.get_model_size_in_bytes(model)
        self.assertTrue(size2 < size)


class TestBenchmarkModel(unittest.TestCase):
    class ToyLinearModel(torch.nn.Module):
        def __init__(self, m=64, n=32, k=64):
            super().__init__()
            self.linear1 = torch.nn.Linear(m, n, bias=False)
            self.linear2 = torch.nn.Linear(n, k, bias=False)

        def example_inputs(self, batch_size=1, dtype=torch.float32, device="cpu"):
            return (
                torch.randn(
                    batch_size, self.linear1.in_features, dtype=dtype, device=device
                ),
            )

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x

    def run_benchmark_model(self, device):
        # params
        dtype = torch.bfloat16
        m = self.ToyLinearModel(1024, 1024, 1024).eval().to(dtype).to(device)
        m_bf16 = copy.deepcopy(m)
        example_inputs = m.example_inputs(dtype=dtype, device=device)
        m_bf16 = torch.compile(m_bf16, mode="max-autotune")
        num_runs = 1
        return benchmark_model(m_bf16, num_runs, example_inputs)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_benchmark_model_cuda(self):
        assert self.run_benchmark_model("cuda") is not None

    def test_benchmark_model_cpu(self):
        assert self.run_benchmark_model("cpu") is not None


if __name__ == "__main__":
    unittest.main()
