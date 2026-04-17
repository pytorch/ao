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
from torchao.quantization import safe_int_mm
from torchao.quantization.granularity import PerGroup

# APIs to be deprecated (used for torch 2.2.2 and 2.3)
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    quantize_,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    dequantize_affine,
)
from torchao.quantization.quantize_.workflows.int4.int4_packing_format import (
    Int4PackingFormat,
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
from torchao.testing.utils import skip_if_rocm, skip_if_xpu
from torchao.utils import (
    benchmark_model,
    get_current_accelerator_device,
    is_fbcode,
    is_ROCM,
    is_sm_at_least_89,
    torch_version_at_least,
    unwrap_tensor_subclass,
)

logger = logging.getLogger("INFO")

torch.manual_seed(0)
config.cache_size_limit = 100

COMMON_DEVICES = ["cpu"] + (
    [get_current_accelerator_device()] if torch.accelerator.is_available() else []
)

COMMON_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

ACT_MAPPING_TYPES = [MappingType.ASYMMETRIC, MappingType.SYMMETRIC]

COMMON_DEVICE_DTYPE = list(itertools.product(COMMON_DEVICES, COMMON_DTYPES)).copy()


def _int8wo_api(mod):
    quantize_(mod, Int8WeightOnlyConfig(set_inductor_config=False))


def _int8wo_groupwise_api(mod):
    group_size = 32
    quantize_(
        mod,
        Int8WeightOnlyConfig(
            granularity=PerGroup(group_size), set_inductor_config=False
        ),
    )


def _int8da_int8w_api(
    mod,
    act_mapping_type=MappingType.SYMMETRIC,
):
    quantize_(
        mod,
        Int8DynamicActivationInt8WeightConfig(
            act_mapping_type=act_mapping_type,
            set_inductor_config=False,
        ),
    )


# TODO: use this to reduce the number of tests
TENSOR_SUBCLASS_APIS = [
    _int8wo_api,
    _int8da_int8w_api,
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

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def wrapper(*args, **kwargs):
        if len(args) < 3:
            raise unittest.SkipTest(
                f"Not enough args. Expected more than or equal to 3, but got {len(args)}"
            )
        device = args[2]
        dtype = kwargs["test_dtype"] if "test_dtype" in kwargs else args[3]
        _DEVICE = get_current_accelerator_device()
        if (
            device == _DEVICE
            and torch.cuda.is_available()
            and dtype == torch.bfloat16
            and torch.cuda.get_device_capability() < (8, 0)
        ):
            raise unittest.SkipTest("Need CUDA and SM80+ available.")
        return test_method(*args, **kwargs)

    return wrapper


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

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    @unittest.skip("AssertionError: Tensor-likes are not close!")
    def test_dynamic_quant_per_channel_numerics_cuda(self):
        device = get_current_accelerator_device()
        test_cases = (
            (-128, 127, torch.int8, torch.qint8, torch.float32, device),
            (-128, 127, torch.int8, torch.qint8, torch.float16, device),
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

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_quantize_per_token_cuda(self):
        device = get_current_accelerator_device()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._test_quantize_per_token_impl(device, dtype)

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
        self.assertTrue(sqnr >= 39.0, f"{sqnr=} too low")

    @unittest.skipIf(is_ROCM(), "Don't test CPU for ROCM version of torch")
    def test_per_token_linear_cpu(self):
        for dtype in (torch.float32,):
            self._test_per_token_linear_impl("cpu", dtype)

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    @skip_if_rocm("ROCm enablement in progress")
    def test_per_token_linear_cuda(self):
        device = get_current_accelerator_device()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._test_per_token_linear_impl(device, dtype)

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test__int_mm(self):
        # TODO(future): figure out what here needs to move to PT core,
        # if it's not already tested there

        m, k, n = 32, 32, 16
        device = get_current_accelerator_device()
        x = torch.randint(-128, 127, (m, k), dtype=torch.int8, device=device)
        w = torch.randint(-128, 127, (k, n), dtype=torch.int8, device=device)

        y_ref = torch.matmul(x.float(), w.float()).to(torch.int32)
        y_raw = safe_int_mm(x, w)

        wrap_in_mm_opt = torch.compile(safe_int_mm, mode="max-autotune")
        # note: triton chokes on the line below on k == 8 and n == 8 with
        # https://www.internalfb.com/phabricator/paste/view/P683467944
        # TODO(future): file an issue
        y_opt = wrap_in_mm_opt(x, w)

        torch.testing.assert_close(y_ref, y_raw, atol=0, rtol=0)
        torch.testing.assert_close(y_ref, y_opt, atol=0, rtol=0)

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test__int_mm_eager_and_torch_compile_numerics(self):
        def __int_mm_ref(x, w):
            x = x.cpu().to(torch.int32)
            w = w.cpu().to(torch.int32)
            y = torch.matmul(x, w)
            device = get_current_accelerator_device()
            return y.to(device)

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

            device = get_current_accelerator_device()
            x = torch.randint(-128, 127, x_shape, dtype=torch.int8, device=device)
            w = torch.randint(-128, 127, w_shape, dtype=torch.int8, device=device)

            z_ref = __int_mm_ref(x, w)
            z_eager = wrap_torch_int_mm(x, w)
            z_torch_compile = wrap_torch_int_mm_opt(x, w)
            # print(z_ref)
            # print(z_eager)
            # print(z_torch_compile)

            torch.testing.assert_close(z_ref, z_eager, atol=0, rtol=0)
            torch.testing.assert_close(z_ref, z_torch_compile, atol=0, rtol=0)


class TestSubclass(unittest.TestCase):
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
    @skip_if_xpu("XPU enablement in progress")
    def test_int4_weight_only_quant_subclass_api_grouped(self, device, dtype):
        if dtype != torch.bfloat16:
            self.skipTest(f"Fails for {dtype}")
        if device == "cpu":
            self.skipTest("Only CUDA is supported for int4 weight only quantization v2")
        ntile_size = 16 if torch.version.hip else 8
        for test_shape in [(256, 256, 16), (256, 256, 8)]:
            for groupsize in [64, 32]:

                def api(mod):
                    quantize_(
                        mod,
                        Int4WeightOnlyConfig(
                            group_size=groupsize,
                            int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
                            int4_tile_packed_ntile=ntile_size,
                        ),
                    )

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
        quantize_(m, Int8DynamicActivationInt8WeightConfig())
        y_test = m(x)

        sqnr = compute_error(y_ref, y_test)
        self.assertGreater(sqnr, 40.0)


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
            self.assertEqual(m[0].weight.qdata.shape, torch.Size([32, 512]))
            self.assertEqual(m[0].weight.scale.shape, torch.Size([32, 16]))
            y_wo = m(x)
            sqnr = compute_error(y_ref, y_wo)
            self.assertGreater(sqnr, 45.0)

    def test_weight_only_groupwise_embedding_quant(self):
        group_size = 64
        m = nn.Embedding(4096, 128)
        input = torch.randint(0, 4096, (1, 6))

        quantize_(
            m,
            Int8WeightOnlyConfig(granularity=PerGroup(group_size)),
            filter_fn=lambda x, *args: isinstance(x, nn.Embedding),
        )
        y_q = m(input)
        y_ref = m.weight.dequantize()[input]

        sqnr = compute_error(y_ref, y_q)

        self.assertGreater(sqnr, 45.0)

    @parameterized.expand(COMMON_DEVICE_DTYPE)
    @torch.no_grad()
    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_weight_only_quant_force_mixed_mm(self, device, dtype):
        undo_recommended_configs()
        if device != get_current_accelerator_device():
            self.skipTest(
                f"weight_only_quant_force_mixed_mm can't be constructed on {device}"
            )
        if (
            torch.cuda.is_available()
            and dtype == torch.bfloat16
            and torch.cuda.get_device_capability() < (8, 0)
        ):
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
    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_weight_only_quant_use_mixed_mm(self, device, dtype):
        undo_recommended_configs()
        if device != get_current_accelerator_device():
            self.skipTest(
                f"weight_only_quant_force_mixed_mm can't be constructed on {device}"
            )
        if (
            torch.cuda.is_available()
            and dtype == torch.bfloat16
            and torch.cuda.get_device_capability() < (8, 0)
        ):
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


@unittest.skipIf(not torch.accelerator.is_available(), "requires gpu")
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
        if not torch_version_at_least("2.7.0"):
            unwrap_tensor_subclass(model)

        # running model
        model(x)

        # make sure it compiles
        torch._inductor.config.mixed_mm_choice = "triton"

        example_inputs = (x,)
        torch._inductor.aoti_compile_and_package(
            torch.export.export(model, example_inputs, strict=True)
        )


@unittest.skipIf(not torch.accelerator.is_available(), "requires gpu")
class TestExport(unittest.TestCase):
    @parameterized.expand(
        list(
            itertools.product(
                TENSOR_SUBCLASS_APIS,
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

        if test_device == "cpu" and is_ROCM():
            self.skipTest("Don't test CPU for ROCM version of torch")

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
        if not torch_version_at_least("2.7.0"):
            unwrap_tensor_subclass(model)

        # running model
        ref = model(x)

        # make sure it compiles
        example_inputs = (x,)
        # TODO: export changes numerics right now, this is because of functionalization according to Zhengxu
        # we can re-enable this after non-functional IR is enabled in export
        # model = torch.export.export(model, example_inputs).module()
        model = torch.export.export(model, example_inputs, strict=True).module()
        after_export = model(x)
        self.assertTrue(torch.equal(after_export, ref))

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

        model = SimpleNetwork().eval().to("cuda")
        inp = torch.randn(2, 32).to("cuda")
        config = Float8DynamicActivationFloat8WeightConfig()
        quantize_(model, config)

        # Need to export with strict=True
        # https://github.com/pytorch/pytorch/issues/167007
        ep = torch.export.export(model, (inp,), strict=True)
        print(ep)
        FileCheck().check_count(
            "torch.ops.torchao.choose_scale_float8.default", 1, exactly=True
        ).run(str(ep.graph))


class TestUtils(unittest.TestCase):
    @parameterized.expand(
        list(itertools.product(TENSOR_SUBCLASS_APIS, COMMON_DEVICES, COMMON_DTYPES)),
    )
    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_get_model_size_aqt(self, api, test_device, test_dtype):
        if test_dtype != torch.bfloat16:
            self.skipTest(f"{api} in {test_dtype} is not supported yet")
        _DEVICE = get_current_accelerator_device()
        if test_device != _DEVICE or not torch.accelerator.is_available():
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

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_benchmark_model_cuda(self):
        device = get_current_accelerator_device()
        assert self.run_benchmark_model(device) is not None

    def test_benchmark_model_cpu(self):
        assert self.run_benchmark_model("cpu") is not None


if __name__ == "__main__":
    unittest.main()
