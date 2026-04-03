# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import tempfile
import unittest

import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

from torchao.core.config import AOBaseConfig
from torchao.quantization import (
    Float8WeightOnlyConfig,
    quantize_,
)
from torchao.testing.utils import skip_if_rocm
from torchao.utils import (
    check_cpu_version,
    check_xpu_version,
    get_current_accelerator_device,
    is_fbcode,
    is_sm_at_least_89,
)

is_cusparselt_available = (
    hasattr(torch.backends, "cusparselt") and torch.backends.cusparselt.is_available()
)


def get_quantization_functions(
    do_sparse: bool, do_int4: bool, device: str = "cuda", int4_zp_int: bool = False
):
    base_functions = []
    if do_int4:
        if check_cpu_version(device):
            pass
        elif check_xpu_version(device):
            pass

    if is_sm_at_least_89():
        base_functions.append(Float8WeightOnlyConfig())

    return base_functions


class TestAffineQuantized(TestCase):
    GPU_DEVICES = (["cuda"] if torch.cuda.is_available() else []) + (
        ["xpu"] if torch.xpu.is_available() else []
    )
    _DEVICE = get_current_accelerator_device() if len(GPU_DEVICES) != 0 else "cpu"

    @unittest.skipIf(len(GPU_DEVICES) == 0, "Need GPU available")
    def test_weights_only(self):
        for device in self.GPU_DEVICES:
            apply_quant_list = get_quantization_functions(
                is_cusparselt_available, True, device, True
            )
            for apply_quant in apply_quant_list:
                linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device=device)
                if isinstance(apply_quant, AOBaseConfig):
                    quantize_(linear, apply_quant)
                    ql = linear
                else:
                    # TODO(#1690): delete this once config migration is done
                    ql = apply_quant(linear)
                with tempfile.NamedTemporaryFile() as f:
                    torch.save(ql.state_dict(), f)
                    f.seek(0)
                    _ = torch.load(f, weights_only=True)

    @skip_if_rocm("ROCm enablement in progress")
    @unittest.skipIf(len(GPU_DEVICES) == 0, "Need GPU available")
    def test_print_quantized_module(self):
        for device in self.GPU_DEVICES:
            apply_quant_list = get_quantization_functions(True, True, device, True)
            for apply_quant in apply_quant_list:
                linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device=device)
                if isinstance(apply_quant, AOBaseConfig):
                    quantize_(linear, apply_quant)
                    ql = linear
                else:
                    # TODO(#1690): delete this once config migration is done
                    ql = apply_quant(linear)
                ql_str = str(ql)
                assert "AffineQuantizedTensor" in ql_str or "Float8Tensor" in ql_str, (
                    f"Expected quantized tensor in repr, got: {ql_str}"
                )


class TestAffineQuantizedBasic(TestCase):
    COMMON_DEVICES = (
        ["cpu"]
        + (["cuda"] if torch.cuda.is_available() else [])
        + (["xpu"] if torch.xpu.is_available() else [])
    )
    COMMON_DTYPES = [torch.bfloat16]

    @common_utils.parametrize("device", COMMON_DEVICES)
    @common_utils.parametrize("dtype", COMMON_DTYPES)
    @skip_if_rocm("ROCm enablement in progress")
    def test_flatten_unflatten(self, device, dtype):
        if device == "cuda" and dtype == torch.bfloat16 and is_fbcode():
            raise unittest.SkipTest("TODO: Failing for cuda + bfloat16 in fbcode")
        apply_quant_list = get_quantization_functions(False, True, device)
        for apply_quant in apply_quant_list:
            linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
            if isinstance(apply_quant, AOBaseConfig):
                quantize_(linear, apply_quant)
                ql = linear
            else:
                # TODO(#1690): delete this once config migration is done
                ql = apply_quant(linear)
            lp_tensor = ql.weight
            tensor_data_name_dict, tensor_attributes = lp_tensor.__tensor_flatten__()
            tensor_data_dict = {
                name: getattr(lp_tensor, name) for name in tensor_data_name_dict
            }
            outer_size = lp_tensor.size()
            outer_stride = lp_tensor.stride()
            reconstructed = type(lp_tensor).__tensor_unflatten__(
                tensor_data_dict, tensor_attributes, outer_size, outer_stride
            )
            example_inputs = (torch.randn(32, 128, dtype=dtype, device=device),)
            ref = ql(*example_inputs)
            ql.weight = torch.nn.Parameter(reconstructed, requires_grad=False)
            reconstruct_res = ql(*example_inputs)
            self.assertEqual(reconstruct_res, ref)


common_utils.instantiate_parametrized_tests(TestAffineQuantized)
common_utils.instantiate_parametrized_tests(TestAffineQuantizedBasic)


if __name__ == "__main__":
    run_tests()
