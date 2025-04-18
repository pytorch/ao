# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import functools
import unittest

import torch
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.testing._internal import common_utils
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

import torchao
from torchao.dtypes import AffineQuantizedTensor, to_affine_quantized_intx
from torchao.quantization import int8_weight_only, quantize_
from torchao.quantization.quant_primitives import MappingType
from torchao.utils import TORCH_VERSION_AT_LEAST_2_6, get_compute_capability

"""
How to use:

import unittest
from torchao.testing.utils import TorchAOBasicTestCase, copy_tests
from torch.testing._internal import common_utils

# TODO: currently there is no way to set COMMON_DEVICES/COMMON_DTYPES
# we can figure out this a bit later

# change arguments
class MyTestCase(TorchAOBasicTestCase):
    TENSOR_SUBCLASS = MyDTypeTensor
    FACTOR_FN = to_my_dtype
    kwargs = {"target_dtype": torch.uint8}
    LINEAR_MIN_SQNR = 30

# copy the instantiated tests
copy_tests(TorchAOBasicTestCase, MyTestCase, "my_test_case")

if __name__ == "__main__":
    unittest.main()
"""


def skip_if_compute_capability_less_than(min_capability):
    import unittest

    def decorator(test_func):
        def wrapper(*args, **kwargs):
            if get_compute_capability() < min_capability:
                raise unittest.SkipTest(
                    f"Compute capability is less than {min_capability}"
                )
            return test_func(*args, **kwargs)

        return wrapper

    return decorator


def skip_if_rocm(message=None):
    """Decorator to skip tests on ROCm platform with custom message.

    Args:
        message (str, optional): Additional information about why the test is skipped.
    """
    import pytest

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if torch.version.hip is not None:
                skip_message = "Skipping the test in ROCm"
                if message:
                    skip_message += f": {message}"
                pytest.skip(skip_message)
            return func(*args, **kwargs)

        return wrapper

    # Handle both @skip_if_rocm and @skip_if_rocm() syntax
    if callable(message):
        func = message
        message = None
        return decorator(func)
    return decorator


def skip_if_no_cuda():
    import unittest

    def decorator(test_func):
        def wrapper(*args, **kwargs):
            if not torch.cuda.is_available():
                raise unittest.SkipTest("No cuda available")
            return test_func(*args, **kwargs)

        return wrapper

    return decorator


# copied from https://github.com/pytorch/pytorch/blob/941d094dd1b507dacf06ddc6ed3485a9537e09b7/test/inductor/test_torchinductor.py#L11389
def copy_tests(my_cls, other_cls, suffix, test_failures=None, xfail_prop=None):  # noqa: B902
    for name, value in my_cls.__dict__.items():
        if name.startswith("test_"):
            # You cannot copy functions in Python, so we use closures here to
            # create objects with different ids. Otherwise, unittest.skip
            # would modify all methods sharing the same object id. Also, by
            # using a default argument, we create a copy instead of a
            # reference. Otherwise, we would lose access to the value.

            @functools.wraps(value)
            def new_test(self, value=value):
                return value(self)

            # Copy __dict__ which may contain test metadata
            new_test.__dict__ = copy.deepcopy(value.__dict__)

            if xfail_prop is not None and hasattr(value, xfail_prop):
                new_test = unittest.expectedFailure(new_test)

            tf = test_failures and test_failures.get(name)
            if tf is not None and suffix in tf.suffixes:
                skip_func = (
                    unittest.skip("Skipped!")
                    if tf.is_skip
                    else unittest.expectedFailure
                )
                new_test = skip_func(new_test)

            setattr(other_cls, f"{name}_{suffix}", new_test)


class TorchAOBasicTestCase(common_utils.TestCase):
    COMMON_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    COMMON_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

    TENSOR_SUBCLASS = AffineQuantizedTensor
    FACTORY_FN = to_affine_quantized_intx
    kwargs = {
        "mapping_type": MappingType.ASYMMETRIC,
        "block_size": (1, 32),
        "target_dtype": torch.uint8,
    }
    # minimum sqnr for linear operation when the weight is quantized to low precision
    # with the above setting
    LINEAR_MIN_SQNR = 40

    def test_flatten_unflatten(self):
        hp_tensor = torch.randn(4, 128)
        lp_tensor = self.FACTORY_FN(hp_tensor, **self.kwargs)
        tensor_data_name_dict, tensor_attributes = lp_tensor.__tensor_flatten__()
        tensor_data_dict = {
            name: getattr(lp_tensor, name) for name in tensor_data_name_dict
        }
        outer_size = lp_tensor.size()
        outer_stride = lp_tensor.stride()
        reconstructed = self.TENSOR_SUBCLASS.__tensor_unflatten__(
            tensor_data_dict, tensor_attributes, outer_size, outer_stride
        )
        self.assertEqual(lp_tensor.dequantize(), reconstructed.dequantize())

    @common_utils.parametrize("device", COMMON_DEVICES)
    @common_utils.parametrize("dtype", COMMON_DTYPES)
    def test_hp_tensor_device_dtype(self, device, dtype):
        hp_tensor = torch.randn(4, 128, device=device, dtype=dtype)
        self.FACTORY_FN(hp_tensor, **self.kwargs)

    @common_utils.parametrize("device1", COMMON_DEVICES)
    @common_utils.parametrize("device2", COMMON_DEVICES)
    def test_device1_to_device2(self, device1, device2):
        """Note: this should be parametrized with device1 and device2
        e.g. device1 = ["cpu", "cuda"], device2 = ["cpu", "cuda"]
        """
        hp_tensor = torch.randn(4, 128, device=device1, dtype=torch.bfloat16)
        lp_tensor = self.FACTORY_FN(hp_tensor, **self.kwargs)
        lp_tensor.to(device=device2)

        hp_tensor = torch.randn(4, 128, device=device1, dtype=torch.bfloat16)
        lp_tensor = self.FACTORY_FN(hp_tensor, **self.kwargs)
        lp_tensor.to(device2)

        hp_tensor = torch.randn(4, 128, device=device1, dtype=torch.bfloat16)
        lp_tensor = self.FACTORY_FN(hp_tensor, **self.kwargs)
        lp_tensor.cuda()

        hp_tensor = torch.randn(4, 128, device=device1, dtype=torch.bfloat16)
        lp_tensor = self.FACTORY_FN(hp_tensor, **self.kwargs)
        lp_tensor.cpu()

    @common_utils.parametrize("device", COMMON_DEVICES)
    @common_utils.parametrize("dtype", COMMON_DTYPES)
    def test_transpose(self, device, dtype):
        hp_tensor = torch.randn(4, 128, device=device, dtype=dtype)
        lp_tensor = self.FACTORY_FN(hp_tensor, **self.kwargs)
        lp_tensor = lp_tensor.t()
        self.assertEqual(lp_tensor.shape, (128, 4))

    @common_utils.parametrize("device", COMMON_DEVICES)
    @common_utils.parametrize("dtype", COMMON_DTYPES)
    def test_linear(self, device, dtype):
        hp_tensor = torch.randn(4, 128, device=device, dtype=dtype)
        lp_tensor = self.FACTORY_FN(hp_tensor, **self.kwargs)

        hp_act_tensor = torch.randn(32, 128, device=device, dtype=dtype)
        hp_res = torch.nn.functional.linear(hp_act_tensor, hp_tensor)
        lp_res = torch.nn.functional.linear(hp_act_tensor, lp_tensor)
        self.assertGreater(
            torchao.quantization.utils.compute_error(hp_res, lp_res),
            self.LINEAR_MIN_SQNR,
        )


class TorchAOCompileTestCase(common_utils.TestCase):
    COMMON_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    COMMON_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

    TENSOR_SUBCLASS = AffineQuantizedTensor
    FACTORY_FN = to_affine_quantized_intx
    kwargs = {
        "mapping_type": MappingType.ASYMMETRIC,
        "block_size": (1, 32),
        "target_dtype": torch.uint8,
    }
    # minimum sqnr for linear operation when the weight is quantized to low precision
    # with the above setting
    LINEAR_MIN_SQNR = 40
    COMPILE_MIN_SQNR = 50

    @common_utils.parametrize("device", COMMON_DEVICES)
    @common_utils.parametrize("dtype", COMMON_DTYPES)
    def test_input_output_tensor_subclass(self, device, dtype):
        hp_tensor = torch.randn(4, 128, device=device, dtype=dtype)
        lp_tensor = self.FACTORY_FN(hp_tensor, **self.kwargs)

        def f(tensor):
            return tensor

        ref = f(lp_tensor)
        f = torch.compile(f)
        compiled = f(lp_tensor)
        self.assertTrue(isinstance(f(lp_tensor), self.TENSOR_SUBCLASS))
        self.assertEqual(ref.dequantize(), compiled.dequantize())

    @common_utils.parametrize("device", COMMON_DEVICES)
    @common_utils.parametrize("dtype", COMMON_DTYPES)
    def test_input_tensor_subclass(self, device, dtype):
        hp_tensor = torch.randn(4, 128, device=device, dtype=dtype)
        lp_tensor = self.FACTORY_FN(hp_tensor, **self.kwargs)

        def f(tensor):
            return tensor.dequantize()

        ref = f(lp_tensor)
        f = torch.compile(f)
        compiled = f(lp_tensor)
        self.assertFalse(isinstance(f(lp_tensor), self.TENSOR_SUBCLASS))
        self.assertEqual(ref, compiled)

    @common_utils.parametrize("device", COMMON_DEVICES)
    @common_utils.parametrize("dtype", COMMON_DTYPES)
    def test_output_tensor_subclass(self, device, dtype):
        hp_tensor = torch.randn(4, 128, device=device, dtype=dtype)

        def f(hp_tensor):
            return self.FACTORY_FN(hp_tensor, **self.kwargs)

        ref = f(hp_tensor)
        f = torch.compile(f)
        compiled = f(hp_tensor)
        self.assertTrue(isinstance(f(hp_tensor), self.TENSOR_SUBCLASS))
        # bfloat16 seems to result in much larger numerical differences
        if dtype != torch.bfloat16:
            self.assertGreater(
                torchao.quantization.utils.compute_error(
                    ref.dequantize(), compiled.dequantize()
                ),
                self.COMPILE_MIN_SQNR,
            )

    @common_utils.parametrize("device", COMMON_DEVICES)
    @common_utils.parametrize("dtype", COMMON_DTYPES)
    def test_linear_compile(self, device, dtype):
        hp_tensor = torch.randn(4, 128, device=device, dtype=dtype)
        lp_tensor = self.FACTORY_FN(hp_tensor, **self.kwargs)

        hp_act_tensor = torch.randn(32, 128, device=device, dtype=dtype)
        hp_res = torch.nn.functional.linear(hp_act_tensor, hp_tensor)
        linear = torch.nn.Linear(128, 4, bias=False, device=device, dtype=dtype)
        linear.weight = torch.nn.Parameter(lp_tensor)
        lp_res = torch.compile(linear)(hp_act_tensor)
        self.assertGreater(
            torchao.quantization.utils.compute_error(hp_res, lp_res),
            self.LINEAR_MIN_SQNR,
        )


class TorchAOTensorParallelTestCase(DTensorTestBase):
    """Basic test case for tensor subclasses"""

    COMMON_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

    TENSOR_SUBCLASS = AffineQuantizedTensor
    QUANT_METHOD_FN = staticmethod(int8_weight_only)
    QUANT_METHOD_KWARGS = {}

    @staticmethod
    def colwise_shard(m: torch.nn.Module, mesh: DeviceMesh) -> torch.nn.Module:
        """
        Shard linear layer of the model in column-wise fashion
        """
        # Column-wise is wrt to A^T, so for A it is row-wise.
        # Number of rows per rank
        orig_weight = m.linear.weight
        n_local_rows = orig_weight.size(0) // mesh.size()
        rank = mesh.get_local_rank()
        local_shard = orig_weight[rank * n_local_rows : (rank + 1) * n_local_rows, :]
        # Construct DTensor from local shard
        dtensor = DTensor.from_local(local_shard, mesh, [Shard(0)])
        # Replace parameter in module
        m.linear.weight = torch.nn.Parameter(dtensor, requires_grad=False)
        return m

    @staticmethod
    def rowwise_shard(m: torch.nn.Module, mesh: DeviceMesh) -> torch.nn.Module:
        """
        Shard linear layer of the model in row-wise fashion
        """
        # Row-wise is wrt to A^T, so for A it is column-wise.
        # Number of rows per rank
        orig_weight = m.linear.weight
        n_local_cols = orig_weight.size(1) // mesh.size()
        rank = mesh.get_local_rank()
        local_shard = orig_weight[:, rank * n_local_cols : (rank + 1) * n_local_cols]
        # Construct DTensor from local shard
        dtensor = DTensor.from_local(local_shard, mesh, [Shard(1)])
        # Replace parameter in module
        m.linear.weight = torch.nn.Parameter(dtensor, requires_grad=False)
        return m

    def quantize(self, m: torch.nn.Module) -> torch.nn.Module:
        """
        Quantize the model
        """
        quantize_(m, self.QUANT_METHOD_FN(**self.QUANT_METHOD_KWARGS))
        return m

    @common_utils.parametrize("dtype", COMMON_DTYPES)
    @with_comms
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_tp(self, dtype):
        device = "cuda"
        # To make sure different ranks create the same module
        torch.manual_seed(5)

        class M(torch.nn.Module):
            def __init__(self, in_features, out_features, **kwargs) -> None:
                super().__init__(**kwargs)
                self.linear = torch.nn.Linear(
                    in_features, out_features, bias=False, device="cuda"
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        # Get rank and device
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")

        # Original model
        proj_up = M(1024, 2048).to(device).to(dtype)
        proj_dn = M(2048, 1024).to(device).to(dtype)
        example_input = 100 * torch.randn(128, 1024, device=device, dtype=dtype)
        proj_dn(proj_up(example_input))

        # Quantize the model
        up_quant = self.quantize(proj_up)
        dn_quant = self.quantize(proj_dn)
        dn_quant(up_quant(example_input))

        mesh = self.build_device_mesh()
        mesh.device_type = "cuda"

        # Shard the models
        up_dist = self.colwise_shard(up_quant, mesh)
        dn_dist = self.rowwise_shard(dn_quant, mesh)

        # We need to turn inputs into DTensor form as well -- just a format change
        input_dtensor = DTensor.from_local(example_input, mesh, [Replicate()])

        dn_dist(up_dist(input_dtensor))

        if not TORCH_VERSION_AT_LEAST_2_6:
            # Need torch 2.6 to support compiled tensor parallelism
            return

        up_compiled = torch.compile(up_dist)
        y_up = up_compiled(input_dtensor)
        dn_compiled = torch.compile(dn_dist)
        dn_compiled(y_up)


common_utils.instantiate_parametrized_tests(TorchAOBasicTestCase)
common_utils.instantiate_parametrized_tests(TorchAOCompileTestCase)
common_utils.instantiate_parametrized_tests(TorchAOTensorParallelTestCase)

if __name__ == "__main__":
    unittest.main()
