import unittest
import functools
import copy
import torch
import torchao

from torch.testing._internal import common_utils
from torchao.dtypes import AffineQuantizedTensor
from torchao.dtypes import to_affine_quantized_intx
from torchao.quantization.quant_primitives import MappingType

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

# copied from https://github.com/pytorch/pytorch/blob/941d094dd1b507dacf06ddc6ed3485a9537e09b7/test/inductor/test_torchinductor.py#L11389
def copy_tests(
    my_cls, other_cls, suffix, test_failures=None, xfail_prop=None
):  # noqa: B902
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
        tensor_data_dict = {name: getattr(lp_tensor, name) for name in tensor_data_name_dict}
        outer_size = lp_tensor.size()
        outer_stride = lp_tensor.stride()
        reconstructed = self.TENSOR_SUBCLASS.__tensor_unflatten__(tensor_data_dict, tensor_attributes, outer_size, outer_stride)
        self.assertEqual(lp_tensor.dequantize(), reconstructed.dequantize())

    @common_utils.parametrize("device", COMMON_DEVICES)
    @common_utils.parametrize("dtype", COMMON_DTYPES)
    def test_hp_tensor_device_dtype(self, device, dtype):
        hp_tensor = torch.randn(4, 128, device=device, dtype=dtype)
        lp_tensor = self.FACTORY_FN(hp_tensor, **self.kwargs)

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
        self.assertGreater(torchao.quantization.utils.compute_error(hp_res, lp_res), self.LINEAR_MIN_SQNR)


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
            self.assertGreater(torchao.quantization.utils.compute_error(ref.dequantize(), compiled.dequantize()), self.COMPILE_MIN_SQNR)

    @common_utils.parametrize("device", COMMON_DEVICES)
    @common_utils.parametrize("dtype", COMMON_DTYPES)
    def test_linear_compile(self, device, dtype):
        hp_tensor = torch.randn(4, 128, device=device, dtype=dtype)
        lp_tensor = self.FACTORY_FN(hp_tensor, **self.kwargs)

        hp_act_tensor = torch.randn(32, 128, device=device, dtype=dtype)
        hp_res = torch.nn.functional.linear(hp_act_tensor, hp_tensor)
        l = torch.nn.Linear(128, 4, bias=False, device=device, dtype=dtype)
        l.weight = torch.nn.Parameter(lp_tensor)
        lp_res = torch.compile(l)(hp_act_tensor)
        self.assertGreater(torchao.quantization.utils.compute_error(hp_res, lp_res), self.LINEAR_MIN_SQNR)



common_utils.instantiate_parametrized_tests(TorchAOBasicTestCase)
common_utils.instantiate_parametrized_tests(TorchAOCompileTestCase)

if __name__ == "__main__":
    unittest.main()
