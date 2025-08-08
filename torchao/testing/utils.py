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
from torchao.quantization.transform_module import (
    _QUANTIZE_CONFIG_HANDLER,
)
from torchao.testing.model_architectures import LlamaModelsLlama4Experts
from torchao.utils import (
    DummyModule,
    get_compute_capability,
)

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
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            if not torch.cuda.is_available():
                raise unittest.SkipTest("No cuda available")
            return test_func(*args, **kwargs)

        return wrapper

    return decorator


def skip_if_no_gemlite():
    import unittest

    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            try:
                import gemlite  # noqa: F401
            except:
                raise unittest.SkipTest("No gemlite available")
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

        up_compiled = torch.compile(up_dist)
        y_up = up_compiled(input_dtensor)
        dn_compiled = torch.compile(dn_dist)
        dn_compiled(y_up)


class TorchAOIntegrationTestCase(common_utils.TestCase):
    def _test_slice_and_copy_similar_to_vllm(self, config):
        # making sure https://github.com/vllm-project/vllm/blob/90bd2ab6e3eb7e83d3f40d99fc23e6e43834743a/vllm/model_executor/layers/linear.py#L483-L495 works properly
        # the test is similar to the linked code, but with some hardcoded arguments
        # and does not use tensor parallelism

        dtype = torch.bfloat16
        device = "cuda"
        l = torch.nn.Linear(1024, 1024, device="cuda", dtype=dtype)
        quantize_(l, config)

        # high level, we do a narrow for both param.data and the loaded_weights
        # and do inplace copy_ to copy from the loaded_weights into param.data

        # simulate loaded_weight
        dummy_l = torch.nn.Linear(1024, 1024).to("cuda").to(torch.bfloat16)
        # making the weight different
        dummy_l.weight = torch.nn.Parameter(
            dummy_l.weight + 2 * torch.randn(1024, 1024, device=device, dtype=dtype),
            requires_grad=False,
        )
        quantize_(dummy_l, config)

        output_dim = 0
        shard_size = 512
        for tp_rank in [0, 1]:
            start_idx = tp_rank * shard_size
            param = l.weight
            param_data = param.data
            param_data = param_data.narrow(output_dim, start_idx, shard_size)
            orig_value = param_data.qdata[0][0].item()
            loaded_weight = dummy_l.weight
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

            # making sure param.data.qdata[0][0] is not the same as loaded_weight.qdata[0][0]
            assert orig_value != loaded_weight.qdata[0][0]
            param_data.copy_(loaded_weight)
            # making sure param.data is updated to loaded_weight
            assert param_data.qdata[0][0] == loaded_weight.qdata[0][0]
            assert torch.equal(param_data.scale, loaded_weight.scale)
            if hasattr(param_data, "zero_point"):
                assert torch.equal(param_data.zero_point, loaded_weight.zero_point)

    def _test_moe_weight_reshape_ops(self, config):
        """This is testing the op call sequence in saving and loading quantization
        checkpoints in llama-models for llama4
        (https://github.com/meta-llama/llama-models/tree/main/models/llama4)
        """
        # only per row quantization is supported for bmm
        dtype = torch.bfloat16
        device = "cuda"

        def _quantize_experts(model, config):
            for _, module in model.named_modules():
                if not isinstance(module, LlamaModelsLlama4Experts):
                    continue

                expert_module = module
                for weight_name in ["w1", "w2", "w3"]:
                    weight = getattr(expert_module, weight_name)
                    config_handler = _QUANTIZE_CONFIG_HANDLER[type(config)]
                    dummy_mod = DummyModule(weight)
                    quant_mod = config_handler(dummy_mod, config)
                    setattr(expert_module, weight_name, quant_mod.weight)

        batch_size = 4
        num_experts = 2
        input_dim = 64
        dim = 128
        hidden_dim = 256

        moe1 = LlamaModelsLlama4Experts(num_experts, dim, hidden_dim, dtype, device)
        moe2 = LlamaModelsLlama4Experts(num_experts, dim, hidden_dim, dtype, device)
        moe_combined = LlamaModelsLlama4Experts(
            num_experts, dim, 2 * hidden_dim, dtype, device
        )
        input = torch.randn(batch_size, input_dim, dim, dtype=dtype, device=device)

        moes = [moe1, moe2]

        for moe in moes:
            moe(input)

            # need to transpose before quantizing
            moe.w1 = torch.nn.Parameter(
                moe.w1.transpose(1, 2).contiguous(), requires_grad=False
            )
            moe.w2 = torch.nn.Parameter(
                moe.w2.transpose(1, 2).contiguous(), requires_grad=False
            )
            moe.w3 = torch.nn.Parameter(
                moe.w3.transpose(1, 2).contiguous(), requires_grad=False
            )

            _quantize_experts(moe, config)

            before = moe(input)

            # transposing for resharding support since only 2D resharding is supported
            new_last_dim = moe.w1.shape[-2]
            moe.w1 = torch.nn.Parameter(
                moe.w1.transpose(1, 2).reshape(-1, new_last_dim).contiguous(),
                requires_grad=False,
            )
            new_last_dim = moe.w2.shape[-2]
            moe.w2 = torch.nn.Parameter(
                moe.w2.transpose(1, 2).reshape(-1, new_last_dim).contiguous(),
                requires_grad=False,
            )
            new_last_dim = moe.w3.shape[-2]
            moe.w3 = torch.nn.Parameter(
                moe.w3.transpose(1, 2).reshape(-1, new_last_dim).contiguous(),
                requires_grad=False,
            )

            moe.w1 = torch.nn.Parameter(
                moe.w1.unflatten(0, (num_experts, -1)).squeeze(dim=0),
                requires_grad=False,
            )
            moe.w2 = torch.nn.Parameter(
                moe.w2.unflatten(0, (num_experts, -1)).squeeze(dim=0),
                requires_grad=False,
            )
            moe.w3 = torch.nn.Parameter(
                moe.w3.unflatten(0, (num_experts, -1)).squeeze(dim=0),
                requires_grad=False,
            )

            # transpose again to recover the original weights
            moe.w1 = torch.nn.Parameter(
                moe.w1.transpose(1, 2).contiguous(), requires_grad=False
            )
            moe.w2 = torch.nn.Parameter(
                moe.w2.transpose(1, 2).contiguous(), requires_grad=False
            )
            moe.w3 = torch.nn.Parameter(
                moe.w3.transpose(1, 2).contiguous(), requires_grad=False
            )

            after = moe(input)
            self.assertEqual(before, after)

        state_dicts = [moe1.state_dict(), moe2.state_dict()]
        # align the scale parameter so they can be concatenated
        for key in ["w1", "w2", "w3"]:
            weights = [st[key] for st in state_dicts]
            for i in range(1, len(weights)):
                weights[i].scale = weights[0].scale
                if hasattr(weights[i], "zero_point"):
                    weights[i].zero_point = weights[0].zero_point

        def process_key(key: str) -> torch.Tensor:
            tensors = [s[key] for s in state_dicts]
            # Note: we have a hacky implementation for cat in user codebase
            # since it is not implemented correctly before
            if key == "w2":
                return torch.cat(tensors, dim=-1)
            else:
                return torch.cat(tensors, dim=-2)

        new_state_dict = {}
        for key in ["w1", "w2", "w3"]:
            new_state_dict[key] = process_key(key)

        moe_combined.w1 = torch.nn.Parameter(
            moe_combined.w1.transpose(1, 2), requires_grad=False
        )
        moe_combined.w2 = torch.nn.Parameter(
            moe_combined.w2.transpose(1, 2), requires_grad=False
        )
        moe_combined.w3 = torch.nn.Parameter(
            moe_combined.w3.transpose(1, 2), requires_grad=False
        )
        moe_combined.load_state_dict(new_state_dict, assign=True)
        # make sure it runs
        moe_combined(input)


common_utils.instantiate_parametrized_tests(TorchAOBasicTestCase)
common_utils.instantiate_parametrized_tests(TorchAOCompileTestCase)
common_utils.instantiate_parametrized_tests(TorchAOTensorParallelTestCase)


if __name__ == "__main__":
    unittest.main()
