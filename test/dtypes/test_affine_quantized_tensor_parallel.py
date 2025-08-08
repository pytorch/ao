# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import pytest
import torch
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchao.quantization import (
    float8_dynamic_activation_float8_weight,
    float8_weight_only,
    int4_weight_only,
    int8_dynamic_activation_int8_weight,
    int8_weight_only,
)
from torchao.quantization.observer import PerRow, PerTensor
from torchao.quantization.quant_api import quantize_

if common_utils.SEED is None:
    common_utils.SEED = 1234

try:
    import gemlite  # noqa: F401

    has_gemlite = True
except ModuleNotFoundError:
    has_gemlite = False

if torch.version.hip is not None:
    pytest.skip("Skipping the test in ROCm", allow_module_level=True)


class TestAffineQuantizedTensorParallel(DTensorTestBase):
    """Basic test case for tensor subclasses"""

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
        dtensor = DTensor.from_local(local_shard, mesh, [Shard(1)], run_check=True)
        # Replace parameter in module
        m.linear.weight = torch.nn.Parameter(dtensor, requires_grad=False)
        return m

    def quantize(self, m: torch.nn.Module) -> torch.nn.Module:
        """
        Quantize the model
        """
        quantize_(m, self.QUANT_METHOD_FN(**self.QUANT_METHOD_KWARGS))
        return m

    def _test_tp(self, dtype):
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


class TestInt8woAffineQuantizedTensorParallel(TestAffineQuantizedTensorParallel):
    QUANT_METHOD_FN = staticmethod(int8_weight_only)
    COMMON_DTYPES = [torch.bfloat16, torch.float16, torch.float32]

    @common_utils.parametrize("dtype", COMMON_DTYPES)
    @with_comms
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_tp(self, dtype):
        return self._test_tp(dtype)


class TestInt4woAffineQuantizedTensorParallel(TestAffineQuantizedTensorParallel):
    QUANT_METHOD_FN = staticmethod(int4_weight_only)
    COMMON_DTYPES = [torch.bfloat16]

    @common_utils.parametrize("dtype", COMMON_DTYPES)
    @with_comms
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skip(
        "This doesn't work right now with the new constraint of aliasing, "
        "we'll look into this later"
    )
    def test_tp(self, dtype):
        return self._test_tp(dtype)


class TestGemliteLayoutTensorParallel(TestAffineQuantizedTensorParallel):
    COMMON_DTYPES = [torch.float16]

    @common_utils.parametrize("dtype", COMMON_DTYPES)
    @with_comms
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(not has_gemlite, "gemlite not available")
    def test_tp_gemlite(self, dtype):
        from torchao.quantization import gemlite_uintx_weight_only

        for packing_bitwidth in [32, 8]:
            for bit_width in [4, 8]:
                for group_size in [64, 32, None] if bit_width == 4 else [None]:
                    api = lambda: gemlite_uintx_weight_only(
                        group_size, bit_width, packing_bitwidth
                    )
                    self.QUANT_METHOD_FN = staticmethod(api)
                    return self._test_tp(dtype)


class TestInt8dqAffineQuantizedTensorParallel(TestAffineQuantizedTensorParallel):
    QUANT_METHOD_FN = staticmethod(int8_dynamic_activation_int8_weight)
    COMMON_DTYPES = [torch.bfloat16]

    @common_utils.parametrize("dtype", COMMON_DTYPES)
    @with_comms
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_tp(self, dtype):
        return self._test_tp(dtype)


common_utils.instantiate_parametrized_tests(TestInt8woAffineQuantizedTensorParallel)
common_utils.instantiate_parametrized_tests(TestInt4woAffineQuantizedTensorParallel)
common_utils.instantiate_parametrized_tests(TestGemliteLayoutTensorParallel)
common_utils.instantiate_parametrized_tests(TestInt8dqAffineQuantizedTensorParallel)

# Run only on H100
if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0):

    class TestFloat8woAffineQuantizedTensorParallel(TestAffineQuantizedTensorParallel):
        QUANT_METHOD_FN = staticmethod(float8_weight_only)
        COMMON_DTYPES = [torch.bfloat16, torch.float16, torch.float32]

        @common_utils.parametrize("dtype", COMMON_DTYPES)
        @with_comms
        @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
        def test_tp(self, dtype):
            return self._test_tp(dtype)

    class TestFloat8dqTensorAffineQuantizedTensorParallel(
        TestAffineQuantizedTensorParallel
    ):
        QUANT_METHOD_FN = staticmethod(float8_dynamic_activation_float8_weight)
        QUANT_METHOD_KWARGS = {"granularity": PerTensor()}
        COMMON_DTYPES = [torch.bfloat16, torch.float16, torch.float32]

        @common_utils.parametrize("dtype", COMMON_DTYPES)
        @with_comms
        @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
        def test_tp(self, dtype):
            return self._test_tp(dtype)

    class TestFloat8dqRowAffineQuantizedTensorParallel(
        TestAffineQuantizedTensorParallel
    ):
        QUANT_METHOD_FN = staticmethod(float8_dynamic_activation_float8_weight)
        QUANT_METHOD_KWARGS = {"granularity": PerRow()}
        COMMON_DTYPES = [torch.bfloat16]

        @common_utils.parametrize("dtype", COMMON_DTYPES)
        @with_comms
        @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
        def test_tp(self, dtype):
            return self._test_tp(dtype)

    common_utils.instantiate_parametrized_tests(
        TestFloat8woAffineQuantizedTensorParallel
    )
    common_utils.instantiate_parametrized_tests(
        TestFloat8dqTensorAffineQuantizedTensorParallel
    )
    common_utils.instantiate_parametrized_tests(
        TestFloat8dqRowAffineQuantizedTensorParallel
    )
if __name__ == "__main__":
    run_tests()
