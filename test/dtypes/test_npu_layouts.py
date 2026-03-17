# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for Huawei Ascend NPU quantization layouts (Int4NPULayout, Float8NPULayout).
These tests require a Huawei Ascend NPU device with torch_npu installed.
"""

import pytest
import torch

from torchao.dtypes import Float8NPULayout, Int4NPULayout

npu_available = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not npu_available, reason="NPU not available")
class TestInt4NPULayout:
    def test_layout_instantiation(self):
        layout = Int4NPULayout()
        assert isinstance(layout, Int4NPULayout)

    def test_quantize_linear_int4_npu(self):
        import torch_npu  # noqa: F401

        from torchao.quantization import Int4WeightOnlyConfig, quantize_

        model = torch.nn.Linear(512, 512, bias=False).npu().to(torch.bfloat16)
        quantize_(model, Int4WeightOnlyConfig(layout=Int4NPULayout()))
        assert model.weight is not None

    def test_linear_forward_int4_npu(self):
        import torch_npu  # noqa: F401

        from torchao.quantization import Int4WeightOnlyConfig, quantize_

        model = torch.nn.Linear(512, 512, bias=False).npu().to(torch.bfloat16)
        quantize_(model, Int4WeightOnlyConfig(layout=Int4NPULayout()))
        x = torch.randn(4, 512, device="npu", dtype=torch.bfloat16)
        y = model(x)
        assert y.shape == (4, 512)
        assert y.dtype == torch.bfloat16

    def test_linear_forward_int4_npu_with_bias(self):
        import torch_npu  # noqa: F401

        from torchao.quantization import Int4WeightOnlyConfig, quantize_

        model = torch.nn.Linear(512, 512, bias=True).npu().to(torch.bfloat16)
        quantize_(model, Int4WeightOnlyConfig(layout=Int4NPULayout()))
        x = torch.randn(4, 512, device="npu", dtype=torch.bfloat16)
        y = model(x)
        assert y.shape == (4, 512)

    def test_tensor_flatten_unflatten_int4_npu(self):
        import torch_npu  # noqa: F401

        from torchao.dtypes.uintx.int4_npu_layout import Int4NPUAQTTensorImpl

        from torchao.quantization import Int4WeightOnlyConfig, quantize_

        model = torch.nn.Linear(512, 512, bias=False).npu().to(torch.bfloat16)
        quantize_(model, Int4WeightOnlyConfig(layout=Int4NPULayout()))

        # Get the tensor impl
        weight = model.weight
        tensor_impl = weight.tensor_impl
        assert isinstance(tensor_impl, Int4NPUAQTTensorImpl)

        # Test flatten/unflatten
        tensor_names, attributes = tensor_impl.__tensor_flatten__()
        assert "packed_weight" in tensor_names

    def test_int4_npu_device_restriction(self):
        """Test that device conversion is restricted to npu -> npu."""
        import torch_npu  # noqa: F401

        from torchao.quantization import Int4WeightOnlyConfig, quantize_

        model = torch.nn.Linear(64, 64, bias=False).npu().to(torch.bfloat16)
        quantize_(model, Int4WeightOnlyConfig(layout=Int4NPULayout()))

        # Moving to CPU should raise ValueError
        with pytest.raises(ValueError, match="does not support conversion"):
            model.cpu()


@pytest.mark.skipif(not npu_available, reason="NPU not available")
class TestFloat8NPULayout:
    def test_layout_instantiation(self):
        layout = Float8NPULayout()
        assert isinstance(layout, Float8NPULayout)

    def test_layout_with_mm_config(self):
        from torchao.float8.inference import Float8MMConfig

        mm_config = Float8MMConfig(use_fast_accum=True)
        layout = Float8NPULayout(mm_config=mm_config)
        assert layout.mm_config is not None

    def test_quantize_linear_float8_npu(self):
        import torch_npu  # noqa: F401

        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            quantize_,
        )

        model = torch.nn.Linear(512, 512, bias=False).npu().to(torch.bfloat16)
        quantize_(
            model,
            Float8DynamicActivationFloat8WeightConfig(layout=Float8NPULayout()),
        )
        assert model.weight is not None

    def test_linear_forward_float8_npu(self):
        import torch_npu  # noqa: F401

        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            quantize_,
        )

        model = torch.nn.Linear(512, 512, bias=False).npu().to(torch.bfloat16)
        quantize_(
            model,
            Float8DynamicActivationFloat8WeightConfig(layout=Float8NPULayout()),
        )
        x = torch.randn(4, 512, device="npu", dtype=torch.bfloat16)
        y = model(x)
        assert y.shape == (4, 512)

    def test_float8_tensor_impl_get_plain(self):
        import torch_npu  # noqa: F401

        from torchao.dtypes.floatx.float8_npu_layout import Float8NPUAQTTensorImpl

        float8_data = torch.zeros(64, 64, dtype=torch.float8_e4m3fn, device="npu")
        scale = torch.ones(64, device="npu", dtype=torch.float32)
        impl = Float8NPUAQTTensorImpl(float8_data, scale, False, Float8NPULayout())
        data, s, zp = impl.get_plain()
        assert data is float8_data
        assert s is scale
        assert zp is None

    def test_float8_npu_device_restriction(self):
        """Test that device conversion is restricted to npu -> npu."""
        import torch_npu  # noqa: F401

        from torchao.dtypes.floatx.float8_npu_layout import Float8NPUAQTTensorImpl

        float8_data = torch.zeros(64, 64, dtype=torch.float8_e4m3fn, device="npu")
        scale = torch.ones(1, device="npu", dtype=torch.float32)
        impl = Float8NPUAQTTensorImpl(float8_data, scale, False, Float8NPULayout())

        with pytest.raises(ValueError, match="does not support conversion"):
            impl.to("cpu")


class TestNPULayoutImports:
    """Tests that run without NPU hardware - just verify imports work."""

    def test_int4_npu_layout_import(self):
        from torchao.dtypes import Int4NPULayout

        layout = Int4NPULayout()
        assert isinstance(layout, Int4NPULayout)

    def test_float8_npu_layout_import(self):
        from torchao.dtypes import Float8NPULayout

        layout = Float8NPULayout()
        assert isinstance(layout, Float8NPULayout)

    def test_int4_npu_layout_in_quant_api(self):
        from torchao.quantization.quant_api import (
            LAYOUT_TO_PRESERVE_ZEROS,
            LAYOUT_TO_ZERO_POINT_DOMAIN,
        )
        from torchao.dtypes import Int4NPULayout

        assert Int4NPULayout in LAYOUT_TO_ZERO_POINT_DOMAIN
        assert Int4NPULayout in LAYOUT_TO_PRESERVE_ZEROS
        assert LAYOUT_TO_PRESERVE_ZEROS[Int4NPULayout] is False
