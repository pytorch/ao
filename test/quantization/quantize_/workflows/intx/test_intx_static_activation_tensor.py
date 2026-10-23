# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import unittest
from typing import Optional

import torch
import torch.nn.functional as F
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import TestCase, run_tests

from torchao.core.config import config_from_dict, config_to_dict
from torchao.quantization import (
    Int8StaticActivationInt8WeightConfig,
    Int8StaticActivationIntxWeightConfig,
    IntxStaticActivationTensor,
    MappingType,
    PerAxis,
    PerGroup,
    PerRow,
    PerTensor,
    quantize_,
)
from torchao.quantization.quant_primitives import (
    dequantize_affine,
    quantize_affine,
)
from torchao.utils import torch_version_at_least

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _static_linear_reference(
    x: torch.Tensor,
    weight_tensor: IntxStaticActivationTensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    orig_dtype = x.dtype
    if weight_tensor.act_quant_scale is not None:
        block_size = list(x.shape)
        scale = weight_tensor.act_quant_scale
        if scale.numel() == 1:
            scale = scale.reshape((1,) * x.ndim)
        if weight_tensor.act_quant_zero_point is not None:
            zp = weight_tensor.act_quant_zero_point
            if zp.numel() == 1:
                zp = zp.reshape((1,) * x.ndim)
        else:
            zp = torch.zeros_like(scale, dtype=torch.int8)
        x_q = quantize_affine(
            x,
            block_size=block_size,
            scale=scale,
            zero_point=zp,
            output_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
        )
        x = dequantize_affine(
            input=x_q,
            block_size=block_size,
            scale=scale,
            zero_point=zp,
            input_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            output_dtype=orig_dtype,
        )

    w_dq = weight_tensor.dequantize().to(orig_dtype)
    y = F.linear(x, w_dq, bias)

    if weight_tensor.output_quant_scale is not None:
        block_size = list(y.shape)
        scale = weight_tensor.output_quant_scale
        if scale.numel() == 1:
            scale = scale.reshape((1,) * y.ndim)
        zp = torch.zeros_like(scale, dtype=torch.int8)
        y_q = quantize_affine(
            y,
            block_size=block_size,
            scale=scale,
            zero_point=zp,
            output_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
        )
        y = dequantize_affine(
            input=y_q,
            block_size=block_size,
            scale=scale,
            zero_point=zp,
            input_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            output_dtype=orig_dtype,
        )
    return y


@unittest.skipIf(not torch_version_at_least("2.7.0"), "Need pytorch 2.7+")
@common_utils.instantiate_parametrized_tests
class TestIntxStaticActivationTensor(TestCase):
    @common_utils.parametrize("device", _DEVICES)
    @common_utils.parametrize("granularity", [PerAxis(0), PerGroup(32)])
    @common_utils.parametrize(
        "weight_mapping_type", [MappingType.SYMMETRIC, MappingType.ASYMMETRIC]
    )
    @common_utils.parametrize("has_act_zp", [False, True])
    @common_utils.parametrize("has_output_scale", [False, True])
    def test_linear_numerical_reference(
        self,
        device,
        granularity,
        weight_mapping_type,
        has_act_zp,
        has_output_scale,
    ):
        torch.manual_seed(42)
        dtype = torch.bfloat16
        x = torch.randn(2, 64, dtype=dtype, device=device)
        linear = torch.nn.Linear(64, 32, dtype=dtype, device=device)

        act_quant_scale = torch.tensor(0.05, dtype=torch.float32, device=device)
        act_quant_zero_point = (
            torch.tensor(5, dtype=torch.int8, device=device) if has_act_zp else None
        )
        output_quant_scale = (
            torch.tensor(0.1, dtype=torch.float32, device=device)
            if has_output_scale
            else None
        )

        config = Int8StaticActivationIntxWeightConfig(
            act_quant_scale=act_quant_scale,
            act_quant_zero_point=act_quant_zero_point,
            output_quant_scale=output_quant_scale,
            weight_dtype=torch.int4,
            weight_granularity=granularity,
            weight_mapping_type=weight_mapping_type,
            act_mapping_type=(
                MappingType.ASYMMETRIC if has_act_zp else MappingType.SYMMETRIC
            ),
        )
        quantize_(linear, config)
        self.assertIsInstance(linear.weight, IntxStaticActivationTensor)

        actual = linear(x)
        expected = _static_linear_reference(x, linear.weight, linear.bias)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    @common_utils.parametrize("device", _DEVICES)
    def test_asymmetric_unit_weight_zero_point(self, device):
        linear = torch.nn.Linear(32, 2, bias=False, dtype=torch.float32, device=device)
        with torch.no_grad():
            linear.weight.fill_(1.0)

        config = Int8StaticActivationIntxWeightConfig(
            act_quant_scale=torch.tensor(1.0, dtype=torch.float32, device=device),
            weight_dtype=torch.int4,
            weight_granularity=PerAxis(0),
            weight_mapping_type=MappingType.ASYMMETRIC,
        )
        quantize_(linear, config)

        x = torch.ones(1, 32, dtype=torch.float32, device=device)
        actual = linear(x)
        expected = _static_linear_reference(x, linear.weight, None)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            actual, torch.full((1, 2), 32.0, device=device), rtol=1e-4, atol=1e-4
        )

    @common_utils.parametrize("device", _DEVICES)
    def test_weight_scale_dtype(self, device):
        torch.manual_seed(0)
        linear = torch.nn.Linear(
            64, 32, bias=False, dtype=torch.bfloat16, device=device
        )
        config = Int8StaticActivationIntxWeightConfig(
            act_quant_scale=torch.tensor(0.05, dtype=torch.float32, device=device),
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(32),
            weight_scale_dtype=torch.float32,
        )
        quantize_(linear, config)
        self.assertIsInstance(linear.weight, IntxStaticActivationTensor)
        self.assertEqual(linear.weight.scale.dtype, torch.float32)

        x = torch.randn(2, 64, dtype=torch.bfloat16, device=device)
        actual = linear(x)
        expected = _static_linear_reference(x, linear.weight, None)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    @common_utils.parametrize("device", _DEVICES)
    def test_custom_weight_scale_and_zero_point(self, device):
        torch.manual_seed(0)
        linear = torch.nn.Linear(64, 16, bias=False, dtype=torch.float32, device=device)
        custom_scale = torch.full((16, 1), 0.1, dtype=torch.float32, device=device)
        custom_zp = torch.full((16, 1), 2, dtype=torch.int8, device=device)

        config = Int8StaticActivationIntxWeightConfig(
            act_quant_scale=torch.tensor(0.05, dtype=torch.float32, device=device),
            weight_dtype=torch.int4,
            weight_granularity=PerAxis(0),
            weight_mapping_type=MappingType.ASYMMETRIC,
            custom_weight_scale=custom_scale,
            custom_weight_zero_point=custom_zp,
        )
        quantize_(linear, config)
        torch.testing.assert_close(linear.weight.scale, custom_scale)
        torch.testing.assert_close(linear.weight.zero_point, custom_zp)

        x = torch.randn(2, 64, dtype=torch.float32, device=device)
        actual = linear(x)
        expected = _static_linear_reference(x, linear.weight, None)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    @common_utils.parametrize("device", _DEVICES)
    def test_config_serialization_roundtrip(self, device):
        # Test Int8StaticActivationIntxWeightConfig with 0D scalar tensors
        intx_config = Int8StaticActivationIntxWeightConfig(
            act_quant_scale=torch.tensor(0.05),
            act_quant_zero_point=torch.tensor(2, dtype=torch.int8),
            output_quant_scale=torch.tensor(0.1),
            weight_dtype=torch.int4,
            weight_granularity=PerAxis(0),
            act_mapping_type=MappingType.ASYMMETRIC,
        )
        restored_intx_config = config_from_dict(config_to_dict(intx_config))
        self.assertIsInstance(restored_intx_config.act_quant_scale, torch.Tensor)
        self.assertIsInstance(restored_intx_config.act_quant_zero_point, torch.Tensor)
        self.assertIsInstance(restored_intx_config.output_quant_scale, torch.Tensor)

        linear1 = torch.nn.Linear(32, 16, dtype=torch.float32, device=device)
        linear2 = copy.deepcopy(linear1)
        quantize_(linear1, intx_config)
        quantize_(linear2, restored_intx_config)

        x = torch.randn(2, 32, dtype=torch.float32, device=device)
        torch.testing.assert_close(linear1(x), linear2(x))

        # Test Int8StaticActivationInt8WeightConfig with 0D scalar tensors
        int8_config = Int8StaticActivationInt8WeightConfig(
            act_quant_scale=torch.tensor(0.05),
            act_quant_zero_point=torch.tensor(0, dtype=torch.int8),
            output_quant_scale=torch.tensor(0.1),
            granularity=[PerTensor(), PerRow()],
        )
        restored_int8_config = config_from_dict(config_to_dict(int8_config))
        self.assertIsInstance(restored_int8_config.act_quant_scale, torch.Tensor)
        self.assertIsInstance(restored_int8_config.act_quant_zero_point, torch.Tensor)
        self.assertIsInstance(restored_int8_config.output_quant_scale, torch.Tensor)

        linear3 = torch.nn.Linear(32, 16, dtype=torch.float32, device=device)
        linear4 = copy.deepcopy(linear3)
        quantize_(linear3, int8_config)
        quantize_(linear4, restored_int8_config)
        torch.testing.assert_close(linear3(x), linear4(x))

    @common_utils.parametrize("device", _DEVICES)
    def test_varying_input_ranks(self, device):
        x_2d = torch.randn(2, 32, dtype=torch.float32, device=device)
        x_3d = x_2d.unsqueeze(0)

        # Int8StaticActivationIntxWeightConfig
        linear_intx = torch.nn.Linear(32, 16, dtype=torch.float32, device=device)
        quantize_(
            linear_intx,
            Int8StaticActivationIntxWeightConfig(
                act_quant_scale=torch.tensor(0.05, device=device),
                act_quant_zero_point=torch.tensor(1, dtype=torch.int8, device=device),
                output_quant_scale=torch.tensor(0.1, device=device),
                weight_dtype=torch.int4,
                weight_granularity=PerAxis(0),
                act_mapping_type=MappingType.ASYMMETRIC,
            ),
        )
        out_2d = linear_intx(x_2d)
        out_3d = linear_intx(x_3d)
        out_2d_again = linear_intx(x_2d)
        self.assertEqual(linear_intx.weight.act_quant_scale.ndim, 0)
        torch.testing.assert_close(out_2d, out_3d.squeeze(0))
        torch.testing.assert_close(out_2d, out_2d_again)

        # Int8StaticActivationInt8WeightConfig
        linear_int8 = torch.nn.Linear(32, 16, dtype=torch.float32, device=device)
        quantize_(
            linear_int8,
            Int8StaticActivationInt8WeightConfig(
                act_quant_scale=torch.tensor(0.05, device=device),
                act_quant_zero_point=torch.tensor(0, dtype=torch.int8, device=device),
                output_quant_scale=torch.tensor(0.1, device=device),
                granularity=[PerTensor(), PerRow()],
            ),
        )
        out8_2d = linear_int8(x_2d)
        out8_3d = linear_int8(x_3d)
        out8_2d_again = linear_int8(x_2d)
        self.assertEqual(linear_int8.weight.act_quant_scale.ndim, 0)
        torch.testing.assert_close(out8_2d, out8_3d.squeeze(0))
        torch.testing.assert_close(out8_2d, out8_2d_again)

    def test_embedding_and_select_and_slice(self):
        dtype = torch.bfloat16
        device = "cpu"
        embedding = torch.nn.Embedding(
            128, 256, padding_idx=0, dtype=dtype, device=device
        )
        with torch.no_grad():
            # Explicitly set nonzero values at padding_idx row to verify forward semantics
            embedding.weight[0].fill_(1.5)

        config = Int8StaticActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(32),
        )
        is_embedding = lambda n, _: isinstance(n, torch.nn.Embedding)
        quantize_(embedding, config, filter_fn=is_embedding)
        self.assertIsInstance(embedding.weight, IntxStaticActivationTensor)

        indices = torch.tensor([[0, 1, 5], [10, 0, 20]], device=device)
        output = embedding(indices)
        dequantized_weight = embedding.weight.dequantize()
        expected_output = F.embedding(indices, dequantized_weight, padding_idx=0)
        torch.testing.assert_close(output, expected_output, atol=1e-5, rtol=1e-5)

        # Verify max_norm fallback matches dequantized F.embedding
        indices_norm = torch.tensor([1, 5, 10], device=device)
        output_norm = F.embedding(indices_norm, embedding.weight, max_norm=1.0)
        expected_norm = F.embedding(
            indices_norm, embedding.weight.dequantize(), max_norm=1.0
        )
        torch.testing.assert_close(output_norm, expected_norm, atol=1e-5, rtol=1e-5)

        # Verify select on dim 0
        selected_row = embedding.weight[5, :]
        torch.testing.assert_close(
            selected_row, dequantized_weight[5, :], atol=1e-3, rtol=1e-3
        )

        # Verify narrow on dim 0 and dim 1
        sliced_0 = embedding.weight.narrow(0, 0, 16)
        torch.testing.assert_close(
            sliced_0.dequantize(), dequantized_weight.narrow(0, 0, 16)
        )
        sliced_1 = embedding.weight.narrow(1, 0, 64)
        torch.testing.assert_close(
            sliced_1.dequantize(), dequantized_weight.narrow(1, 0, 64)
        )

    def test_state_dict_serialization(self):
        model = torch.nn.Sequential(torch.nn.Linear(64, 32))
        model2 = torch.nn.Sequential(torch.nn.Linear(64, 32))
        x = torch.randn(2, 64)
        config = Int8StaticActivationIntxWeightConfig(
            act_quant_scale=torch.tensor(0.05),
            output_quant_scale=torch.tensor(0.1),
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(32),
        )
        quantize_(model, config)
        expected = model(x)

        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.save(model.state_dict(), f"{tmpdirname}/model.pt")
            state_dict = torch.load(
                f"{tmpdirname}/model.pt", map_location="cpu", weights_only=True
            )
            model2.load_state_dict(state_dict, assign=True)
            actual = model2(x)
            torch.testing.assert_close(expected, actual)

    @common_utils.parametrize("device", _DEVICES)
    @common_utils.parametrize("granularity", [PerAxis(0), PerGroup(32)])
    def test_compile(self, device, granularity):
        torch.compiler.reset()
        torch.manual_seed(0)
        dtype = torch.float32
        model = torch.nn.Sequential(torch.nn.Linear(64, 32, dtype=dtype, device=device))
        config = Int8StaticActivationIntxWeightConfig(
            act_quant_scale=torch.tensor(0.05, dtype=torch.float32, device=device),
            act_quant_zero_point=torch.tensor(2, dtype=torch.int8, device=device),
            output_quant_scale=torch.tensor(0.01, dtype=torch.float32, device=device),
            weight_dtype=torch.int4,
            weight_granularity=granularity,
            weight_mapping_type=MappingType.ASYMMETRIC,
            act_mapping_type=MappingType.ASYMMETRIC,
        )
        quantize_(model, config)
        x = torch.randn(32, 64, dtype=dtype, device=device)
        expected = model(x)
        compiled_model = torch.compile(model)
        actual = compiled_model(x)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    run_tests()
