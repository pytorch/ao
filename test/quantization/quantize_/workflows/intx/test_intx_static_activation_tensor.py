# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import unittest
from typing import Optional
from unittest import mock

import torch
import torch.nn.functional as F
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import TestCase, run_tests

from torchao.core.config import config_from_dict, config_to_dict
from torchao.quantization import (
    Int8StaticActivationIntxWeightConfig,
    IntxStaticActivationTensor,
    MappingType,
    PerAxis,
    PerGroup,
    quantize_,
)
from torchao.quantization.quant_primitives import (
    dequantize_affine,
    quantize_affine,
)
from torchao.utils import torch_version_at_least

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_M = 32


def _static_linear_reference(
    x: torch.Tensor,
    weight_tensor: IntxStaticActivationTensor,
    bias: Optional[torch.Tensor] = None,
    return_pre_quant: bool = False,
) -> torch.Tensor:
    orig_dtype = x.dtype
    if weight_tensor.act_quant_scale is not None:
        scale = weight_tensor.act_quant_scale
        if scale.numel() == 1:
            scale = scale.reshape((1,) * x.ndim)
            block_size = list(x.shape)
        else:
            while scale.ndim < x.ndim:
                scale = scale.unsqueeze(0)
            block_size = [x.shape[i] // scale.shape[i] for i in range(x.ndim)]
        if weight_tensor.act_quant_zero_point is not None:
            zp = weight_tensor.act_quant_zero_point
            if zp.numel() == 1:
                zp = zp.reshape((1,) * x.ndim)
            else:
                while zp.ndim < x.ndim:
                    zp = zp.unsqueeze(0)
            zp = zp.expand_as(scale)
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
    y_pre_quant = y

    if weight_tensor.output_quant_scale is not None:
        scale = weight_tensor.output_quant_scale
        if scale.numel() == 1:
            scale = scale.reshape((1,) * y.ndim)
            block_size = list(y.shape)
        else:
            while scale.ndim < y.ndim:
                scale = scale.unsqueeze(0)
            block_size = [y.shape[i] // scale.shape[i] for i in range(y.ndim)]
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
    if return_pre_quant:
        return y, y_pre_quant
    return y


def _assert_close_static(
    test: TestCase,
    actual: torch.Tensor,
    x: torch.Tensor,
    weight_tensor: IntxStaticActivationTensor,
    bias: Optional[torch.Tensor],
    rtol: float,
    atol: float,
):
    expected, pre_quant = _static_linear_reference(
        x, weight_tensor, bias, return_pre_quant=True
    )
    if weight_tensor.output_quant_scale is None:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        return

    actual_f, expected_f, pre_f = (
        t.to(torch.float32) for t in (actual, expected, pre_quant)
    )
    out_scale, _ = _broadcast_output_scale(weight_tensor, pre_f)
    diff = (actual_f - expected_f).abs()
    mismatch = diff > atol + rtol * expected_f.abs()
    if not bool(mismatch.any()):
        return
    step = out_scale.expand_as(pre_f)[mismatch]
    # (a) exactly one output step apart
    test.assertTrue(
        bool((diff[mismatch] <= step * 1.01 + atol).all()),
        f"mismatches exceed one output step: max {diff.max().item()}",
    )
    # (b) reference pre-quant value sits on a rounding boundary (x.5 in step units)
    u = pre_f[mismatch] / step
    dist_to_boundary = (u - torch.floor(u) - 0.5).abs() * step
    tol = atol + rtol * pre_f[mismatch].abs()
    test.assertTrue(
        bool((dist_to_boundary <= tol).all()),
        f"{int(mismatch.sum())} mismatched elements, "
        f"{int((dist_to_boundary > tol).sum())} not at a rounding boundary "
        f"(max distance {dist_to_boundary.max().item():.4g} vs tol {tol.max().item():.4g})",
    )


def _broadcast_output_scale(weight_tensor: IntxStaticActivationTensor, y: torch.Tensor):
    scale = weight_tensor.output_quant_scale.to(torch.float32)
    if scale.numel() == 1:
        return scale.reshape((1,) * y.ndim), True
    return scale.reshape((1,) * (y.ndim - 1) + (scale.numel(),)), False


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
    @common_utils.parametrize("scale_dtype", [torch.float32, torch.bfloat16])
    @common_utils.parametrize("batch", [2, _M])
    def test_linear_numerical_reference(
        self,
        device,
        granularity,
        weight_mapping_type,
        has_act_zp,
        has_output_scale,
        scale_dtype,
        batch,
    ):
        torch.manual_seed(42)
        dtype = torch.bfloat16
        x = torch.randn(batch, 64, dtype=dtype, device=device)
        linear = torch.nn.Linear(64, 32, dtype=dtype, device=device)

        act_quant_scale = torch.tensor(0.05, dtype=scale_dtype, device=device)
        act_quant_zero_point = (
            torch.tensor(5, dtype=torch.int8, device=device) if has_act_zp else None
        )
        output_quant_scale = (
            torch.tensor(0.1, dtype=scale_dtype, device=device)
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
        _assert_close_static(
            self, actual, x, linear.weight, linear.bias, rtol=1e-2, atol=1e-2
        )

    @common_utils.parametrize("device", _DEVICES)
    @common_utils.parametrize("out_features", [2, 16])
    def test_asymmetric_unit_weight_zero_point(self, device, out_features):
        linear = torch.nn.Linear(
            32, out_features, bias=False, dtype=torch.float32, device=device
        )
        with torch.no_grad():
            linear.weight.fill_(1.0)

        config = Int8StaticActivationIntxWeightConfig(
            act_quant_scale=torch.tensor(1.0, dtype=torch.float32, device=device),
            weight_dtype=torch.int4,
            weight_granularity=PerAxis(0),
            weight_mapping_type=MappingType.ASYMMETRIC,
        )
        quantize_(linear, config)

        x = torch.ones(_M, 32, dtype=torch.float32, device=device)
        actual = linear(x)
        expected = _static_linear_reference(x, linear.weight, None)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            actual,
            torch.full((_M, out_features), 32.0, device=device),
            rtol=1e-4,
            atol=1e-4,
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

        x = torch.randn(_M, 64, dtype=torch.bfloat16, device=device)
        actual = linear(x)
        expected = _static_linear_reference(x, linear.weight, None)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    @common_utils.parametrize("device", _DEVICES)
    def test_from_hp_custom_scale_and_zero_point(self, device):
        torch.manual_seed(0)
        weight = torch.randn(16, 64, dtype=torch.float32, device=device)
        custom_scale = torch.full((16, 1), 0.1, dtype=torch.float32, device=device)
        custom_zp = torch.full((16, 1), 2, dtype=torch.int8, device=device)

        w = IntxStaticActivationTensor.from_hp(
            weight,
            block_size=(1, 64),
            target_dtype=torch.int4,
            mapping_type=MappingType.ASYMMETRIC,
            intx_choose_qparams_algorithm=None,
            custom_scale=custom_scale,
            custom_zero_point=custom_zp,
            act_quant_scale=torch.tensor(0.05, dtype=torch.float32, device=device),
        )
        torch.testing.assert_close(w.scale, custom_scale)
        torch.testing.assert_close(w.zero_point, custom_zp)

        x = torch.randn(_M, 64, dtype=torch.float32, device=device)
        actual = F.linear(x, w)
        expected = _static_linear_reference(x, w, None)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    @common_utils.parametrize("granularity", [PerAxis(0), PerGroup(32)])
    def test_qat_range_learning_convert_pass_qparams(self, granularity):
        from torchao.quantization.qat import (
            IntxFakeQuantizeConfig,
            QATConfig,
            initialize_fake_quantizers,
        )

        torch.manual_seed(0)
        in_features, out_features = 64, 16
        group_size = (
            granularity.group_size if isinstance(granularity, PerGroup) else in_features
        )
        m = torch.nn.Sequential(torch.nn.Linear(in_features, out_features, bias=False))
        fq_config = IntxFakeQuantizeConfig(
            torch.int4,
            group_size=group_size,
            is_symmetric=True,
            is_dynamic=False,
            range_learning=True,
        )
        quantize_(m, QATConfig(weight_config=fq_config, step="prepare"))
        initialize_fake_quantizers(m, (torch.randn(4, in_features),))
        learned_scale = m[0].weight_fake_quantizer.scale.detach().clone()

        base_config = Int8StaticActivationIntxWeightConfig(
            act_quant_scale=torch.tensor(0.05),
            weight_dtype=torch.int4,
            weight_granularity=granularity,
        )
        quantize_(m, QATConfig(base_config, step="convert"))
        self.assertIsInstance(m[0].weight, IntxStaticActivationTensor)
        torch.testing.assert_close(m[0].weight.scale, learned_scale)
        self.assertTrue(torch.all(m[0].weight.zero_point == 0))

        x = torch.randn(_M, in_features)
        torch.testing.assert_close(
            m(x), _static_linear_reference(x, m[0].weight, None), rtol=1e-4, atol=1e-4
        )

    @common_utils.parametrize("device", _DEVICES)
    def test_config_serialization_roundtrip(self, device):
        intx_config = Int8StaticActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerAxis(0),
            weight_mapping_type=MappingType.ASYMMETRIC,
            act_mapping_type=MappingType.ASYMMETRIC,
        )
        restored_intx_config = config_from_dict(config_to_dict(intx_config))
        self.assertEqual(restored_intx_config.weight_dtype, torch.int4)
        self.assertEqual(restored_intx_config.weight_granularity, PerAxis(0))
        self.assertEqual(
            restored_intx_config.weight_mapping_type, MappingType.ASYMMETRIC
        )
        self.assertEqual(restored_intx_config.act_mapping_type, MappingType.ASYMMETRIC)

        linear1 = torch.nn.Linear(32, 16, dtype=torch.float32, device=device)
        linear2 = copy.deepcopy(linear1)
        quantize_(linear1, intx_config)
        quantize_(linear2, restored_intx_config)

        x = torch.randn(_M, 32, dtype=torch.float32, device=device)
        torch.testing.assert_close(linear1(x), linear2(x))

    @common_utils.parametrize("device", _DEVICES)
    def test_varying_input_ranks(self, device):
        x_2d = torch.randn(_M, 32, dtype=torch.float32, device=device)
        x_3d = x_2d.unsqueeze(0)

        # Per-tensor calibration tensors may carry singleton dims (e.g. (1, 1, 1)) as
        # exported by some checkpoints; they must behave exactly like 0-d scalars.
        linear_intx = torch.nn.Linear(32, 16, dtype=torch.float32, device=device)
        quantize_(
            linear_intx,
            Int8StaticActivationIntxWeightConfig(
                act_quant_scale=torch.full((1, 1, 1), 0.05, device=device),
                act_quant_zero_point=torch.full(
                    (1, 1, 1), 1, dtype=torch.int8, device=device
                ),
                output_quant_scale=torch.full((1, 1, 1), 0.1, device=device),
                weight_dtype=torch.int4,
                weight_granularity=PerAxis(0),
                act_mapping_type=MappingType.ASYMMETRIC,
            ),
        )
        out_2d = linear_intx(x_2d)
        out_3d = linear_intx(x_3d)
        out_2d_again = linear_intx(x_2d)
        self.assertEqual(linear_intx.weight.act_quant_scale.shape, (1, 1, 1))
        torch.testing.assert_close(out_2d, out_3d.squeeze(0))
        torch.testing.assert_close(out_2d, out_2d_again)
        _assert_close_static(
            self,
            out_2d,
            x_2d,
            linear_intx.weight,
            linear_intx.bias,
            rtol=1e-4,
            atol=1e-4,
        )

    @common_utils.parametrize("device", _DEVICES)
    def test_rejects_unsupported_calibration_shapes(self, device):
        def make(**kwargs):
            linear = torch.nn.Linear(32, 16, dtype=torch.float32, device=device)
            quantize_(
                linear,
                Int8StaticActivationIntxWeightConfig(
                    weight_dtype=torch.int4,
                    weight_granularity=PerAxis(0),
                    act_mapping_type=MappingType.ASYMMETRIC,
                    **kwargs,
                ),
            )
            return linear

        # Supported: per-tensor and per-channel
        make(act_quant_scale=torch.tensor(0.05, device=device))
        make(act_quant_scale=torch.full((32,), 0.05, device=device))
        make(act_quant_scale=torch.full((1, 32), 0.05, device=device))
        make(output_quant_scale=torch.full((16,), 0.1, device=device))

        # Rejected: per-token or wrong channel count
        with self.assertRaisesRegex(AssertionError, "act_quant_scale"):
            make(act_quant_scale=torch.full((_M, 1), 0.05, device=device))
        with self.assertRaisesRegex(AssertionError, "act_quant_scale"):
            make(act_quant_scale=torch.full((16,), 0.05, device=device))
        with self.assertRaisesRegex(AssertionError, "output_quant_scale"):
            make(output_quant_scale=torch.full((32,), 0.1, device=device))
        with self.assertRaisesRegex(AssertionError, "act_quant_zero_point"):
            make(
                act_quant_scale=torch.full((32,), 0.05, device=device),
                act_quant_zero_point=torch.full(
                    (16,), 1, dtype=torch.int8, device=device
                ),
            )

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

        # Verify narrow on dim 0 and dim 1, and full-dimension slice w[:16, :]
        sliced_0 = embedding.weight.narrow(0, 0, 16)
        torch.testing.assert_close(
            sliced_0.dequantize(), dequantized_weight.narrow(0, 0, 16)
        )
        sliced_full = embedding.weight[:16, :]
        torch.testing.assert_close(sliced_full.dequantize(), dequantized_weight[:16, :])
        sliced_1 = embedding.weight.narrow(1, 0, 64)
        torch.testing.assert_close(
            sliced_1.dequantize(), dequantized_weight.narrow(1, 0, 64)
        )
        with self.assertRaises(NotImplementedError):
            _ = embedding.weight[::2]

    @common_utils.parametrize("device", _DEVICES)
    def test_to_device_and_dtype(self, device):
        torch.manual_seed(0)
        # 1. Pure device move preserves float32 calibration scales on a bfloat16 layer
        linear_bf16 = torch.nn.Linear(
            64, 32, bias=False, dtype=torch.bfloat16, device="cpu"
        )
        quantize_(
            linear_bf16,
            Int8StaticActivationIntxWeightConfig(
                act_quant_scale=torch.tensor(0.05, dtype=torch.float32),
                act_quant_zero_point=torch.tensor(3, dtype=torch.int8),
                output_quant_scale=torch.tensor(0.1, dtype=torch.float32),
                weight_dtype=torch.int4,
                weight_granularity=PerAxis(0),
                weight_mapping_type=MappingType.ASYMMETRIC,
                act_mapping_type=MappingType.ASYMMETRIC,
            ),
        )
        linear_bf16 = linear_bf16.to(device)
        self.assertEqual(linear_bf16.weight.act_quant_scale.dtype, torch.float32)
        self.assertEqual(linear_bf16.weight.output_quant_scale.dtype, torch.float32)
        self.assertEqual(linear_bf16.weight.act_quant_zero_point.dtype, torch.int8)

        # 2. Explicit dtype move (.to(torch.bfloat16)) casts scales and preserves numerical accuracy
        linear_fp32 = torch.nn.Linear(
            64, 32, bias=True, dtype=torch.float32, device=device
        )
        quantize_(
            linear_fp32,
            Int8StaticActivationIntxWeightConfig(
                act_quant_scale=torch.tensor(0.05, dtype=torch.float32, device=device),
                act_quant_zero_point=torch.tensor(5, dtype=torch.int8, device=device),
                output_quant_scale=torch.tensor(
                    0.1, dtype=torch.float32, device=device
                ),
                weight_dtype=torch.int4,
                weight_granularity=PerAxis(0),
                weight_mapping_type=MappingType.ASYMMETRIC,
                act_mapping_type=MappingType.ASYMMETRIC,
            ),
        )
        linear_cast = linear_fp32.to(torch.bfloat16)
        self.assertEqual(linear_cast.weight.act_quant_scale.dtype, torch.bfloat16)
        self.assertEqual(linear_cast.weight.output_quant_scale.dtype, torch.bfloat16)

        x = torch.randn(_M, 64, dtype=torch.bfloat16, device=device)
        actual = linear_cast(x)
        _assert_close_static(
            self, actual, x, linear_cast.weight, linear_cast.bias, rtol=1e-2, atol=1e-2
        )

    @common_utils.parametrize("device", _DEVICES)
    def test_slice_per_axis_and_calibration_scales(self, device):
        torch.manual_seed(0)
        linear = torch.nn.Linear(64, 32, bias=False, dtype=torch.float32, device=device)
        act_scale = torch.linspace(0.01, 0.05, 64, dtype=torch.float32, device=device)
        act_zp = torch.arange(-32, 32, dtype=torch.int8, device=device)
        out_scale = torch.linspace(0.02, 0.1, 32, dtype=torch.float32, device=device)

        quantize_(
            linear,
            Int8StaticActivationIntxWeightConfig(
                act_quant_scale=act_scale,
                act_quant_zero_point=act_zp,
                output_quant_scale=out_scale,
                weight_dtype=torch.int4,
                weight_granularity=PerAxis(0),
                weight_mapping_type=MappingType.ASYMMETRIC,
                act_mapping_type=MappingType.ASYMMETRIC,
            ),
        )

        # Row-parallel slice (dim=0): slices output_quant_scale along dim 0
        w_row = linear.weight[:16, :]
        self.assertEqual(w_row.shape, (16, 64))
        self.assertEqual(w_row.output_quant_scale.shape, (16,))
        x_full = torch.randn(_M, 64, dtype=torch.float32, device=device)
        torch.testing.assert_close(
            F.linear(x_full, w_row),
            _static_linear_reference(x_full, w_row, None),
            rtol=1e-4,
            atol=1e-4,
        )

        # Column-parallel slice (dim=1) on PerAxis(0): keeps single-block row scale & slices per-feature act scales
        w_col = linear.weight[:, :32]
        self.assertEqual(w_col.shape, (32, 32))
        self.assertEqual(w_col.block_size, (1, 32))
        self.assertEqual(w_col.act_quant_scale.shape, (32,))
        self.assertEqual(w_col.act_quant_zero_point.shape, (32,))
        torch.testing.assert_close(
            w_col.dequantize(), linear.weight.dequantize()[:, :32]
        )
        x_half = torch.randn(_M, 32, dtype=torch.float32, device=device)
        torch.testing.assert_close(
            F.linear(x_half, w_col),
            _static_linear_reference(x_half, w_col, None),
            rtol=1e-4,
            atol=1e-4,
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

    def test_safetensors_flatten_unflatten(self):
        from torchao.prototype.safetensors.safetensors_support import (
            flatten_tensor_state_dict,
            unflatten_tensor_state_dict,
        )
        from torchao.prototype.safetensors.safetensors_utils import is_metadata_torchao

        torch.manual_seed(0)
        config = Int8StaticActivationIntxWeightConfig(
            act_quant_scale=torch.tensor(0.05),
            act_quant_zero_point=torch.tensor(2, dtype=torch.int8),
            output_quant_scale=torch.tensor(0.1),
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(32),
            weight_mapping_type=MappingType.ASYMMETRIC,
            act_mapping_type=MappingType.ASYMMETRIC,
        )
        model = torch.nn.Sequential(torch.nn.Linear(64, 32))
        quantize_(model, config)
        x = torch.randn(2, 64)
        expected = model(x)

        tensors_data, metadata = flatten_tensor_state_dict(model.state_dict())
        self.assertTrue(is_metadata_torchao(metadata))
        restored_state_dict, leftover = unflatten_tensor_state_dict(
            tensors_data, metadata
        )
        self.assertEqual(len(leftover), 0)

        # Verify both assign=True and assign=False (aten.copy_)
        model_assign = torch.nn.Sequential(torch.nn.Linear(64, 32))
        model_assign.load_state_dict(restored_state_dict, assign=True)
        torch.testing.assert_close(model_assign(x), expected)

        model_copy = torch.nn.Sequential(torch.nn.Linear(64, 32))
        quantize_(model_copy, config)
        model_copy.load_state_dict(restored_state_dict, assign=False)
        torch.testing.assert_close(model_copy(x), expected)

    @common_utils.parametrize("device", _DEVICES)
    @common_utils.parametrize("granularity", [PerAxis(0), PerGroup(32)])
    @common_utils.parametrize("batch", [2, _M])
    def test_compile(self, device, granularity, batch):
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
        x = torch.randn(batch, 64, dtype=dtype, device=device)
        expected = model(x)
        compiled_model = torch.compile(model, fullgraph=True)
        actual = compiled_model(x)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_cuda_int_mm_path_selection(self):
        """The int8 fast path must be used for M > 16 with K, N multiples of 8, and the
        dequantize path otherwise (safe_int_mm cannot handle those shapes under compile)."""
        import importlib

        mod = importlib.import_module(
            "torchao.quantization.quantize_.workflows.intx.intx_static_activation_tensor"
        )

        calls = []
        orig_safe_int_mm = mod.safe_int_mm

        def spy(a, b):
            calls.append(tuple(a.shape))
            return orig_safe_int_mm(a, b)

        def make_linear(in_features, out_features):
            linear = torch.nn.Linear(in_features, out_features, device="cuda")
            quantize_(
                linear,
                Int8StaticActivationIntxWeightConfig(
                    act_quant_scale=torch.tensor(0.05, device="cuda"),
                    act_quant_zero_point=torch.tensor(
                        2, dtype=torch.int8, device="cuda"
                    ),
                    weight_dtype=torch.int4,
                    weight_granularity=PerAxis(0),
                    weight_mapping_type=MappingType.ASYMMETRIC,
                    act_mapping_type=MappingType.ASYMMETRIC,
                ),
            )
            return linear

        aligned = make_linear(64, 32)
        unaligned_n = make_linear(64, 12)
        with mock.patch.object(mod, "safe_int_mm", spy):
            aligned(torch.randn(_M, 64, device="cuda"))
            self.assertEqual(calls, [(_M, 64)])
            aligned(torch.randn(2, 3, 64, device="cuda"))  # 3D input, 6 tokens <= 16
            self.assertEqual(len(calls), 1)
            aligned(torch.randn(3, 8, 64, device="cuda"))  # 3D input, 24 tokens
            self.assertEqual(calls[-1], (24, 64))
            unaligned_n(torch.randn(_M, 64, device="cuda"))
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    run_tests()
