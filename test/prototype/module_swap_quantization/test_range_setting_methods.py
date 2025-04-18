import copy
import unittest
from typing import Union

import torch
from torch import nn

from torchao.prototype.quantization.module_swap import (
    IntQuantizer,
    QuantizedLinear,
)
from torchao.prototype.quantization.module_swap.range_setting_methods import (
    quantize_per_group_scales,
    set_activation_min_max,
    set_weight_min_max,
    set_weight_mse,
    set_weight_range_activation_loss,
)


class SimpleTestNetwork(nn.Module):
    def __init__(self, weight_group_size: Union[int, str] = "per_channel") -> None:
        super().__init__()
        weight_quantizer = IntQuantizer(
            num_bits=4,
            group_size=weight_group_size,
            dynamic=False,
            quantization_mode="symmetric",
            range_learning=False,
        )
        self.linear = QuantizedLinear(
            in_features=16,
            out_features=8,
            weight_quantizer=weight_quantizer,
            bias=False,
            activation_bits=8,
            input_quantization=False,
            output_quantization=False,
            weight_quantization=True,
            activation_quantization=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SimpleTestNetworkStaticActivation(nn.Module):
    def __init__(self, weight_group_size: Union[int, str] = "per_channel") -> None:
        super().__init__()
        weight_quantizer = IntQuantizer(
            num_bits=4,
            group_size=weight_group_size,
            dynamic=False,
            quantization_mode="symmetric",
            range_learning=False,
        )
        self.linear = QuantizedLinear(
            in_features=16,
            out_features=8,
            weight_quantizer=weight_quantizer,
            bias=False,
            activation_bits=8,
            input_quantization=True,
            output_quantization=True,
            weight_quantization=True,
            activation_quantization=True,
            dynamic_activations=False,
            activation_group_size="per_tensor",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestSetWeightMinMax(unittest.TestCase):
    def test_set_weight_min_max(self) -> None:
        model = SimpleTestNetwork()

        set_weight_min_max(model)

    def test_set_weight_min_max_grouped(self) -> None:
        model = SimpleTestNetwork(weight_group_size=8)

        set_weight_min_max(model)


class TestSetWeightMSE(unittest.TestCase):
    def test_set_weight_mse(self) -> None:
        model = SimpleTestNetwork()
        set_weight_mse(model, num_points=5)

    def test_set_weight_mse_grouped(self) -> None:
        model = SimpleTestNetwork(weight_group_size=8)
        set_weight_mse(model, num_points=5)


class TestSetWeightRangeActivationLoss(unittest.TestCase):
    def test_set_weight_range_activation_loss(self) -> None:
        model = SimpleTestNetwork()
        test_data = torch.rand(2, 16)
        set_weight_range_activation_loss(
            model,
            test_data,
            batch_size=1,
            num_points=5,
            progressive=False,
        )

    def test_set_weight_range_activation_loss_progressive(self) -> None:
        model = SimpleTestNetwork(weight_group_size=8)
        test_data = torch.rand(2, 16)
        set_weight_range_activation_loss(
            model,
            test_data,
            batch_size=1,
            num_points=5,
            progressive=True,
        )


class TestStaticActivationRangeSetting(unittest.TestCase):
    def test_static_activation_range_setting(self) -> None:
        model = SimpleTestNetworkStaticActivation()

        test_data = torch.rand(2, 16)
        set_activation_min_max(model, test_data, batch_size=1)

    def test_static_activation_range_setting_no_input(self) -> None:
        model = SimpleTestNetworkStaticActivation()

        test_data = torch.rand(2, 16)
        set_activation_min_max(model, test_data, batch_size=1)


class TestQuantizePerGroupScales(unittest.TestCase):
    def test_quantize_per_group_scales(self) -> None:
        model = SimpleTestNetwork(weight_group_size=8)

        set_weight_min_max(model)
        assert model.linear.weight_quantizer.scale is not None
        scale_before = copy.deepcopy(model.linear.weight_quantizer.scale)
        quantize_per_group_scales(model, bit_width=4)
        assert model.linear.weight_quantizer.scale is not None
        assert not torch.allclose(scale_before, model.linear.weight_quantizer.scale)

    def test_quantize_per_group_scales_dont_change_per_channel(self) -> None:
        model = SimpleTestNetwork(weight_group_size="per_channel")

        set_weight_min_max(model)
        assert model.linear.weight_quantizer.scale is not None
        scale_before = copy.deepcopy(model.linear.weight_quantizer.scale)
        quantize_per_group_scales(model, bit_width=4)
        assert model.linear.weight_quantizer.scale is not None
        assert torch.allclose(scale_before, model.linear.weight_quantizer.scale)


if __name__ == "__main__":
    unittest.main()
