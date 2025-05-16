import unittest
from itertools import product

import torch

from torchao.prototype.quantization.module_swap import (
    IntQuantizer,
    QuantizedEmbedding,
    QuantizedLinear,
)


class TestQuantizedLinear(unittest.TestCase):
    def test_quantized_linear_init(self) -> None:
        in_features = 16
        out_features = 8
        weight_bits = 8
        weight_group_size = "per_channel"
        activation_bits = 8
        activation_group_size = "per_token"
        input_quantization = True
        output_quantization = False
        weight_quantization = True
        activation_quantization = True
        dynamic_weights = False
        range_learning = False
        scale_eps = 1e-6
        weight_quantizer = IntQuantizer(
            num_bits=weight_bits,
            group_size=weight_group_size,
            dynamic=dynamic_weights,
            quantization_mode="symmetric",
            range_learning=range_learning,
        )
        QuantizedLinear(
            in_features=in_features,
            out_features=out_features,
            weight_quantizer=weight_quantizer,
            activation_bits=activation_bits,
            activation_group_size=activation_group_size,
            input_quantization=input_quantization,
            output_quantization=output_quantization,
            weight_quantization=weight_quantization,
            activation_quantization=activation_quantization,
            scale_eps=scale_eps,
        )

    def test_quantized_linear(self) -> None:
        for (
            weight_group_size,
            activation_group_size,
            input_quantization,
            output_quantization,
            weight_quantization,
            dynamic_weights,
        ) in product(
            ["per_channel", "per_tensor", 4],
            ["per_token", "per_tensor", 4],
            [True, False],
            [True, False],
            [True, False],
            [True, False],
        ):
            for x in [
                torch.FloatTensor(torch.randn(2, 16)),
                torch.FloatTensor(torch.randn(2, 2, 16)),
            ]:
                weight_quantizer = IntQuantizer(
                    num_bits=4,
                    group_size=weight_group_size,
                    dynamic=dynamic_weights,
                    quantization_mode="symmetric",
                    range_learning=False,
                )
                linear = QuantizedLinear(
                    in_features=16,
                    out_features=8,
                    weight_quantizer=weight_quantizer,
                    activation_bits=8,
                    activation_group_size=activation_group_size,
                    input_quantization=input_quantization,
                    output_quantization=output_quantization,
                    weight_quantization=weight_quantization,
                    activation_quantization=True,
                )
                if not dynamic_weights:
                    assert isinstance(linear.weight_quantizer, IntQuantizer)
                    linear.weight_quantizer.set_scale_offset_to_min_max(linear.weight)
                linear(x)

    def test_quantized_linear_passes_gradients(self) -> None:
        for (
            weight_group_size,
            activation_group_size,
            input_quantization,
            output_quantization,
            weight_quantization,
            dynamic_weights,
        ) in product(
            ["per_channel", "per_tensor", 4],
            ["per_token", "per_tensor", 4],
            [True, False],
            [True, False],
            [True, False],
            [True, False],
        ):
            for x in [
                torch.FloatTensor(torch.randn(2, 16)),
                torch.FloatTensor(torch.randn(2, 2, 16)),
            ]:
                x = x.requires_grad_(True)
                weight_quantizer = IntQuantizer(
                    num_bits=4,
                    group_size=weight_group_size,
                    dynamic=dynamic_weights,
                    quantization_mode="symmetric",
                    range_learning=False,
                )
                linear = QuantizedLinear(
                    in_features=16,
                    out_features=8,
                    weight_quantizer=weight_quantizer,
                    activation_bits=8,
                    activation_group_size=activation_group_size,
                    input_quantization=input_quantization,
                    output_quantization=output_quantization,
                    weight_quantization=weight_quantization,
                    activation_quantization=True,
                )
                if not dynamic_weights:
                    assert isinstance(linear.weight_quantizer, IntQuantizer)
                    linear.weight_quantizer.set_scale_offset_to_min_max(linear.weight)
                y = linear(x)
                (y.sum() ** 2).backward()
                assert linear.weight.grad is not None
                assert x.grad is not None

    def test_quantized_linear_passes_gradients_to_weight_scale(self) -> None:
        in_features = 16
        out_features = 8
        weight_bits = 8
        weight_group_size = "per_channel"
        activation_bits = 8
        activation_group_size = "per_token"
        input_quantization = True
        output_quantization = False
        weight_quantization = True
        activation_quantization = True
        scale_eps = 1e-6
        weight_quantizer = IntQuantizer(
            num_bits=weight_bits,
            group_size=weight_group_size,
            dynamic=False,
            quantization_mode="symmetric",
            range_learning=True,
        )
        linear = QuantizedLinear(
            in_features=in_features,
            out_features=out_features,
            weight_quantizer=weight_quantizer,
            activation_bits=activation_bits,
            activation_group_size=activation_group_size,
            input_quantization=input_quantization,
            output_quantization=output_quantization,
            weight_quantization=weight_quantization,
            activation_quantization=activation_quantization,
            scale_eps=scale_eps,
            range_learning=True,
        )
        x = torch.FloatTensor(torch.randn(2, 6, 16))
        y = linear(x)
        loss = y.sum()
        loss.backward()
        assert linear.weight_quantizer.scale is not None
        scale = linear.weight_quantizer.scale
        assert isinstance(scale, torch.Tensor)
        assert scale.grad is not None

    def test_quantized_linear_passes_gradients_to_activation_scale(self) -> None:
        in_features = 16
        out_features = 8
        weight_bits = 8
        weight_group_size = "per_channel"
        activation_bits = 8
        activation_group_size = "per_tensor"
        input_quantization = True
        output_quantization = False
        weight_quantization = False
        activation_quantization = True
        scale_eps = 1e-6
        weight_quantizer = IntQuantizer(
            num_bits=weight_bits,
            group_size=weight_group_size,
            dynamic=False,
            quantization_mode="symmetric",
            range_learning=True,
        )
        linear = QuantizedLinear(
            in_features=in_features,
            out_features=out_features,
            weight_quantizer=weight_quantizer,
            activation_bits=activation_bits,
            activation_group_size=activation_group_size,
            input_quantization=input_quantization,
            output_quantization=output_quantization,
            weight_quantization=weight_quantization,
            activation_quantization=activation_quantization,
            scale_eps=scale_eps,
            dynamic_activations=False,
            range_learning=True,
        )
        x = torch.FloatTensor(torch.randn(2, 8, 16))
        assert linear.input_quantizer is not None
        linear.input_quantizer.set_scale_offset_to_min_max(x)
        y = linear(x)
        loss = y.sum()
        loss.backward()
        assert linear.input_quantizer is not None
        assert linear.input_quantizer._range_learning is True

        assert linear.input_quantizer is not None
        assert linear.input_quantizer.scale is not None
        assert linear.input_quantizer.offset is not None
        assert linear.input_quantizer.scale.requires_grad is True

        assert linear.input_quantizer.scale.grad is not None, (
            linear.input_quantizer.scale
        )
        assert linear.input_quantizer.offset.grad is not None

    def test_set_weight_scale_to_min_max_test_all_options(self) -> None:
        for (
            x,
            weight_group_size,
        ) in product(
            [
                torch.FloatTensor(torch.randn(2, 16)),
                torch.FloatTensor(torch.randn(2, 2, 16)),
            ],
            ["per_channel", "per_tensor", 4],
        ):
            x = x.requires_grad_(True)
            weight_quantizer = IntQuantizer(
                num_bits=4,
                group_size=weight_group_size,
                dynamic=False,
                quantization_mode="symmetric",
                range_learning=True,
            )
            linear = QuantizedLinear(
                in_features=16,
                out_features=8,
                weight_quantizer=weight_quantizer,
                activation_bits=8,
                activation_group_size="per_token",
                input_quantization=False,
                output_quantization=False,
                weight_quantization=True,
                activation_quantization=False,
            )
            assert isinstance(linear.weight_quantizer, IntQuantizer)
            linear.weight_quantizer.set_scale_offset_to_min_max(linear.weight)

    def test_set_weight_scale_to_min_max_test_correct(self) -> None:
        weight_group_size = "per_channel"

        weight_quantizer = IntQuantizer(
            num_bits=4,
            group_size=weight_group_size,
            dynamic=False,
            quantization_mode="symmetric",
            range_learning=True,
        )
        linear = QuantizedLinear(
            in_features=16,
            out_features=1,
            weight_quantizer=weight_quantizer,
            activation_bits=8,
            activation_group_size="per_token",
            input_quantization=False,
            output_quantization=False,
            weight_quantization=True,
            activation_quantization=False,
        )

        linear.weight.data = torch.ones_like(linear.weight.data) * 7
        linear.weight.data[0][0] = -8

        assert isinstance(linear.weight_quantizer, IntQuantizer)
        linear.weight_quantizer.set_scale_offset_to_min_max(linear.weight)
        assert linear.weight_quantizer.scale is not None
        scale = linear.weight_quantizer.scale
        assert isinstance(scale, torch.Tensor)
        assert torch.allclose(scale, torch.FloatTensor([1.0]))

    def test_quantize_dynamic(self) -> None:
        weight_quantizer = IntQuantizer(
            num_bits=4,
            group_size="per_channel",
            dynamic=False,
            quantization_mode="symmetric",
            range_learning=True,
        )
        linear = QuantizedLinear(
            in_features=32,
            out_features=64,
            weight_quantizer=weight_quantizer,
            activation_bits=8,
            activation_group_size="per_token",
            input_quantization=True,
            output_quantization=False,
            weight_quantization=True,
            activation_quantization=True,
        )

        x = torch.FloatTensor([0, 5, 10, 15])
        assert linear.input_quantizer is not None
        linear.input_quantizer(x)

        torch.testing.assert_close(x, torch.FloatTensor([0, 5, 10, 15]))

    def test_quantize_dynamic_vectorized(self) -> None:
        weight_quantizer = IntQuantizer(
            num_bits=4,
            group_size="per_channel",
            dynamic=False,
            quantization_mode="symmetric",
            range_learning=True,
        )
        linear = QuantizedLinear(  # noqa
            in_features=32,
            out_features=64,
            weight_quantizer=weight_quantizer,
            activation_bits=8,
            activation_group_size="per_token",
            input_quantization=True,
            output_quantization=False,
            weight_quantization=True,
            activation_quantization=True,
        )

        x = torch.FloatTensor([0, 5, 10, 15, 0, 10, 20, 30]).reshape([2, 4])
        assert linear.input_quantizer is not None
        linear.input_quantizer(x)

        torch.testing.assert_close(
            x, torch.FloatTensor([0, 5, 10, 15, 0, 10, 20, 30]).reshape([2, 4])
        )


class TestQuantizedEmbedding(unittest.TestCase):
    def test_quantized_embedding(self) -> None:
        for weight_group_size in ["per_channel", "per_tensor", 4]:
            linear = QuantizedEmbedding(
                num_embeddings=16,
                embedding_dim=12,
                num_bits=4,
                group_size=weight_group_size,
                quantization_mode="symmetric",
            )
            x = torch.Tensor(torch.zeros(2, 16).to(torch.int32))
            x[0][0] = 1
            x[1][0] = 1
            linear.weight_quantizer.set_scale_offset_to_min_max(linear.weight)
            linear(x)


if __name__ == "__main__":
    unittest.main()
