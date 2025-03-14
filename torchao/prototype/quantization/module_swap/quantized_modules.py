from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchao.prototype.quantization.module_swap.quantizers import (
    CodeBookQuantizer,
    IntQuantizer,
    __supported_group_size_strings__,
)

SupportedQuantizers = Union[CodeBookQuantizer, IntQuantizer]


class WeightModuleQuantizerBase:
    def set_weight_scale_to_min_max(self) -> None:
        if not self.weight_quantizer.dynamic:
            self.weight_quantizer.set_scale_offset_to_min_max(self.weight)
        else:
            raise ValueError(
                "Weights are quantized dynamically, no range/scale is used"
            )

    @property
    def weight_scale(self) -> torch.Tensor:
        return self.weight_quantizer.scale

    @property
    def quantized_weight(self) -> torch.Tensor:
        return self.weight_quantizer(self.weight)


class QuantizedLinear(nn.Linear, WeightModuleQuantizerBase):
    def __init__(
        self,
        activation_bits: int,
        weight_quantizer: SupportedQuantizers,
        input_quantization: bool = True,
        output_quantization: bool = False,
        activation_group_size: Union[int, str] = "per_token",
        weight_quantization: bool = True,
        activation_quantization: bool = True,
        dynamic_activations: bool = True,
        range_learning: bool = False,
        scale_eps: float = 1e-9,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        weight_group_size, activation_group_size = self.validate_group_sizes(
            weight_quantizer.group_size, activation_group_size, dynamic_activations
        )

        self.weight_quantizer: SupportedQuantizers = weight_quantizer
        self.weight_quantizer.scale_eps = scale_eps
        self.input_quantizer: Optional[IntQuantizer] = None
        self.output_quantizer: Optional[IntQuantizer] = None

        if input_quantization:
            self.input_quantizer = IntQuantizer(
                num_bits=activation_bits,
                group_size=activation_group_size,
                dynamic=dynamic_activations,
                quantization_mode="asymmetric",
                range_learning=range_learning,
                scale_eps=scale_eps,
            )
        else:
            self.input_quantizer = None
        if output_quantization:
            self.output_quantizer = IntQuantizer(
                num_bits=activation_bits,
                group_size=activation_group_size,
                dynamic=dynamic_activations,
                quantization_mode="asymmetric",
                range_learning=range_learning,
                scale_eps=scale_eps,
            )
        else:
            self.output_quantizer = None

        self.input_quantization = input_quantization
        self.output_quantization = output_quantization
        self.weight_quantization = weight_quantization
        self.activation_quantization = activation_quantization

        if not weight_quantizer.dynamic:
            assert isinstance(self.weight_quantizer, IntQuantizer)
            self.weight_quantizer.set_scale_offset_to_min_max(self.weight)

        self.pre_transforms: nn.ModuleList = nn.ModuleList()
        self.post_transforms: nn.ModuleList = nn.ModuleList()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input

        if self.input_quantization:
            assert self.input_quantizer is not None
            self.input_quantizer.quantize = self.activation_quantization
            x = self.input_quantizer(x)

        for transform in self.pre_transforms:
            x = transform(x)

        if self.weight_quantization:
            weight = self.quantized_weight
        else:
            weight = self.weight

        x = F.linear(x, weight, self.bias)

        for transform in self.post_transforms:
            x = transform(x)

        if self.output_quantization:
            assert self.output_quantizer is not None
            self.output_quantizer.quantize = self.activation_quantization
            x = self.output_quantizer(x)
        return x

    @staticmethod
    def validate_group_sizes(
        weight_group_size: Union[int, str],
        activation_group_size: Union[int, str],
        dynamic_activations: bool,
    ) -> Tuple[Union[int, str], Union[int, str]]:
        assert (
            isinstance(weight_group_size, int)
            or weight_group_size in __supported_group_size_strings__
        )
        if weight_group_size == "per_token":
            raise ValueError(
                "per_token is only available for dynamic activation quantization"
            )

        assert (
            isinstance(activation_group_size, int)
            or activation_group_size in __supported_group_size_strings__
        )
        if activation_group_size == "per_channel":
            raise ValueError("per_channel is not supported for activatins")
        if not dynamic_activations and activation_group_size != "per_tensor":
            raise ValueError("Only per-tensor supported for static activations")

        return weight_group_size, activation_group_size

    def __repr__(self) -> str:
        output_string = "QuantizedLinear("
        empty_space = " " * len(output_string)
        output_string += (
            f"weight_quantizer={self.weight_quantizer} - {self.weight_quantization}, \n"
        )
        if self.input_quantizer is not None:
            output_string += (
                empty_space
                + f"input_quant={self.input_quantizer} - {self.activation_quantization}, \n"
            )
        if self.output_quantizer is not None:
            output_string += (
                empty_space
                + f"output_quant={self.output_quantizer} - {self.activation_quantization}, \n"
            )
        if self.pre_transforms:
            output_string += empty_space + f"pre_transforms={self.pre_transforms}, \n"
        if self.post_transforms:
            output_string += empty_space + f"post_transforms={self.post_transforms}, \n"
        output_string = output_string[:-3]
        output_string += ")"
        return output_string


class QuantizedEmbedding(nn.Embedding, WeightModuleQuantizerBase):
    def __init__(
        self,
        num_bits: int,
        group_size: Union[int, str],
        quantization_mode: str,
        range_learning: bool = False,
        scale_eps: float = 1e-9,
        dynamic_weights: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.weight_quantizer = IntQuantizer(
            num_bits=num_bits,
            group_size=group_size,
            dynamic=dynamic_weights,
            quantization_mode="symmetric",
            range_learning=range_learning,
            scale_eps=scale_eps,
        )
        self.weight_quantization = True
        self.dynamic_weights = dynamic_weights

        if not self.dynamic_weights:
            self.weight_quantizer.set_scale_offset_to_min_max(self.weight)
        self._range_learning = range_learning

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight_quantization:
            weight = self.weight_quantizer(self.weight)
        else:
            weight = self.weight

        return torch.nn.functional.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
