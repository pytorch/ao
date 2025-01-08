from typing import Optional

import torch
import torch.nn.functional as F

from torchao.dtypes import to_affine_quantized_intx
from torchao.dtypes.uintx.uintx_layout import UintxLayout
from torchao.quantization.granularity import Granularity
from torchao.quantization.observer import (
    AffineQuantizedObserverBase,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
)


class AWQObserver(AffineQuantizedObserverBase):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        quantization_granularity: Granularity,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        n_validation_examples: int,
        validation_sequence_len: int,
        scale_search_space_size: int = 20,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: Optional[bool] = True,
        zero_point_domain=ZeroPointDomain.INT,
    ):
        """
        A custom observer for Activation aware Weight Quantization (AWQ)

        Args:
            weight: The weight tensor to be observed.
            bias: The bias tensor to be observed.
            quantization_granularity: Granularity which specifies how many weights share the same scale/zero point
            input_dtype: The data type of the input tensor.
            mapping_type: Always set to asymmetric
            target_dtype: The target data type of the quantized tensor
            n_validation_examples: Number of examples used to calibrate observer
            validation_sequence_len: Number of tokens in each example
            scale_search_space_size: The number of scales to search for.
            quant_min: The minimum quantized value
            quant_max: The maximum quantized value
            eps: The minimum scale.
            scale_dtype: The data type of the scale tensor.
            zero_point_dtype: The data type of the zero point tensor.
            preserve_zero: A flag to indicate whether we need zero to be exactly
                representable or not.
            zero_point_domain: The domain of the zero point.
        """
        super().__init__(
            mapping_type,
            target_dtype,
            quantization_granularity,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            scale_dtype=scale_dtype,
            zero_point_dtype=zero_point_dtype,
            preserve_zero=preserve_zero,
            zero_point_domain=zero_point_domain,
        )
        self.quantization_granularity = quantization_granularity
        self.weight = weight
        self.bias = bias
        self.n_validation_examples = n_validation_examples
        self.validation_sequence_len = validation_sequence_len
        self.calibration_token_count = 0
        self.inputs = []
        self.outputs = []
        self.scale_options = scale_search_space_size
        self.device = self.weight.device
        self.average = torch.zeros((1, weight.shape[1]), device=self.device)
        if self.bias is not None:
            self.bias.to(self.device)

    @torch.no_grad()
    def forward(self, input: torch.Tensor, output: torch.Tensor):
        # import pdb
        # pdb.set_trace()
        # print(input.shape, input.abs().sum(1).shape, self.average.shape)
        if len(self.inputs) < self.n_validation_examples:
            self.inputs.append(input.to("cpu"))
            self.outputs.append(output.to("cpu"))
        self.calibration_token_count += input.shape[-2]
        self.average += input.abs().sum(-2)

    def calculate_qparams(self):
        # import pdb
        # pdb.set_trace()
        assert (
            self.outputs != None
        ), "calibrate observer first by running model on exemplar data"
        self.average /= self.calibration_token_count
        for i in range(self.n_validation_examples):
            self.inputs[i] = self.inputs[i].to(self.device)
            self.outputs[i] = self.outputs[i].to(self.device)

        best_loss = float("inf")
        best_scales = None
        for i in range(self.scale_options):
            ratio = i * 1 / self.scale_options
            scales = self.average.pow(ratio).to(self.weight.dtype)
            scales = scales / (scales.max() * scales.min()).sqrt()
            layout = UintxLayout(self.target_dtype)
            # regardless of weight dtype, we have to store as packed uint8 tensors
            tensor_dtype = torch.uint8
            w = to_affine_quantized_intx(
                self.weight * scales,
                self.mapping_type,
                (1, self.quantization_granularity.group_size),
                tensor_dtype,
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                eps=self.eps,
                scale_dtype=self.scale_dtype,
                zero_point_dtype=self.zero_point_dtype,
                preserve_zero=self.preserve_zero,
                zero_point_domain=self.zero_point_domain,
                _layout=layout,
            )
            loss = 0
            for i in range(self.n_validation_examples):
                q_out = F.linear(self.inputs[i] / scales, w, self.bias)
                loss += (self.outputs[i] - q_out).pow(2).mean().item()
            if loss < best_loss:
                best_scales = scales
                best_loss = loss
            for i in range(self.n_validation_examples):
                self.inputs[i].to("cpu")
                self.outputs[i].to("cpu")
        return best_scales.detach()


class AWQObservedLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_obs: torch.nn.Module,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_obs = act_obs

    def forward(self, input: torch.Tensor):
        output = F.linear(input, self.weight, self.bias)
        self.act_obs(input, output)
        return output

    @classmethod
    def from_float(cls, float_linear: torch.nn.Linear, act_obs: AWQObserver):
        observed_linear = cls(
            float_linear.in_features,
            float_linear.out_features,
            act_obs,
            False,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
        )
        observed_linear.weight = float_linear.weight
        observed_linear.bias = float_linear.bias
        return observed_linear
