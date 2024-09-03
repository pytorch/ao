from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from torchao.dtypes.uintx.Uintx import to_uintx
from torchao.dtypes.affine_quantized_tensor import (
    to_affine_quantized_intx,
    LayoutType,
    register_layout_cls,
    PlainAQTLayout,
    register_aqt_quantized_linear_dispatch

) 
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
)
from torchao.quantization.observer import (
    AffineQuantizedObserverBase,
)


class AWQObserver(AffineQuantizedObserverBase):
    def __init__(self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        block_size: Tuple,
        input_dtype: torch.dtype,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        device: str,
        scale_search_space_size: int = 20,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: Optional[bool] = True,
        zero_point_domain = ZeroPointDomain.INT,
    ):
        super().__init__(
            mapping_type,
            target_dtype,
            block_size = block_size, 
            quant_min = quant_min,
            quant_max = quant_max,
            eps = eps,
            scale_dtype = scale_dtype,
            zero_point_dtype = zero_point_dtype,
            preserve_zero = preserve_zero,
            zero_point_domain = zero_point_domain,
        )
        self.weight = weight
        self.bias = bias
        self.scale_options = scale_search_space_size
        self.scales = None
        self.device = device

    @torch.no_grad()
    def forward(self, input: torch.Tensor, output: torch.Tensor):
        average = input.abs().view(-1,input.shape[-1]).mean(0)
        
        best_loss = float('inf')
        scaleopts = []
        for i in range(self.scale_options):
            ratio = i * 1 / self.scale_options
            scales = average.pow(ratio)
            scales = scales / (scales.max() * scales.min()).sqrt()
            layout = AWQLayoutType(scales, self.target_dtype)
            tensor_dtype = torch.int8 if self.target_dtype == torch.int8 else torch.uint8
            w = to_affine_quantized_intx(
                self.weight.data,
                self.mapping_type,
                self.block_size,
                tensor_dtype,
                quant_min = self.quant_min,
                quant_max = self.quant_max,
                eps = self.eps,
                scale_dtype = self.scale_dtype,
                zero_point_dtype = self.zero_point_dtype,
                preserve_zero = self.preserve_zero,
                zero_point_domain = self.zero_point_domain,
                layout_type = layout
            )
            q_out = F.linear(input/scales, w, self.bias)
            scaleopts.append(q_out.mean().item())
            loss = (output - q_out).pow(2).mean().item()
            if loss < best_loss:
                self.scales = scales
                best_loss = loss

    def calculate_qparams(self):
        return self.scales.detach()


class ObservedLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, act_obs: torch.nn.Module, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_obs = act_obs

    def forward(self, input: torch.Tensor):
        output = F.linear(input, self.weight, self.bias)
        self.act_obs(input, output)
        return output

    @classmethod
    def from_float(cls, float_linear: torch.nn.Linear, act_obs: AWQObserver):
        observed_linear = cls(float_linear.in_features, float_linear.out_features, act_obs, False, device=float_linear.weight.device, dtype=float_linear.weight.dtype)
        observed_linear.weight = float_linear.weight
        observed_linear.bias = float_linear.bias
        return observed_linear
    

@dataclass(frozen=True)
class AWQLayoutType(LayoutType):
    equalization_scale: torch.Tensor
    dtype: torch.dtype

    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.equalization_scale

    def post_process(self, input: torch.Tensor) -> torch.Tensor:
        
        return to_uintx(input, self.dtype)
    
    def _quantized_linear_impl(input_tensor, weight_tensor, bias):
        return F.linear(input_tensor / weight_tensor.layout_tensor.layout_type.equalization_scale, weight_tensor.dequantize(), bias)
    
    def _linear_awq_check(input_tensor, weight_tensor, bias):
        return isinstance(weight_tensor.layout_tensor, AWQ_AQTLayout)

register_aqt_quantized_linear_dispatch(AWQLayoutType._linear_awq_check, AWQLayoutType._quantized_linear_impl)

@register_layout_cls(AWQLayoutType)
class AWQ_AQTLayout(PlainAQTLayout):
    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.int_data.get_plain(), self.scale, self.zero_point
    
    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        layout_type: LayoutType,
    ):
        return cls(int_data, scale, zero_point, layout_type)