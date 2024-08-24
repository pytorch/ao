import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from copy import deepcopy
from torchao.dtypes.utils import (
    LayoutType,
)
from torchao.dtypes.affine_quantized_tensor import (
    PlainAQTLayout, 
    register_layout_cls, 
    to_affine_quantized
    
) 

from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
)
from torchao.quantization.observer import (
    PerAxis,
    AffineQuantizedObserverBase,
)

class AWQObserver(AffineQuantizedObserverBase):
    def __init__(self,
        weight: torch.Tensor,
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
        self.block_size = (1, -1)
        super().__init__(
            mapping_type,
            target_dtype,
            block_size = self.block_size, 
            quant_min = quant_min,
            quant_max = quant_max,
            eps = eps,
            scale_dtype = scale_dtype,
            zero_point_dtype = zero_point_dtype,
            preserve_zero = preserve_zero,
            zero_point_domain = zero_point_domain,
        )
        self.weight = weight
        self.scale_options = scale_search_space_size
        self.losses = [0] * self.scale_options
        self.average = torch.zeros(weight.shape[-1], dtype=torch.float32).to(device)
        self.counter = 0

    def forward(self, input: torch.Tensor):
        self.average = self.average * self.counter / (self.counter + input.shape[0])  + input.abs().sum(dim=1).squeeze(0) / (self.counter + input.shape[0])
        self.counter += input.shape[0]
        for i in range(self.scale_options):
            unquantized_result = F.linear(input, self.weight)
            ratio = i *1.0 / self.scale_options
            scales = self.average.pow(ratio).clamp(min=1e-4)
            scales = scales / (scales.max() * scales.min()).sqrt()
            quantized_weight = to_affine_quantized(
                self.weight.data * scales,
                self.mapping_type,
                self.block_size,
                self.target_dtype,
                quant_min = self.quant_min,
                quant_max = self.quant_max,
                eps = self.eps,
                scale_dtype = self.scale_dtype,
                zero_point_dtype = self.zero_point_dtype,
                preserve_zero = self.preserve_zero,
                zero_point_domain = self.zero_point_domain,
                layout_type = AWQLayoutType(scales)
            ) 
            scaled_activation = (input / scales)
            out = F.linear(scaled_activation, quantized_weight)
            self.losses[i] += (unquantized_result - out).pow(2).mean().item()
    
    def calculate_qparams(self):
        ratio = torch.argmin(self.losses) * 1.0 / self.scale_options
        scales = self.average.pow(ratio).clamp(min=1e-4)
        scales = scales / (scales.max() * scales.min()).sqrt() 
        return scales.detach()

@dataclass(frozen=True)
class AWQLayoutType(LayoutType):
    equalization_scale: torch.Tensor
    
    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.equalization_scale
    
@register_layout_cls(AWQLayoutType)
class AWQ_AQTLayout(PlainAQTLayout):
    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        layout_type: LayoutType,
    ):
        assert isinstance(layout_type, AWQLayoutType)
        return cls(int_data, scale, zero_point, layout_type)
    