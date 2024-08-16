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
        weight_shape: Tuple[int, int],
        mapping_type: MappingType,
        target_dtype: torch.dtype,
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
        self.average = torch.zeros(weight_shape[-1], dtype=torch.float32)
        self.counter = 0
        self.calibration_data = []

    def forward(self, input: torch.Tensor):
        self.average = self.average * self.counter / (self.counter + input.shape[0])  + input.abs().sum(dim=0) / (self.counter + input.shape[0])
        self.counter += 1
        self.calibration_data.append(input)

    
    def calculate_qparams(self, orig_weight):
        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []
        calibration_data = torch.cat(self.calibration_data, dim=0)
        unquantized_result = F.linear(calibration_data, orig_weight).sum(dim=0)
        x_max = self.average 
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            weight = deepcopy(orig_weight)
            scales = x_max.pow(ratio)
            scales = scales / (scales.max() * scales.min()).sqrt()
            weight.mul_(scales)
            quantized_weight = to_affine_quantized(
                weight,
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
            scaled_activation = (calibration_data)
            out = F.linear(scaled_activation, quantized_weight).sum(dim=0)
            
            loss = (
                (unquantized_result - out).pow(2).mean().item()
            )  # float prevents overflow
            # print(f"ratio: {ratio}, loss: {loss}")
            # print(ratio, loss)
            history.append(loss)
            
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
        # print(f"best error: {best_error}")
                
        
        # print(best_ratio)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

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

    
    

    
    
    
    