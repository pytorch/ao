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
        self.output_sum = torch.zeros(weight_shape[0], dtype=torch.float32)

    def forward(self, input: torch.Tensor, output: torch.Tensor):
        self.average = self.average * self.counter / (self.counter + input.shape[0])  + input.abs().sum(dim=0) / (self.counter + input.shape[0])
        self.counter += 1
        self.output_sum += output.sum(dim=0)

    
    def calculate_qparams(self, weight, calibration_data):
        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []
        x_max = self.average 
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
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
            scaled_activation = (calibration_data).to(torch.bfloat16)
            out = F.linear(scaled_activation, quantized_weight).sum(dim=0)

            loss = (
                (self.output_sum - out).float().pow(2).mean().item()
            )  # float prevents overflow
            # print(f"ratio: {ratio}, loss: {loss}")
            history.append(loss)
            
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
        print(f"best scale: {best_scales}, best error: {best_error}")
                
        if best_ratio == -1:
            print(history)
            raise Exception
        # print(best_ratio)
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

@dataclass(frozen=True)
class AWQLayoutType(LayoutType):
    scales: torch.Tensor
    
    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        return input / (self.scales.view(1, -1))
    
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

    
    

    
    
    
    