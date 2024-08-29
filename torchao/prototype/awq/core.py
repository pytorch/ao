import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from tqdm import tqdm
from copy import deepcopy
from torchao.dtypes.utils import (
    LayoutType,
)
from torchao.dtypes.affine_quantized_tensor import (
    PlainAQTLayout,
    TensorCoreTiledAQTLayout, 
    register_layout_cls, 
    to_affine_quantized,
    AWQ_INT4_LayoutType, 
    AWQLayoutType
    
) 
from torchao.utils import find_multiple

from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
)
from torchao.quantization.quant_api import int4_weight_only
from torchao.dtypes.uintx.Uintx import to_uintx
from torchao.quantization.observer import (
    PerAxis,
    AffineQuantizedObserverBase,
)
import pdb
def _awq_quant(w, n_bit=8, q_group_size=-1, get_scale_zp=False):
    # pdb.set_trace()
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
    # perform dequant
    if not get_scale_zp:
        w = (w - zeros) * scales
        w = w.reshape(org_w_shape)
    assert torch.isnan(w).sum() == 0

    

    if get_scale_zp:
        return w, scales, zeros
    else:
        return w

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
        # print(input.shape)
        average = input.abs().view(-1,input.shape[-1]).mean(0)
        
        best_loss = float('inf')
        best_ratio = -1
        scaleopts = []
        ws = []
        for i in range(self.scale_options):
            ratio = i *1 / self.scale_options
            scales = average.pow(ratio)
            scales = scales / (scales.max() * scales.min()).sqrt()
            layout = AWQLayoutType(scales) if self.zero_point_domain == ZeroPointDomain.INT else AWQ_INT4_LayoutType(scales)
            w = to_affine_quantized(
                self.weight.data,
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
                layout_type = layout
            )
            # w = deepcopy(self.weight) * scales
            # w = _awq_quant(w, q_group_size=128, n_bit=4) / scales
            # ws.append(w.mean().item())
            q_out = F.linear(input/scales, w, self.bias)
            scaleopts.append(q_out.mean().item())
            loss = (output - q_out).pow(2).mean().item()
            if loss < best_loss:
                self.scales = scales
                best_ratio = ratio
                best_loss = loss
        # print(f"x: {input.mean().item(): .03f} w_: {torch.tensor(ws).sum().item()} o: {torch.tensor(scaleopts).sum().item(): .05f} ratio: {self.best_ratio}")

    def calculate_qparams(self):
        return self.scales.detach()
