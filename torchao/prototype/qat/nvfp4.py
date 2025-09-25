from dataclasses import dataclass
from typing import Optional

import torch

from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    per_tensor_amax_to_scale,
)
from torchao.quantization.qat import (
    FakeQuantizeConfigBase,
    FakeQuantizerBase,
)


class _NVFP4FakeQuantize(torch.autograd.Function):
    """
    Fake quantize a high precision tensor to nvfp4 and back with backward STE.
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, per_tensor_scale: Optional[torch.Tensor]
    ) -> torch.Tensor:
        q = NVFP4Tensor.to_nvfp4(x, per_tensor_scale=per_tensor_scale)
        dq = q.to_dtype(x.dtype)
        return dq

    @staticmethod
    def backward(ctx, gy: torch.Tensor) -> torch.Tensor:
        return gy, None


@dataclass
class NVFP4FakeQuantizeConfig(FakeQuantizeConfigBase):
    """
    Config for fake quantizing weights or activations to NVIDIA's NVFP4 format
    according to https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/.

    Fake quantization numerics follow `NVFP4Tensor` closely: https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/nvfp4_tensor.py.

    Args:
        use_per_tensor_scale (bool): Whether to use two-level per-tensor fp32 scaling
            after the initial fp8 (e4m3) block-wise scaling (default True)
    """

    use_per_tensor_scale: bool = True


class NVFP4FakeQuantizer(FakeQuantizerBase):
    """
    (Prototype) Generic module for applying NVFP4 fake quantization to a tensor, as specified in the config.
    """

    def __init__(self, config: NVFP4FakeQuantizeConfig):
        super().__init__()
        torch._C._log_api_usage_once("torchao.quantization.qat.NVFP4FakeQuantizer")
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])
        if self.config.use_per_tensor_scale:
            tensor_amax = torch.max(torch.abs(x))
            per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)
        else:
            per_tensor_scale = None
        fq = _NVFP4FakeQuantize.apply(x, per_tensor_scale)
        assert fq.dtype == x.dtype
        return fq.view(original_shape)
