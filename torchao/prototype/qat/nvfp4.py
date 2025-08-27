from dataclasses import dataclass

import torch

from torchao.prototype.mx_formats.nvfp4_tensor import (
    _nvfp4_quantize,
    per_tensor_amax_to_scale,
)
from torchao.quantization.qat import (
    FakeQuantizeConfigBase,
    FakeQuantizerBase,
)


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
        block_size = 16
        original_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])
        if self.config.use_per_tensor_scale:
            tensor_amax = torch.max(torch.abs(x))
            per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)
        else:
            per_tensor_scale = None

        # quantize
        scale, q = _nvfp4_quantize(
            x,
            block_size=block_size,
            per_tensor_scale=per_tensor_scale,
            skip_dtype_cast_and_packing=True,
        )
        if self.config.use_per_tensor_scale:
            scale = scale * per_tensor_scale
        assert q.dtype == x.dtype
        assert scale.dtype == torch.float32

        # dequantize
        M, K = q.shape[0], q.shape[1]
        q = q.view(M, K // block_size, block_size)
        scale = scale.view(M, K // block_size, 1)
        dq = q * scale
        return dq.view(original_shape).to(x.dtype)
