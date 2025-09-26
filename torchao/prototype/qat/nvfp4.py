from dataclasses import dataclass
from typing import Optional

import torch

from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    _addmm_nvfp4_dispatch,
    per_tensor_amax_to_scale,
)
from torchao.quantization.qat import FakeQuantizeConfigBase


@dataclass
class NVFP4FakeQuantizeConfig(FakeQuantizeConfigBase):
    """
    Config for fake quantizing weights or activations to NVIDIA's NVFP4 format
    according to https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/.

    Fake quantization numerics follow `NVFP4Tensor` closely: https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/nvfp4_tensor.py.

    Args:
        use_per_tensor_scale (bool): Whether to use two-level per-tensor fp32 scaling
            after the initial fp8 (e4m3) block-wise scaling (default True)
        use_swizzled_scales (bool): Whether scales are stored in swizzled (blocked) format
        use_triton_kernel (bool): Whether to use triton kernels during fake quantization
    """

    use_per_tensor_scale: bool = True
    use_swizzled_scales: bool = False
    use_triton_kernel: bool = False


class _NVFP4FakeQuantizedLinearForward(torch.autograd.Function):
    """
    Autograd function for NVFP4 fake quantization + addmm.
    """

    @staticmethod
    def forward(
        ctx,
        _input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation_config: NVFP4FakeQuantizeConfig,
        weight_config: NVFP4FakeQuantizeConfig,
    ) -> torch.Tensor:
        # quantize input activations
        if activation_config.use_per_tensor_scale:
            tensor_amax = torch.max(torch.abs(_input))
            per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)
        else:
            per_tensor_scale = None
        _input = NVFP4Tensor.to_nvfp4(
            _input,
            per_tensor_scale=per_tensor_scale,
            is_swizzled_scales=activation_config.use_swizzled_scales,
            use_triton_kernel=activation_config.use_triton_kernel,
        )

        # quantize weights
        if weight_config.use_per_tensor_scale:
            tensor_amax = torch.max(torch.abs(weight))
            per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)
        else:
            per_tensor_scale = None
        weight = NVFP4Tensor.to_nvfp4(
            weight,
            per_tensor_scale=per_tensor_scale,
            is_swizzled_scales=weight_config.use_swizzled_scales,
            use_triton_kernel=False,
        )

        # Follow `NVFP4InferenceConfig`, always use traditional construction
        # for weights and set `use_triton_kernel` afterwards
        weight.use_triton_kernel = weight_config.use_triton_kernel

        ctx.save_for_backward(_input, weight)

        return _addmm_nvfp4_dispatch(
            _input,
            weight.t(),
            None,  # aten_op, not used
            bias,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        _input, weight = ctx.saved_tensors
        assert isinstance(_input, NVFP4Tensor)
        assert isinstance(weight, NVFP4Tensor)
        _input = _input.to_dtype(_input._orig_dtype)
        weight = weight.to_dtype(weight._orig_dtype)
        grad_input = torch.mm(grad_output, weight)
        grad_weight = torch.mm(grad_output.t(), _input)
        return grad_input, grad_weight, None, None, None


class NVFP4FakeQuantizedLinear(torch.nn.Linear):
    """
    Linear module for fake quantized NVFP4 weights and/or activations.

    The forward pass follows quantization and addmm numerics in `NVFP4Tensor` exactly.

    Example usage::

        from torchao.quantization import quantize_
        from torchao.prototype.mx_formats import NVFP4InferenceConfig

        base_config = NVFP4InferenceConfig()
        quantize_(model, QATConfig(base_config, step="prepare"))
        # Model contains `NVFP4FakeQuantizedLinear` now

        train_loop(model)
        quantize_(model, QATConfig(base_config, step="convert"))
        # Model contains `nn.Linear` with `NVFP4Tensor` weights now
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_config: Optional[NVFP4FakeQuantizeConfig] = None,
        weight_config: Optional[NVFP4FakeQuantizeConfig] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            *args,
            **kwargs,
        )
        if weight_config is None:
            raise ValueError("Must specify `weight_config`")
        if activation_config is None:
            raise ValueError("Weight only NVFP4 QAT not supported yet")
        self.activation_config = activation_config
        self.weight_config = weight_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            batch_size = x.shape[0]
            x = x.view(-1, x.shape[-1])
        else:
            batch_size = None
        fq = _NVFP4FakeQuantizedLinearForward.apply(
            x, self.weight, self.bias, self.activation_config, self.weight_config
        )
        assert fq.dtype == x.dtype
        if batch_size is not None:
            return fq.view(batch_size, -1, fq.shape[-1])
        else:
            return fq

    @classmethod
    def from_linear(
        cls,
        mod: torch.nn.Linear,
        activation_config: Optional[NVFP4FakeQuantizeConfig] = None,
        weight_config: Optional[NVFP4FakeQuantizeConfig] = None,
    ):
        new_linear = NVFP4FakeQuantizedLinear(
            mod.in_features,
            mod.out_features,
            mod.bias is not None,
            activation_config=activation_config,
            weight_config=weight_config,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
        )
        # In distributed training, the model may be instantiated
        # on the meta device, in which case there is no need to
        # copy the weights, and doing so will result in an error
        if mod.weight.device != torch.device("meta"):
            new_linear.weight = mod.weight
            new_linear.bias = mod.bias
        return new_linear
