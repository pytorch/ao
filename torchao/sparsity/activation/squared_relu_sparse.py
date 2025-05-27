
import types
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.sparse import to_sparse_semi_structured

from torchao.core.config import AOBaseConfig
from torchao.float8.inference import Float8MMConfig
from torchao.prototype.sparsity.sparsifier.weight_norm_sparsifier import (
    WeightNormSparsifier,
)
from torchao.quantization.quant_api import (
    _is_linear,
    _linear_extra_repr,
    _replace_with_custom_fn_if_matches_filter,
)
from torchao.quantization.transform_module import (
    _QUANTIZE_CONFIG_HANDLER,
    register_quantize_module_handler,
)
from torchao.sparsity.blocksparse import BlockSparseTensor
from dataclasses import dataclass

import torch
from torch import nn

from torchao.core.config import AOBaseConfig
from torchao.ops import (
    rowwise_scaled_linear_sparse_cutlass_f8f8,
)
from torchao.quantization.quant_api import (
    _float8_cutlass_quant,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)

from torchao.kernel.splitk_sparse_gemv import splitk_sparse_gemv
from torch.utils._python_dispatch import return_and_correct_aliasing
def _to_fp8_rowwise(x: torch.Tensor, dtype):
    max_v = torch.finfo(dtype).max
    x_scale = (x.abs().max(1, keepdim=True)[0].clip(1e-12) / max_v).float()
    x = (x.float() / x_scale).clamp(min=-max_v, max=max_v).to(dtype)
    return x, x_scale


from torchao.utils import TorchAOBaseTensor
from torchao.quantization import LinearActivationQuantizedTensor

from torchao.float8.float8_utils import tensor_to_scale
from torchao.float8.float8_scaling_utils import (
    get_maybe_axiswise_dim,
    hp_tensor_to_float8_dynamic,
    hp_tensor_and_scale_to_float8,
)
from torchao.float8.config import CastConfig, ScalingGranularity

@dataclass
class ActivationSparseLinearConfig(AOBaseConfig):
    """
    Adds in acceleration for activation sparsity to linear layers for decode. 

    Args:
        `activation_dtype`: data type for quantized activation tensor.
        `weight_dtype`: data type for quantized weight tensor.
    """

    activation_dtype: torch.dtype = torch.float8_e4m3fn
    weight_dtype: torch.dtype = torch.float8_e4m3fn

@register_quantize_module_handler(
    ActivationSparseLinearConfig)
def _(
    module: torch.nn.Module,
    config: ActivationSparseLinearConfig,
):
    new_weight = ActivationSparseTensor.from_dense(module.weight.data)
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


class ActivationSparseTensor(TorchAOBaseTensor):
    data: Optional[torch.Tensor]
    scale: Optional[torch.Tensor]

    __slots__ = ["data", "scale"]

    @staticmethod
    def __new__(  # noqa: PYI034
        cls,
        shape: torch.Size,
        data: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        requires_grad: bool = False,
    ):
        assert data is not None
        kwargs = {
            "device": data.device,
            "dtype": data.dtype,
            "layout": data.layout,
            "requires_grad": requires_grad,
        }
        tensor = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]
        tensor.data = data
        tensor.scale = scale
        return tensor

    def __repr__(self) -> str:  # type: ignore[override]
        assert hasattr(self, "shape")
        return f"{self.__class__.__name__}(shape={self.shape})"

    def __tensor_flatten__(self):
        inner_tensors = list(
            filter(lambda x: getattr(self, x) is not None, self.__slots__)
        )
        tensor_meta = (self.shape, self.requires_grad)
        return inner_tensors, tensor_meta

    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors,
        tensor_meta,
        outer_size,
        outer_stride,
    ) -> torch.Tensor:
        shape, requires_grad = tensor_meta
        return cls(
            shape=shape,
            data=inner_tensors.get("data", None),
            scale=inner_tensors.get("scale", None),
            requires_grad=requires_grad,
        )

    @classmethod
    def from_dense(cls, weight, use_fp8=True):
        if use_fp8:
            # weight, scale = _to_fp8_rowwise(weight, torch.float8_e4m3fn)
            # scale = None
            scale = tensor_to_scale(
                weight,
                torch.float8_e4m3fn,
                reduce_amax=False,
                device_mesh=None,
                scaling_granularity=ScalingGranularity.TENSORWISE,
                axiswise_dim=-1,
                round_scales_to_power_of_2=False,
            )
            x2_lp = hp_tensor_and_scale_to_float8(weight, scale, torch.float8_e4m3fn)
            return cls(weight.shape,
                    data=x2_lp,
                    scale=None,
                    requires_grad=False)
        else:
            return cls(weight.shape,
                    data=weight.data.t().contiguous().t(),
                    scale=None,
                    requires_grad=False)

    def apply_fn_to_shard(self, func):
        return ActivationSparseTensor(
            shape=self.shape,
            data=func(self.data),
            scale=func(self.scale),
            requires_grad=self.requires_grad,
        )

# Subclass op dispatch registration
implements = ActivationSparseTensor.implements
aten = torch.ops.aten


@implements(
    [
        aten.detach.default,
        aten.slice.Tensor,
    ]
)
def _(func, types, args, kwargs):
    new_data = func(args[0].data, *args[1:], **kwargs)
    if args[0].scale is None:
        new_scale = None
    else:
        new_scale = func(args[0].scale, *args[1:], **kwargs)
    return ActivationSparseTensor(
        new_data.shape,
        data=new_data,
        scale=new_scale,
        requires_grad=False,
    )

@implements(
    [aten.copy_.default]
)
def _(func, types, args, kwargs):
    self = args[0]
    src = args[1]
    if not isinstance(src, ActivationSparseTensor):
        src_subclass = ActivationSparseTensor.from_dense(src)

    self.data.copy_(src.data)
    # slef.scale.copy_(src.scale)
    if self.scale is None:
        self.scale = None
    else:
        self.scale.copy_(src.scale)
    return

@implements(torch.nn.functional.linear)
def sparse_activation_linear(func, types, args, kwargs):
    x_orig, w, bias = args
    assert bias is None
    x = x_orig.view(-1, x_orig.size(-1))
    # M = w.shape[0]
    # K = w.shape[1]

    if x.shape[0] % 64 != 0:
        # w_dequantized = (w.data.to(torch.bfloat16))
        # x_relu = torch.square(torch.nn.functional.relu(x))
        return torch.nn.functional.linear(x_orig, w.data.to_original_precision().to(torch.bfloat16), bias)
        # res = torch.ops.torchao.splitk_sparse_gemv(x_relu,
        #                                             w.data)
        # return res.view(*x_orig.shape[:-1], w.shape[0])
    else:
        # X_scale = torch.empty([x.shape[0], 1], dtype=torch.float32, device=x.device)
        # Xq_sparse, X_meta = torch.ops.torchao.sparse24_sm90_sparsify(
        #     x,
        #     "cutlass",
        #     "identity",
        #     "largest",
        #     dtype=torch.float8_e4m3fn,
        #     scale=X_scale,
        # )
        x_ast = ActivationSparseTensor.from_dense(x_orig) # .data.to_original_precision().to(torch.bfloat16)
        # x_orig = 
        # return torch.nn.functional.linear(x_ast, w.data.to_original_precision().to(torch.bfloat16), bias)
        breakpoint()
        return torch._scaled_mm(x_ast.data._data, w.data._data.T, scale_a=x_ast.data._scale, scale_b=w.data._scale.T, out_dtype=torch.bfloat16)


        out_sparse = torch.ops.torchao.sparse24_fp8_sm90_cutlass_gemm(
            Xq_sparse, X_meta, w.data._data.T, a_scale=X_scale, b_scale=w.data._scale.T,
        )
        out_sparse = out_sparse.view(*x_orig.shape[:-1], w.shape[0])

        return out_sparse

        # For normal linear
        # x_orig_relu = torch.square(torch.nn.functional.relu(x_orig))
        # return torch.nn.functional.linear(x_orig_relu, w.data, bias)
