import types
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

import torchao
from torchao.core.config import AOBaseConfig
from torchao.dtypes import (
    CutlassSemiSparseLayout,
    Float8Layout,
    to_affine_quantized_floatx,
)
from torchao.float8.config import e4m3_dtype
from torchao.float8.inference import (
    Float8MMConfig,
    FP8Granularity,
    _check_hardware_support,
    _normalize_granularity,
)
from torchao.quantization.observer import get_block_size
from torchao.quantization.quant_api import (
    PerRow,
    _float8_cutlass_quant,
    _linear_extra_repr,
    to_linear_activation_quantized,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.utils import (
    is_MI300,
    is_sm_at_least_89,
)

import torch.nn.functional as F

from torchao.utils import TorchAOBaseTensor


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

    mm_config = Float8MMConfig(use_fast_accum=True)


@register_quantize_module_handler(ActivationSparseLinearConfig)
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
            W_aqt = _float8_cutlass_quant(weight, torch.float8_e4m3fn)
            W = W_aqt.tensor_impl.float8_data
            W_scale = W_aqt.tensor_impl.scale
            return cls(weight.shape, data=W, scale=W_scale, requires_grad=False)
        else:
            return cls(
                weight.shape,
                data=weight.data.t().contiguous().t(),
                scale=None,
                requires_grad=False,
            )

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


@implements([aten.copy_.default])
def _(func, types, args, kwargs):
    self = args[0]
    src = args[1]
    if not isinstance(src, ActivationSparseTensor):
        src_subclass = ActivationSparseTensor.from_dense(src)
        self.data.copy_(src_subclass.data)
        self.scale.copy_(src_subclass.scale)
    return




@implements(torch.nn.functional.linear)
def sparse_activation_linear(func, types, args, kwargs):
    x_orig, w, bias = args
    assert bias is None
    x = x_orig.view(-1, x_orig.size(-1))
    m, n = x.shape

    # # # if x input is the right shape, we use sparse matmul
    # x_padded = _pad_dense_input(x)
    # if (x.size(0) % 64) == 0:
    # if (x.size(0) == 64) or (x.size(0) == 128) or (x.size(0) ==256) or (x.size(0)==512):
    if False:
        X_scale = torch.empty(
            [x.shape[0], 1], dtype=torch.float32, device=x_orig.device
        )
        Xq_sparse, X_meta = torch.ops.torchao.sparse24_sm90_sparsify(
            x,
            "cutlass",
            "identity",
            "largest",
            dtype=torch.float8_e4m3fn,
            scale=X_scale,
        )

        out_sparse = torch.ops.torchao.sparse24_fp8_sm90_cutlass_gemm(
            Xq_sparse,
            X_meta,
            w.data.t(),
            a_scale=X_scale,
            b_scale=w.scale.t(),
        )
        # print(out_sparse.shape)
        out_sparse = out_sparse.reshape(*x_orig.shape[:-1], w.shape[0])
        return out_sparse
    else:
        w_dequantized = (w.data.to(torch.float32) * w.scale).to(torch.bfloat16)
        return torch.nn.functional.linear(x_orig, w_dequantized, bias)


from torchao.quantization.quant_api import (
    Float8Layout,
    _check_hardware_support,
    _fp8_mm_compat,
    to_affine_quantized_floatx,
)


@dataclass
class Float8DynamicSemiSparseActivationFloat8WeightConfig(AOBaseConfig):
    """
    Configuration for applying float8 dynamic symmetric quantization to both activations and weights of linear layers.

    Args:
        activation_dtype (torch.dtype): The target data type for activation quantization. Default is torch.float8_e4m3fn.
        weight_dtype (torch.dtype): The target data type for weight quantization. Default is torch.float8_e4m3fn.
        granularity:
            The granularity for quantization. Can be either a single granularity (applied to both
            activations and weights) or a tuple of two granularities (one for activations, one for weights).
            If None, defaults to PerTensor for both. Currently both quantizations need to be the same type. And
            only PerTensor and PerRow are supported.
        mm_config (Float8MMConfig): Configuration for the matrix multiplication. Default uses fast accumulation.
        set_inductor_config (bool): if True, adjusts `torchinductor` settings to recommended values.

    """

    activation_dtype: torch.dtype = e4m3_dtype
    weight_dtype: torch.dtype = e4m3_dtype
    granularity: Optional[Union[FP8Granularity, List[FP8Granularity]]] = None
    mm_config: Optional[Float8MMConfig] = None
    set_inductor_config: bool = True

    def __post_init__(self):
        if self.mm_config is None:
            self.mm_config = Float8MMConfig(use_fast_accum=True)

        activation_granularity, weight_granularity = _normalize_granularity(
            self.granularity
        )
        self.granularity = [activation_granularity, weight_granularity]


def _float8_dynamic_sparse_activation_float8_weight_quantize_tensor(weight, config):
    activation_dtype = config.activation_dtype
    weight_dtype = config.weight_dtype
    granularity = config.granularity
    mm_config = config.mm_config

    # Ensure works on device
    _check_hardware_support(granularity)
    activation_granularity, weight_granularity = granularity

    if not _fp8_mm_compat(weight):
        # TODO(future PR): this should really throw an exception instead of silently
        # not doing what the user asked
        return weight
    if isinstance(weight_granularity, PerRow):
        assert weight.dtype == torch.bfloat16, (
            "PerRow quantization only works for bfloat16 precision input weight"
        )
    block_size = get_block_size(weight.shape[-2:], weight_granularity)
    if weight.dim() == 3:
        block_size = tuple([1] + list(block_size))
    quantized_weight = to_affine_quantized_floatx(
        input_float=weight,
        block_size=block_size,
        target_dtype=weight_dtype,
        scale_dtype=torch.float32,
        _layout=Float8Layout(mm_config=mm_config),
    )

    # input_quant_func = torch.compile(_input_activation_quant_func_fp8_sparse, fullgraph=True)
    input_quant_func = _input_activation_quant_func_fp8_sparse
    input_quant_kwargs = {
        "activation_granularity": activation_granularity,
        "activation_dtype": activation_dtype,
    }


    quantized_weight = to_linear_activation_quantized(
        quantized_weight, input_quant_func, quant_kwargs=input_quant_kwargs
    )
    return quantized_weight


@register_quantize_module_handler(Float8DynamicSemiSparseActivationFloat8WeightConfig)
def _float8_dynamic_activation_sparse_float8_weight_transform(
    module: torch.nn.Module, config: Float8DynamicSemiSparseActivationFloat8WeightConfig
):
    assert is_sm_at_least_89() or is_MI300(), (
        "Float8 dynamic activation quantization is only supported on CUDA>=8.9 and MI300+"
    )
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    assert hasattr(module, "weight"), (
        "applying float8 dynamic activation quant requires module to have weight attribute"
        + f"but {module} does not have one"
    )
    quantized_weight = _float8_dynamic_sparse_activation_float8_weight_quantize_tensor(
        module.weight, config
    )
    module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module

from collections import Counter
from pprint import pprint

SEEN = Counter()
def _input_activation_quant_func_fp8_sparse(
    x: torch.Tensor,
    activation_granularity,
    activation_dtype: torch.dtype,
    scale: Optional[torch.Tensor] = None,
    zero_point: Optional[torch.Tensor] = None,
):
    """This function is used to quantize the input activation tensor for an aqt_float variant. If scale
    is not provided it will be dynamically calculate the scales otherwise it will use the provided scale.
    """
    # print(x.shape)
    # x_2d = x.view(-1, x.size(-1))

    assert zero_point is None, (
        "Zero point is not supported for dynamic FP8 quantization"
    )
    if isinstance(activation_granularity, PerRow):
        assert x.dtype == torch.bfloat16, (
            "PerRow quantization only works for bfloat16 precision input activation"
        )

    # x_2d = _pad_dense_input(x_2d)
    # if x.shape not in SEEN:
    #     SEEN[x.shape] += 1
    #     pprint(SEEN)
    # else:
    #     SEEN[x.shape] += 1


    # if (
    #     (x.size(0) == 64) or
    #     (x.size(0) == 128) or
    #     (x.size(0) == 192) or
    #     (x.size(0) == 256) or
    #     (x.size(0) == 320) or
    #     (x.size(0) == 384) or
    #     (x.size(0) == 448) or
    #     (x.size(0) == 512) 
    # ):
        # print(x.shape)
    # if x.shape[0] % 64 == 0:
    # else:
    #     layout=Float8Layout(mm_config=None)
    layout=CutlassSemiSparseLayout()

    block_size = get_block_size(x.shape, activation_granularity)
    activation = to_affine_quantized_floatx(
        input_float=x,
        block_size=block_size,
        target_dtype=activation_dtype,
        scale_dtype=torch.float32,
        _layout=layout,
    )
    return activation
