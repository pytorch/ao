import functools
import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple, Type

import torch
import torch.utils._pytree as pytree
from torch import nn
from torch._prims_common import suggest_memory_format
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy

from torchao.prototype.moe_training.utils import (
    unwrap_weight,
)
from torchao.quantization.granularity import PerRow, Granularity
from torchao.quantization.qat.fake_quantize_config import (
    FakeQuantizeConfigBase,
    Float8FakeQuantizeConfig,
)
from torchao.quantization.quant_primitives import (
    _choose_scale_float8,
    _dequantize_affine_float8,
    _quantize_affine_float8,
)
from torchao.quantization.utils import get_block_size
from torchao.utils import TorchAOBaseTensor

"""
Registry mapping a :class:`FakeQuantizeConfigBase` type to a handler that
wraps or unwraps a single ``nn.Parameter`` for MoE QAT. The handler receives
the parameter and the :class:`MoEQATConfig` and returns the transformed parameter.

This is the parameter-level analogue of ``_QUANTIZE_CONFIG_HANDLER`` in
:mod:`torchao.quantization.transform_module`.
"""
_MoE_QAT_PARAMETER_QUANTIZE_CONFIG_HANDLER: Dict[
    Type[FakeQuantizeConfigBase],
    Callable[[nn.Module, str, nn.Parameter, Tuple[Any, ...]], nn.Parameter],
] = {}


def register_MoE_QAT_quantize_parameter_handler(
    config_type: Type[FakeQuantizeConfigBase],
):
    """
    Decorator to register a handler for a specific :class:`FakeQuantizeConfigBase`
    type. The handler takes an ``nn.Parameter`` and a :class:`MoEQATConfig` and
    returns the transformed ``nn.Parameter``.

    Example usage::

        @register_MoE_QAT_quantize_parameter_handler(Float8FakeQuantizeConfig)
        def _float8_parameter_handler(
            param: nn.Parameter, config: MoEQATConfig
        ) -> nn.Parameter:
            ...
    """

    @functools.wraps(config_type)
    def decorator(func):
        _MoE_QAT_PARAMETER_QUANTIZE_CONFIG_HANDLER[config_type] = func
        return func

    return decorator


"""
ATen ops that should preserve the wrapper subclass identity. When any of these
ops is called on a FakeQuantizedWeightWrapperBaseTensor, the output is re-wrapped
in the same subclass with the operated-on ``_data``.

Design: deferred fake quantization. Slicing or indexing a wrapped weight returns
a new wrapper with the sliced ``_data`` — no fake quantization is applied at
this point. Fake quantization is deferred until computation time, when
``__torch_function__`` intercepts computation ops (``torch.mm``, ``torch.bmm``,
``torch._grouped_mm``, etc.) and applies fake quantization just before the op.
This avoids double fake quantization.

Indexing patterns on a 3D weight tensor and their ATen ops (all preserved):

  w[0]                  → aten.select.int
  w[0:5]                → aten.slice.Tensor
  w[ids]                → aten.index.Tensor / aten._unsafe_index.Tensor
  w[[0,2,3]]            → aten.index.Tensor / aten._unsafe_index.Tensor
  w[mask]               → aten.index.Tensor / aten._unsafe_index.Tensor
  w[ids, :, :]          → aten.index.Tensor / aten._unsafe_index.Tensor
  w[ids, [0,1]]         → aten.index.Tensor / aten._unsafe_index.Tensor
  w[...]                → aten.slice.Tensor
  w[None]               → aten.unsqueeze.default

Dimension-manipulation ops follow the same deferred design — they return a new
wrapper with the reshaped ``_data``, no fake quantization applied:
  permute, squeeze, view, as_strided, transpose, t, split
"""
_ops_to_preserve_subclass = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.select.int,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.index.Tensor,
    torch.ops.aten._unsafe_index.Tensor,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,  # for *.to(dtype)
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.default,
    # required for TP - scatter_ is used to distribute weights
    torch.ops.c10d.scatter_.default,
}


class FakeQuantizedWeightWrapperBaseTensor(TorchAOBaseTensor):
    """
    Base class for wrapper tensor subclasses that apply fake quantization to
    MoE expert weights during QAT.

    Wraps a 3D weight tensor ``_data`` of shape ``[num_experts, in_features, out_features]``
    and a :class:`~torchao.quantization.qat.fake_quantize_config.FakeQuantizeConfigBase`
    that specifies the fake quantization recipe.

    Supports FSDP2 via :meth:`fsdp_pre_all_gather` and :meth:`fsdp_post_all_gather`,
    which handle mixed-precision casting and wrapper reconstruction after all-gather.

    Subclasses must override :meth:`__torch_function__` to intercept computation
    ops and apply their precision-specific fake quantization.

    Not intended to be used directly.
    """

    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        activation_config: Optional[FakeQuantizeConfigBase] = None,
        weight_config: Optional[FakeQuantizeConfigBase] = None,
    ):
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned() if not isinstance(tensor, DTensor) else False,
            requires_grad=tensor.requires_grad,
        )
        return self

    def __init__(
        self,
        tensor: torch.Tensor,
        activation_config: Optional[FakeQuantizeConfigBase] = None,
        weight_config: Optional[FakeQuantizeConfigBase] = None,
    ):
        self._data = tensor
        if weight_config is None:
            raise ValueError(
                f"Must specify `weight_config` in {type(self).__name__}."
            )

        self.activation_config = activation_config
        self.weight_config = weight_config

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs=None):
        raise NotImplementedError(
            f"{cls.__name__} is not intended to be used directly, please override `__torch_function__` in a tensor subclass for your intended derived dtype."
        )

    @classmethod
    def _fake_quantize(cls, weight: torch.Tensor, config: FakeQuantizeConfigBase) -> torch.Tensor:
        raise NotImplementedError(
            f"{cls.__name__} is not intended to be used directly, please override `_fake_quantize` in a tensor subclass for your intended derived dtype."
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # In MoE training (TrainingWeightWrapperBaseTensor), all operands are
        # assumed to be wrapped and share the same quantization config. In QAT,
        # operands may not all be fake-quantized — e.g., activations can be
        # plain tensors with no config. So check to ensure at least one operand
        # carries a config.
        weight_config = None
        activation_config = None

        # All wrapped tensors in a dispatch call come from the same prepare step,
        # so each config must be uniform: either absent on all wrappers or identical
        # across all wrappers.
        def unwrap(t: FakeQuantizedWeightWrapperBaseTensor):
            nonlocal weight_config, activation_config
            if weight_config is None:
                weight_config = t.weight_config
            else:
                assert t.weight_config == weight_config, (
                    "All FakeQuantizedWeightWrapperBaseTensor instances must have the same weight_config"
                )
            if activation_config is None:
                activation_config = t.activation_config
            else:
                assert t.activation_config == activation_config, (
                    "All FakeQuantizedWeightWrapperBaseTensor instances must have the same activation_config"
                )
            return t._data

        args_unwrapped, kwargs_unwrapped = pytree.tree_map_only(
            FakeQuantizedWeightWrapperBaseTensor, unwrap, (args, kwargs or {})
        )

        assert weight_config is not None, (
            f"__torch_dispatch__ called on {func.__name__} without any FakeQuantizedWeightWrapperBaseTensor arguments"
        )

        # The treatment below for "detach" is different from that of TrainingWeightWrapperBaseTensor.
        # To align with the semantics of "detach" and avoid the "dual-nature" problem of a wrapper, we
        # choose to also detach _data. The config is shared since in the "detach" of torch.nn.Tensor,
        # most of metadata is also shared except the metadata related to autograd.
        #
        # TODO: Configs are assumed immutable. If configs gain trainable parameters,
        #       a newly-created config should be used instead of sharing the reference.
        #       The actual logic depends on the design of these quantization parameters in the future.
        if func == torch.ops.aten.detach.default:
            return cls(
                args_unwrapped[0].detach(),
                activation_config=activation_config,
                weight_config=weight_config,
            )

        # Perform op
        out = func(*args_unwrapped, **kwargs_unwrapped)

        # Return regular tensors for ops that don't preserve subclass
        if func not in _ops_to_preserve_subclass:
            return out

        # Return the original wrapper to maintain in-place semantics.
        # Unlike copy_ where the wrapper is both input and output, scatter_
        # writes into pre-allocated output buffers that did not exist as
        # wrappers before the call — so scatter_ falls through to the generic
        # re-wrap path below to create new wrappers from the result tensors.
        if func == torch.ops.aten.copy_.default:
            return args[0]

        # Wrap outputs back into the same subclass for the remaining preserved ops.
        # Configs are captured during unwrapping (above), which handles both single-input
        # ops (select, slice, view, etc.) and multi-input ops (scatter_) correctly.
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: cls(
                x,
                activation_config=activation_config,
                weight_config=weight_config,
            ),
            out,
        )

    def to_tensor(self) -> torch.Tensor:
        """Return the underlying raw tensor, unwrapping the subclass."""
        return self._data

    def __repr__(self):
        return f"{type(self).__name__}(data={self._data}, activation_config={self.activation_config}, weight_config={self.weight_config})"

    def __tensor_flatten__(self):
        metadata = {
            "activation_config": self.activation_config,
            "weight_config": self.weight_config,
        }
        return ["_data"], metadata

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        return cls(
            tensor_data_dict["_data"],
            activation_config=tensor_attributes["activation_config"],
            weight_config=tensor_attributes["weight_config"],
        )

    # FSDP hooks based on torchao/prototype/moe_training/tensor.py:156
    def fsdp_pre_all_gather(
        self,
        mesh: DeviceMesh,
        outer_size: torch.Size,
        outer_stride: Tuple[int, ...],
        module: nn.Module,
        mp_policy: MixedPrecisionPolicy,
    ):
        # Cast to mixed precision dtype prior to all-gather
        all_gather_inputs = (self._data.to(mp_policy.param_dtype),)
        all_gather_metadata = ()
        return all_gather_inputs, all_gather_metadata

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        (data,) = all_gather_outputs

        # For training step 0, out=None, create a new wrapper.
        if out is None:
            output = type(self)(
                data,
                activation_config=self.activation_config,
                weight_config=self.weight_config,
            )
            inner_tensors = (data,)
            return output, inner_tensors
        else:
            # For training step 1+, out=unsharded param. FSDP2 creates a shallow copy
            # of the wrapper; we restore configs from self. out may be a bare subclass
            # or wrapped in DTensor.
            if isinstance(out, FakeQuantizedWeightWrapperBaseTensor):
                out_data = out._data
                out.activation_config = self.activation_config
                out.weight_config = self.weight_config
            elif isinstance(out, DTensor) and isinstance(
                out._local_tensor, FakeQuantizedWeightWrapperBaseTensor
            ):
                out_data = out._local_tensor._data
                out._local_tensor.activation_config = self.activation_config
                out._local_tensor.weight_config = self.weight_config
            else:
                raise RuntimeError(
                    f"expected out to be {type(self).__name__} or DTensor with local_tensor={type(self).__name__}, but got {type(out)}"
                )

            # If `data` (all gather outputs) is already in the mixed precision policy param_dtype,
            # verify it has underlying storage as `out` (pre-allocated unsharded param),
            # and then we can just return directly.
            if data.dtype == param_dtype:
                assert (
                    data.untyped_storage().data_ptr()
                    == out_data.untyped_storage().data_ptr()
                )
            else:
                # Otherwise, verify that `out` (pre-allocated unsharded param) has the
                # mixed precision policy param_dtype, then copy `data` to `out`.
                assert out_data.dtype == param_dtype, f"{out_data.dtype} {param_dtype}"
                out_data.copy_(data)

            return


"""
Functions intercepted by __torch_function__ for fake quantization:

func                  args                  weight pos  keyword-only        shapes
──────────────────────────────────────────────────────────────────────────────────────────
torch._grouped_mm     (A, B)                args[1]     offs, out           A: [S,K], B: [E,K,N] or [E,N,K], offs: [E] int32
torch.bmm             (A, B)                args[1]     out                 A: [S,1,K], B: [S,K,N]
torch.mm              (A, B)                args[1]     out                 A: [M,K], B: [K,N]
torch.matmul          (A, B)                args[1]     —                   any shapes compatible with matmul broadcasting
F.linear              (A, B [, bias])       args[1]     —                   A: [*,K], B: [N,K], bias: [N]
torch.addmm           (bias, A, B)          args[2]     beta, alpha, out    bias: [N] or [M,N], A: [M,K], B: [K,N]
"""
_func_to_prepend_fake_quantization = {
    torch._grouped_mm,
    torch.addmm,
    torch.bmm,
    torch.matmul,
    torch.mm,
    torch.nn.functional.linear,
}

class Float8FakeQuantizedWeightWrapperTensor(FakeQuantizedWeightWrapperBaseTensor):
    """
    Applies FP8 row-wise fake quantization during MoE QAT.

    Intercepts computation ops via :meth:`__torch_function__`, applies per-row
    FP8 fake quantization (quantize → dequant in high precision with STE
    gradient) to the weights and optionally the activations, and delegates to
    the standard op.

    Both ``weight_config`` and ``activation_config`` (if set) must be
    :class:`~torchao.quantization.qat.fake_quantize_config.Float8FakeQuantizeConfig`.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        activation_config: Optional[FakeQuantizeConfigBase] = None,
        weight_config: Optional[FakeQuantizeConfigBase] = None,
    ):
        if activation_config is not None and not isinstance(activation_config, Float8FakeQuantizeConfig):
            raise ValueError(
                f"Only `Float8FakeQuantizeConfig` is supported for `activation_config` in {type(self).__name__}."
            )
        if not isinstance(weight_config, Float8FakeQuantizeConfig):
            raise ValueError(
                f"Only `Float8FakeQuantizeConfig` is supported for `weight_config` in {type(self).__name__}."
            )
        elif weight_config.granularity != PerRow(dim=-1):
            raise ValueError(
                f"Only the row-wise granularity is supported."
            )
        super().__init__(tensor, activation_config=activation_config, weight_config=weight_config)

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs=None):
        """
        Because we defer fake-quantization in the case of slicing, indexing, transposing,
        and permuting the wrapper tensor, the following fake-quantization is fragile when
        complicated transpositions and permutations are applied before real computations.
        After a general sequence of transpositions and permutations, it is NOT guaranteed
        that the fake-quantization is still carried out along the desired dimension.

        1. During prepare, the default or user-defined params_filter_fn may also wrap bias
           parameters (1D or 2D) that appear in F.linear/linear/addmm. In either case, the
           fake-quantization of these wrapped biases is always bypassed here: we only unpack
           the weight position for fake quantization. The bias passes through other positions
           and is unwrapped transparently by __torch_dispatch__ at add time.

        2. For torch.bmm, the wrapped weight may be at args[1] (is_transposed=True) or
           args[0] (is_transposed=False, as in HF's _batched_linear). We detect the
           position at runtime and correctly select the contracted dimension.
        """

        if kwargs is None:
            kwargs = {}
        
        if func in (torch._grouped_mm, torch.matmul, torch.mm):
            # weight at args[1], contracted dim=-2
            _need_fake_quantization = True
            a_pos, b_pos = 0, 1
            granularity=PerRow(dim=-2)
        elif func is torch.nn.functional.linear:
            # weight at args[1], contracted dim=-1
            _need_fake_quantization = True
            a_pos, b_pos = 0, 1
            granularity=PerRow(dim=-1)
        elif func is torch.bmm:
            _need_fake_quantization = True
            if isinstance(args[1], cls):
                # weight at args[1], shape [B, K, N], contracted dim=-2
                a_pos, b_pos = 0, 1
                granularity = PerRow(dim=-2)
            else:
                # weight at args[0], shape [B, N, K], contracted dim=-1
                a_pos, b_pos = 1, 0
                granularity = PerRow(dim=-1)
        elif func is torch.addmm:
            # weight at args[2], contracted dim=-2
            _need_fake_quantization = True
            a_pos, b_pos = 1, 2
            granularity = PerRow(dim=-2)
        else:
            _need_fake_quantization = False

        if _need_fake_quantization:
            A, B = args[a_pos], args[b_pos]

            assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
            assert isinstance(B, cls), f"B should be a {cls.__name__}"

            # Fake-quantize the activation if B.activation_config exists. With torch._grouped_mm, activation
            # is quantized once for the shared 3D weight. In a per-expert loop pattern, this repeats per expert.
            # Activation fake quantization is skipped if the activation is empty. This is a possible case when
            # a loop over experts instead of grouped_mm is used and some experts don't receive any tokens.
            if B.activation_config is not None and A.numel() > 0:
                assert not isinstance(A, TorchAOBaseTensor), \
                    f"When an activation config is specified, the activation must not be a quantized tensor, got {type(A)}"
                fq_A = cls._fake_quantize(A, B.activation_config, PerRow(dim=-1)) # always quantize the last dimension of activations
            else:
                fq_A = A

            # Fake-quantize the weight
            B_data = unwrap_weight(B)
            if B.weight_config is not None:
                fq_B_data = cls._fake_quantize(B_data, B.weight_config, granularity)
            else:
                fq_B_data = B_data

            new_args = list(args)
            new_args[a_pos] = fq_A
            new_args[b_pos] = fq_B_data
        else:
            new_args = args

        with torch._C.DisableTorchFunctionSubclass():
            return func(*new_args, **kwargs)

    @classmethod
    def _fake_quantize(
        cls, weight: torch.Tensor,
        config: Float8FakeQuantizeConfig,
        granularity: Granularity,
    ) -> torch.Tensor:
        original_dtype = weight.dtype
        block_size = get_block_size(weight.shape, granularity)
        scale = _choose_scale_float8(
            weight,
            block_size,
            config.dtype,
            hp_value_lb=config.hp_value_lb,
            hp_value_ub=config.hp_value_ub,
        ).detach()
        q = _quantize_affine_float8(weight, scale, config.dtype)
        dq = _dequantize_affine_float8(q, scale, original_dtype)
        return dq


@register_MoE_QAT_quantize_parameter_handler(Float8FakeQuantizeConfig)
def _(
    module: nn.Module,
    param_fqn: str,
    param: nn.Parameter,
    extra_args: Tuple[Any, ...] = (),
):
    from torchao.prototype.moe_qat.config import MoEQATConfig

    config: MoEQATConfig = extra_args[0]

    assert isinstance(config, MoEQATConfig), "extra_args[0] must be a MoEQATConfig"

    return nn.Parameter(
        data=Float8FakeQuantizedWeightWrapperTensor(
            param.data,
            activation_config=config.activation_config,
            weight_config=config.weight_config,
        ),
        requires_grad=param.requires_grad,
    )
