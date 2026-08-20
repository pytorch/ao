"""Intel XPU MXTensor subclass for MX format."""

import torch
import torch.nn.functional as F
from torch.nn.functional import ScalingType

from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import (
    MXTensor,
    QuantizeTensorToMXKwargs,
    _addmm_mx_dispatch,
    _get_gemm_choice,
    tensor_size_hp_to_fp4x2,
    to_mx,
)
from torchao.prototype.mx_formats.utils import _swizzle_aware_slice
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.utils import (
    fill_defaults,
    register_ao_tensor,
    return_and_correct_aliasing,
)

aten = torch.ops.aten


class MXTensorXPU(MXTensor):
    """MXTensor subclass for Intel XPU: no scale swizzling, device-specific gemm."""

    @staticmethod
    @torch._dynamo.allow_in_graph
    def to_mx(
        data_hp,
        elem_dtype,
        block_size=32,
        scaling_mode=ScaleCalculationMode.FLOOR,
        kernel_preference=KernelPreference.EMULATED,
        act_quant_kwargs=None,
        is_swizzled_scales=False,
        mxfp8_dim0_cast_kernel_choice=None,
    ):
        """XPU override: always is_swizzled_scales=False."""
        # Force no swizzle for XPU
        if act_quant_kwargs is not None and act_quant_kwargs.is_swizzled_scales:
            act_quant_kwargs = QuantizeTensorToMXKwargs(
                elem_dtype=act_quant_kwargs.elem_dtype,
                block_size=act_quant_kwargs.block_size,
                scaling_mode=act_quant_kwargs.scaling_mode,
                kernel_preference=act_quant_kwargs.kernel_preference,
                is_swizzled_scales=False,
            )
        scale, data_lp = to_mx(data_hp, elem_dtype, block_size, scaling_mode, False)
        return MXTensorXPU(
            data_lp,
            scale,
            elem_dtype,
            block_size,
            data_hp.dtype,
            kernel_preference,
            act_quant_kwargs,
            False,
        )


def _xpu_addmm_dispatch(a, b, aten_op, bias=None):
    """XPU-specific MX gemm dispatch."""
    if not isinstance(a, MXTensor):
        assert b.act_quant_kwargs is not None, "weight-only quant not yet supported"
        k = b.act_quant_kwargs
        a = MXTensorXPU.to_mx(
            a, k.elem_dtype, k.block_size, k.scaling_mode, k.kernel_preference
        )

    gemm_choice = _get_gemm_choice(a.kernel_preference, b.kernel_preference)

    if gemm_choice == KernelPreference.EMULATED:
        return _addmm_mx_dispatch(a, b, aten_op, bias)

    # AUTO: XPU-specific gemm
    M, K, N = a.shape[0], a.shape[1], b.shape[1]
    assert a.block_size == 32 and b.block_size == 32

    a_scale = a.scale.view(M, K // 32)
    b_scale = b.scale.t().view(N, K // 32)

    if a.elem_dtype == torch.float8_e4m3fn:
        assert b.elem_dtype == torch.float8_e4m3fn
        a_scale_e8m0 = a_scale.view(torch.float8_e8m0fnu)
        b_scale_e8m0 = b_scale.view(torch.float8_e8m0fnu).t().contiguous()
        return torch._scaled_mm(
            a.qdata,
            b.qdata,
            a_scale_e8m0,
            b_scale_e8m0,
            bias=bias,
            out_dtype=torch.bfloat16,
        )
    else:
        assert a.elem_dtype == torch.float4_e2m1fn_x2
        assert b.elem_dtype == torch.float4_e2m1fn_x2
        return F.scaled_mm(
            a.qdata.view(torch.float4_e2m1fn_x2),
            b.qdata.view(torch.float4_e2m1fn_x2),
            scale_a=a_scale,
            scale_recipe_a=ScalingType.BlockWise1x32,
            scale_b=b_scale.contiguous(),
            scale_recipe_b=ScalingType.BlockWise1x32,
            swizzle_a=None,
            swizzle_b=None,
            bias=bias,
            output_dtype=torch.bfloat16,
        )


xpu_implements = MXTensorXPU.implements


@xpu_implements([aten.mm.default, aten.matmul.default])
def xpu_mx_mm(func, types, args, kwargs):
    return _xpu_addmm_dispatch(args[0], args[1], func)


@xpu_implements([aten.addmm.default])
def xpu_mx_addmm(func, types, args, kwargs):
    return _xpu_addmm_dispatch(args[1], args[2], func, bias=args[0])


@xpu_implements([aten.linear.default])
def xpu_mx_linear(func, types, args, kwargs):
    a = args[0]
    orig_shape = a.shape
    a_2d = a.view(-1, orig_shape[-1])
    b = args[1].t()
    bias = args[2] if len(args) > 2 else None
    if bias is not None:
        res = _xpu_addmm_dispatch(a_2d, b, aten.addmm.default, bias)
    else:
        res = _xpu_addmm_dispatch(a_2d, b, aten.mm.default)
    return res.view(*orig_shape[:-1], res.shape[-1])


@xpu_implements([aten._pin_memory.default])
def xpu_mx_pin_memory(func, types, args, kwargs):
    tensor = args[0]
    return MXTensorXPU(
        tensor.qdata.pin_memory(),
        tensor.scale.pin_memory(),
        tensor.elem_dtype,
        tensor.block_size,
        tensor.orig_dtype,
        tensor.kernel_preference,
        tensor.act_quant_kwargs,
        tensor.is_swizzled_scales,
    )


@xpu_implements([aten.t.default])
def xpu_mx_t(func, types, args, kwargs):
    old = args[0]
    return MXTensorXPU(
        old.qdata.t(),
        old.scale.t(),
        old.elem_dtype,
        old.block_size,
        old.orig_dtype,
        old.kernel_preference,
        old.act_quant_kwargs,
        old.is_swizzled_scales,
    )


@xpu_implements([aten.view.default])
def xpu_mx_view_op(func, types, args, kwargs):
    data = args[0].qdata
    new_size = args[1]
    if args[0].elem_dtype == torch.float4_e2m1fn_x2:
        new_size = tensor_size_hp_to_fp4x2(new_size, data.is_contiguous())
    new_data = func(data, new_size, *args[2:], **kwargs)
    return MXTensorXPU(
        new_data,
        args[0].scale,
        args[0].elem_dtype,
        args[0].block_size,
        args[0].orig_dtype,
        args[0].kernel_preference,
        args[0].act_quant_kwargs,
        args[0].is_swizzled_scales,
    )


@xpu_implements([aten.slice.Tensor])
def xpu_mx_slice(func, types, args, kwargs):
    x, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    if step != 1:
        raise ValueError("Only support aten.slice with step=1")
    sliced_data, sliced_scale = _swizzle_aware_slice(x, dim, start, end, step)
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        MXTensorXPU(
            sliced_data,
            sliced_scale,
            x.elem_dtype,
            x.block_size,
            x.orig_dtype,
            x.kernel_preference,
            x.act_quant_kwargs,
            x.is_swizzled_scales,
        ),
    )


@xpu_implements([torch.ops._c10d_functional.all_gather_into_tensor.default])
def xpu_mx_all_gather(func, types, args, kwargs):
    mx_tensor = args[0]
    group_tag = args[1]
    gathered_qdata = func(mx_tensor.qdata, group_tag, *args[2:], **kwargs)
    scale_uint8 = mx_tensor.scale.view(torch.uint8)
    gathered_scale = func(scale_uint8, group_tag, *args[2:], **kwargs)
    gathered_scale = gathered_scale.view(torch.float8_e8m0fnu)
    return MXTensorXPU(
        gathered_qdata,
        gathered_scale,
        mx_tensor.elem_dtype,
        mx_tensor.block_size,
        mx_tensor.orig_dtype,
        mx_tensor.kernel_preference,
        mx_tensor.act_quant_kwargs,
        mx_tensor.is_swizzled_scales,
    )


@xpu_implements([torch.ops._c10d_functional.wait_tensor.default])
def xpu_mx_wait_tensor(func, types, args, kwargs):
    mx_tensor = args[0]
    waited_qdata = torch.ops._c10d_functional.wait_tensor.default(
        mx_tensor.qdata, *args[1:], **kwargs
    )
    waited_scale = torch.ops._c10d_functional.wait_tensor.default(
        mx_tensor.scale, *args[1:], **kwargs
    )
    return MXTensorXPU(
        waited_qdata,
        waited_scale,
        mx_tensor.elem_dtype,
        mx_tensor.block_size,
        mx_tensor.orig_dtype,
        mx_tensor.kernel_preference,
        mx_tensor.act_quant_kwargs,
        mx_tensor.is_swizzled_scales,
    )


# Register XPU class
register_ao_tensor(MXTensor, "xpu", MXTensorXPU)

# Allow safe serialization
torch.serialization.add_safe_globals([MXTensorXPU])
