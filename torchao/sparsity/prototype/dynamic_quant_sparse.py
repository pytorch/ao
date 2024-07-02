import torch
import torch.nn as nn
from typing import Tuple, Optional

from torchao.quantization.utils import (
    dynamically_quantize_per_channel,
    quant_int8_dynamic_per_token_linear,
    quantize_activation_per_token_absmax,
    dequantize_per_channel,
)

from torchao.quantization.subclass import (
    Int8DynamicallyQuantizedLinearWeight,
    QuantizedLinearWeightBase,
)

from torch.sparse import to_sparse_semi_structured

# Quant + Sparse helper functinos
def sparse_quant_int8_dynamic_linear(
    x : torch.Tensor,
    w_vals_int8_packed : torch.Tensor,
    w_meta_int32 : Optional[torch.Tensor],
    w_scales : torch.Tensor,
    bias : Optional[torch.Tensor],
    out_dtype : torch.dtype,
    fuse_mul=False,
):
    x_vals_int8, x_scales = quantize_activation_per_token_absmax(x)
    # w_meta_int32 is either None or meta tensor
    if w_meta_int32 is None:
        if fuse_mul:
            mm_out = sparse_quant_int8_cslt_matmul_fuse_mul(
                x_vals_int8, x_scales, w_vals_int8_packed, w_scales, out_dtype,
            )
        else:
            mm_out = sparse_quant_int8_cslt_matmul(
                x_vals_int8, x_scales, w_vals_int8_packed, w_scales, out_dtype,
            )
    else:
        mm_out = sparse_quant_int8_cutlass_matmul(
            x_vals_int8, x_scales, w_vals_int8_packed, w_meta_int32, w_scales, out_dtype,
        )

    if bias is not None:
        mm_out += bias
    return mm_out

def sparse_quant_int8_cslt_matmul_fuse_mul(
    x_vals_int8,
    x_scales,
    w_vals_int8,
    w_scales,
    out_dtype,
):

    assert (
        x_vals_int8.dtype == torch.int8
    ), f"x dtype {x_vals_int8.dtype} not yet supported"
    assert (
        w_vals_int8.dtype == torch.int8
    ), f"w dtype {w_vals_int8.dtype} not yet supported"
    # assert w_scales.dtype == out_dtype, f'{w_scales.dtype} does not match {out_dtype}'

    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1]).contiguous()

    assert x_scales.dtype in [
        torch.float,
        torch.bfloat16,
    ], f"x_scales needs to be a torch.float32 or torch.bfloat16 but got {x_scales.dtype}"

    y_dot_bf16_w_scales_fused = torch._cslt_sparse_mm(
        w_vals_int8, tmp.t(), alpha=w_scales.to(torch.float32), out_dtype=torch.bfloat16
    ).t()
    y = (y_dot_bf16_w_scales_fused * x_scales.reshape(-1, 1)).reshape(
        *x_vals_int8.shape[:-1], y_dot_bf16_w_scales_fused.shape[-1]
    )
    y = y.to(out_dtype)

    return y

def sparse_quant_int8_cslt_matmul(
    x_vals_int8,
    x_scales,
    w_vals_int8,
    w_scales,
    out_dtype,
):

    assert (
        x_vals_int8.dtype == torch.int8
    ), f"x dtype {x_vals_int8.dtype} not yet supported"
    assert (
        w_vals_int8.dtype == torch.int8
    ), f"w dtype {w_vals_int8.dtype} not yet supported"
    # assert w_scales.dtype == out_dtype, f'{w_scales.dtype} does not match {out_dtype}'

    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1]).contiguous()

    assert x_scales.dtype in [
        torch.float,
        torch.bfloat16,
    ], f"x_scales needs to be a torch.float32 or torch.bfloat16 but got {x_scales.dtype}"

    y_dot_bf16_w_scales_fused = torch._cslt_sparse_mm(
        w_vals_int8, tmp.t(), out_dtype=torch.bfloat16
    ).t()
    y = (y_dot_bf16_w_scales_fused * x_scales.reshape(-1, 1) * w_scales).reshape(
        *x_vals_int8.shape[:-1], y_dot_bf16_w_scales_fused.shape[-1]
    )
    y = y.to(out_dtype)

    return y


def sparse_quant_int8_cutlass_matmul(
    x_vals_int8,
    x_scales,
    w_vals_int8,
    w_meta_int32,
    w_scales,
    out_dtype,
):
    assert (
        x_vals_int8.dtype == torch.int8
    ), f"x dtype {x_vals_int8.dtype} not yet supported"
    assert (
        w_vals_int8.dtype == torch.int8
    ), f"w dtype {w_vals_int8.dtype} not yet supported"
    assert w_scales.dtype == out_dtype, f"{w_scales.dtype} does not match {out_dtype}"
    assert w_meta_int32.dtype == torch.int32, f"{w_meta_int32.dtype} not yet supported"

    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1]).contiguous()

    assert x_scales.dtype in [
        torch.float,
        torch.bfloat16,
    ], f"x_scales needs to be a torch.float32 or torch.bfloat16 but got {x_scales.dtype}"

    y_dot_int32 = torch._sparse_semi_structured_linear(
        tmp, w_vals_int8, w_meta_int32.view(torch.int32), out_dtype=torch.int32
    )
    y = (y_dot_int32 * x_scales.reshape(-1, 1) * w_scales).reshape(
        *x_vals_int8.shape[:-1], y_dot_int32.shape[-1]
    )
    y = y.to(out_dtype)
    return y

class Int8DynamicallyQuantized24CusparseltLinearFuseMulWeight(
    Int8DynamicallyQuantizedLinearWeight
):
    def dequantize(self, dtype=None):
        # overload dequantize op for __repr__
        zero_points = torch.zeros(self.q_scales.shape, device=self.q_scales.device, dtype=self.q_scales.dtype)
        int_data_expanded = torch._cslt_sparse_mm(self.int_data, torch.eye(self.shape[1],
                                                                           dtype=self.int_data.dtype,
                                                                           device=self.int_data.device))
        dq_t = dequantize_per_channel(
            int_data_expanded, self.q_scales, zero_points, self.dtype if dtype is None else dtype
        ).to(self.dtype)

        return dq_t if not self.transposed else dq_t.t()

    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        return sparse_quant_int8_dynamic_linear(
            act_mat, w_qtensor.int_data, None, w_qtensor.q_scales, bias, act_mat.dtype,
            fuse_mul=True
        )

    @classmethod
    def from_float(cls, input_float, qmin=-128, qmax=127):

        assert input_float.is_cuda

        w_int_repr, w_scales, _ = dynamically_quantize_per_channel(
            input_float, qmin, qmax, torch.int8
        )

        int_data = w_int_repr.contiguous()
        int_data = torch._cslt_compress(int_data)

        return cls(
            int_data,
            w_scales,
            False,
            input_float.shape,
            dtype=input_float.dtype,
        )


class Int8DynamicallyQuantizedSemiStructuredSparseLinearWeight(QuantizedLinearWeightBase):

    @staticmethod
    def __new__(cls, int_data, mask_meta, q_scales, transposed, shape, **kwargs):
        kwargs["dtype"] = kwargs.get("dtype", q_scales.dtype)
        return super().__new__(cls, int_data, transposed, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, int_data, mask_meta, q_scales, transposed, shape, **kwargs):
        self.q_scales = q_scales
        self.mask_meta = mask_meta
        super().__init__(int_data, transposed)

    def dequantize(self, dtype=None):
        """
        Obtain the dequantized version of the quantized tensor subclass
        """
        dq_t = dequantize_per_channel(
            self.int_data, self.q_scales, 0, self.dtype if dtype is None else dtype
        ).to(self.dtype)
        # data was transposed to dequantize so make sure shape is correct
        return dq_t if not self.transposed else dq_t.t()

    def int_repr(self):
        """
        Get the internal integer representation of the quantized tensor
        """
        return self.int_data if self.transposed else self.int_data.t()

    def q_params(self):
        """
        Get the quantization scales for the quantized tensor
        """
        return {"q_scales": self.q_scales}

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.int_data.to(kwargs["device"]),
            self.mask_meta.to(kwargs["device"]),
            self.q_scales.to(kwargs["device"]),
            self.transposed,
            self.shape,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.int_data),
            fn(self.mask_meta),
            fn(self.q_scales),
            self.transposed,
            self.shape,
            dtype=self.dtype,
        )

    def _change_shape(self, shape):
        return self.__class__(
            self.int_data,
            self.mask_meta,
            self.q_scales,
            self.transposed,
            shape,
            dtype=self.dtype,
        )

    def __tensor_flatten__(self):
        return ["int_data", "mask_meta", "q_scales"], [
            self.transposed,
            self.dtype,
            self.shape,
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None
    ):
        int_data, q_scales = tensor_data_dict["int_data"], tensor_data_dict["q_scales"]
        mask_meta = tensor_data_dict["mask_meta"]
        transposed, dtype, shape = tensor_attributes
        return cls(
            int_data,
            mask_meta,
            q_scales,
            transposed,
            shape if outer_size is None else outer_size,
            dtype=dtype,
            strides=outer_stride,
        )

    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        return sparse_quant_int8_dynamic_linear(
            act_mat,
            w_qtensor.int_data,
            w_qtensor.mask_meta,
            w_qtensor.q_scales,
            bias,
            act_mat.dtype,
        )

    @classmethod
    def from_float(cls, input_float, qmin=-128, qmax=127):

        assert input_float.is_cuda

        w_int_repr, w_scales, _ = dynamically_quantize_per_channel(
            input_float, qmin, qmax, torch.int8
        )

        int_data = w_int_repr.contiguous()
        sparse_tensor = to_sparse_semi_structured(int_data)

        return cls(
            sparse_tensor.packed,
            sparse_tensor.meta,
            w_scales,
            False,
            input_float.shape,
            dtype=input_float.dtype,
        )
