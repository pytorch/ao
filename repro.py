import torch
from torch.utils._python_dispatch import return_and_correct_aliasing


aten = torch.ops.aten


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

def pack_tinygemm_scales_and_zeros(scales, zeros):
    assert scales.shape == zeros.shape
    assert scales.dtype == torch.bfloat16
    assert zeros.dtype == torch.bfloat16
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )

def unpack_tinygemm_scales_and_zeros(scales_and_zeros):
    assert len(scales_and_zeros.shape) == 3 and scales_and_zeros.shape[2] == 2
    assert scales_and_zeros.dtype == torch.float
    return torch.split(scales_and_zeros.transpose(0, 1), 1, 2)

def get_groupwise_affine_qparams(w, n_bit=4, groupsize=128):
    """ """
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    # assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    zeros = min_val + scales * (2 ** (n_bit - 1))
    return scales.to(torch.bfloat16).reshape(w.shape[0], -1), zeros.to(
        torch.bfloat16
    ).reshape(w.shape[0], -1)

def groupwise_affine_quantize_tensor_from_qparams(
    w,
    scales,
    zeros,
    n_bit=4,
    groupsize=128,
):
    assert groupsize > 1
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    # assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    min_val = zeros - scales * (2 ** (n_bit - 1))
    max_int = 2**n_bit - 1
    min_int = 0
    w_int4x8 = (
        to_quant.sub(min_val)
        .div(scales)
        .round()
        .clamp_(min_int, max_int)
        .to(torch.int32)
        .reshape_as(w)
    )

    return w_int4x8

def get_group_qparams_symmetric(w, n_bit=4, groupsize=128, precision=torch.float32):
    # needed for GPTQ with padding
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

    max_val_abs = torch.max(-min_val_neg, max_val_pos)
    max_int = 2 ** (n_bit - 1) - 1
    min_int = -(2 ** (n_bit - 1))

    scales = max_val_abs / (float(max_int - min_int) / 2)
    scales = torch.max(scales, torch.full_like(scales, torch.finfo(torch.float32).eps))
    # TODO: make sure abs(scales) is not too small?
    zeros = torch.full_like(scales, 0)
    return scales.to(precision).reshape(w.shape[0], -1), zeros.to(precision).reshape(
        w.shape[0], -1
    )

def groupwise_affine_quantize_tensor(w, n_bit=4, groupsize=128):
    scales, zeros = get_groupwise_affine_qparams(w, n_bit, groupsize)
    w_int4x8 = groupwise_affine_quantize_tensor_from_qparams(
        w, scales, zeros, n_bit, groupsize
    )
    scales_and_zeros = pack_tinygemm_scales_and_zeros(scales, zeros)
    return w_int4x8, scales_and_zeros

class QuantizedLinearWeightBase(torch.Tensor):
    """
    Base quantized tensor subclass for quantized linear weights. When the from_float method is used,
    to create an instance of any QuantizedLinearWeightBase, we assume the input
    weight is oriented the way it is in a normal linear op, i.e. out-channels x in-channels.

    The shape and dtype of the tensor subclass represent how the tensor subclass looks externally,
    regardless of the internal representation's type or orientation.
    """

    int_data: torch.Tensor
    transposed: bool

    @staticmethod
    def __new__(cls, int_data, transposed, shape, *args, **kwargs):
        kwargs["device"] = int_data.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else int_data.layout
        )
        assert "dtype" in kwargs
        assert not kwargs.get("requires_grad", False)
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, int_data, transposed, *args, **kwargs):

        self.int_data = int_data

        self.transposed = transposed

    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.dequantize()}, shape={self.shape}, "
            f"device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def dequantize(self):
        pass

    def int_repr(self):
        pass

    def q_params(self):
        pass

    def half(self):
        return self.to(torch.float16)

    def _get_to_kwargs(self, *args, **kwargs):
        device, dtype, _, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        memory_format = (
            memory_format if memory_format is not None else torch.preserve_format
        )
        kwargs = {
            "device": device,
            "dtype": dtype,
            "memory_format": memory_format,
        }
        return kwargs

    def _apply_fn_to_data(self, fn):
        pass

    def _change_shape(self):
        pass

    def __tensor_flatten__(self):
        pass

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        pass

    @classmethod
    def from_float(cls, input_float):
        pass

    # __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func is torch.nn.functional.linear:
            mat1, w_qtensor, bias = (
                args[0],
                args[1],
                args[2] if len(args) > 2 else None,
            )
            assert w_qtensor.transposed == False
            return cls._quantized_op(mat1, w_qtensor, bias)

        try:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        except:
            print(f"ERR: subclass doesn't implement {func}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # two scenarios where we currently fall back to vanilla mm:
        # 1 - when tensor is on CPU: we are missing qmm for CPU, but we should have a CPU implementation
        #     for consistency and to allow people to test
        # 2 - we're given non-floats - quantizing long to int8 is crazy
        if (
            func in [aten.mm.default, aten.addmm.default]
            and args[0].is_floating_point()
            and args[0].is_cuda
        ):
            if func == aten.addmm.default:
                assert args[1].shape[-1] == args[2].shape[0], (
                    f"need mat1 shape: {args[1].shape} final"
                    f"dim to match mat2 shape: {args[2].shape} first dim "
                )
                mat1, w_qtensor, bias = (
                    args[1],
                    args[2],
                    args[0],
                )
            else:
                assert args[0].shape[-1] == args[1].shape[0], (
                    f"need mat1 shape: {args[0].shape} final dim"
                    f"to match mat2 shape: {args[1].shape} first dim"
                )
                mat1, w_qtensor, bias = (
                    args[0],
                    args[1],
                    None if len(args) == 2 else args[2],
                )
            # call the quantized op for the specific type
            # of quantized tensor subclass
            return cls._quantized_op(mat1, w_qtensor, bias)

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        if func is aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )

        if func is aten.t.default:
            args[0].transposed = not args[0].transposed
            new = args[0]._change_shape(args[0].shape[::-1])
            return return_and_correct_aliasing(func, args, kwargs, new)

        if func is aten._to_copy.default:
            return return_and_correct_aliasing(
                func,
                args,
                kwargs,
                args[0].to(*args[1:], **kwargs)._apply_fn_to_data(torch.clone),
            )


class Int4WeightOnlyQuantizedLinearWeight(QuantizedLinearWeightBase):
    """
    A Tensor subclass that when applied to a weight used in a linear op/module,
    changes that linear op to a weight-only int4 quantized linear op with groupwise
    affine quantization on the weight.
    """

    scales_and_zeros: torch.Tensor
    groupsize: int
    inner_k_tiles: int

    @staticmethod
    def __new__(
        cls,
        int_data,
        scales_and_zeros,
        transposed,
        shape,
        groupsize=128,
        inner_k_tiles=8,
        **kwargs,
    ):
        kwargs["dtype"] = kwargs.get("dtype", scales_and_zeros.dtype)
        return super().__new__(cls, int_data, transposed, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        int_data,
        scales_and_zeros,
        transposed,
        shape,
        groupsize,
        inner_k_tiles,
        **kwargs,
    ):
        # the transposed flag tracks whether the tensor subclass has been transposed relative
        # to how a weight is normally stored in a linear i.e. [out_features, in_features].
        # tracking both transposed and shape is slightly redundant but corner cases like
        # square matrices can cause issues otherwise

        self.scales_and_zeros = scales_and_zeros

        self.groupsize = groupsize

        self.inner_k_tiles = inner_k_tiles
        super().__init__(int_data, transposed)

    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        orig_act_size = act_mat.size()
        orig_dtype = act_mat.dtype

        # reshape and pad activation
        act_mat = act_mat.reshape(-1, act_mat.shape[-1]).to(torch.bfloat16)
        pad_size = find_multiple(act_mat.shape[-1], 1024)
        act_mat = torch.nn.functional.pad(act_mat, (0, pad_size - act_mat.shape[-1]))

        # matmul
        y = aten._weight_int4pack_mm(
            act_mat.contiguous(),
            w_qtensor.int_data,
            w_qtensor.groupsize,
            w_qtensor.scales_and_zeros,
        )

        # remove out_feature padding
        orig_out_features = (
            w_qtensor.shape[-1] if w_qtensor.transposed else w_qtensor.shape[-2]
        )
        y = y[:, :orig_out_features]

        y = y.reshape(*orig_act_size[:-1], orig_out_features)
        if bias is not None:
            y += bias
        return y.to(orig_dtype)

    def dequantize(self):
        eye_shape = self.shape[1] if not self.transposed else self.shape[0]
        w_dq = self._quantized_op(
            torch.eye(eye_shape, device=self.device, dtype=self.dtype), self, None
        )
        # we dequantized using linear with the identity matrix, output has shape [in_channels, out_channels]
        # so we need to transpose back to get the original shape unless self.transposed is set.
        w_dq = w_dq if self.transposed else w_dq.t()
        return w_dq.to(self.dtype)

    def int_repr(self):
        return self.int_data

    def q_params(self):
        scales, zero_points = unpack_tinygemm_scales_and_zeros(
            self.scales_and_zeros,
        )
        return {"q_scales": scales, "q_zero_points": zero_points}

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.int_data.to(kwargs["device"]),
            self.scales_and_zeros.to(kwargs["device"]),
            self.transposed,
            self.shape,
            self.groupsize,
            self.inner_k_tiles,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.int_data),
            fn(self.scales_and_zeros),
            self.transposed,
            self.shape,
            self.groupsize,
            self.inner_k_tiles,
            dtype=self.dtype,
        )

    #  `QuantizedLinearWeightBase` inconsistently.

    def _change_shape(self, shape):
        return self.__class__(
            self.int_data,
            self.scales_and_zeros,
            self.transposed,
            shape,
            self.groupsize,
            self.inner_k_tiles,
            dtype=self.dtype,
        )

    def __tensor_flatten__(self):
        return ["int_data", "scales_and_zeros"], (
            self.transposed,
            self.groupsize,
            self.inner_k_tiles,
            self.dtype,
            self.shape,
        )

    @classmethod

    #  `QuantizedLinearWeightBase` inconsistently.

    def __tensor_unflatten__(
        cls, tensor_data_dict, attributes, outer_size=None, outer_stride=None
    ):
        int_data, scales_and_zeros = (
            tensor_data_dict["int_data"],
            tensor_data_dict["scales_and_zeros"],
        )
        transposed, groupsize, inner_k_tiles, dtype, shape = attributes
        return cls(
            int_data,
            scales_and_zeros,
            transposed,
            shape if outer_size is None else outer_size,
            groupsize,
            inner_k_tiles,
            dtype=dtype,
            strides=outer_stride,
        )

    @classmethod
    def from_float(cls, input_float, groupsize=128, inner_k_tiles=8):
        """
        Method used to convert a linear weight tensor to an instance of the
        Int4WeightOnlyQuantizedLinearWeight subclass.

        Example usage::

            model.lin_mod.weight = (
                Int4WeightOnlyQuantizedLinearWeight.from_float(model.lin_mod.weight)
            )
        """
        assert groupsize in [256, 128, 64, 32]
        assert inner_k_tiles in [8, 4, 2]
        orig_shape = input_float.shape
        orig_out_features, orig_in_features = input_float.shape

        # padding
        in_features = find_multiple(orig_in_features, 1024)
        out_features = find_multiple(orig_out_features, 8)
        input_float = torch.nn.functional.pad(
            input_float,
            (0, in_features - orig_in_features, 0, out_features - orig_out_features),
        )

        # quantization and packing
        input_int4x8, scales_and_zeros = groupwise_affine_quantize_tensor(
            input_float, 4, groupsize
        )
        int_data = aten._convert_weight_to_int4pack(input_int4x8, inner_k_tiles)

        return cls(
            int_data,
            scales_and_zeros,
            False,
            orig_shape,
            groupsize,
            inner_k_tiles,
            dtype=input_float.dtype,
        )


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_mod = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.lin_mod(x)


model = M().eval()
example_inputs = torch.randn(1, 10)
ref = model(*example_inputs)
model.lin_mod.weight = torch.nn.Parameter(Int4WeightOnlyQuantizedLinearWeight.from_float(
    model.lin_mod.weight
))
torch.save(model.state_dict(), "model.pt")
loaded_state_dict = torch.load("model.pt")


model2 = M().eval()
model2.lin_mod.weight = torch.nn.Parameter(Int4WeightOnlyQuantizedLinearWeight.from_float(
    model2.lin_mod.weight
))
print("weight before loading:", model2.lin_mod.weight.data)
model2.load_state_dict(torch.load("model.pt", mmap=True, weights_only=False))
res = model2(*example_inputs)
print("weight after loading:", model2.lin_mod.weight.data)

# print(model2)
# print("ref:", ref, "\nres:", res, "\ndiff:", ref - res)
torch.testing.assert_allclose(ref, res)
