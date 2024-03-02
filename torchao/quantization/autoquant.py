import torch

from .subclass import ( # noqa
    Int8DynamicallyQuantizedLinearWeight,
    Int8WeightOnlyQuantizedLinearWeight,
    QuantizedLinearWeightBase,
)
from torch.utils._python_dispatch import return_and_correct_aliasing
from .utils import benchmark
from .quant_primitives import (
    quantize_activation_per_token_absmax,
    safe_int_mm,
)
import torch.nn.functional as F

aten = torch.ops.aten

AUTOQUANT_CACHE = {}

def check_cache(cls, shape, dtype):
    return AUTOQUANT_CACHE.get((cls, shape, dtype), None)

def update_cache(cls, shape, dtype, res):
    AUTOQUANT_CACHE[(cls, shape, dtype)] = res

class AutoQuantizableLinearWeight(torch.Tensor):
    """
    when run, finds best type of quantization for this tensor and swaps itself with that
    """
    @staticmethod
    def __new__(cls, weight, qtensor_class_list, *args, **kwargs):
        kwargs["device"] = weight.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else weight.layout
        )
        kwargs["dtype"] = (
            kwargs.get("dtype") if kwargs.get("dtype", False) else weight.dtype
        )
        kwargs["requires_grad"] = False
        shape = kwargs.pop("shape", weight.shape)
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, weight, qtensor_class_list, *args, **kwargs):
        self.weight = weight
        self.qtensor_class_list = qtensor_class_list
        self.logged_shape = None
        self.logged_dtype = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.weight}, shape={self.shape}, "
            f"device={self.device}, dtype={self.dtype}, qtensor_class_list={self.qtensor_class_list})"
        )

    @staticmethod
    def log_shape(act_mat, w_autoquant, bias):
        orig_shape = act_mat.shape
        act_mat = act_mat.reshape(-1, act_mat.shape[-1])
        logged_shape = (act_mat.shape, w_autoquant.shape, None if bias is None else bias.shape)
        logged_dtype = act_mat.dtype
        w_autoquant.logged_shape = logged_shape
        w_autoquant.logged_dtype = logged_dtype
        for q_cls in w_autoquant.qtensor_class_list:
            if check_cache(q_cls, logged_shape, logged_dtype) is None:
                update_cache(q_cls, logged_shape, logged_dtype, None)
        y = torch.mm(act_mat, w_autoquant.weight.t())
        y = y.reshape(*orig_shape[:-1], y.shape[-1])
        if bias is not None:
            y += bias
        return y

    def tune_autoquant(self, q_cls):
        act_shape, w_shape, bias_shape = self.logged_shape
        if check_cache(q_cls, self.logged_shape, self.logged_dtype) is None:
            with torch.no_grad():
                act_mat = torch.randn(act_shape, dtype=self.logged_dtype, device=self.device)
                bias = None if bias_shape is None else torch.randn(bias_shape, dtype=self.logged_dtype, device=self.device)
                res = q_cls._autoquant_test(act_mat, self.weight, bias)
                update_cache(q_cls, self.logged_shape, self.logged_dtype, res)

    def to_quantized(self, error_on_unseen, **kwargs):
        if error_on_unseen and (self.logged_shape is None or self.logged_dtype is None):
            raise RuntimeError("must run module normally to get shape, dtype info for autoquant")
        elif (self.logged_shape is None or self.logged_dtype is None) and not error_on_unseen:
            # default back to non-quantized weight if not seen
            self = AQFloatLinearWeight.from_float(self.weight)
            return  self
        best_time = torch.inf
        best_cls = None
        do_print=False
        for q_cls in self.qtensor_class_list:
            if check_cache(q_cls, self.logged_shape, self.logged_dtype) is None:
                do_print=True
                self.tune_autoquant(q_cls)
                torch._dynamo.reset()
            cls_res = AUTOQUANT_CACHE.get((q_cls, self.logged_shape, self.logged_dtype), torch.inf)
            if best_time >= cls_res:
                best_time = cls_res
                best_cls = q_cls
        if do_print:
            print(f"shape={self.logged_shape}, dtype={self.logged_dtype}, best_cls={best_cls}")
        # TODO handle random cls args/kwargs? or should they be curried
        self = best_cls.from_float(self.weight)
        return self

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.weight), self.qtensor_class_list, dtype=self.dtype
        )

    def __tensor_flatten__(self):
        return ["weight"], [self.qtensor_class_list, self.dtype, self.shape]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        weight = tensor_data_dict["weight"]
        qtensor_class_list, dtype, shape = tensor_attributes[0]
        return cls(weight, qtensor_class_list, shape=shape if outer_size is None else outer_size, dtype=dtype, strides=outer_stride)

    @classmethod
    def from_float(cls, weight, qtensor_class_list):
        return cls(weight, qtensor_class_list)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func is torch.nn.functional.linear:
            mat1, w_autoquant, bias = (
                args[0],
                args[1],
                args[2] if len(args)>2 else None
            )
            return cls.log_shape(mat1, w_autoquant, bias)

        try:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        except:
            print(f"ERR: subclass doesn't implement {func}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
         if func is aten.detach.default:
            return return_and_correct_aliasing(func, args, kwargs, args[0]._apply_fn_to_data(torch.detach))

class AQMixin():
    """
    Mixin to turn normal quantized subclasses into autoquantizable ones
    """
    @classmethod
    def _autoquant_test(cls, act_mat, weight, bias):
        w_qtensor = cls.from_float(weight)
        func = lambda act_mat, w_qtensor, bias: F.relu(cls._quantized_op(F.relu(act_mat), w_qtensor, bias))
        q_c_op = torch.compile(func, mode="max-autotune")
        # q_c_op = torch.compile(cls._quantized_op, mode="max-autotune")
        with torch.no_grad():
            torch.cuda.synchronize()
            res = benchmark(q_c_op, act_mat, w_qtensor, bias)
        print(cls, res)
        return res

class AQInt8DynamicallyQuantizedLinearWeight(AQMixin, Int8DynamicallyQuantizedLinearWeight):
    """
    AutoQuantizable version of Int8DynamicallyQuantizedLinearWeight
    """
    @classmethod
    def _autoquant_test(cls, act_mat, weight, bias):
        res = super()._autoquant_test(act_mat, weight, bias)
        w_qtensor = cls.from_float(weight)
        x_vals_int8, x_scales = quantize_activation_per_token_absmax(
            act_mat.reshape(-1, act_mat.shape[-1])
        )
        quantized_matmul = (
            lambda x_vals_int8, x_scales, w_vals_int8:
                safe_int_mm(x_vals_int8, w_vals_int8) * x_scales
        )
        q_c_matmul=torch.compile(quantized_matmul, mode="max-autotune")
        with torch.no_grad():
            res2=benchmark(q_c_matmul, x_vals_int8, x_scales, w_qtensor.int_data)
        print(cls, "matmul", res2)
        # for SAM best is between .458-.499, SDXL .45=3.094 .47=2.880 .48=3.036 .5=2.930
        return res


class AQWeightOnlyQuantizedLinearWeight(Int8WeightOnlyQuantizedLinearWeight, AQMixin):
    """
    AutoQuantizable version of Int8WeightOnlyQuantizedLinearWeight
    """

class AQWeightOnlyQuantizedLinearWeight2(Int8WeightOnlyQuantizedLinearWeight, AQMixin):
    """
    AutoQuantizable version of Int8WeightOnlyQuantizedLinearWeight that
    uses a different kernel
    """
    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        orig_dtype = act_mat.dtype
        orig_shape = act_mat.shape
        act_mat = act_mat.reshape(-1, act_mat.shape[-1], 1)
        y = (act_mat*w_qtensor.int_data.unsqueeze(0)).sum(dim=-2)
        y = y.reshape(*orig_shape[:-1], y.shape[-1])
        if bias is not None:
            y += bias
        return y.to(orig_dtype)

    @classmethod
    def _autoquant_test(cls, act_mat, weight, bias):
        # if act_mat has batchsize>2 don't use this kernel
        if act_mat.reshape(-1, act_mat.shape[-1]).shape[0]>2:
            return torch.inf
        return super()._autoquant_test(act_mat, weight, bias)

class AQWeightOnlyQuantizedLinearWeight3(Int8WeightOnlyQuantizedLinearWeight, AQMixin):
    def _quantized_op(act_mat, w_qtensor, bias):
        orig_shape = act_mat.shape
        y = torch.mm(act_mat.reshape(-1, orig_shape[-1]), w_qtensor.int_data*w_qtensor.q_scales)
        y=y.reshape(*orig_shape[:-1], y.shape[-1])
        if bias is not None:
            y += bias
        return y


class AQFloatLinearWeight(torch.Tensor, AQMixin):
    """
    A class to be used in concert with AutoQuantizableLinearWeight to provide a
    default/non-quantized option. Only implements the bare minimum needed to work with the
    AutoQuantizableLinearWeight class using the same interfaces that would normally be
    used by QTensor subclasses but for a default linear op instead.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        return torch.nn.functional.linear(act_mat, w_qtensor, bias)

    @classmethod
    def from_float(cls, weight):
        return weight

DEFAULT_CLASS_LIST = [
    AQFloatLinearWeight,
    AQInt8DynamicallyQuantizedLinearWeight,
    AQWeightOnlyQuantizedLinearWeight,
    AQWeightOnlyQuantizedLinearWeight2,
    AQWeightOnlyQuantizedLinearWeight3,
]

if False:
    # def _get_to_kwargs(self, *args, **kwargs):
    #     device, dtype, _, memory_format = torch._C._nn._parse_to(*args, **kwargs)
    #     device = self.device if device is None else device
    #     dtype = self.dtype if dtype is None else dtype
    #     memory_format = (
    #         memory_format if memory_format is not None else torch.preserve_format
    #     )
    #     kwargs = {
    #         "device": device,
    #         "dtype": dtype,
    #         "memory_format": memory_format,
    #     }
    #     return kwargs

    # def to(self, *args, **kwargs):
    #     kwargs = self._get_to_kwargs(*args, **kwargs)
    #     return self.__class__(
    #         self.int_data.to(kwargs["device"]),
    #         self.q_scales.to(kwargs["device"]),
    #         self.transposed,
    #         self.shape,
    #         **kwargs,
    #     )

    # def _apply_fn_to_data(self, fn):
    #     return self.__class__(
    #         fn(self.int_data), fn(self.q_scales), self.transposed, self.shape, dtype=self.dtype
    #     )

    # def _change_shape(self, shape):
    #     return self.__class__(
    #         self.int_data, self.q_scales, self.transposed, shape, dtype=self.dtype
    #     )

    # def half(self):
    #     return self.to(torch.float16)
    pass
