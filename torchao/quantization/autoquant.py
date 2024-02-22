import torch

from .subclass import ( # noqa
    Int8DynamicallyQuantizedLinearWeight,
    Int8WeightOnlyQuantizedLinearWeight,
    QuantizedLinearWeightBase,
)
from torch.utils._python_dispatch import return_and_correct_aliasing
from .utils import benchmark

aten = torch.ops.aten

AUTOQUANT_CACHE = {}

def check_cache(shape, cls):
    if shape in AUTOQUANT_CACHE:
        return AUTOQUANT_CACHE[shape].get(cls, None)
    else:
        return None

def update_cache(shape, cls, res):
    if not shape in AUTOQUANT_CACHE:
        AUTOQUANT_CACHE[shape] = {}
    AUTOQUANT_CACHE[shape][cls] = res

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
        self.cache_shape = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.weight}, shape={self.shape}, "
            f"device={self.device}, dtype={self.dtype}, qtensor_class_list={self.qtensor_class_list})"
        )

    @staticmethod
    def tune_autoquant(act_mat, w_autoquant, bias):
        orig_shape = act_mat.shape
        act_mat = act_mat.reshape(-1, act_mat.shape[-1])
        cache_shape = (act_mat.shape, w_autoquant.shape, None if bias is None else bias.shape)
        w_autoquant.cache_shape = cache_shape
        for cur_cls in w_autoquant.qtensor_class_list:
            if check_cache(cache_shape, cur_cls) is None:
                with torch.no_grad():
                    print(cur_cls, cache_shape)
                    print(torch.cuda.max_memory_allocated()/1e6, torch.cuda.memory_usage())
                    res = cur_cls._autoquant_test(act_mat.clone(), w_autoquant.weight.clone(), None if bias is None else bias.clone())
                    update_cache(cache_shape, cur_cls, res)
                    print(torch.cuda.max_memory_allocated()/1e6, torch.cuda.memory_usage())
        y = torch.mm(act_mat, w_autoquant.weight.t())
        y = y.reshape(*orig_shape[:-1], y.shape[-1])
        if bias is not None:
            y += bias
        return y

    def to_quantized(self):
        if self.cache_shape is None or self.cache_shape not in AUTOQUANT_CACHE:
            raise RuntimeError("must run module normally to find best quantization option")
        best_time = torch.inf
        best_cls = None
        for cur_cls in self.qtensor_class_list:
            cls_res = AUTOQUANT_CACHE[self.cache_shape].get(cur_cls, torch.inf)
            if best_time >= cls_res:
                best_time = cls_res
                best_cls = cur_cls
        # need to handle random cls args/kwargs?
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
            return cls.tune_autoquant(mat1, w_autoquant, bias)

        try:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        except:
            print(f"ERR: subclass doesn't implement {func}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
         if func is aten.detach.default:
            return return_and_correct_aliasing(func, args, kwargs, args[0]._apply_fn_to_data(torch.detach))


class DefaultLinear(torch.Tensor):
    """
    An class to be used in concert with AutoQuantizableLinearWeight to provide a
    default/non-quantized option. Only implements the bare minimum needed to work with the
    AutoQuantizableLinearWeight class using the same interfaces that would normally be
    used by QTensor subclasses but for a default linear op instead.
    """
    def __init__(self):
        super().__init__()

    @classmethod
    def _autoquant_test(cls, act_mat, weight, bias):
        w_qtensor = cls.from_float(weight)
        q_c_op = torch.compile(cls._quantized_op, mode="max-autotune")
        with torch.no_grad():
            res=benchmark(q_c_op, act_mat, w_qtensor, bias)
        print(cls, res)
        return res

    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        return torch.nn.functional.linear(act_mat, w_qtensor, bias)

    @classmethod
    def from_float(cls, weight):
        return weight

DEFAULT_CLASS_LIST = [
    DefaultLinear,
    Int8WeightOnlyQuantizedLinearWeight,
    Int8DynamicallyQuantizedLinearWeight,
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
