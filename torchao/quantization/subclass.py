import torch
from .subclass import (  # noqa
    Int8DynamicallyQuantizedLinearWeight,
    Int8WeightOnlyQuantizedLinearWeight,
    QuantizedLinearWeightBase,
)
from torch.utils._python_dispatch import return_and_correct_aliasing
from .quant_primitives import (
    quantize_activation_per_token_absmax,
    safe_int_mm,
)
from .utils import TORCH_VERSION_AFTER_2_4
import torch.nn.functional as F

try:
    from torch._inductor.utils import do_bench
except ImportError:
    from torch._inductor.runtime.runtime_utils import do_bench

aten = torch.ops.aten

AUTOQUANT_CACHE = {}

def check_cache(cls, shapes_and_dtype):
    """
    Checks the cache for a specific quantization class and shapes/dtype combination.

    Args:
        cls: The quantization class.
        shapes_and_dtype: A tuple containing the shapes and dtype of the tensors.

    Returns:
        The cached result if found, otherwise None.
    """
    return AUTOQUANT_CACHE.get((cls,) + shapes_and_dtype, None)

def update_cache(cls, shapes_and_dtype, res):
    """
    Updates the cache with a result for a specific quantization class and shapes/dtype combination.

    Args:
        cls: The quantization class.
        shapes_and_dtype: A tuple containing the shapes and dtype of the tensors.
        res: The result to cache.
    """
    AUTOQUANT_CACHE[(cls,) + shapes_and_dtype] = res

class AutoQuantizableLinearWeight(torch.Tensor):
    """
    A tensor subclass that finds the best type of quantization and swaps itself with that.

    This class extends `torch.Tensor` to enable automatic quantization by
    dynamically determining the best quantization method based on runtime benchmarks.
    """

    @staticmethod
    def __new__(cls, weight, qtensor_class_list, *args, mode=["relu", None], **kwargs):
        """
        Creates a new AutoQuantizableLinearWeight instance.

        Args:
            weight (torch.Tensor): The original weight tensor.
            qtensor_class_list (list): List of quantization classes to consider.
            *args: Additional positional arguments.
            mode (list): The mode for quantization.
            **kwargs: Additional keyword arguments.

        Returns:
            AutoQuantizableLinearWeight: The created instance.
        """
        kwargs["device"] = weight.device
        kwargs["layout"] = kwargs.get("layout", weight.layout)
        kwargs["dtype"] = kwargs.get("dtype", weight.dtype)
        kwargs["requires_grad"] = False
        shape = kwargs.pop("shape", weight.shape)
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, weight, qtensor_class_list, *args, mode=["relu", None], **kwargs):
        """
        Initializes the AutoQuantizableLinearWeight instance.

        Args:
            weight (torch.Tensor): The original weight tensor.
            qtensor_class_list (list): List of quantization classes to consider.
            *args: Additional positional arguments.
            mode (list): The mode for quantization.
            **kwargs: Additional keyword arguments.
        """
        self.weight = weight
        self.qtensor_class_list = qtensor_class_list
        self.logged_data = {}
        self.mode = mode

    def __repr__(self):
        """
        Returns a string representation of the AutoQuantizableLinearWeight instance.

        Returns:
            str: The string representation.
        """
        return (
            f"{self.__class__.__name__}(data={self.weight}, shape={self.shape}, "
            f"device={self.device}, dtype={self.dtype}, qtensor_class_list={self.qtensor_class_list})"
        )

    @staticmethod
    def log_shape(act_mat, w_autoquant, bias):
        """
        Logs the shape of the activation matrix, weight, and bias.

        Args:
            act_mat (torch.Tensor): The activation matrix.
            w_autoquant (AutoQuantizableLinearWeight): The weight tensor with autoquantization.
            bias (torch.Tensor or None): The bias tensor.
        """
        act_mat = act_mat.reshape(-1, act_mat.shape[-1])
        logged_dtype = act_mat.dtype
        logged_shapes = (act_mat.shape, w_autoquant.shape, None if bias is None else bias.shape)
        shapes_and_dtype = logged_shapes + (logged_dtype,)
        w_autoquant.logged_data[shapes_and_dtype] = 1 + w_autoquant.logged_data.get(shapes_and_dtype, 0)
        for q_cls in w_autoquant.qtensor_class_list:
            if check_cache(q_cls, shapes_and_dtype) is None:
                update_cache(q_cls, shapes_and_dtype, None)

    def tune_autoquant(self, q_cls, shapes_and_dtype, best_time):
        """
        Tunes the autoquantization for a specific quantization class.

        Args:
            q_cls: The quantization class.
            shapes_and_dtype: A tuple containing the shapes and dtype of the tensors.
            best_time: The best time recorded so far.
        """
        act_shape, w_shape, bias_shape, act_dtype = shapes_and_dtype
        if check_cache(q_cls, shapes_and_dtype) is None:
            with torch.no_grad():
                act_mat = torch.randn(act_shape, dtype=act_dtype, device=self.device)
                bias = None if bias_shape is None else torch.randn(bias_shape, dtype=act_dtype, device=self.device)
                res = q_cls._autoquant_test(act_mat, self.weight, bias, best_time, self.mode)
                update_cache(q_cls, shapes_and_dtype, res)

    @torch.no_grad()
    def to_quantized(self, error_on_unseen, **kwargs):
        """
        Converts the weight to a quantized version.

        Args:
            error_on_unseen (bool): Raise an error if no shape and dtype information is logged.
            **kwargs: Additional keyword arguments.

        Returns:
            The quantized weight tensor.
        """
        if error_on_unseen and not self.logged_data:
            raise RuntimeError("Must run module normally to get shape, dtype info for autoquant")
        elif not self.logged_data and not error_on_unseen:
            return AQFloatLinearWeight.from_float(self.weight)

        best_time = torch.inf
        best_cls = None
        for q_cls in self.qtensor_class_list:
            cur_time = 0
            for shapes_and_dtype, times_seen in self.logged_data.items():
                if check_cache(q_cls, shapes_and_dtype) is None:
                    self.tune_autoquant(q_cls, shapes_and_dtype, best_time)
                cur_time += check_cache(q_cls, shapes_and_dtype) * times_seen
            if best_time >= cur_time:
                best_time = cur_time
                best_cls = q_cls
        self = best_cls.from_float(self.weight)
        return self

    def _apply_fn_to_data(self, fn):
        """
        Applies a function to the weight data.

        Args:
            fn: The function to apply.

        Returns:
            AutoQuantizableLinearWeight: A new instance with the function applied to the weight.
        """
        return self.__class__(
            fn(self.weight), self.qtensor_class_list, dtype=self.dtype, mode=self.mode
        )

    def __tensor_flatten__(self):
        """
        Flattens the tensor for serialization.

        Returns:
            list: List of tensor names.
            list: List of tensor attributes.
        """
        return ["weight"], [self.qtensor_class_list, self.mode, self.dtype, self.shape]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        """
        Unflattens the tensor from serialized data.

        Args:
            tensor_data_dict (dict): Dictionary of tensor data.
            tensor_attributes (list): List of tensor attributes.
            outer_size: Outer size for the tensor.
            outer_stride: Outer stride for the tensor.

        Returns:
            AutoQuantizableLinearWeight: The unflattened tensor.
        """
        weight = tensor_data_dict["weight"]
        qtensor_class_list, mode, dtype, shape = tensor_attributes[0]
        return cls(weight, qtensor_class_list, mode, shape=shape if outer_size is None else outer_size, dtype=dtype, strides=outer_stride)

    @classmethod
    def from_float(cls, weight, qtensor_class_list, **kwargs):
        """
        Creates an AutoQuantizableLinearWeight from a floating-point tensor.

        Args:
            weight (torch.Tensor): The original weight tensor.
            qtensor_class_list (list): List of quantization classes to consider.
            **kwargs: Additional keyword arguments.

        Returns:
            AutoQuantizableLinearWeight: The created instance.
        """
        return cls(weight, qtensor_class_list, **kwargs)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Overrides the torch function for this class.

        Args:
            func: The torch function to override.
            types: The types of the arguments.
            args: The positional arguments.
            kwargs: The keyword arguments.

        Returns:
            The result of the torch function.
        """
        kwargs = {} if kwargs is None else kwargs

        if func is torch.nn.functional.linear:
            mat1, w_autoquant, bias = (
                args[0],
                args[1],
                args[2] if len(args) > 2 else None,
            )
            cls.log_shape(mat1, w_autoquant, bias)
            return func(mat1, w_autoquant.weight, bias)
        try:
            with torch._C.DisableTorchFunctionSubclass
