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
except:
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

# TODO: Document the modes there seems to be relu, median and interpolate but the mode is a list not a string which is confusing
class AutoQuantizableLinearWeight(torch.Tensor):
    """
    A tensor subclass that finds the best type of quantization and swaps itself with that.

    This class extends `torch.Tensor` to enable automatic quantization by
    dynamically determining the best quantization method based on runtime benchmarks.

    Attributes:
        weight (torch.Tensor): The original weight tensor.
        qtensor_class_list (list): List of quantization classes to consider.
        logged_data (dict): Dictionary to log shape and dtype information.
        mode (list): The mode for quantization.
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
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        except Exception as e:
            print(f"ERR: subclass doesn't implement {func}, {str(e)}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        """
        Overrides the torch dispatch for this class.

        Args:
            func: The torch function to override.
            types: The types of the arguments.
            args: The positional arguments.
            kwargs: The keyword arguments.

        Returns:
            The result of the torch dispatch.
        """
        if func is aten.detach.default:
            return return_and_correct_aliasing(func, args, kwargs, args[0]._apply_fn_to_data(torch.detach))

@torch.no_grad()
def do_autoquant_bench(op, *args, **kwargs):
    """
    Runs benchmark op(*args, **kwargs) avoiding torch.compile overhead.

    Args:
        op: The operation to benchmark.
        *args: Additional positional arguments for the operation.
        **kwargs: Additional keyword arguments for the operation.

    Returns:
        float: The benchmark result.
    """
    rep = kwargs.pop("rep", 100)
    warmup = kwargs.pop("warmup", 25)
    with torch.no_grad():
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op(*args, **kwargs)
        stream.synchronize()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            op(*args, **kwargs)
        if TORCH_VERSION_AFTER_2_4:
            from torch._inductor.runtime.runtime_utils import do_bench_gpu
            res = do_bench_gpu(lambda: graph.replay(), warmup=warmup, rep=rep, return_mode="median")
        else:
            res = do_bench(lambda: graph.replay(), warmup=warmup, rep=rep, return_mode="median")
    return res

def _is_interpolate_mode(mode):
    """
    Checks if the mode is interpolate.

    Args:
        mode: The mode to check.

    Returns:
        bool: True if the mode is interpolate, False otherwise.
    """
    return isinstance(mode, list) and mode[0] == "interpolate" and len(mode) == 2 and isinstance(mode[1], float)

class AQMixin:
    """
    Mixin to turn normal quantized subclasses into autoquantizable ones.

    This mixin provides the necessary methods to enable automatic quantization
    for quantized tensor subclasses.
    """

    @classmethod
    def _autoquant_test(cls, act_mat, weight, bias, best_time, mode=["relu", None]):
        """
        Tests the autoquantization for a specific quantization class.

        Args:
            act_mat (torch.Tensor): The activation matrix.
            weight (torch.Tensor): The weight tensor.
            bias (torch.Tensor or None): The bias tensor.
            best_time (float): The best time recorded so far.
            mode (list): The mode for quantization.

        Returns:
            float: The benchmark result.
        """
        w_qtensor = cls.from_float(weight)
        if _is_interpolate_mode(mode):
            q_c_op = torch.compile(cls._quantized_op, mode="max-autotune-no-cudagraphs")
        else:
            func = lambda a, b, c: F.relu(cls._quantized_op(F.relu(a), b, c))
            q_c_op = torch.compile(func, mode="max-autotune-no-cudagraphs")
        res = do_autoquant_bench(q_c_op, act_mat, w_qtensor, bias, warmup=25, rep=100)
        if res < best_time * 1.1:
            res2 = do_autoquant_bench(q_c_op, act_mat, w_qtensor, bias, warmup=25, rep=900)
            res = (res2 * 0.9 + res * 0.1)
        print(f">>time: {res:0.3f}ms for {cls}, to_beat: {best_time:0.3f}ms ")
        return res

class AQInt8DynamicallyQuantizedLinearWeight(AQMixin, Int8DynamicallyQuantizedLinearWeight):
    """
    AutoQuantizable version of Int8DynamicallyQuantizedLinearWeight.
    """

    @classmethod
    def _autoquant_test(cls, act_mat, weight, bias, best_time, mode=["relu", None]):
        """
        Tests the autoquantization for the Int8DynamicallyQuantizedLinearWeight class.

        Args:
            act_mat (torch.Tensor): The activation matrix.
            weight (torch.Tensor): The weight tensor.
            bias (torch.Tensor or None): The bias tensor.
            best_time (float): The best time recorded so far.
            mode (list): The mode for quantization.

        Returns:
            float: The benchmark result.
        """
        if not _is_interpolate_mode(mode):
            return super()._autoquant_test(act_mat, weight, bias, best_time, mode)

        INTERPOLATION_CONSTANT = mode[1]
        w_qtensor = cls.from_float(weight)
        x_vals_int8, x_scales = quantize_activation_per_token_absmax(
            act_mat.reshape(-1, act_mat.shape[-1])
        )
        quantized_matmul = (
            lambda x_vals_int8, x_scales, w_vals_int8:
                safe_int_mm(x_vals_int8, w_vals_int8) * x_scales
        )
        q_c_matmul = torch.compile(quantized_matmul, mode="max-autotune-no-cudagraphs")
        with torch.no_grad():
            res_matmul = do_autoquant_bench(q_c_matmul, x_vals_int8, x_scales.reshape(-1, 1), w_qtensor.int_data)
        print(f">>time: {res_matmul:0.3f}ms for {cls} matmul, to_beat: {best_time:0.3f}ms")

        if res_matmul >= best_time:
            return res_matmul

        to_beat = best_time + INTERPOLATION_CONSTANT / (1 - INTERPOLATION_CONSTANT) * (best_time - res_matmul)
        res = super()._autoquant_test(act_mat, weight, bias, to_beat)
        max_int_const_win = (best_time - res_matmul) / (res - res_matmul)
        res_f = INTERPOLATION_CONSTANT * res + (1 - INTERPOLATION_CONSTANT) * res_matmul
        print(f">>time: {res_f:0.3f}ms for {cls} interpolated, breakeven constant: {max_int_const_win:0.2f}")
        return res_f

class AQWeightOnlyQuantizedLinearWeight(Int8WeightOnlyQuantizedLinearWeight, AQMixin):
    """
    AutoQuantizable version of Int8WeightOnlyQuantizedLinearWeight.
    """

class AQWeightOnlyQuantizedLinearWeight2(Int8WeightOnlyQuantizedLinearWeight, AQMixin):
    """
    AutoQuantizable version of Int8WeightOnlyQuantizedLinearWeight that uses a different kernel.
    """

    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        """
        Defines the quantized operation for the weight.

        Args:
            act_mat (torch.Tensor): The activation matrix.
            w_qtensor (torch.Tensor): The quantized weight tensor.
            bias (torch.Tensor or None): The bias tensor.

        Returns:
            torch.Tensor: The result of the quantized operation.
        """
        orig_dtype = act_mat.dtype
        orig_shape = act_mat.shape
        act_mat = act_mat.reshape(-1, act_mat.shape[-1], 1)
        y = (act_mat * w_qtensor.int_data.unsqueeze(0)).sum(dim=-2)
        y = y.reshape(*orig_shape[:-1], y.shape[-1]) * w_qtensor.q_scales
        if bias is not None:
            y += bias
        return y.to(orig_dtype)

    @classmethod
    def _autoquant_test(cls, act_mat, *args):
        """
        Tests the autoquantization for the AQWeightOnlyQuantizedLinearWeight2 class.

        Args:
            act_mat (torch.Tensor): The activation matrix.
            *args: Additional arguments.

        Returns:
            float: The benchmark result.
        """
        if act_mat.reshape(-1, act_mat.shape[-1]).shape[0] > 32:
            return torch.inf
        return super()._autoquant_test(act_mat, *args)

class AQWeightOnlyQuantizedLinearWeight3(Int8WeightOnlyQuantizedLinearWeight, AQMixin):
    """
    AutoQuantizable version of Int8WeightOnlyQuantizedLinearWeight that uses a different kernel.
    """

    def _quantized_op(act_mat, w_qtensor, bias):
        """
        Defines the quantized operation for the weight.

        Args:
            act_mat (torch.Tensor): The activation matrix.
            w_qtensor (torch.Tensor): The quantized weight tensor.
            bias (torch.Tensor or None): The bias tensor.

        Returns:
            torch.Tensor: The result of the quantized operation.
        """
        orig_shape = act_mat.shape
        y = torch.mm(act_mat.reshape(-1, orig_shape[-1]), w_qtensor.int_data * w_qtensor.q_scales)
        y = y.reshape(*orig_shape[:-1], y.shape[-1])
        if bias is not None:
            y += bias
        return y

class AQFloatLinearWeight(torch.Tensor, AQMixin):
    """
    A class to be used in concert with AutoQuantizableLinearWeight to provide a default/non-quantized option.

    This class implements the bare minimum needed to work with the AutoQuantizableLinearWeight class using
    the same interfaces that would normally be used by QTensor subclasses but for a default linear operation instead.
    The result of from_float is not a tensor subclass, but rather the float tensor.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        """
        Defines the linear operation for the weight.

        Args:
            act_mat (torch.Tensor): The activation matrix.
            w_qtensor (torch.Tensor): The weight tensor.
            bias (torch.Tensor or None): The bias tensor.

        Returns:
            torch.Tensor: The result of the linear operation.
        """
        return torch.nn.functional.linear(act_mat, w_qtensor, bias)

    @classmethod
    def from_float(cls, weight):
        """
        Creates an AQFloatLinearWeight from a floating-point tensor.

        Args:
            weight (torch.Tensor): The original weight tensor.

        Returns:
            AQFloatLinearWeight: The created instance.
        """
        return weight

DEFAULT_CLASS_LIST = [
    AQFloatLinearWeight,
    AQInt8DynamicallyQuantizedLinearWeight,
    AQWeightOnlyQuantizedLinearWeight,
    AQWeightOnlyQuantizedLinearWeight2,
    # AQWeightOnlyQuantizedLinearWeight3,
]

def change_linears_to_autoquantizable(model, **kwargs):
    """
    Converts all linear weight tensors to the AutoQuantizableLinearWeight tensor subclass.

    Expectation is that this is followed by running the model and then calling change_autoquantizable_to_quantized.

    Args:
        model (torch.nn.Module): The model containing linear weight tensors to convert.
        **kwargs: Additional keyword arguments.
    """
    from torchao.quantization.quant_api import _is_linear
    filter_fn = kwargs.pop("filter_fn", _is_linear)
    _ = kwargs.pop("error_on_unseen", True)  # same kwargs used for this and to_quantized
    kwargs["qtensor_class_list"] = kwargs.get("qtensor_class_list", DEFAULT_CLASS_LIST)
    kwargs["mode"] = kwargs.get("mode", ["relu", None])
    from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
    from torchao.quantization.quant_api import _get_subclass_inserter
    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(AutoQuantizableLinearWeight, **kwargs),
        filter_fn if filter_fn is not None else _is_linear,
    )

def change_autoquantizable_to_quantized(model, **kwargs):
    """
    Converts AutoQuantizableLinearWeight tensor subclasses to various quantized/non-quantized tensor subclasses
    depending on benchmark results.

    Expectation is that these modules are torch.compiled afterwards.

    Args:
        model (torch.nn.Module): The model containing AutoQuantizableLinearWeight tensors to convert.
        **kwargs: Additional keyword arguments.
    """
    hold = torch._dynamo.config.automatic_dynamic_shapes
    torch._dynamo.config.automatic_dynamic_shapes = False

    filter_fn = kwargs.pop(
        "filter_fn",
        lambda mod, *args: hasattr(mod, "weight") and isinstance(mod.weight, AutoQuantizableLinearWeight)
    )
    error_on_unseen = kwargs.pop("error_on_unseen", True)
    from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
    from torchao.quantization.quant_api import _get_subclass_inserter
    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(
            AutoQuantizableLinearWeight, method="to_quantized", error_on_unseen=error_on_unseen, **kwargs
        ),
        filter_fn,
    )
    torch._dynamo.config.automatic_dynamic_shapes = hold
    torch._dynamo.reset()

@torch.no_grad()
def autoquant(model, example_input=None, qtensor_class_list=DEFAULT_CLASS_LIST, filter_fn=None, mode=["interpolate", .85], **aq_kwargs):
    """
    Wraps model in AutoQuantWrapper, if example_input is provided, runs forward on it, otherwise returns the wrapped model.

    AutoQuantWrapper handles instances where model is torch.compiled by first performing autoquantization on the original
    model and then letting the torch.compile run/tracing occur.

    Args:
        model (torch.nn.Module): The model to wrap and potentially autoquantize.
        example_input (torch.Tensor or list or tuple, optional): Example input to run forward pass with.
        qtensor_class_list (list): List of quantization classes to consider.
        filter_fn (callable, optional): Filter function to apply to model layers.
        mode (list): The mode for quantization
        **aq_kwargs: Additional keyword arguments for autoquantization.

    Returns:
        torch.nn.Module: The wrapped and potentially autoquantized model.

    Example usage:
        torchao.autoquant(torch.compile(model))
        model(*example_input)
    """
    def autoquant_prehook(module, args, kwargs):
        module.forward_log_only(*args, **kwargs)
        change_autoquantizable_to_quantized(
            module,
            **aq_kwargs,
        )
        module.clean_up_autoquant_hooks_and_attrs()
        return args, kwargs

    change_linears_to_autoquantizable(
        model,
        filter_fn=filter_fn,
        qtensor_class_list=qtensor_class_list,
        mode=mode,
        **aq_kwargs
    )

    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        real_model = model._orig_mod
    else:
        real_model = model

    model.forward_log_only = lambda *args, **kwargs: real_model.forward(*args, **kwargs)

    handle = model.register_forward_pre_hook(autoquant_prehook, with_kwargs=True)

    def clean_up_autoquant_hooks_and_attrs():
        try:
            handle.remove()
            delattr(real_model, "clean_up_autoquant_hooks_and_attrs")
            delattr(real_model, "forward_log_only")
        except:
            pass
    model.clean_up_autoquant_hooks_and_attrs = clean_up_autoquant_hooks_and_attrs

    if isinstance(example_input, torch.Tensor):
        example_input = [example_input]
    if isinstance(example_input, (tuple, list)):
        model(*example_input)

    return model
