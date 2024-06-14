import torch
from .subclass import ( # noqa
    Int8DynamicallyQuantizedLinearWeight,
    Int8WeightOnlyQuantizedLinearWeight,
    QuantizedLinearWeightBase,
)
from torch.utils._python_dispatch import return_and_correct_aliasing
from .quant_primitives import (
    safe_int_mm,
)
from torchao.utils import TORCH_VERSION_AFTER_2_4
from torchao.quantization.utils import quantize_activation_per_token_absmax

import torch.nn.functional as F
try:
    from torch._inductor.utils import do_bench
except:
    from torch._inductor.runtime.runtime_utils import do_bench

aten = torch.ops.aten

AUTOQUANT_CACHE = {}

def check_cache(cls, shapes_and_dtype):
    return AUTOQUANT_CACHE.get((cls,)+shapes_and_dtype, None)

def update_cache(cls, shapes_and_dtype, res):
    AUTOQUANT_CACHE[(cls,)+shapes_and_dtype] = res

# TODO: Document the methods
class AutoQuantizableLinearWeight(torch.Tensor):
    """
    A subclass of torch.Tensor that, when run, finds the best type of quantization for itself and swaps
    its data with the quantized version.

    Args:
        weight (torch.Tensor): The initial weight tensor.
        qtensor_class_list (list): A list of tensor classes to be considered for quantization.
        *args: Additional positional arguments.
        mode (list, optional): A list containing mode settings for quantization. The first element is the mode type
                               (e.g., "relu"), and the second element is the mode value (e.g., None). Defaults to ["relu", None].
        **kwargs: Additional keyword arguments.
    """

    @staticmethod
    def __new__(cls, weight, qtensor_class_list, *args, mode=["relu", None], **kwargs):
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

    def __init__(self, weight, qtensor_class_list, *args, mode=["relu", None], **kwargs):
        self.weight = weight
        self.qtensor_class_list = qtensor_class_list
        self.logged_data = {}
        self.mode = mode

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.weight}, shape={self.shape}, "
            f"device={self.device}, dtype={self.dtype}, qtensor_class_list={self.qtensor_class_list})"
        )

    @staticmethod
    def log_shape(act_mat, w_autoquant, bias):
        act_mat = act_mat.reshape(-1, act_mat.shape[-1])
        logged_dtype = act_mat.dtype
        logged_shapes = (act_mat.shape, w_autoquant.shape, None if bias is None else  bias.shape,)
        shapes_and_dtype = logged_shapes + (logged_dtype,)
        w_autoquant.logged_data[shapes_and_dtype] = 1 + w_autoquant.logged_data.get(shapes_and_dtype, 0)
        for q_cls in w_autoquant.qtensor_class_list:
            if check_cache(q_cls, shapes_and_dtype) is None:
                update_cache(q_cls, shapes_and_dtype, None)

    def tune_autoquant(self, q_cls, shapes_and_dtype, best_time):
        act_shape, w_shape, bias_shape, act_dtype = shapes_and_dtype
        if check_cache(q_cls, shapes_and_dtype) is None:
            with torch.no_grad():
                act_mat = torch.randn(act_shape, dtype=act_dtype, device=self.device)
                bias = None if bias_shape is None else torch.randn(bias_shape, dtype=act_dtype, device=self.device)
                res = q_cls._autoquant_test(act_mat, self.weight, bias, best_time, self.mode)
                update_cache(q_cls, shapes_and_dtype, res)

    @torch.no_grad()
    def to_quantized(self, error_on_unseen, **kwargs):
        if error_on_unseen and self.logged_data == {}:
            raise RuntimeError("must run module normally to get shape, dtype info for autoquant")
        elif (self.logged_data == {}) and not error_on_unseen:
            # default back to non-quantized weight if not seen
            self = AQFloatLinearWeight.from_float(self.weight)
            return self


        # only want to do shape+final print a single time if multiple layers
        # see/have same shapes so we gate on check_cache being empty for
        # at least one of the class/shape combinations.
        do_final_print = False
        print_once = True

        def count_shapes(self, do_print=True):
            differe_shape_count=0
            for shapes_and_dtype, times_seen in self.logged_data.items():
                differe_shape_count += 1
                if do_print:
                    act_shape, weight_shape, bias_shape, dtype = shapes_and_dtype
                    print(f"activation_shapes: {act_shape}, times_seen: {times_seen}")
            if do_print:
                print(f"weight_shape: {weight_shape}, dtype: {dtype}, bias_shape: {bias_shape}")
            return differe_shape_count

        # check each class
        best_time = torch.inf
        best_cls = None
        for q_cls in self.qtensor_class_list:
            # for each logged shape+dtype, benchmark
            cur_time=0
            shape_count = count_shapes(self, do_print=False)
            for shapes_and_dtype, times_seen in self.logged_data.items():
                if check_cache(q_cls, shapes_and_dtype) is None:
                    # only do final print if we have to autotune at least one cls/shape pair
                    do_final_print=True

                    # only print shapes once
                    if print_once == True:
                        print_once = False
                        count_shapes(self, do_print=True)

                    time_for_best_shape = check_cache(best_cls, shapes_and_dtype)
                    time_for_best_shape = torch.inf if time_for_best_shape is None else time_for_best_shape
                    self.tune_autoquant(q_cls, shapes_and_dtype, time_for_best_shape)
                    torch._dynamo.reset()
                cur_time += check_cache(q_cls, shapes_and_dtype) * times_seen
            if shape_count is not None and shape_count > 1:
                print(f">time (all shapes): {cur_time:0.3f}ms for {q_cls}, prev_best: {best_time:0.3f}ms")
            if best_time >= cur_time:
                best_time = cur_time
                best_cls = q_cls
        # only print if this is the first time seeing some cls+shape combo,
        # otherwise we will print the same thing for every layer.
        if do_final_print:
            print(f"best_cls={best_cls}\n")
        # TODO handle random cls args/kwargs? or should they be curried?
        self = best_cls.from_float(self.weight)
        return self

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.weight), self.qtensor_class_list, dtype=self.dtype, mode=self.mode
        )

    def __tensor_flatten__(self):
        return ["weight"], [self.qtensor_class_list, self.mode, self.dtype, self.shape]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        weight = tensor_data_dict["weight"]
        qtensor_class_list, mode, dtype, shape = tensor_attributes[0]
        return cls(weight, qtensor_class_list, mode, shape=shape if outer_size is None else outer_size, dtype=dtype, strides=outer_stride)

    @classmethod
    def from_float(cls, weight, qtensor_class_list, **kwargs):
        return cls(weight, qtensor_class_list, **kwargs)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func is torch.nn.functional.linear:
            mat1, w_autoquant, bias = (
                args[0],
                args[1],
                args[2] if len(args)>2 else None
            )
            cls.log_shape(mat1, w_autoquant, bias)
            return func(mat1, w_autoquant.weight, bias)
        try:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        except:
            print(f"ERR: subclass doesn't implement {func}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
         if func is aten.detach.default:
            return return_and_correct_aliasing(func, args, kwargs, args[0]._apply_fn_to_data(torch.detach))

@torch.no_grad()
def do_autoquant_bench(op, *args, **kwargs):
    """
    runs benchmark op(*args, **kwargs) avoiding torch.compile overhead
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
    if isinstance(mode, list) and mode[0]=="interpolate" and len(mode)==2 and isinstance(mode[1], float):
        return True
    return False

class AQMixin():
    """
    Tests and benchmarks the autoquantization process for the given activation matrix, weight, and bias.

    Args:
        act_mat (torch.Tensor): The activation matrix.
        weight (torch.Tensor): The weight tensor.
        bias (torch.Tensor or None): The bias tensor.
        best_time (float): The best time to beat for the quantization process.
        mode (list, optional): A list containing mode settings for quantization. The first element is the mode type
                                (e.g., "relu"), and the second element is the mode value (e.g., None). Defaults to ["relu", None].

    Returns:
        float: The benchmarked time for the autoquantization process.
    """
    @classmethod
    def _autoquant_test(cls, act_mat, weight, bias, best_time, mode=["relu", None]):
        w_qtensor = cls.from_float(weight)
        if _is_interpolate_mode(mode):
            q_c_op = torch.compile(cls._quantized_op, mode="max-autotune-no-cudagraphs")
        else:
            func = lambda a,b,c: F.relu(cls._quantized_op(F.relu(a), b, c))
            q_c_op = torch.compile(func, mode="max-autotune-no-cudagraphs")
        res = do_autoquant_bench(q_c_op, act_mat, w_qtensor, bias, warmup=25, rep=100)
        if res < best_time*1.1:
            res2 = do_autoquant_bench(q_c_op, act_mat, w_qtensor, bias, warmup=25, rep=900)
            res=(res2*.9+res*.1)
        print(f">>time: {res:0.3f}ms for {cls}, to_beat: {best_time:0.3f}ms ")
        return res

class AQInt8DynamicallyQuantizedLinearWeight(AQMixin, Int8DynamicallyQuantizedLinearWeight):
    """
    AutoQuantizable version of Int8DynamicallyQuantizedLinearWeight
    """
    @classmethod
    def _autoquant_test(cls, act_mat, weight, bias, best_time, mode=["relu", None]):
        """
        Tests and benchmarks the autoquantization process with special handling for interpolate mode.

        Args:
            act_mat (torch.Tensor): The activation matrix.
            weight (torch.Tensor): The weight tensor.
            bias (torch.Tensor or None): The bias tensor.
            best_time (float): The best time to beat for the quantization process.
            mode (list, optional): A list containing mode settings for quantization. The first element is the mode type
                                   (e.g., "relu"), and the second element is the mode value (e.g., None). Defaults to ["relu", None].

        Returns:
            float: The benchmarked time for the autoquantization process.
        """
        if not _is_interpolate_mode(mode):
            return super()._autoquant_test(act_mat, weight, bias, best_time, mode)

        # SAM best is between .8 and 1, SDXL also performs best in this range
        INTERPOLATION_CONSTANT = mode[1]
        w_qtensor = cls.from_float(weight)
        x_vals_int8, x_scales = quantize_activation_per_token_absmax(
            act_mat.reshape(-1, act_mat.shape[-1])
        )
        quantized_matmul = (
            lambda x_vals_int8, x_scales, w_vals_int8:
                safe_int_mm(x_vals_int8, w_vals_int8) * x_scales
        )
        q_c_matmul=torch.compile(quantized_matmul, mode="max-autotune-no-cudagraphs")
        with torch.no_grad():
            res_matmul = do_autoquant_bench(q_c_matmul, x_vals_int8, x_scales.reshape(-1,1), w_qtensor.int_data)
        print(f">>time: {res_matmul:0.3f}ms for {cls} matmul, to_beat: {best_time:0.3f}ms")

        # if the (much faster) matmul kernel is already beat, don't bother benchmarking full op
        if res_matmul>=best_time:
            return res_matmul

        # calculate what time full op needs to beat for dynamic quant to be best given INTERPOLATION_CONSTANT
        to_beat = best_time + INTERPOLATION_CONSTANT/(1-INTERPOLATION_CONSTANT)*(best_time-res_matmul)
        res = super()._autoquant_test(act_mat, weight, bias, to_beat)
        max_int_const_win = (best_time-res_matmul)/(res-res_matmul)
        res_f = INTERPOLATION_CONSTANT*res+(1-INTERPOLATION_CONSTANT)*res_matmul
        print(f">>time: {res_f:0.3f}ms for {cls} interpolated, breakeven constant: {max_int_const_win:0.2f}")
        return res_f

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
        """
        Performs the quantized linear operations

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
        y = (act_mat*w_qtensor.int_data.unsqueeze(0)).sum(dim=-2)
        y = y.reshape(*orig_shape[:-1], y.shape[-1]) * w_qtensor.q_scales
        if bias is not None:
            y += bias
        return y.to(orig_dtype)

    @classmethod
    def _autoquant_test(cls, act_mat, *args):
        # if act_mat has batchsize>2 don't use this kernel
        if act_mat.reshape(-1, act_mat.shape[-1]).shape[0]>32:
            return torch.inf
        return super()._autoquant_test(act_mat, *args)

class AQWeightOnlyQuantizedLinearWeight3(Int8WeightOnlyQuantizedLinearWeight, AQMixin):
    """
    AutoQuantizable version of Int8WeightOnlyQuantizedLinearWeight that
    uses a different kernel
    """
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
    used by QTensor subclasses but for a default linear op instead. Result of from_float
    is not a tensor subclass, but rather the float tensor.
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
    # AQWeightOnlyQuantizedLinearWeight3,
    # TODO this gets picked in places where it makes perf worse, why?
]

def change_linears_to_autoquantizable(model, **kwargs):
    """
    Converts all linear weight tensors to the
    AutoQuantizableLinearWeight tensor subclass. Expectation is that this is followed
    by running the model and then calling change_autoquantizable_to_quantized
    """
    from torchao.quantization.quant_api import _is_linear
    filter_fn = kwargs.pop("filter_fn", _is_linear)
    _ = kwargs.pop("error_on_unseen", True) # same kwargs used for this and to_quantized
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
    Converts AutoQuantizableLinearWeight tensor subclasses
    to various quantized/non-quantized tensor subclasses depending
    on benchmark results. Expectation is that these modules are
    torch.compiled afterwards.
    """
    hold =  torch._dynamo.config.automatic_dynamic_shapes
    torch._dynamo.config.automatic_dynamic_shapes = False

    filter_fn = kwargs.pop(
        "filter_fn",
        lambda mod, *args:
            hasattr(mod, "weight") and isinstance(mod.weight, AutoQuantizableLinearWeight)
    )
    error_on_unseen=kwargs.pop("error_on_unseen", True)
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

# TODO: example_input seems weird to include in the API
# TODO: Document all the modes
# TODO: Mode being a list is weird, should be a string or some object
@torch.no_grad()
def autoquant(model, example_input=None, qtensor_class_list=DEFAULT_CLASS_LIST, filter_fn=None, mode=["interpolate", .85], **aq_kwargs):
    """
    Wraps the given model in an AutoQuantWrapper. If `example_input` is provided, performs a forward pass on the input.
    Otherwise, returns the wrapped model. The AutoQuantWrapper manages cases where the model is torch-compiled by first
    performing autoquantization on the original model and then allowing the torch.compile run/tracing to occur.

    Args:
        model (torch.nn.Module): The model to be autoquantized.
        example_input (Any, optional): An example input for the model. If provided, the function performs a forward pass
                                       on this input. Defaults to None.
        qtensor_class_list (list, optional): A list of tensor classes to be used for quantization. Defaults to DEFAULT_CLASS_LIST.
        filter_fn (callable, optional): A filter function to apply to the model parameters. Defaults to None.
        mode (list, optional): A list containing mode settings for quantization. The first element is the mode type (e.g., "interpolate"),
                               and the second element is the mode value (e.g., 0.85). Defaults to ["interpolate", .85].
        **aq_kwargs: Additional keyword arguments for the autoquantization process.

    Returns:
        torch.nn.Module: The autoquantized and wrapped model. If `example_input` is provided, the function performs a forward pass
                         on the input and returns the result of the forward pass.

    Example usage:
        torchao.autoquant(torch.compile(model))
        model(*example_input)
    """
    # the hook we will use to intercept the model forward and perform
    # autoquantization
    def autoquant_prehook(module, args, kwargs):
        module.forward_log_only(*args, **kwargs)
        change_autoquantizable_to_quantized(
            module,
            **aq_kwargs,
        )
        module.clean_up_autoquant_hooks_and_attrs()
        return args, kwargs

    # perform initial swap from linear weights
    # to AutoQuantizableLinearWeight
    change_linears_to_autoquantizable(
        model,
        filter_fn=filter_fn,
        qtensor_class_list=qtensor_class_list,
        mode=mode,
        **aq_kwargs
    )

    # access actual model of torch.compile wrapper if needed
    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        real_model = model._orig_mod
    else:
        real_model = model

    # we need a consistent way to run the model which bypasses both
    # A) the torch.compile tracing (so we need to run the inner model directly)
    # B) the autoquant_prehook we're about to register (so we call forward directly)
    model.forward_log_only = lambda *args, **kwargs: real_model.forward(*args, **kwargs)

    # the autoquant_prehook intercepts the forward call and performs autoquantization
    # and then deletes the hook. if model is a torch.compile wrapper, it then
    # does the tracing/compile since the prehook is naturally followed by the normal.
    # model run.
    handle = model.register_forward_pre_hook(autoquant_prehook, with_kwargs=True)

    # note the torch.compile wrapper eval_frame moved the assignment of any assigned
    # attributes to the inner model, so we have to call delattr on the inner model
    def clean_up_autoquant_hooks_and_attrs():
        try:
            handle.remove()
            delattr(real_model, "clean_up_autoquant_hooks_and_attrs")
            delattr(real_model, "forward_log_only")
        except:
            pass
    model.clean_up_autoquant_hooks_and_attrs = clean_up_autoquant_hooks_and_attrs

    # if example input was provided, check it and run it
    if isinstance(example_input, torch.Tensor):
        example_input = [example_input]
    if isinstance(example_input, (tuple, list)):
        model(*example_input)

    return model

class AutoQuantConfig:
    """Configuration that can be used to fully define how a model is quantized"""
    def __init__(self, start=None):
        self.config = {}
        if start is not None:
            if isinstance(start, torch.nn.Module):
                self.from_autoquantized_model(start)
            elif isinstance(start, str):
                self.load(start)

    def from_autoquantized_model(self, model):
        for name, child in model.named_modules():
            print(name, child)
            if isinstance(child, torch.nn.Linear):
                self.add_layer_to_config(child, name)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            import pickle
            pickle.dump(self.config, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            import pickle
            self.config = pickle.load(f)

    def add_layer_to_config(self, layer, fqn):
        if isinstance(layer.weight, QuantizedLinearWeightBase):
            self.config[fqn] = layer.weight.__class__

    def apply_to_model(self, model):
        from torchao.quantization.quant_api import _get_subclass_inserter
        for name, child in model.named_modules():
            if name in self.config:
                subclass_inserter = _get_subclass_inserter(self.config[name])
                child = subclass_inserter(child)
        return model
                
