import torch
import torchao
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
)
from .subclass import ( # noqa
    Int8DynamicallyQuantizedLinearWeight,
    Int8WeightOnlyQuantizedLinearWeight,
    QuantizedLinearWeightBase,
)
from torchao.dtypes import AffineQuantizedTensor, PlainLayout, TensorCoreTiledLayout, Float8Layout
from torchao.quantization.linear_activation_quantized_tensor import LinearActivationQuantizedTensor
from torch.utils._python_dispatch import return_and_correct_aliasing
from .granularity import (
    PerAxis,
    PerRow,
    PerTensor,
)
from .quant_primitives import safe_int_mm
from torchao.utils import TORCH_VERSION_AT_LEAST_2_3, TORCH_VERSION_AT_LEAST_2_5
from torchao.quantization.utils import quantize_activation_per_token_absmax
from torchao.float8.inference import Float8MMConfig
from torch.fx._symbolic_trace import Tracer
import copy

import torch.nn.functional as F

__all__ = [
    "AutoQuantizableLinearWeight",
    "autoquant",
    "DEFAULT_AUTOQUANT_CLASS_LIST",
    "DEFAULT_INT4_AUTOQUANT_CLASS_LIST",
    "OTHER_AUTOQUANT_CLASS_LIST",
]


aten = torch.ops.aten

AUTOQUANT_CACHE = {}
AUTOQUANT_CACHE2 = {}

class QuantizationTracer(Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return (
            (m.__module__.startswith("torch.nn") or m.__module__.startswith("torch.ao.nn"))
            and not isinstance(m, torch.nn.Sequential)
            # and not isinstance(m, torch.nn.MultiheadAttention)
        )

def check_cache(cls, shapes_and_dtype):
    return AUTOQUANT_CACHE.get((cls,)+shapes_and_dtype, None)

def update_cache(cls, shapes_and_dtype, res):
    AUTOQUANT_CACHE[(cls,)+shapes_and_dtype] = res

def check_cache2(original_weight_id):
    return AUTOQUANT_CACHE2.get(original_weight_id, None)

def update_cache2(original_weight_id, cls, res):
    AUTOQUANT_CACHE2[original_weight_id] = (cls, res)

def _remove_all_nodes_after(end_node, model):
    nodes = list(model.graph.nodes)
    for n in nodes[nodes.index(end_node)+1:]:
        if n.op == "output":
            n.args = (end_node,)
        else:
            for user in dict(n.users):
                user.replace_input_with(n, n.args[0])
            model.graph.erase_node(n)

    # model.graph.eliminate_dead_code()
    # model.recompile()
    return model

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
    def __new__(cls, weight, qtensor_class_list, *args, mode=["relu", None], model=None, fqn=None, example_inputs=None, **kwargs):
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

    def __init__(self, weight, qtensor_class_list, *args, mode=["relu", None], model=None, fqn=None, example_inputs=None, **kwargs):
        self.weight = weight
        self.qtensor_class_list = qtensor_class_list
        self.logged_data = {}
        self.mode = mode
        self.model = model
        self.fqn = fqn
        self.example_inputs = example_inputs

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

    def tune_autoquant(self, q_cls):
        # act_shape, w_shape, bias_shape, act_dtype = shapes_and_dtype
        with torch.no_grad():
            # act_mat = torch.randn(act_shape, dtype=act_dtype, device=self.device)

            print("fqn:", self.fqn)
            try:
                model = torch.fx.GraphModule(self.model, copy.deepcopy(self.model.graph))
                end_node = None
                for n in model.graph.nodes:
                    if n.op != "call_module":
                        continue
                    module = model.graph.owning_module.get_submodule(n.target)
                    node_fqn = n.target
                    if isinstance(module, torch.nn.Linear):
                        if node_fqn == self.fqn:
                            print("found end node:", n)
                            end_node = n
                            break

                if end_node is not None:
                    model = _remove_all_nodes_after(end_node, model)
                    # res = q_cls._autoquant_test(act_mat, self.weight, bias, best_time, self.mode)
                    linear_module = self.model.get_submodule(self.fqn)
                    cur_time = q_cls._autoquant_test2(model, linear_module, self.example_inputs, self.weight)
                    original_weight_id = id(self.weight)
                    prev_res = check_cache2(original_weight_id)
                    print(f">>time: {cur_time:0.3f}ms for {q_cls}, to_beat: {prev_res}")
                    if prev_res is None or cur_time < prev_res[1]:
                        update_cache2(original_weight_id, q_cls, cur_time)

                else:
                    res = torch.inf
                    print("end node None:", self.fqn)

            except Exception as e:
                print(f"warning: failed to autoquant {q_cls.__name__} due to {e}")
                res = torch.inf

    @torch.no_grad()
    def to_quantized(self, error_on_unseen, **kwargs):
        if error_on_unseen and self.logged_data == {}:
            raise RuntimeError("must run module normally to get shape, dtype info for autoquant")
        elif (self.logged_data == {}) and not error_on_unseen:
            # default back to non-quantized weight if not seen
            self = AQFloatLinearWeight.from_float(self.weight)
            return self


        # only want to print shape (at start) and final result (at end)
        # once per shape+quantization subclass combination.
        ran_new_benchmarks = False
        print_shape_once = True

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
            total_seen=0
            # shape_count = count_shapes(self, do_print=False)
            # for shapes_and_dtype, times_seen in self.logged_data.items():
            # only print shapes once
            # if print_shape_once == True:
            #     print_shape_once = False
            #     count_shapes(self, do_print=True)

            # time_for_best_shape = check_cache(best_cls, shapes_and_dtype)
            # time_for_best_shape = check_cache2((id(self.weight), q_cls))
            # time_for_best_shape = torch.inf if time_for_best_shape is None else time_for_best_shape
            self.tune_autoquant(q_cls)
            torch._dynamo.reset()
        bench_res = check_cache2(id(self.weight))
        if bench_res is None:
            self = AQFloatLinearWeight.from_float(self.weight)
            return self

        cls, best_time = bench_res
        print("cls, time:", cls, best_time)
        if best_time is not None:
            best_cls = cls
        else:
            best_cls = AQFloatLinearWeight
        # if no new benchmarking was done, don't print the final result, it will be the same as for another layer
        print(f"best_cls={best_cls}, best_time={best_time}\n")
        # TODO handle random cls args/kwargs? or should they be curried?
        self = best_cls.from_float(self.weight)
        return self

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.weight), self.qtensor_class_list, dtype=self.dtype, mode=self.mode, model=self.model, fqn=self.fqn, example_inputs=self.example_inputs
        )

    def __tensor_flatten__(self):
        return ["weight"], [self.qtensor_class_list, self.mode, self.model, self.fqn, self.example_inputs, self.dtype, self.shape]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        weight = tensor_data_dict["weight"]
        qtensor_class_list, mode, model, fqn, example_inputs, dtype, shape = tensor_attributes[0]
        return cls(weight, qtensor_class_list, mode, shape=shape if outer_size is None else outer_size, dtype=dtype, strides=outer_stride, fqn=fqn, model=model, example_inputs=example_inputs)

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
            # print("logging:", mat1.shape, w_autoquant.shape)
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
        if TORCH_VERSION_AT_LEAST_2_5:
            from torch._inductor.runtime.benchmarking import benchmarker
            res = benchmarker.benchmark_gpu(
                lambda: graph.replay(), warmup=warmup, rep=rep, return_mode="median"
            )
        elif TORCH_VERSION_AT_LEAST_2_3:
            from torch._inductor.runtime.runtime_utils import do_bench_gpu
            res = do_bench_gpu(lambda: graph.replay(), warmup=warmup, rep=rep, return_mode="median")
        else:
            from torch._inductor.utils import do_bench
            res = do_bench(lambda: graph.replay(), warmup=warmup, rep=rep, return_mode="median")
    return res

@torch.no_grad()
def do_autoquant_bench2(op, *args, **kwargs):
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
        if TORCH_VERSION_AT_LEAST_2_5:
            from torch._inductor.runtime.benchmarking import benchmarker
            res = benchmarker.benchmark_gpu(
                lambda: graph.replay(), warmup=warmup, rep=rep, return_mode="median"
            )
        elif TORCH_VERSION_AT_LEAST_2_3:
            from torch._inductor.runtime.runtime_utils import do_bench_gpu
            res = do_bench_gpu(lambda: graph.replay(), warmup=warmup, rep=rep, return_mode="median")
        else:
            from torch._inductor.utils import do_bench
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
            q_c_op = torch.compile(cls._quantized_linear_op, mode="max-autotune-no-cudagraphs")
        else:
            func = lambda a,b,c: F.relu(cls._quantized_linear_op(F.relu(a), b, c))
            q_c_op = torch.compile(func, mode="max-autotune-no-cudagraphs")
        res = do_autoquant_bench(q_c_op, act_mat, w_qtensor, bias, warmup=25, rep=100)
        if res < best_time*1.1:
            res2 = do_autoquant_bench(q_c_op, act_mat, w_qtensor, bias, warmup=25, rep=900)
            res=(res2*.9+res*.1)
        print(f">>time: {res:0.3f}ms for {cls}, to_beat: {best_time:0.3f}ms ")
        return res

    @classmethod
    def _autoquant_test2(cls, sub_model, linear_module, example_inputs, weight):
        # overwrite the weight
        orig_weight = linear_module.weight
        linear_module.weight = torch.nn.Parameter(cls.from_float(weight), requires_grad=False)
        torch._dynamo.reset()
        sub_model_compiled = torch.compile(sub_model, mode="max-autotune-no-cudagraphs")
        res = do_autoquant_bench2(sub_model_compiled, *example_inputs, warmup=5, rep=20)
        linear_module.weight = torch.nn.Parameter(orig_weight, requires_grad=False)
        return res



class AQInt8DynamicallyQuantizedLinearWeight(AQMixin, LinearActivationQuantizedTensor):
    """
    AutoQuantizable version of Int8DynamicallyQuantizedLinearWeight
    """
    @classmethod
    def from_float(cls, weight):
        # TODO test if this is valid
        # in_features = weight.shape[1]
        # int8 dynamic quantization only has benefit when in_feature > 16
        # if in_features <= 16:
            # return weight

        # avoid circular dep
        from torchao.dtypes import to_affine_quantized_intx
        # weight settings
        mapping_type = MappingType.SYMMETRIC
        def get_weight_block_size(x):
            return (1, x.shape[1])
        target_dtype = torch.int8
        eps = torch.finfo(torch.float32).eps
        zero_point_dtype = torch.int64

        # input settings
        def get_per_token_block_size(x):
            block_size = list(x.shape)
            for i in range(len(block_size)-1):
                block_size[i] = 1
            return block_size

        input_mapping_type = MappingType.SYMMETRIC
        input_target_dtype = torch.int8
        input_eps = 1e-5
        input_quant_min = -127
        input_quant_max = 127
        _layout = PlainLayout()
        input_quant_func = lambda x: to_affine_quantized_intx(x, input_mapping_type, get_per_token_block_size(x), input_target_dtype, eps=input_eps, quant_min=input_quant_min, quant_max=input_quant_max, scale_dtype=torch.float32 if x.dtype == torch.float16 else None)

        block_size = get_weight_block_size(weight)
        weight = to_affine_quantized_intx(weight, mapping_type, block_size, target_dtype, eps=eps, zero_point_dtype=zero_point_dtype, _layout=_layout)
        weight = super(AQInt8DynamicallyQuantizedLinearWeight, cls).from_float(weight, input_quant_func)
        return weight

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
            w_vals_int8 = w_qtensor.original_weight_tensor.tensor_impl.int_data.contiguous().t()
            res_matmul = do_autoquant_bench(q_c_matmul, x_vals_int8, x_scales.reshape(-1,1), w_vals_int8)
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

class AQInt8WeightOnlyQuantizedLinearWeight(AffineQuantizedTensor, AQMixin):
    """
    AutoQuantizable version of Int8WeightOnlyQuantizedLinearWeight
    """
    @classmethod
    def from_float(cls, weight):
        mapping_type = MappingType.SYMMETRIC
        target_dtype = torch.int8
        eps = torch.finfo(torch.float32).eps
        zero_point_dtype = torch.int64
        block_size = (1, weight.shape[1])
        return super(AQInt8WeightOnlyQuantizedLinearWeight, cls).from_hp_to_intx(weight, mapping_type, block_size, target_dtype, eps=eps, zero_point_dtype=zero_point_dtype)


class AQInt8WeightOnlyQuantizedLinearWeight2(AQInt8WeightOnlyQuantizedLinearWeight, AQMixin):
    """
    AutoQuantizable version of Int8WeightOnlyQuantizedLinearWeight that
    uses a different kernel
    """
    @staticmethod
    def _quantized_linear_op(act_mat, w_qtensor, bias):
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
        y = (act_mat*w_qtensor.tensor_impl.int_data.t().unsqueeze(0)).sum(dim=-2)
        y = y.reshape(*orig_shape[:-1], y.shape[-1]) * w_qtensor.tensor_impl.scale
        if bias is not None:
            y += bias
        return y.to(orig_dtype)

    @classmethod
    def _autoquant_test(cls, act_mat, *args):
        # if act_mat has batchsize>2 don't use this kernel
        if act_mat.reshape(-1, act_mat.shape[-1]).shape[0]>32:
            return torch.inf
        return super()._autoquant_test(act_mat, *args)

class AQInt8WeightOnlyQuantizedLinearWeight3(AQInt8WeightOnlyQuantizedLinearWeight, AQMixin):
    """
    AutoQuantizable version of Int8WeightOnlyQuantizedLinearWeight that
    uses a different kernel
    """
    @staticmethod
    def _quantized_linear_op(act_mat, w_qtensor, bias):
        orig_shape = act_mat.shape
        y = torch.mm(act_mat.reshape(-1, orig_shape[-1]), w_qtensor.tensor_impl.int_data.t()*w_qtensor.tensor_impl.scale)
        y=y.reshape(*orig_shape[:-1], y.shape[-1])
        if bias is not None:
            y += bias
        return y


class AQInt4G32WeightOnlyQuantizedLinearWeight(AffineQuantizedTensor, AQMixin):
    """
    AutoQuantizable version of Int4WeightOnlyQuantizedLinearWeight
    """
    group_size: int = 32
    @classmethod
    def from_float(cls, weight):
        group_size = cls.group_size
        _layout = TensorCoreTiledLayout(inner_k_tiles=8)

        if weight.shape[-1] % group_size != 0:
            return weight
        use_hqq = True
        mapping_type = MappingType.ASYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.int32
        quant_min = 0
        quant_max = 15
        eps = 1e-6
        preserve_zero = False
        zero_point_dtype = torch.bfloat16
        zero_point_domain = ZeroPointDomain.FLOAT
        return super(AQInt4G32WeightOnlyQuantizedLinearWeight, cls).from_hp_to_intx(weight, mapping_type, block_size, target_dtype, quant_min, quant_max, eps, zero_point_dtype=zero_point_dtype, preserve_zero=preserve_zero, zero_point_domain=zero_point_domain, _layout=_layout, use_hqq=use_hqq)

class AQInt4G64WeightOnlyQuantizedLinearWeight(AQInt4G32WeightOnlyQuantizedLinearWeight):
    group_size: int = 64

class AQInt4G128WeightOnlyQuantizedLinearWeight(AQInt4G32WeightOnlyQuantizedLinearWeight):
    group_size: int = 128

class AQInt4G256WeightOnlyQuantizedLinearWeight(AQInt4G32WeightOnlyQuantizedLinearWeight):
    group_size: int = 256

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
    def _quantized_linear_op(act_mat, w_qtensor, bias):
        return torch.nn.functional.linear(act_mat, w_qtensor, bias)

    @classmethod
    def from_float(cls, weight):
        return weight

class AQFloat8WeightOnlyQuantizedLinearWeight(AffineQuantizedTensor, AQMixin):
    """
    AutoQuantizable version of Float8WeightOnlyQuantizedLinearWeight for target_dtype=torch.float8_e4m3fn
    """
    target_dtype: torch.dtype = torch.float8_e4m3fn

    @staticmethod
    def _quantized_linear_op(act_mat, w_qtensor, bias):
        return torch.nn.functional.linear(act_mat, w_qtensor.dequantize(), bias)

    @classmethod
    def from_float(cls, weight):
        block_size = (1, weight.shape[1])
        return super(AQFloat8WeightOnlyQuantizedLinearWeight, cls).from_hp_to_floatx(weight, block_size, target_dtype=cls.target_dtype, _layout=Float8Layout())

class AQFloat8PerRowScalingDynamicallyQuantizedLinearWeight(AQMixin, LinearActivationQuantizedTensor):
    """
    AutoQuantizable version of Float8DynamicallyQuantizedLinearWeight using per row scaling
    """
    activation_granularity: str = PerRow()
    @classmethod
    def from_float(cls, weight):

        # avoid circular dep
        from torchao.dtypes import to_affine_quantized_floatx
        from torchao.quantization.quant_api import _input_activation_quant_func_fp8
        # weight settings
        def get_weight_block_size(x):
            return (1, x.shape[1])
        target_dtype = torch.float8_e4m3fn

        # input settings
        def get_per_token_block_size(x):
            block_size = list(x.shape)
            for i in range(len(block_size)-1):
                block_size[i] = 1
            return block_size

        input_target_dtype = torch.float8_e4m3fn
        _layout = Float8Layout(mm_config=Float8MMConfig(use_fast_accum=True))
        input_quant_func = lambda x: _input_activation_quant_func_fp8(
            x=x,
            activation_granularity=cls.activation_granularity,
            activation_dtype=input_target_dtype,
        )
        block_size = get_weight_block_size(weight)
        weight = to_affine_quantized_floatx(
                    input_float=weight,
                    block_size=block_size,
                    target_dtype=target_dtype,
                    _layout=_layout,
                    scale_dtype=torch.float32,
        )
        weight = super(AQFloat8PerRowScalingDynamicallyQuantizedLinearWeight, cls).from_float(weight, input_quant_func)
        return weight


# here we don't include int4 quantization in since int8 tends to be a better apples to apples comparison
DEFAULT_AUTOQUANT_CLASS_LIST = [
    AQFloatLinearWeight,
    AQInt8WeightOnlyQuantizedLinearWeight,
    AQInt8WeightOnlyQuantizedLinearWeight2,
    # AQInt8WeightOnlyQuantizedLinearWeight3,
    # TODO this gets picked in places where it makes perf worse, why?
    AQInt8DynamicallyQuantizedLinearWeight,
]

DEFAULT_INT4_AUTOQUANT_CLASS_LIST = [
    AQFloatLinearWeight,
    AQInt8DynamicallyQuantizedLinearWeight,
    AQInt4G64WeightOnlyQuantizedLinearWeight
]

OTHER_AUTOQUANT_CLASS_LIST = [
    AQFloat8WeightOnlyQuantizedLinearWeight,
    AQFloat8PerRowScalingDynamicallyQuantizedLinearWeight,
]

def _replace_with_custom_fn_if_matches_filter(
    model,
    replacement_fn,
    filter_fn,
    cur_fqn="",
    device=None,
) -> None:
    """
    Recursively replaces each child module in `model` with the result of `replacement_fn(child)`
    if `filter_fn(child)` returns `True`.

    Args:
        model (torch.nn.Module): The model containing modules to be replaced.
        replacement_fn (Callable[[torch.nn.Module], torch.nn.Module]): The function to replace matching modules.
        filter_fn (Callable[[torch.nn.Module], bool]): The filter function to determine which modules to replace.
        cur_fqn (str, optional): The current fully qualified name of the module being processed. Defaults to "".
        device (device, optional): Device to move the model to before applying `filter_fn`. Defaults to None.

    Returns:
        None
    """
    if filter_fn(model, cur_fqn[:-1]):
        if device is not None:
            model.to(device=device)  # move to device before quantization
        model = replacement_fn(model, cur_fqn[:-1])
        return model
    else:
        for name, child in model.named_children():
            new_child = _replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, f"{cur_fqn}{name}.", device
            )
            if new_child is not child:
                setattr(model, name, new_child)
        if device is not None:
            model.to(device=device)  # move parent module to device
        return model


from itertools import chain
def dict_union(*args):
    return dict(chain.from_iterable(d.items() for d in args))

def _change_linears_to_autoquantizable(model, example_input, **kwargs):
    """
    Converts all linear weight tensors to the
    AutoQuantizableLinearWeight tensor subclass. Expectation is that this is followed
    by running the model and then calling _change_autoquantizable_to_quantized
    """
    from torchao.quantization.quant_api import _is_linear
    filter_fn = kwargs.pop("filter_fn", _is_linear)
    _ = kwargs.pop("error_on_unseen", True) # same kwargs used for this and to_quantized
    kwargs["qtensor_class_list"] = kwargs.get("qtensor_class_list", DEFAULT_AUTOQUANT_CLASS_LIST)
    kwargs["mode"] = kwargs.get("mode", ["relu", None])
    tracer = QuantizationTracer()
    kwargs["model"] = torch.fx.GraphModule(model, tracer.trace(model))
    kwargs["example_inputs"] = example_input
    from torchao.quantization.quant_api import _get_subclass_inserter
    _replace_with_custom_fn_if_matches_filter(
        model,
        lambda model, fqn: _get_subclass_inserter(AutoQuantizableLinearWeight, **dict_union(kwargs, {"fqn": fqn}))(model),
        filter_fn if filter_fn is not None else _is_linear,
    )

def _change_autoquantizable_to_quantized(model, supress_autoquant_errors=True, **kwargs):
    """
    Converts AutoQuantizableLinearWeight tensor subclasses
    to various quantized/non-quantized tensor subclasses depending
    on benchmark results. Expectation is that these modules are
    torch.compiled afterwards.
    """
    hold_automatic_dynamic_shapes =  torch._dynamo.config.automatic_dynamic_shapes
    torch._dynamo.config.automatic_dynamic_shapes = False

    if supress_autoquant_errors:
        hold_supress_errors = torch._dynamo.config.suppress_errors
        torch._dynamo.config.suppress_errors = True
        import logging
        torch._logging.set_logs(inductor=logging.CRITICAL, dynamo=logging.CRITICAL)
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
    # undo dynamic shape change
    torch._dynamo.config.automatic_dynamic_shapes = hold_automatic_dynamic_shapes

    # undo error supression
    if supress_autoquant_errors:
        torch._dynamo.config.suppress_errors = hold_supress_errors
        torch._logging.set_logs()
    torch._dynamo.reset()

# TODO: example_input seems weird to include in the API
# TODO: Document all the modes
# TODO: Mode being a list is weird, should be a string or some object
@torch.no_grad()
def autoquant(
    model,
    example_input=None,
    qtensor_class_list=DEFAULT_AUTOQUANT_CLASS_LIST,
    filter_fn=None,
    mode=["interpolate", .85],
    manual=False,
    set_inductor_config=True,
    supress_autoquant_errors=True,
    **aq_kwargs
):
    """
    Autoquantization is a process which identifies the fastest way to quantize each layer of a model over some set of potential
    qtensor subclasses.

    Autoquantization happens in three steps:

    1-Prepare Model: the model is searched for Linear layers whose weights are exchanged for AutoQuantizableLinearWeight.
    2-Shape Calibration: the user runs the model on one or more inputs, the details of the activation shape/dtype seen by
        the AutoQuantizableLinearWeight are recorded so we know what shapes/dtypes to use in order to optimize the quantized op in step 3
    3-Finalize Autoquantization: for each AutoQuantizableLinearWeight, benchmarks are run for each shape/dtype on each member of the qtensor_class_list.
        the fastest option is picked, resulting in a highly performant model

    This autoquant function performs step 1. Steps 2 and 3 can be completed by simply running the model.
    If `example_input` is provided, this function also runs the model (which completes steps 2 and 3).
    This autoquant api can handle models which have already had torch.compile applied to them, in which case, once the model is run and quantized,
    the torch.compile process normally proceeds as well.

    To optimize over a combination of input shapes/dtypes, the user can set manual=True, run the model with all desired shapes/dtypes, then
    call model.finalize_autoquant to finalize the quantization once the desired set of inputs have been logged.

    Args:
        model (torch.nn.Module): The model to be autoquantized.
        example_input (Any, optional): An example input for the model. If provided, the function performs a forward pass
                                       on this input (which fully autoquantizes the model unless manual=True). Defaults to None.
        qtensor_class_list (list, optional): A list of tensor classes to be used for quantization. Defaults to DEFAULT_AUTOQUANT_CLASS_LIST.
        filter_fn (callable, optional): A filter function to apply to the model parameters. Defaults to None.
        mode (list, optional): A list containing mode settings for quantization. The first element is the mode type (e.g., "interpolate"),
                               and the second element is the mode value (e.g., 0.85). Defaults to ["interpolate", .85].
        manual (bool, optional): Whether to stop shape calibration and do autoquant after a single run (default, False) or to wait for
                                the user to call model.finalize_autoquant (True) so inputs with several shapes/dtypes can be logged.
        set_inductor_config (bool, optional): Whether to automatically use recommended inductor config settings (defaults to True)
        supress_autoquant_errors (bool, optional): Whether to suppress errors during autoquantization. (defaults to True)
        **aq_kwargs: Additional keyword arguments for the autoquantization process.

    Returns:
        torch.nn.Module: The autoquantized and wrapped model. If `example_input` is provided, the function performs a forward pass
                         on the input and returns the result of the forward pass.

    Example usage:
        torchao.autoquant(torch.compile(model))
        model(*example_input)

        # multiple input shapes
        torchao.autoquant(model, manual=True)
        model(*example_input1)
        model(*example_input2)
        model.finalize_autoquant()
    """
    if set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    if qtensor_class_list is OTHER_AUTOQUANT_CLASS_LIST:
        assert torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9), "float8 requires CUDA arch >= 8.9"

    # perform initial swap from linear weights
    # to AutoQuantizableLinearWeight
    assert example_input is not None
    _change_linears_to_autoquantizable(
        model,
        example_input,
        filter_fn=filter_fn,
        qtensor_class_list=qtensor_class_list,
        mode=mode,
        **aq_kwargs
    )

    # access actual model of torch.compile wrapper if needed
    is_compiled = isinstance(model, torch._dynamo.eval_frame.OptimizedModule)
    if is_compiled:
        real_model = model._orig_mod
    else:
        real_model = model

    if manual:
        # we don't want model.forward to trigger
        # torch.compilation
        if is_compiled:
            real_model.old_forward = model.forward
            model.forward = real_model.forward

    # we want to automatically do autoquant after a single model run
    # and have it occur before torch.compilation if applicable
    else:
        # the hook we will use to intercept the model forward and perform
        # autoquantization
        def autoquant_prehook(module, args, kwargs):
            real_model.forward(*args, **kwargs)
            module.finalize_autoquant()
            return args, kwargs

        # the autoquant_prehook intercepts the forward call, performs logging then
        # does autoquantization. if model is a torch.compile wrapper, it then
        # does the tracing/compile since the prehook is naturally followed by the normal.
        # model run.
        handle = model.register_forward_pre_hook(autoquant_prehook, with_kwargs=True)

    # note the torch.compile wrapper (eval_frame) moves the assignment of any assigned
    # attributes to the inner model that didn't exist before, so we have to call delattr on the inner model
    def finalize_autoquant():
        _change_autoquantizable_to_quantized(
            real_model,
            supress_autoquant_errors,
            **aq_kwargs,
        )
        if hasattr(real_model, "old_forward"):
            model.forward = real_model.old_forward
            delattr(real_model, "old_forward")
        if hasattr(real_model, "finalize_autoquant"):
            delattr(real_model, "finalize_autoquant")
        if not manual:
            handle.remove()

    real_model.finalize_autoquant = finalize_autoquant

    # if example input was provided, check it and run it
    if isinstance(example_input, torch.Tensor):
        example_input = [example_input]
    if isinstance(example_input, (tuple, list)):
        model(*example_input)

    return model
