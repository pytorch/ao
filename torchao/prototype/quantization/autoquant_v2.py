import copy
import csv
import logging
import os
import re
from itertools import chain

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import return_and_correct_aliasing
from torch.utils._pytree import tree_map

import torchao
from torchao.dtypes import (
    AffineQuantizedTensor,
    Float8Layout,
    PlainLayout,
    TensorCoreTiledLayout,
)
from torchao.float8.inference import Float8MMConfig
from torchao.kernel import safe_int_mm
from torchao.prototype.quantization.subgraph_utils.extract_subgraphs import (
    debug_linears_for_float8,
    prepare_target_folder,
)
from torchao.quantization import LinearActivationQuantizedTensor
from torchao.quantization.autoquant import (
    AutoQuantizableLinearWeight as AutoQuantizableLinearWeightV1,
)
from torchao.quantization.granularity import (
    PerRow,
    PerTensor,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
)
from torchao.quantization.subclass import (  # noqa
    Int8DynamicallyQuantizedLinearWeight,
    Int8WeightOnlyQuantizedLinearWeight,
    QuantizedLinearWeightBase,
)
from torchao.quantization.utils import quantize_activation_per_token_absmax
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_3,
    TORCH_VERSION_AT_LEAST_2_5,
    TorchAOBaseTensor,
    is_sm_at_least_89,
    is_sm_at_least_90,
)

logging.basicConfig(level=logging.ERROR)  # Set the root logger level to ERROR


target_folder = "/home/jerryzh/local/tmp/20241104_dynamo_test"

__all__ = [
    "AutoQuantizableLinearWeight",
    "autoquant_v2",
    "DEFAULT_AUTOQUANT_CLASS_LIST",
    "DEFAULT_INT4_AUTOQUANT_CLASS_LIST",
    "DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST",
    "OTHER_AUTOQUANT_CLASS_LIST",
    "ALL_AUTOQUANT_CLASS_LIST",
    "_is_linear",
]


def _is_linear(mod, *args):
    # avoid circular dependencies
    from torchao.quantization.qat.affine_fake_quantized_tensor import (
        AffineFakeQuantizedTensor,
    )

    # adding weight tensor subclass isinstance check to make sure the weight is only quantized once
    # when it is shared by multiple linear modules
    return (
        isinstance(mod, torch.nn.Linear)
        and hasattr(mod, "weight")
        and not isinstance(mod.weight, QuantizedLinearWeightBase)
        and not isinstance(mod.weight, AutoQuantizableLinearWeightV1)
        and not isinstance(mod.weight, AffineQuantizedTensor)
        and not isinstance(mod.weight, LinearActivationQuantizedTensor)
        and not isinstance(mod.weight, AffineFakeQuantizedTensor)
        and not isinstance(mod, torch.nn.modules.linear.NonDynamicallyQuantizableLinear)
    )


# TODO: use SubgraphMatcher
def _graph_equals(g1, g2):
    if len(g1.nodes) != len(g2.nodes):
        return False

    for n1, n2 in zip(g1.nodes, g2.nodes):
        if n1.op != n2.op:
            return False

        if n1.op in ["call_function", "call_method"] and n1.target != n2.target:
            return False

        if len(n1.args) != len(n2.args):
            return False
    return True


aten = torch.ops.aten

AUTOQUANT_CACHE = {}

# This is a flag to control whether we do some rewrite for graph
# to account for different batch sizes, it's a temporary solution for llama model
# we'll need to think about how to support this more generally
LLAMA = True


def check_cache(gm, cls, shapes_and_dtype):
    for gm_, cls_, shapes_and_dtype_ in AUTOQUANT_CACHE.keys():
        graph_equals = _graph_equals(gm_.graph, gm.graph)
        if graph_equals and cls_ is cls and shapes_and_dtype_ == shapes_and_dtype:
            return AUTOQUANT_CACHE[(gm_, cls_, shapes_and_dtype_)]
    return None


def update_cache(gm, cls, shapes_and_dtype, res):
    AUTOQUANT_CACHE[(gm, cls, shapes_and_dtype)] = res


# adjust each input's bsz to target_bsz
# enable grad
# a hacky solution but should work in the use cases we are testing now
# we went through the list of sizes and swap the dimension that matches extracted_bsz to target_bsz
def resize_input(t, extracted_bsz, target_bsz):
    if len(t.shape) > 1:
        new_shape = []
        for i in range(len(t.size())):
            if t.size(i) == extracted_bsz:
                new_shape.append(target_bsz)
            else:
                new_shape.append(t.size(i))
        t = torch.randn(*new_shape, dtype=t.dtype, device=t.device)
    return t


# a hacky solution but should work in the use cases we are testing now
# we went through the list of sizes and swap the dimension that matches extracted_bsz to target_bsz
def maybe_adjust_model_bsz(m, extracted_bsz, target_bsz):
    """
    Makes guesses on how to adjust the model graph to account for the
    fact that we changed the batch size. Note: this is very brittle
    """
    for n in m.graph.nodes:
        if n.op == "call_method" and n.target == "view":
            new_args = []
            for arg in n.args:
                if arg == extracted_bsz:
                    new_args.append(target_bsz)
                else:
                    new_args.append(arg)
            n.args = tuple(new_args)

    m.recompile()


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
    def __new__(
        cls,
        weight,
        qtensor_class_list,
        *args,
        mode=["relu", None],
        model=None,
        fqn=None,
        example_inputs=None,
        fqn_to_submodule=None,
        batch_size=None,
        **kwargs,
    ):
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

    def __init__(
        self,
        weight,
        qtensor_class_list,
        *args,
        mode=["relu", None],
        model=None,
        fqn=None,
        example_inputs=None,
        fqn_to_submodule=None,
        batch_size=None,
        **kwargs,
    ):
        self.weight = weight
        self.qtensor_class_list = qtensor_class_list
        self.logged_data = {}
        self.mode = mode
        self.model = model
        self.fqn = fqn
        self.example_inputs = example_inputs
        self.fqn_to_submodule = fqn_to_submodule
        self.batch_size = batch_size

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.weight}, shape={self.shape}, "
            f"device={self.device}, dtype={self.dtype}, qtensor_class_list={self.qtensor_class_list})"
        )

    @staticmethod
    def log_shape(act_mat, w_autoquant, bias):
        act_mat = act_mat.reshape(-1, act_mat.shape[-1])
        logged_dtype = act_mat.dtype
        logged_shapes = (
            act_mat.shape,
            w_autoquant.shape,
            None if bias is None else bias.shape,
        )
        shapes_and_dtype = logged_shapes + (logged_dtype,)
        w_autoquant.logged_data[shapes_and_dtype] = 1 + w_autoquant.logged_data.get(
            shapes_and_dtype, 0
        )

    def tune_autoquant2(
        self, fqn, m, batch_size, inputs, q_cls, shapes_and_dtype, time_for_best_shape
    ):
        act_shape, w_shape, bias_shape, act_dtype = shapes_and_dtype

        with torch.no_grad():
            try:
                m_copy = copy.deepcopy(m)
                for name, module in m_copy.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        linear_module = module
                weight = q_cls.from_float(linear_module.weight)
                linear_module.weight = torch.nn.Parameter(weight, requires_grad=False)
                if batch_size is not None:
                    extracted_bsz = batch_size
                    target_bsz = act_shape[0]
                    inputs = tree_map(
                        lambda t: resize_input(t, extracted_bsz, target_bsz), inputs
                    )
                    maybe_adjust_model_bsz(m_copy, extracted_bsz, target_bsz)

                m_copy = torch.compile(m_copy, mode="max-autotune-no-cudagraphs")

                if isinstance(inputs, (list, tuple)):
                    cur_time = do_autoquant_bench(m_copy, *inputs, warmup=25, rep=100)
                else:
                    cur_time = do_autoquant_bench(m_copy, **inputs, warmup=25, rep=100)
                print(
                    f">>time: {cur_time:0.3f}ms for {q_cls}, to_beat: {time_for_best_shape}"
                )
                if cur_time < time_for_best_shape:
                    update_cache(m, q_cls, shapes_and_dtype, cur_time)
                res = cur_time
                return res
            except Exception as e:
                print(f"warning: failed to autoquant {q_cls.__name__} due to {e}")
                return None

    @torch.no_grad()
    def to_quantized(self, error_on_unseen, **kwargs):
        if error_on_unseen and self.logged_data == {}:
            raise RuntimeError(
                "must run module normally to get shape, dtype info for autoquant"
            )
        elif (self.logged_data == {}) and not error_on_unseen:
            # default back to non-quantized weight if not seen
            self = AQDefaultLinearWeight.from_float(self.weight)
            return self

        # only want to print shape (at start) and final result (at end)
        # once per shape+quantization subclass combination.
        ran_new_benchmarks = False
        print_shape_once = True

        def count_shapes(self, do_print=True):
            differe_shape_count = 0
            for shapes_and_dtype, times_seen in self.logged_data.items():
                differe_shape_count += 1
                if do_print:
                    act_shape, weight_shape, bias_shape, dtype = shapes_and_dtype
                    print(f"activation_shapes: {act_shape}, times_seen: {times_seen}")
            if do_print:
                print(
                    f"weight_shape: {weight_shape}, dtype: {dtype}, bias_shape: {bias_shape}"
                )
            return differe_shape_count

        # check each class
        best_time = torch.inf
        best_cls = None
        fqn = self.fqn
        print(f"autoquant for {fqn}")
        for q_cls in self.qtensor_class_list:
            # for each logged shape+dtype, benchmark
            cur_time = 0
            total_seen = 0
            shape_count = count_shapes(self, do_print=False)
            # copied from https://github.com/pytorch/pytorch/blob/75eeefbfab3862abe887e1d85a0b1b18c227d9f3/torch/_dynamo/variables/builder.py#L963
            modified_fqn = "L__self___" + re.sub(r"[^a-zA-Z0-9]+", "_", fqn)
            m, inputs = self.fqn_to_submodule[modified_fqn]
            for shapes_and_dtype, times_seen in self.logged_data.items():
                if check_cache(m, q_cls, shapes_and_dtype) is None:
                    # only print shapes once
                    if print_shape_once is True:
                        print_shape_once = False
                        count_shapes(self, do_print=True)

                    time_for_best_shape = check_cache(m, q_cls, shapes_and_dtype)
                    time_for_best_shape = (
                        torch.inf
                        if time_for_best_shape is None
                        else time_for_best_shape
                    )
                    self.tune_autoquant2(
                        fqn,
                        m,
                        self.batch_size,
                        inputs,
                        q_cls,
                        shapes_and_dtype,
                        time_for_best_shape,
                    )
                    ran_new_benchmarks = True
                    torch._dynamo.reset()
                if check_cache(m, q_cls, shapes_and_dtype) is not None:
                    cur_time += check_cache(m, q_cls, shapes_and_dtype) * times_seen
                    total_seen += times_seen

            if total_seen != 0:
                cur_time = cur_time / total_seen

                # print aggregated time if there were multiple shapes to aggregate and some new benchmarking was done
                if shape_count is not None and shape_count > 1 and ran_new_benchmarks:
                    print(
                        f">time (all shapes): {cur_time:0.4f}ms for {q_cls}, prev_best: {best_time:0.4f}ms"
                    )
                if best_time >= cur_time:
                    best_time = cur_time
                    best_cls = q_cls
        # if no new benchmarking was done, don't print the final result, it will be the same as for another layer
        if ran_new_benchmarks:
            print(f"best_cls={best_cls}\n")
        # TODO handle random cls args/kwargs? or should they be curried?
        if best_cls is None:
            best_cls = AQDefaultLinearWeight

        self = best_cls.from_float(self.weight)
        return self

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.weight),
            self.qtensor_class_list,
            dtype=self.dtype,
            mode=self.mode,
            model=self.model,
            fqn=self.fqn,
            example_inputs=self.example_inputs,
            fqn_to_submodule=self.fqn_to_submodule,
            batch_size=self.batch_size,
        )

    def __tensor_flatten__(self):
        return ["weight"], [
            self.qtensor_class_list,
            self.mode,
            self.model,
            self.fqn,
            self.example_inputs,
            self.fqn_to_submodule,
            self.batch_size,
            self.dtype,
            self.shape,
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None
    ):
        weight = tensor_data_dict["weight"]
        (
            qtensor_class_list,
            mode,
            model,
            fqn,
            example_inputs,
            fqn_to_submodule,
            batch_size,
            dtype,
            shape,
        ) = tensor_attributes
        return cls(
            weight,
            qtensor_class_list,
            mode,
            model=model,
            fqn=fqn,
            example_inputs=example_inputs,
            fqn_to_submodule=fqn_to_submodule,
            batch_size=batch_size,
            shape=shape if outer_size is None else outer_size,
            dtype=dtype,
            strides=outer_stride,
        )

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
                args[2] if len(args) > 2 else None,
            )
            cls.log_shape(mat1, w_autoquant, bias)
            return func(mat1, w_autoquant.weight, bias)
        try:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        except Exception:
            print(f"ERR: subclass doesn't implement {func}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )


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

            res = do_bench_gpu(
                lambda: graph.replay(), warmup=warmup, rep=rep, return_mode="median"
            )
        else:
            from torch._inductor.utils import do_bench

            res = do_bench(
                lambda: graph.replay(), warmup=warmup, rep=rep, return_mode="median"
            )
    return res


def _is_interpolate_mode(mode):
    if (
        isinstance(mode, list)
        and mode[0] == "interpolate"
        and len(mode) == 2
        and isinstance(mode[1], float)
    ):
        return True
    return False


class AQMixin:
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
            q_c_op = torch.compile(
                cls._quantized_linear_op, mode="max-autotune-no-cudagraphs"
            )
        else:
            func = lambda a, b, c: F.relu(cls._quantized_linear_op(F.relu(a), b, c))
            q_c_op = torch.compile(func, mode="max-autotune-no-cudagraphs")
        res = do_autoquant_bench(q_c_op, act_mat, w_qtensor, bias, warmup=25, rep=100)
        if res < best_time * 1.1:
            res2 = do_autoquant_bench(
                q_c_op, act_mat, w_qtensor, bias, warmup=25, rep=900
            )
            res = res2 * 0.9 + res * 0.1
        print(f">>time: {res:0.3f}ms for {cls}, to_beat: {best_time:0.3f}ms ")
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
            for i in range(len(block_size) - 1):
                block_size[i] = 1
            return block_size

        input_mapping_type = MappingType.SYMMETRIC
        input_target_dtype = torch.int8
        input_eps = 1e-5
        input_quant_min = -127
        input_quant_max = 127
        _layout = PlainLayout()
        input_quant_func = lambda x: to_affine_quantized_intx(
            x,
            input_mapping_type,
            get_per_token_block_size(x),
            input_target_dtype,
            eps=input_eps,
            quant_min=input_quant_min,
            quant_max=input_quant_max,
            scale_dtype=torch.float32 if x.dtype == torch.float16 else None,
        )

        block_size = get_weight_block_size(weight)
        weight = to_affine_quantized_intx(
            weight,
            mapping_type,
            block_size,
            target_dtype,
            eps=eps,
            zero_point_dtype=zero_point_dtype,
            _layout=_layout,
        )
        weight = super(AQInt8DynamicallyQuantizedLinearWeight, cls).from_float(
            weight, input_quant_func
        )
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
            lambda x_vals_int8, x_scales, w_vals_int8: safe_int_mm(
                x_vals_int8, w_vals_int8
            )
            * x_scales
        )
        q_c_matmul = torch.compile(quantized_matmul, mode="max-autotune-no-cudagraphs")
        with torch.no_grad():
            w_vals_int8 = (
                w_qtensor.original_weight_tensor.tensor_impl.int_data.contiguous().t()
            )
            res_matmul = do_autoquant_bench(
                q_c_matmul, x_vals_int8, x_scales.reshape(-1, 1), w_vals_int8
            )
        print(
            f">>time: {res_matmul:0.3f}ms for {cls} matmul, to_beat: {best_time:0.3f}ms"
        )

        # if the (much faster) matmul kernel is already beat, don't bother benchmarking full op
        if res_matmul >= best_time:
            return res_matmul

        # calculate what time full op needs to beat for dynamic quant to be best given INTERPOLATION_CONSTANT
        to_beat = best_time + INTERPOLATION_CONSTANT / (1 - INTERPOLATION_CONSTANT) * (
            best_time - res_matmul
        )
        res = super()._autoquant_test(act_mat, weight, bias, to_beat)
        max_int_const_win = (best_time - res_matmul) / (res - res_matmul)
        res_f = INTERPOLATION_CONSTANT * res + (1 - INTERPOLATION_CONSTANT) * res_matmul
        print(
            f">>time: {res_f:0.3f}ms for {cls} interpolated, breakeven constant: {max_int_const_win:0.2f}"
        )
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
        return super(AQInt8WeightOnlyQuantizedLinearWeight, cls).from_hp_to_intx(
            weight,
            mapping_type,
            block_size,
            target_dtype,
            eps=eps,
            zero_point_dtype=zero_point_dtype,
        )


class AQInt8WeightOnlyQuantizedLinearWeight2(
    AQInt8WeightOnlyQuantizedLinearWeight, AQMixin
):
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
        y = (act_mat * w_qtensor.tensor_impl.int_data.t().unsqueeze(0)).sum(dim=-2)
        y = y.reshape(*orig_shape[:-1], y.shape[-1]) * w_qtensor.tensor_impl.scale
        if bias is not None:
            y += bias
        return y.to(orig_dtype)

    @classmethod
    def _autoquant_test(cls, act_mat, *args):
        # if act_mat has batchsize>2 don't use this kernel
        if act_mat.reshape(-1, act_mat.shape[-1]).shape[0] > 32:
            return torch.inf
        return super()._autoquant_test(act_mat, *args)


class AQInt8WeightOnlyQuantizedLinearWeight3(
    AQInt8WeightOnlyQuantizedLinearWeight, AQMixin
):
    """
    AutoQuantizable version of Int8WeightOnlyQuantizedLinearWeight that
    uses a different kernel
    """

    @staticmethod
    def _quantized_linear_op(act_mat, w_qtensor, bias):
        orig_shape = act_mat.shape
        y = torch.mm(
            act_mat.reshape(-1, orig_shape[-1]),
            w_qtensor.tensor_impl.int_data.t() * w_qtensor.tensor_impl.scale,
        )
        y = y.reshape(*orig_shape[:-1], y.shape[-1])
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
        return super(AQInt4G32WeightOnlyQuantizedLinearWeight, cls).from_hp_to_intx(
            weight,
            mapping_type,
            block_size,
            target_dtype,
            quant_min,
            quant_max,
            eps,
            zero_point_dtype=zero_point_dtype,
            preserve_zero=preserve_zero,
            zero_point_domain=zero_point_domain,
            _layout=_layout,
            use_hqq=use_hqq,
        )


class AQInt4G64WeightOnlyQuantizedLinearWeight(
    AQInt4G32WeightOnlyQuantizedLinearWeight
):
    group_size: int = 64


class AQInt4G128WeightOnlyQuantizedLinearWeight(
    AQInt4G32WeightOnlyQuantizedLinearWeight
):
    group_size: int = 128


class AQInt4G256WeightOnlyQuantizedLinearWeight(
    AQInt4G32WeightOnlyQuantizedLinearWeight
):
    group_size: int = 256


class AQDefaultLinearWeight(torch.Tensor, AQMixin):
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


class Float32Tensor(TorchAOBaseTensor):
    """Tensor subclass tensor for fp32 dtype"""

    def __init__(self, weight):
        self.weight = weight.to(torch.float32)

    @staticmethod
    def _quantized_linear_op(act_mat, w_qtensor, bias):
        _DTYPE = torch.float32
        orig_dtype = act_mat.dtype
        return torch.nn.functional.linear(
            act_mat.to(_DTYPE),
            w_qtensor.weight,
            bias.to(_DTYPE) if bias is not None else bias,
        ).to(dtype=orig_dtype)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.weight),
        )

    @classmethod
    def from_float(cls, weight):
        return cls(weight)


@Float32Tensor.implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    return weight_tensor._quantized_linear_op(input_tensor, weight_tensor, bias)


@Float32Tensor.implements(aten.detach.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )


@Float32Tensor.implements(aten.clone.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )


@Float32Tensor.implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].to(*args[1:], **kwargs)._apply_fn_to_data(torch.clone),
    )


class BFloat16Tensor(Float32Tensor):
    def __init__(self, weight):
        self.weight = weight.to(torch.bfloat16)

    @staticmethod
    def _quantized_linear_op(act_mat, w_qtensor, bias):
        _DTYPE = torch.bfloat16
        orig_dtype = act_mat.dtype
        return torch.nn.functional.linear(
            act_mat.to(_DTYPE),
            w_qtensor.weight,
            bias.to(_DTYPE) if bias is not None else bias,
        ).to(dtype=orig_dtype)


class Float16Tensor(Float32Tensor):
    def __init__(self, weight):
        self.weight = weight.to(torch.float16)

    @staticmethod
    def _quantized_linear_op(act_mat, w_qtensor, bias):
        _DTYPE = torch.float16
        orig_dtype = act_mat.dtype
        return torch.nn.functional.linear(
            act_mat.to(_DTYPE),
            w_qtensor.weight,
            bias.to(_DTYPE) if bias is not None else bias,
        ).to(dtype=orig_dtype)


class AQFloat32LinearWeight(Float32Tensor, AQMixin):
    """
    AutoQuantizable version for float32 precision weight

    (also converts input activation and bias to float32, and restores the original precision after
    linear)
    """

    @classmethod
    def from_float(cls, weight):
        return super(AQFloat32LinearWeight, cls).from_float(weight)


class AQBFloat16LinearWeight(BFloat16Tensor, AQMixin):
    """
    AutoQuantizable version for bfloat16 precision weight

    (also converts input activation and bias to bfloat16, and restores the original precision after
    linear)
    """

    @classmethod
    def from_float(cls, weight):
        return super(AQBFloat16LinearWeight, cls).from_float(weight)


class AQFloat16LinearWeight(Float16Tensor, AQMixin):
    """
    AutoQuantizable version for float16 precision weight

    (also converts input activation and bias to float16, and restores the original precision after
    linear)
    """

    @classmethod
    def from_float(cls, weight):
        return super(AQFloat16LinearWeight, cls).from_float(weight)


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
        return super(AQFloat8WeightOnlyQuantizedLinearWeight, cls).from_hp_to_floatx(
            weight, block_size, target_dtype=cls.target_dtype, _layout=Float8Layout()
        )


class AQFloat8PerRowScalingDynamicallyQuantizedLinearWeight(
    AQMixin, LinearActivationQuantizedTensor
):
    """
    AutoQuantizable version of Float8DynamicallyQuantizedLinearWeight using per row scaling
    """

    activation_granularity = PerRow()

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
            for i in range(len(block_size) - 1):
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
        weight = super(
            AQFloat8PerRowScalingDynamicallyQuantizedLinearWeight, cls
        ).from_float(weight, input_quant_func)
        return weight


class AQFloat8PerTensorScalingDynamicallyQuantizedLinearWeight(
    AQMixin, LinearActivationQuantizedTensor
):
    """
    AutoQuantizable version of Float8DynamicallyQuantizedLinearWeight using per tensor scaling
    """

    activation_granularity = PerTensor()

    @classmethod
    def from_float(cls, weight):
        # avoid circular dep
        from torchao.dtypes import to_affine_quantized_floatx
        from torchao.quantization.quant_api import _input_activation_quant_func_fp8

        # weight settings
        def get_weight_block_size(x):
            assert x.ndim == 2, "Only works for 2D tensors"
            return x.shape

        target_dtype = torch.float8_e4m3fn

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
        weight = super(
            AQFloat8PerTensorScalingDynamicallyQuantizedLinearWeight, cls
        ).from_float(weight, input_quant_func)
        return weight


# here we don't include int4 quantization in since int8 tends to be a better apples to apples comparison
DEFAULT_AUTOQUANT_CLASS_LIST = [
    AQDefaultLinearWeight,
    AQInt8WeightOnlyQuantizedLinearWeight,
    AQInt8WeightOnlyQuantizedLinearWeight2,
    # AQInt8WeightOnlyQuantizedLinearWeight3,
    # TODO this gets picked in places where it makes perf worse, why?
    AQInt8DynamicallyQuantizedLinearWeight,
]

DEFAULT_INT4_AUTOQUANT_CLASS_LIST = [
    AQDefaultLinearWeight,
    AQInt8DynamicallyQuantizedLinearWeight,
    AQInt4G64WeightOnlyQuantizedLinearWeight,
]

DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST = [
    AQFloat32LinearWeight,
    AQBFloat16LinearWeight,
    AQFloat16LinearWeight,
]

OTHER_AUTOQUANT_CLASS_LIST = [
    AQFloat8WeightOnlyQuantizedLinearWeight,
    AQFloat8PerRowScalingDynamicallyQuantizedLinearWeight,
    AQFloat8PerTensorScalingDynamicallyQuantizedLinearWeight,
]

ALL_AUTOQUANT_CLASS_LIST = list(
    set(
        DEFAULT_AUTOQUANT_CLASS_LIST
        + DEFAULT_INT4_AUTOQUANT_CLASS_LIST
        + DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST
    )
)
if is_sm_at_least_89():
    ALL_AUTOQUANT_CLASS_LIST += [
        AQFloat8WeightOnlyQuantizedLinearWeight,
        AQFloat8PerTensorScalingDynamicallyQuantizedLinearWeight,
    ]

if is_sm_at_least_90():
    ALL_AUTOQUANT_CLASS_LIST += [AQFloat8PerRowScalingDynamicallyQuantizedLinearWeight]


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


def dict_union(*args):
    return dict(chain.from_iterable(d.items() for d in args))


def _change_linears_to_autoquantizable(
    model, example_input, fqn_to_submodule, batch_size, **kwargs
):
    """
    Converts all linear weight tensors to the
    AutoQuantizableLinearWeight tensor subclass. Expectation is that this is followed
    by running the model and then calling _change_autoquantizable_to_quantized
    """
    # from torchao.quantization.quant_api import _is_linear

    filter_fn = kwargs.pop("filter_fn", _is_linear)
    _ = kwargs.pop(
        "error_on_unseen", True
    )  # same kwargs used for this and to_quantized
    kwargs["qtensor_class_list"] = kwargs.get(
        "qtensor_class_list", DEFAULT_AUTOQUANT_CLASS_LIST
    )
    kwargs["mode"] = kwargs.get("mode", ["relu", None])
    kwargs["model"] = model
    kwargs["example_inputs"] = example_input
    kwargs["fqn_to_submodule"] = fqn_to_submodule
    kwargs["batch_size"] = batch_size
    from torchao.quantization.quant_api import _get_subclass_inserter

    _replace_with_custom_fn_if_matches_filter(
        model,
        lambda model, fqn: _get_subclass_inserter(
            AutoQuantizableLinearWeight, **dict_union(kwargs, {"fqn": fqn})
        )(model),
        filter_fn if filter_fn is not None else _is_linear,
    )


def _change_autoquantizable_to_quantized(
    model, supress_autoquant_errors=True, **kwargs
):
    """
    Converts AutoQuantizableLinearWeight tensor subclasses
    to various quantized/non-quantized tensor subclasses depending
    on benchmark results. Expectation is that these modules are
    torch.compiled afterwards.
    """
    hold_automatic_dynamic_shapes = torch._dynamo.config.automatic_dynamic_shapes
    torch._dynamo.config.automatic_dynamic_shapes = False

    if supress_autoquant_errors:
        hold_supress_errors = torch._dynamo.config.suppress_errors
        torch._dynamo.config.suppress_errors = True
        import logging

        torch._logging.set_logs(inductor=logging.CRITICAL, dynamo=logging.CRITICAL)
    filter_fn = kwargs.pop(
        "filter_fn",
        lambda mod, *args: hasattr(mod, "weight")
        and isinstance(mod.weight, AutoQuantizableLinearWeight),
    )
    error_on_unseen = kwargs.pop("error_on_unseen", True)
    from torchao.quantization.quant_api import (
        _get_subclass_inserter,
        _replace_with_custom_fn_if_matches_filter,
    )

    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(
            AutoQuantizableLinearWeight,
            method="to_quantized",
            error_on_unseen=error_on_unseen,
            **kwargs,
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
def autoquant_v2(
    model,
    example_input=None,
    qtensor_class_list=DEFAULT_AUTOQUANT_CLASS_LIST,
    filter_fn=None,
    mode=["interpolate", 0.85],
    manual=False,
    set_inductor_config=True,
    supress_autoquant_errors=True,
    batch_size=None,
    **aq_kwargs,
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
        assert torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
            8,
            9,
        ), "float8 requires CUDA arch >= 8.9"

    assert example_input is not None

    prepare_target_folder(target_folder)
    torch._dynamo.reset()
    # TODO: explore using node.meta to retrieve the subgraph and fqn information
    # disable nn module inlining, our subgraph extraction logic depends on this
    torch._dynamo.config.inline_inbuilt_nn_modules = False
    torch._inductor.config.pre_grad_custom_pass = lambda g: debug_linears_for_float8(
        g, target_folder
    )
    model = torch.compile(model)
    if isinstance(example_input, torch.Tensor):
        example_input = [example_input]
    if isinstance(example_input, (list, tuple)):
        model(*example_input)
    elif isinstance(example_input, dict):
        model(**example_input)
    else:
        raise Exception("Unexpected example_input:", example_input)

    torch._inductor.config.pre_grad_custom_pass = None

    # verify debug logs and summary got saved
    assert os.path.isfile(
        os.path.join(target_folder, "debug_logs_0.txt")
    ), "No debug log saved, autoquant_v2 can't work for this model right now"
    assert os.path.isfile(
        os.path.join(target_folder, "summary_0.csv")
    ), "No debug log saved, autoquant_v2 can't work for this model right now"

    # first, find how many torch.compile'd regions we have
    extraction_idxs = []
    for f in os.listdir(target_folder):
        match = re.match(r"summary_([0-9]+).csv", f)
        if match:
            extraction_idxs.append(int(match.group(1)))
    extraction_idxs.sort()

    fqn_to_submodule = {}

    for extraction_idx in extraction_idxs:
        summary_filename = os.path.join(target_folder, f"summary_{extraction_idx}.csv")
        summary_rows = []
        with open(summary_filename, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                summary_rows.append(row)

        # [1:] to skip header row
        for row_idx, row in enumerate(summary_rows[1:]):
            subgraph_idx = row[2]
            fqn = row[-1]
            subgraph_fname = f"subgraph_with_inputs_{extraction_idx}_{subgraph_idx}.pt"
            print(f"loading {subgraph_fname} fqn {fqn}")
            subgraph_fname = os.path.join(target_folder, subgraph_fname)
            m, inputs = torch.load(subgraph_fname, weights_only=False)

            # for now, force cast to bf16
            # TODO(future): configure this
            m = m.to(torch.bfloat16)
            inputs = tree_map(lambda x: x.to(torch.bfloat16), inputs)

            m = m.to(torch.bfloat16)
            inputs = tree_map(lambda x: x.to(torch.bfloat16), inputs)

            fqn_to_submodule[fqn] = m, inputs

    model = model._orig_mod

    # perform initial swap from linear weights
    # to AutoQuantizableLinearWeight
    _change_linears_to_autoquantizable(
        model,
        example_input,
        fqn_to_submodule,
        batch_size,
        filter_fn=filter_fn,
        qtensor_class_list=qtensor_class_list,
        mode=mode,
        **aq_kwargs,
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
    elif isinstance(example_input, dict):
        model(**example_input)

    return model
