import time
from typing import Optional, List, Dict, Any

import torch
from torch.profiler import ProfilerActivity, profile

from torchao.quantization import (
    MappingType,
    PerRow,
    PerTensor,
    float8_dynamic_activation_float8_weight,
    float8_weight_only,
    fpx_weight_only,
    int4_weight_only,
    int8_dynamic_activation_int4_weight,
    int8_dynamic_activation_int8_weight,
    int8_weight_only,
    quantize_,
    uintx_weight_only,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_3,
    TORCH_VERSION_AT_LEAST_2_5,
    unwrap_tensor_subclass,
)

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


class BenchmarkConfig:
    def __init__(
        self,
        quantization: str,
        params: Dict[str, Any],
        shape_name: str,
        shape: List[int],
        output_dir: str,
    ):
        self.quantization = quantization
        self.m, self.k, self.n = shape
        self.shape_name = shape_name
        self.precision = self._parse_precision(params["precision"])
        self.compile = params.get("compile", False)
        self.device = params.get("device", get_default_device())
        self.model_type = params.get("model_type", "linear")
        self.output_dir = output_dir
        self.name = f"benchmark_{self.quantization}_{self.model_type}_m{self.m}_k{self.k}_n{self.n}"

    @staticmethod
    def _parse_precision(precision_str: str) -> torch.dtype:
        """Convert string precision to torch dtype"""
        return getattr(torch, precision_str.split(".")[-1])

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for main function"""
        return {
            "quantization": self.quantization,
            "m": self.m,
            "k": self.k,
            "n": self.n,
            "precision": self.precision,
            "compile": self.compile,
            "device": self.device,
            "model_type": self.model_type,
            "output_dir": self.output_dir,
        }


# TODO: add more models
class ToyLinearModel(torch.nn.Module):
    def __init__(self, k=64, n=32, dtype=torch.bfloat16):
        super().__init__()
        self.linear1 = torch.nn.Linear(k, n, bias=False).to(dtype)

    def forward(self, x):
        x = self.linear1(x)
        return x


class LNLinearSigmoid(torch.nn.Module):
    def __init__(self, fc_dim1, fc_dim2, dtype=torch.bfloat16):
        super().__init__()
        self.ln = torch.nn.LayerNorm(fc_dim1, elementwise_affine=False)
        self.fc = torch.nn.Linear(fc_dim1, fc_dim2, bias=False).to(dtype)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.ln(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


def get_default_device() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.xpu.is_available():
            return "xpu"
        elif torch.backends.mps.is_available():
            return "mps"
    except (AssertionError, AttributeError):
        print("Warning: Running on CPU as no GPU support was found")
        return "cpu"


def benchmark_func_call(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to run {func.__name__}: {elapsed_time:.2f} seconds")
    return result


def ffn_only(mod, fqn):
    return isinstance(mod, torch.nn.Linear) and "feed_forward" in fqn


def not_ffn_only(mod, fqn):
    return isinstance(mod, torch.nn.Linear) and not ffn_only(mod, fqn)


def ffn_or_attn_only(mod, fqn):
    return isinstance(mod, torch.nn.Linear) and (
        "feed_forward" in fqn or "attention" in fqn
    )


def quantize_model(
    model: torch.nn.Module,
    quantization: str,
    **kwargs,
):
    """Quantize a model inplace or return a new quantized model.

    Args:
        model (torch.nn.Module): model to be quantized
        quantization (str): quantization method to be used
        kwargs: additional arguments to be passed to the quantization method
    """
    if "int4wo" in quantization and not HAS_TRITON:
        print("Warning: Triton not available, falling back to baseline")
        return model
    
    # Define kwargs
    sparsity = kwargs.get("sparsity", None)
    precision = kwargs.get("precision", None)

    # Quantization techniques
    if "baseline" in quantization:
        return model
    if "int8wo" in quantization:
        quantize_(model, int8_weight_only())
    if "int8dq" in quantization:
        if sparsity and "semi" in sparsity:
            from torchao.dtypes import SemiSparseLayout

            quantize_(
                model,
                int8_dynamic_activation_int8_weight(layout=SemiSparseLayout()),
                filter_fn=ffn_only,
            )
            quantize_(
                model, int8_dynamic_activation_int8_weight(), filter_fn=not_ffn_only
            )
        elif "int8dq_prefill_wo_decode" in quantization:
            quantize_(
                model, int8_dynamic_activation_int8_weight(weight_only_decode=True)
            )
        else:
            quantize_(model, int8_dynamic_activation_int8_weight())
    if "int4wo" in quantization:
        use_hqq = False
        if "hqq" in quantization:
            use_hqq = True
        group_size = int(quantization.split("-")[1])
        assert group_size in [
            32,
            64,
            128,
            256,
        ], f"int4wo group_size needs to be one of [32,64,128,256] but got {group_size}"
        quantize_(model, int4_weight_only(group_size=group_size, use_hqq=use_hqq))
    elif "int8adq-int4w-symm" in quantization:
        from torchao.dtypes import CutlassInt4PackedLayout

        quantize_(
            model,
            int8_dynamic_activation_int4_weight(
                group_size=None,
                mapping_type=MappingType.SYMMETRIC,
                act_mapping_type=MappingType.SYMMETRIC,
                layout=CutlassInt4PackedLayout(),
            ),
        )
    if "marlin" in quantization:
        if "qqq" in quantization:
            from torchao.dtypes import MarlinQQQLayout

            quantize_(
                model,
                int8_dynamic_activation_int4_weight(
                    group_size=128,
                    mapping_type=MappingType.SYMMETRIC,
                    act_mapping_type=MappingType.SYMMETRIC,
                    layout=MarlinQQQLayout(),
                ),
            )
        elif "semi" in sparsity:
            from torchao.dtypes import MarlinSparseLayout

            quantize_(
                model,
                int4_weight_only(layout=MarlinSparseLayout()),
                filter_fn=ffn_or_attn_only,
            )
    if "fp6" in quantization:
        quantize_(model, fpx_weight_only(3, 2))
    elif "embed-int8wo" in quantization:
        quantize_(
            model,
            int8_weight_only(group_size=64),
            filter_fn=lambda x, *args: isinstance(x, torch.nn.Embedding),
        )
    elif "uintx" in quantization:
        # uintx-nbits-group_size, e.g. "uintx-2-64"
        if "hqq" in quantization:
            # uintx-nbits-group_size-hqq
            use_hqq = True
        else:
            use_hqq = False
        _quant_args = quantization.split("-")
        nbits = int(_quant_args[1])
        assert nbits >= 1 and nbits <= 8, "nbits must be 1 to 8"
        _NBITS_TO_DTYPE = {
            1: torch.uint1,
            2: torch.uint2,
            3: torch.uint3,
            4: torch.uint4,
            5: torch.uint5,
            6: torch.uint6,
            7: torch.uint7,
            8: torch.uint8,
        }
        dtype = _NBITS_TO_DTYPE[nbits]
        group_size = int(_quant_args[2])
        quantize_(model, uintx_weight_only(dtype, group_size, use_hqq=use_hqq))
    elif "int8_dynamic_activation_intx_weight" in quantization:
        from torchao.experimental.quant_api import (
            int8_dynamic_activation_intx_weight,
        )
        from torchao.quantization.granularity import PerGroup

        assert (
            precision == torch.float32
        ), "int8_dynamic_activation_intx_weight requires using precision=torch.float32"

        # Quantize model
        _quant_args = quantization.split("-")
        weight_dtype = getattr(torch, f"int{_quant_args[1]}")
        granularity = PerGroup(int(_quant_args[2]))
        has_weight_zeros = bool(_quant_args[3])
        quantize_(
            model,
            int8_dynamic_activation_intx_weight(
                weight_dtype=weight_dtype,
                granularity=granularity,
                has_weight_zeros=has_weight_zeros,
            ),
        )
    elif "float8wo" in quantization:
        quantize_(model, float8_weight_only())
    elif "float8dq" in quantization:
        granularity = str(quantization.split("-")[-1])
        if granularity == "tensor":
            granularity = PerTensor()
        elif granularity == "row":
            granularity = PerRow()
        else:
            granularity = PerTensor()
        quantize_(
            model, float8_dynamic_activation_float8_weight(granularity=granularity)
        )
    else:
        if not TORCH_VERSION_AT_LEAST_2_5:
            unwrap_tensor_subclass(model)
    return model


# Function to benchmark model evaluation - e2e eval run
def benchmark_model_inference_in_seconds(model, input_data):
    # Returns model run time in seconds
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # warm up
    for _ in range(2):
        model(input_data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    num_iters = 5
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(input_data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    end_time = time.perf_counter()

    return (end_time - start_time) / num_iters


def benchmark_model_op_with_profiler_in_microseconds(model, input_data, op_name: str):
    """Benchmarks model inference using PyTorch profiler to measure GPU kernel execution times.

    This function profiles the model execution and measures the time spent in specific GPU operations
    versus overhead time. It performs warmup runs before profiling to ensure accurate measurements.

    Args:
        model (torch.nn.Module): PyTorch model to benchmark
        input_data (torch.Tensor): Input tensor to run through the model
        op_name (str): Name of the GPU operation to measure time for

    Returns:
        tuple[float, float]: A tuple containing:
            - gpu_op_time (float): Time spent in the specified GPU operation in microseconds
            - gpu_overhead_time (float): Time spent in other GPU operations in microseconds
    """
    # Warm up
    for _ in range(2):
        model(input_data)

    # Profile model execution
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        with torch.no_grad():
            _ = model(input_data)
            torch.cuda.synchronize()

    # Get event data from profiler
    event_data = [
        (event.key, event.device_time)
        for event in prof.key_averages()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]

    # Calculate op time and overhead time
    gpu_op_time, gpu_overhead_time = 0, 0
    for event in event_data:
        if op_name in event[0]:
            gpu_op_time += event[1]
        else:
            gpu_overhead_time += event[1]

    return gpu_op_time, gpu_overhead_time


def create_model_and_input(
    model_type: str,
    m: int,
    k: int,
    n: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = get_default_device(),
):
    """Create a model and input data for benchmarking.

    Args:
        model_type (str): type of the model to be created
        batch_size (int): batch size of the input data
        device (str): device to run the model on
        dtype (torch.dtype): data
        m, k, n (int): dimensions of the model and input data
    """
    if model_type == "linear":
        model = ToyLinearModel(k, n, dtype).to(device)
        input_data = torch.randn(m, k, device=device, dtype=dtype)
    elif model_type == "ln_linear_sigmoid":
        model = LNLinearSigmoid(k, n, dtype).to(device)
        input_data = torch.randn(m, k, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model, input_data


@torch.no_grad()
def benchmark_op_with_cuda_graph(op, *args, **kwargs):
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


def clean_caches():
    import gc

    # Clear everything before starting
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if hasattr(torch, '_dynamo'):
        torch._dynamo.reset()


def get_name_to_shapes_iter(
    shape_gen_name: str,
    M: Optional[int],
    K: Optional[int],
    N: Optional[int],
):
    if shape_gen_name == "llama":
        assert (
            M == K == N == None
        ), f"M, K, N arguments not supported for shape_gen_name {shape_gen_name}"
        bsz, seq_len = 4, 4096
        M = bsz * seq_len
        # LLaMa 2 70B single-node weight shapes
        # assumes fused attn.wqkv and ffn.w13
        # source: https://fburl.com/gsheet/g8onr7rh
        name_to_shapes_70b = {
            "attn.wqkv": (M, 8192, 1280),
            "attn.w0": (M, 1024, 8192),
            "ffn.w13": (M, 8192, 7168),
            "ffn.w2": (M, 3584, 8192),
        }
        return name_to_shapes_70b.items()

    elif shape_gen_name == "square":
        assert (
            M == K == N == None
        ), f"M, K, N arguments not supported for shape_gen_name {shape_gen_name}"
        name_to_shapes = {}
        min_power_of_2 = 8  # 256
        max_power_of_2 = 15  # 32,768
        for idx, power_of_2 in enumerate(range(min_power_of_2, max_power_of_2 + 1)):
            val = 2**power_of_2
            name_to_shapes[idx] = val, val, val
        return name_to_shapes.items()

    elif shape_gen_name == "sweep":
        assert (
            M == K == N == None
        ), f"M, K, N arguments not supported for shape_gen_name {shape_gen_name}"
        name_to_shapes = {}
        min_p2 = 8  # 256
        max_p2 = 15  # 32,768
        counter = 0
        for M_p2 in range(min_p2, max_p2 + 1):
            M = 2**M_p2
            for K_p2 in range(min_p2, max_p2 + 1):
                K = 2**K_p2
                for N_p2 in range(min_p2, max_p2 + 1):
                    N = 2**N_p2
                    name_to_shapes[counter] = M, K, N
                    counter += 1
        return name_to_shapes.items()

    elif shape_gen_name == "custom":
        assert (
            M is not None and K is not None and N is not None
        ), "M, K, N must be specified for custom shape_gen"
        name_to_shapes = {
            1: (M, K, N),
        }
        return name_to_shapes.items()

    raise AssertionError(f"unknown shape_gen_name {shape_gen_name}")