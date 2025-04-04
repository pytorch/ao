# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import csv
import json
import os
import subprocess
import uuid
from typing import Any, Dict, List, Optional

import torch
from tabulate import tabulate
from torch.profiler import ProfilerActivity
from torch.utils.benchmark import Timer

from torchao.core.config import AOBaseConfig
from torchao.quantization import (
    Float8DynamicActivationFloat8SemiSparseWeightConfig,
    Float8DynamicActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    FPXWeightOnlyConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    MappingType,
    PerRow,
    PerTensor,
    UIntXWeightOnlyConfig,
)
from torchao.sparsity.sparse_api import BlockSparseWeightConfig, SemiSparseWeightConfig

try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def get_default_device(device: str = "cuda") -> str:
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif device == "xpu" and torch.xpu.is_available():
        return "xpu"
    elif device == "mps" and torch.backends.mps.is_available():
        return "mps"
    elif device == "cpu":
        return "cpu"
    else:
        print(f"Warning: Running on CPU as {device} support was not found")
        return "cpu"


def upload_trace_file(local_path: str, overwrite: bool = False) -> Optional[str]:
    MANIFOLD_FOLDER = "perfetto_internal_traces/tree/shared_trace"
    DEFAULT_TTL_SEC = 28 * 24 * 60 * 60
    file_name = os.path.basename(local_path)
    manifold_path = os.path.join(
        MANIFOLD_FOLDER, f"{os.getlogin()}_{str(uuid.uuid4())}_{file_name}"
    )
    cmd = [
        "manifold",
        "put",
        local_path,
        manifold_path,
        "--ttl",
        str(DEFAULT_TTL_SEC),
        "--userData",
        "false",
    ]
    ret = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    if ret.returncode == 0:
        print("Upload trace successfully.")
        return manifold_path
    else:
        print("[ERROR] Upload failed, maybe the trace file exists.")
        return None


def print_perfetto_ui_url(manifold_path: str) -> Optional[str]:
    """Generate and print the Perfetto UI URL for a Manifold trace file.

    Args:
        manifold_path: Path to the trace file in Manifold

    Returns:
        The URL to the Perfetto UI or None if there was an error
    """
    try:
        url = (
            "https://interncache-all.fbcdn.net/manifold/perfetto-artifacts/tree/ui/index.html#!/?url=https://interncache-all.fbcdn.net/manifold/"
            + manifold_path
        )
        print(f"The trace is accessible at:\n{url}")
        return url
    except Exception as e:
        print(f"Error generating Perfetto UI URL: {e}")
        return None


def generate_model_profile(model, input_data, profile_file_path):
    """Function to benchmark model evaluation with profiling.

    Args:
        model: The model to profile
        input_data: Input data for the model
        profile_file_path: Path to save the profiler output

    Returns:
        Tuple of (profile_file_path, perfetto_url)
    """
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(profile_file_path), exist_ok=True)

    # Set up profiler activities based on device
    activities = [ProfilerActivity.CPU]
    device = next(model.parameters()).device
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # Run profiler with minimal settings to ensure compatibility
    prof = torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,  # Disable stack traces to reduce overhead
        profile_memory=False,  # Disable memory profiling as it's not reliable across all devices
    )

    # Warm up
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_data)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Profile
    with prof:
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_data)
                if device.type == "cuda":
                    torch.cuda.synchronize()

    # Save profiling details
    prof.export_chrome_trace(profile_file_path)
    print(f"Profile saved to: {profile_file_path}")

    # Try to upload to Perfetto UI
    perfetto_url = None
    try:
        manifold_path = upload_trace_file(profile_file_path)
        if manifold_path:
            perfetto_url = print_perfetto_ui_url(manifold_path)
    except Exception as e:
        print(f"Warning: Failed to upload profile to Perfetto UI: {e}")

    return profile_file_path, perfetto_url


# def visualize_memory_profile(snapshot, output_html_path) -> Optional[str]:
#     from torch.cuda._memory_viz import trace_plot

#     # Convert to HTML
#     html = trace_plot(snapshot)

#     # Save to file
#     with open(output_html_path, "w") as f:
#         f.write(html)


def generate_memory_profile(model, input_data, profile_file_path):
    """Function to generate memory profile for model evaluation.

    Args:
        model: The model to profile
        input_data: Input data for the model
        profile_file_path: Path to save the memory profile output

    Returns:
        Tuple of (profile_file_path, memory_stats)
    """
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(profile_file_path), exist_ok=True)

    device = next(model.parameters()).device
    memory_stats = {
        "peak_memory_allocated": 0,
        "peak_memory_reserved": 0,
        "total_memory_allocated": 0,
        "total_memory_reserved": 0,
        "memory_events": [],
    }

    if device.type == "cuda":
        # Enable memory history recording for CUDA
        torch.cuda.memory._record_memory_history(
            True, trace_alloc_max_entries=250000, trace_alloc_record_context=True
        )

        # Reset CUDA memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_data)
                torch.cuda.synchronize()

        # Profile memory
        with torch.no_grad():
            _ = model(input_data)
            torch.cuda.synchronize()

        # Collect memory stats
        memory_stats.update(
            {
                "peak_memory_allocated": torch.cuda.max_memory_allocated()
                / 1024**2,  # Convert to MB
                "peak_memory_reserved": torch.cuda.max_memory_reserved() / 1024**2,
                "total_memory_allocated": torch.cuda.memory_allocated() / 1024**2,
                "total_memory_reserved": torch.cuda.memory_reserved() / 1024**2,
            }
        )

        # Get detailed memory snapshot
        snapshot = torch.cuda.memory._snapshot()

        # Save memory profile as pickle file
        pickle_path = profile_file_path.replace(".json", ".pickle")
        with open(pickle_path, "wb") as f:
            from pickle import dump

            dump(snapshot, f)

        print(f"Memory profile saved to: {pickle_path}")

        # TODO: Add memory visualization
        # visualize_memory_profile(snapshot, pickle_path.replace(".pickle", ".html"))
        # print(f"Memory visualization saved to: {pickle_path.replace('.pickle', '.html')}")

        # Disable memory history recording
        torch.cuda.memory._record_memory_history(False)

    else:
        print("Memory profiling only works on CUDA devices")
        # TODO: Add XPU support when available
        return profile_file_path, memory_stats

    # Save basic stats as JSON for easy access
    with open(profile_file_path, "w") as f:
        json.dump(memory_stats, f, indent=2)

    return profile_file_path, memory_stats


class BenchmarkConfig:
    def __init__(
        self,
        quantization: Optional[
            str
        ],  # Quantization string format is similar to the format being used for llama/generate.py
        sparsity: Optional[str],  # Specify the type of sparsity to be used
        params: Dict[str, Any],
        shape_name: str,
        shape: List[int],
        output_dir: str,
        benchmark_mode: str,
    ):
        self.benchmark_mode = benchmark_mode
        self.quantization = quantization
        self.sparsity = sparsity
        self.m, self.k, self.n = shape
        self.shape_name = shape_name
        self.high_precision_dtype = self._parse_precision(
            params.get("high_precision_dtype", "torch.bfloat16")
        )
        self.use_torch_compile = bool(params.get("use_torch_compile", False))
        self.torch_compile_mode = (
            params.get("torch_compile_mode", "default")
            if self.use_torch_compile
            else None
        )
        self.device = get_default_device(params.get("device", None))
        self.model_type = params.get("model_type", "linear")
        self.output_dir = output_dir
        self.name = params.get(
            "name",
            f"benchmark_{self.quantization}_{self.model_type}_m{self.m}_k{self.k}_n{self.n}{'_compile' if self.use_torch_compile else ''}",
        )
        self.enable_profiler = bool(params.get("enable_profiler", False))
        self.enable_memory_profile = bool(params.get("enable_memory_profile", False))
        # Create profiler directory path without leading slash
        profiler_dir = os.path.join(self.output_dir, "profiler")
        os.makedirs(profiler_dir, exist_ok=True)
        file_name = f"{self.name}_{self.m}_{self.k}_{self.n}_quant_{self.quantization}_sparsity_{self.sparsity}"
        self.profiler_file_name = os.path.join(
            profiler_dir, f"{file_name}_profile.json"
        )
        self.memory_profile_file_name = os.path.join(
            profiler_dir, f"{file_name}_memory_profile.json"
        )

    @staticmethod
    def _parse_precision(precision_str: str) -> torch.dtype:
        """Convert string precision to torch dtype"""
        return getattr(torch, precision_str.split(".")[-1])

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for main function"""
        return {
            "name": self.name,
            "quantization": self.quantization,
            "sparsity": self.sparsity,
            "m": self.m,
            "k": self.k,
            "n": self.n,
            "high_precision_dtype": self.high_precision_dtype,
            "use_torch_compile": self.use_torch_compile,
            "torch_compile_mode": self.torch_compile_mode,
            "device": self.device,
            "model_type": self.model_type,
            "output_dir": self.output_dir,
            "enable_profiler": self.enable_profiler,
            "enable_memory_profile": self.enable_memory_profile,
        }


class BenchmarkResult:
    def __init__(
        self,
        config: BenchmarkConfig,
    ):
        self.config = config
        self.output_dir = config.output_dir
        self.model_inference_time_in_ms = 0.0
        self.profiler_json_path: Optional[str] = None
        self.perfetto_url: Optional[str] = None
        self.memory_profile_path: Optional[str] = None
        self.memory_stats: Optional[Dict[str, Any]] = None
        # self.memory_visualization_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for main function"""
        result_dict = {
            **self.config.to_dict(),
            "model_inference_time_in_ms": self.model_inference_time_in_ms,
            "profiler_json_path": self.profiler_json_path,
            "perfetto_url": self.perfetto_url,
            "memory_profile_path": self.memory_profile_path,
            "memory_stats": self.memory_stats,
            # "memory_visualization_path": self.memory_visualization_path,
        }
        return result_dict


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


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RMSNormLinearActivation(torch.nn.Module):
    def __init__(self, fc_dim1, fc_dim2, dtype=torch.bfloat16, activation="gelu"):
        super().__init__()
        self.rms_norm = RMSNorm(fc_dim1, dtype=dtype)
        self.fc = torch.nn.Linear(fc_dim1, fc_dim2, bias=False).to(dtype)
        
        if activation == "gelu":
            self.activation = torch.nn.GELU()
        elif activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "silu":
            self.activation = torch.nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        x = self.rms_norm(x)
        x = self.fc(x)
        x = self.activation(x)
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads=8, mlp_ratio=4, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Self-attention
        self.qkv = torch.nn.Linear(hidden_dim, 3 * hidden_dim, bias=False).to(dtype)
        self.proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(dtype)
        
        # MLP
        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp_fc1 = torch.nn.Linear(hidden_dim, self.mlp_hidden_dim, bias=False).to(dtype)
        self.mlp_fc2 = torch.nn.Linear(self.mlp_hidden_dim, hidden_dim, bias=False).to(dtype)
        
        # Layer norms
        self.norm1 = RMSNorm(hidden_dim, dtype=dtype)
        self.norm2 = RMSNorm(hidden_dim, dtype=dtype)
        
        # Activation
        self.activation = torch.nn.GELU()

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        # Reshape qkv projection for better memory layout
        qkv = self.qkv(x)  # [batch_size, seq_len, 3 * hidden_dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv  # Each has shape [batch_size, num_heads, seq_len, head_dim]
        
        # Scaled dot-product attention with proper reshaping
        # Reshape for better memory layout and avoid broadcasting issues
        q = q.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        k = k.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        v = v.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention to values
        x = attn @ v  # [batch_size * num_heads, seq_len, head_dim]
        
        # Reshape back to original dimensions
        x = x.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        x = x.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        
        # Project back to hidden dimension
        x = self.proj(x)
        x = residual + x
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp_fc1(x)
        x = self.activation(x)
        x = self.mlp_fc2(x)
        x = residual + x
        
        return x


def string_to_config(
    quantization: Optional[str], sparsity: Optional[str], **kwargs
) -> AOBaseConfig:
    """Get quantization config based on quantization string.

    Args:
        quantization (str): Quantization method to be used. The quantiation string format is similar to the format being used for llama/generate.py.
        sparsity (str): Sparsity method to be used. The sparsity string format is similar to the format being used for llama/generate.py.
        **kwargs: Additional arguments to be passed to the quantization method

    Returns:
        AOBaseConfig: Quantization configuration object
    """
    # Handle block sparsity case - with block sparsity, quantization should always be "none" or "baseline"
    if sparsity is not None and sparsity == "block":
        return BlockSparseWeightConfig()

    # Handle other sparsity cases
    if quantization is None and sparsity is not None:
        if "semi" in sparsity or "2:4" in sparsity:
            return SemiSparseWeightConfig()
        else:
            raise ValueError(f"Unknown sparsity type: {sparsity}")
    if quantization is None and sparsity is None:
        return None
    high_precision_dtype = kwargs.get("high_precision_dtype", torch.bfloat16)

    if "int4wo" in quantization and not HAS_TRITON:
        print("Warning: Triton not available, falling back to baseline")
        return None

    # Quantization techniques
    if "baseline" in quantization:
        return None
    if "int8wo" in quantization:
        return Int8WeightOnlyConfig()
    if "int8dq" in quantization:
        if sparsity is not None and ("semi" in sparsity or "2:4" in sparsity):
            from torchao.dtypes import SemiSparseLayout

            return Int8DynamicActivationInt8WeightConfig(layout=SemiSparseLayout())
        elif "int8dq_prefill_wo_decode" in quantization:
            return Int8DynamicActivationInt8WeightConfig(weight_only_decode=True)
        else:
            return Int8DynamicActivationInt8WeightConfig()
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
        return Int4WeightOnlyConfig(group_size=group_size, use_hqq=use_hqq)
    elif "int8adq-int4w-symm" in quantization:
        from torchao.dtypes import CutlassInt4PackedLayout

        return Int8DynamicActivationInt4WeightConfig(
            group_size=None,
            mapping_type=MappingType.SYMMETRIC,
            act_mapping_type=MappingType.SYMMETRIC,
            layout=CutlassInt4PackedLayout(),
        )
    if "marlin" in quantization:
        if "qqq" in quantization:
            from torchao.dtypes import MarlinQQQLayout

            return Int8DynamicActivationInt4WeightConfig(
                group_size=128,
                mapping_type=MappingType.SYMMETRIC,
                act_mapping_type=MappingType.SYMMETRIC,
                layout=MarlinQQQLayout(),
            )
        elif sparsity is not None and ("semi" in sparsity or "2:4" in sparsity):
            from torchao.dtypes import MarlinSparseLayout

            return Int4WeightOnlyConfig(layout=MarlinSparseLayout())
    if "fp6" in quantization:
        return FPXWeightOnlyConfig(3, 2)
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
        return UIntXWeightOnlyConfig(dtype, group_size, use_hqq=use_hqq)
    elif "int8_dynamic_activation_intx_weight" in quantization:
        from torchao.experimental.quant_api import (
            Int8DynamicActivationIntxWeightConfig,
        )
        from torchao.quantization.granularity import PerGroup

        assert (
            high_precision_dtype == torch.float32
        ), "int8_dynamic_activation_intx_weight requires using high_precision_dtype=torch.float32"

        # Quantize model
        _quant_args = quantization.split("-")
        weight_dtype = getattr(torch, f"int{_quant_args[1]}")
        granularity = PerGroup(int(_quant_args[2]))
        has_weight_zeros = bool(_quant_args[3])
        return Int8DynamicActivationIntxWeightConfig(
            weight_dtype=weight_dtype,
            granularity=granularity,
            has_weight_zeros=has_weight_zeros,
        )
    elif "float8wo" in quantization:
        return Float8WeightOnlyConfig()
    elif "float8dq" in quantization:
        if sparsity and "semi" in sparsity:
            return Float8DynamicActivationFloat8SemiSparseWeightConfig()
        granularity = str(quantization.split("-")[-1])
        if granularity == "tensor":
            granularity = PerTensor()
        elif granularity == "row":
            granularity = PerRow()
        else:
            granularity = PerTensor()
        return Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
    return None


@torch.no_grad()
def model_inference_time_in_ms(model, input_data):
    """Benchmark model inference time without compile overhead.

    Args:
        model: The model to benchmark
        input_data: Input data for the model

    Returns:
        float: Median inference time in microseconds
    """
    # First run to trigger any compilation/lazy initialization

    timer = Timer(
        stmt="model(input_data)",
        globals={"model": model, "input_data": input_data},
        num_threads=1,
    )

    # warmup
    timer.timeit(number=100)
    # actual measurement
    measurement = timer.timeit(number=100)
    res = measurement.mean

    # Convert to microseconds
    return res * 1e6


def create_model_and_input(
    model_type: str,
    m: int,
    k: int,
    n: int,
    high_precision_dtype: torch.dtype = torch.bfloat16,
    device: str = get_default_device(),
):
    """Create a model and input data for benchmarking.

    Args:
        model_type (str): type of the model to be created
        batch_size (int): batch size of the input data
        device (str): device to run the model on
        high_precision_dtype (torch.dtype): data type of the model
        m, k, n (int): dimensions of the model and input data
    """
    if model_type == "linear":
        model = ToyLinearModel(k, n, high_precision_dtype).to(device)
        input_data = torch.randn(m, k, device=device, dtype=high_precision_dtype)
    elif model_type == "ln_linear_sigmoid":
        model = LNLinearSigmoid(k, n, high_precision_dtype).to(device)
        input_data = torch.randn(m, k, device=device, dtype=high_precision_dtype)
    elif model_type == "rms_norm_linear_activation":
        model = RMSNormLinearActivation(k, n, high_precision_dtype).to(device)
        input_data = torch.randn(m, k, device=device, dtype=high_precision_dtype)
    elif model_type == "transformer_block":
        # For transformer block, k is the hidden dimension
        model = TransformerBlock(k, num_heads=8, mlp_ratio=4, dtype=high_precision_dtype).to(device)
        # Input shape for transformer is [batch_size, seq_len, hidden_dim]
        input_data = torch.randn(m, 16, k, device=device, dtype=high_precision_dtype)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model, input_data


def clean_caches():
    import gc

    # Clear everything before starting
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()


def generate_results_csv(
    results: List[BenchmarkResult],
    output_dir: str,
    file_name: str = "results.csv",
):
    """Generate a CSV file with the results of the benchmarking.

    Args:
        results (List[BenchmarkResult]): List Dictionary containing the results of the benchmarking with the config.
        output_dir (str): Directory to save the CSV file.
        file_name (str, optional): Name of the CSV file. Defaults to "results.csv".
    """
    # Check if results list is empty
    if not results:
        print("No results to save to CSV.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)

    # Create a CSV file with the results
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        header = results[0].to_dict().keys()
        writer.writerow(header)
        for result in results:
            writer.writerow(result.to_dict().values())

    print(f"Results saved to {file_path}")


def print_results(results: List[BenchmarkResult]):
    """Print results in a table format"""
    if not results:
        print("No results to display")
        return

    table_data = []
    for result in results:
        if result is None:
            continue

        row = [
            result.config.name,
            result.config.quantization or "baseline",
            result.config.sparsity or "none",
            f"{result.config.shape_name} ({result.config.m}, {result.config.k}, {result.config.n})",
            f"{result.model_inference_time_in_ms:.2f}",
            str(result.config.enable_profiler),
            str(result.config.enable_memory_profile),
        ]

        # Add memory profile data if enabled
        if result.config.enable_memory_profile:
            if result.memory_stats:
                row.append(
                    f"Peak memory: {result.memory_stats['peak_memory_allocated']:.2f}MB"
                )
            else:
                row.append("Memory profiling failed")

        table_data.append(row)

    # Define headers
    headers = [
        "Name",
        "Quantization",
        "Sparsity",
        "Shape",
        "Inference Time (ms)",
        "Profiler Enabled",
        "Memory Profiling Enabled",
    ]
    if any(r.config.enable_memory_profile for r in results if r is not None):
        headers.append("Memory Profile Data")

    if table_data:
        print("\nBenchmark Results:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print("\nNo valid results to display")
