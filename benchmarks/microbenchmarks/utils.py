# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import csv
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from tabulate import tabulate
from torch.utils.benchmark import Timer

from torchao.core.config import AOBaseConfig
from torchao.quantization import (
    Float8DynamicActivationFloat8SemiSparseWeightConfig,
    Float8DynamicActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    FPXWeightOnlyConfig,
    GemliteUIntXWeightOnlyConfig,
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
        self.output_dir = f"{output_dir}/{self.benchmark_mode}"
        self.name = params.get(
            "name",
            f"benchmark_{self.quantization}_{self.model_type}_m{self.m}_k{self.k}_n{self.n}{'_compile' if self.use_torch_compile else ''}",
        )
        self.enable_profiler = bool(params.get("enable_profiler", False))
        self.enable_memory_profiler = bool(params.get("enable_memory_profiler", False))
        # Create profiler directory path without leading slash
        profiler_dir = os.path.join(self.output_dir, "profiler")
        os.makedirs(profiler_dir, exist_ok=True)
        self._file_name = f"{self.name}_{self.m}_{self.k}_{self.n}_quant_{self.quantization}_sparsity_{self.sparsity}"

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
            "enable_memory_profiler": self.enable_memory_profiler,
        }


class BenchmarkResult:
    def __init__(
        self,
        config: BenchmarkConfig,
    ):
        self.config = config
        self.output_dir = config.output_dir
        self.baseline_inference_time_in_ms = 0.0
        self.model_inference_time_in_ms = 0.0
        self.speedup = 0.0
        self.profiler_json_path: Optional[str] = None
        self.memory_profile_path: Optional[str] = None
        self.memory_visualization_path: Optional[str] = None
        self.memory_stats: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for main function"""
        result_dict = {
            **self.config.to_dict(),
            "baseline_inference_time_in_ms": self.baseline_inference_time_in_ms,
            "model_inference_time_in_ms": self.model_inference_time_in_ms,
            "speedup": self.speedup,
            "profiler_json_path": self.profiler_json_path,
            "memory_profile_path": self.memory_profile_path,
            "memory_visualization_path": self.memory_visualization_path,
            "memory_stats": self.memory_stats,
        }
        return result_dict


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
        assert high_precision_dtype == torch.float32, (
            "int8_dynamic_activation_intx_weight requires using high_precision_dtype=torch.float32"
        )

        from torchao.dtypes import PackedLinearInt8DynamicActivationIntxWeightLayout
        from torchao.quantization.granularity import PerAxis, PerGroup
        from torchao.quantization.quant_api import (
            Int8DynamicActivationIntxWeightConfig,
        )

        # Quantize model
        _quant_args = quantization.split("-")
        weight_dtype = getattr(torch, f"int{_quant_args[1]}")
        group_size = int(_quant_args[2])
        granularity = PerGroup(group_size) if group_size > 0 else PerAxis(0)
        is_asymmetric = bool(_quant_args[3])
        return Int8DynamicActivationIntxWeightConfig(
            weight_dtype=weight_dtype,
            weight_granularity=granularity,
            weight_mapping_type=MappingType.ASYMMETRIC
            if is_asymmetric
            else MappingType.SYMMETRIC,
            weight_scale_dtype=torch.bfloat16,
            layout=PackedLinearInt8DynamicActivationIntxWeightLayout(),
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
    if "gemlitewo" in quantization:
        params = quantization.split("-")
        bit_width = int(params[1]) if len(params) > 1 else 4
        group_size = (
            int(params[2])
            if len(params) > 2 and bit_width == 4
            else None
            if bit_width == 8
            else 64
        )
        assert group_size in [
            32,
            64,
            128,
            256,
        ], f"int4wo group_size needs to be one of [32,64,128,256] but got {group_size}"
        return GemliteUIntXWeightOnlyConfig(group_size=group_size, bit_width=bit_width)
    return None


@torch.no_grad()
def model_inference_time_in_ms(model, input_data):
    """Benchmark model inference time without compile overhead.

    Args:
        model: The model to benchmark
        input_data: Input data for the model

    Returns:
        float: Median inference time in milliseconds
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

    # Convert to milliseconds
    return (res * 1e6) / 1000  # Convert microseconds to milliseconds


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
    file_name: Optional[str] = None,
):
    """Generate a CSV file with the results of the benchmarking.

    Args:
        results (List[BenchmarkResult]): List Dictionary containing the results of the benchmarking with the config.
        output_dir (str): Directory to save the CSV file.
        file_name (str, optional): Name of the CSV file. Defaults to "results.csv".
    """
    # Check if results list is empty
    if len(results) == 0:
        print("No results to save to CSV.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Generate the filename with the current date and time in the specified format
    if file_name is None:
        file_name = datetime.now().strftime("results_%d%m%Y_%H%M%S.csv")

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
            f"{result.baseline_inference_time_in_ms:.2f}",
            f"{result.model_inference_time_in_ms:.2f}",
            f"{result.speedup:.2f}x",
            str(result.config.enable_profiler),
        ]

        table_data.append(row)

    # Define headers
    headers = [
        "Name",
        "Quantization",
        "Sparsity",
        "Shape",
        "Baseline Inference Time (ms)",
        "Inference Time (ms)",
        "Speedup",
        "Profiler Enabled",
    ]

    if table_data:
        print("\nBenchmark Results:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print("\nNo valid results to display")
