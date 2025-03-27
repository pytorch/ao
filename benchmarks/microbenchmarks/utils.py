# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import csv
import os
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
        self.output_dir = output_dir
        self.name = params.get(
            "name",
            f"benchmark_{self.quantization}_{self.model_type}_m{self.m}_k{self.k}_n{self.n}{'_compile' if self.use_torch_compile else ''}",
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
        }


class BenchmarkResult:
    def __init__(
        self,
        config: BenchmarkConfig,
    ):
        self.config = config
        self.output_dir = config.output_dir
        self.model_inference_time_in_ms = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for main function"""
        return {
            **self.config.to_dict(),
            "model_inference_time_in_ms": self.model_inference_time_in_ms,
        }


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
    """Print benchmark results in a formatted table.

    Args:
        results (List[BenchmarkResult]): List of benchmark results
    """
    if not results:
        print("No results to display")
        return

    # Extract relevant columns for display
    display_columns = [
        "quantization",
        "sparsity",
        "model_type",
        "m",
        "k",
        "n",
        "model_inference_time_in_ms",
        "use_torch_compile",
    ]

    # Format data for tabulate
    headers = {
        "quantization": "Quantization",
        "sparsity": "Sparsity",
        "model_type": "Model Type",
        "m": "M",
        "k": "K",
        "n": "N",
        "model_inference_time_in_ms": "Time (μs)",
        "use_torch_compile": "Compile Mode",
    }

    # Extract and format data
    table_data = []
    for result in results:
        result_dict = result.to_dict()
        row = []
        for col in display_columns:
            value = result_dict.get(col, "N/A")
            if value is None:
                value = "N/A"
            if col == "model_inference_time_in_ms":
                value = f"{value:.2f}" if isinstance(value, (int, float)) else value
            elif col == "use_torch_compile":
                # Show compile mode if compile is True, otherwise show False
                value = (
                    result_dict.get("torch_compile_mode", "default")
                    if result_dict.get("use_torch_compile")
                    else "False"
                )
            row.append(value)
        table_data.append(row)

    # Print formatted table
    print("\nBenchmark Results:")
    print(
        tabulate(
            table_data,
            headers=[headers[col] for col in display_columns],
            tablefmt="grid",
            floatfmt=".2f",
        )
    )
    print()
