import csv
import os
import time
from typing import Any, Dict, List

import torch

from torchao.quantization import (
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
    quantize_,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    unwrap_tensor_subclass,
)

try:
    import triton  # noqa: F401

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
        self.high_precision_dtype = self._parse_precision(
            params["high_precision_dtype"]
        )
        self.compile = params.get("compile", False)
        self.compile_mode = params.get("compile_mode", "default")
        self.device = params.get("device", get_default_device())
        self.model_type = params.get("model_type", "linear")
        self.output_dir = output_dir
        self.name = f"benchmark_{self.quantization}_{self.model_type}_m{self.m}_k{self.k}_n{self.n}{'_compile' if self.compile else ''}"

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
            "high_precision_dtype": self.high_precision_dtype,
            "compile": self.compile,
            "compile_mode": "default",
            "device": self.device,
            "model_type": self.model_type,
            "output_dir": self.output_dir,
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


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        print("Warning: Running on CPU as no GPU support was found")
        return "cpu"


def ffn_only(mod, fqn):
    return isinstance(mod, torch.nn.Linear) and "feed_forward" in fqn


def not_ffn_only(mod, fqn):
    return isinstance(mod, torch.nn.Linear) and not ffn_only(mod, fqn)


def ffn_or_attn_only(mod, fqn):
    return isinstance(mod, torch.nn.Linear) and (
        "feed_forward" in fqn or "attention" in fqn
    )


def quantization_string_to_quantized_model(
    model: torch.nn.Module, quantization: str, **kwargs
) -> torch.nn.Module:
    """Quantize a model inplace or return a new quantized model.

    Args:
        model (torch.nn.Module): model to be quantized
        quantization (str): quantization method to be used
        **kwargs: additional arguments to be passed to the quantization method
    """
    high_precision_dtype = kwargs.get("high_precision_dtype", torch.bfloat16)
    if "int4wo" in quantization and not HAS_TRITON:
        print("Warning: Triton not available, falling back to baseline")
        return model

    # Quantization techniques
    if "baseline" in quantization:
        return model
    if "int8wo" in quantization:
        quantize_(model, Int8WeightOnlyConfig())
    if "int8dq" in quantization:
        if "int8dq_prefill_wo_decode" in quantization:
            quantize_(
                model, Int8DynamicActivationInt8WeightConfig(weight_only_decode=True)
            )
        else:
            quantize_(model, Int8DynamicActivationInt8WeightConfig())
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
        quantize_(model, Int4WeightOnlyConfig(group_size=group_size, use_hqq=use_hqq))
    elif "int8adq-int4w-symm" in quantization:
        from torchao.dtypes import CutlassInt4PackedLayout

        quantize_(
            model,
            Int8DynamicActivationInt4WeightConfig(
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
                Int8DynamicActivationInt4WeightConfig(
                    group_size=128,
                    mapping_type=MappingType.SYMMETRIC,
                    act_mapping_type=MappingType.SYMMETRIC,
                    layout=MarlinQQQLayout(),
                ),
            )
    if "fp6" in quantization:
        quantize_(model, FPXWeightOnlyConfig(3, 2))
    elif "embed-int8wo" in quantization:
        quantize_(
            model,
            Int8WeightOnlyConfig(group_size=64),
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
        quantize_(model, UIntXWeightOnlyConfig(dtype, group_size, use_hqq=use_hqq))
    elif "int8_dynamic_activation_intx_weight" in quantization:
        from torchao.experimental.quant_api import (
            int8_dynamic_activation_intx_weight,
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
        quantize_(
            model,
            int8_dynamic_activation_intx_weight(
                weight_dtype=weight_dtype,
                granularity=granularity,
                has_weight_zeros=has_weight_zeros,
            ),
        )
    elif "float8wo" in quantization:
        quantize_(model, Float8WeightOnlyConfig())
    elif "float8dq" in quantization:
        granularity = str(quantization.split("-")[-1])
        if granularity == "tensor":
            granularity = PerTensor()
        elif granularity == "row":
            granularity = PerRow()
        else:
            granularity = PerTensor()
        quantize_(
            model, Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
        )
    else:
        if not TORCH_VERSION_AT_LEAST_2_5:
            unwrap_tensor_subclass(model)
    return model


# Function to benchmark model evaluation - e2e eval run
def benchmark_model_inference_in_microseconds(model, input_data):
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

    return ((end_time - start_time) / num_iters) * 1e6


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
    results: List[Dict[str, Any]],
    output_dir: str,
    file_name: str = "results.csv",
):
    """Generate a CSV file with the results of the benchmarking.

    Args:
        results (List[Dict[str, Any]]): List Dictionary containing the results of the benchmarking with the config.
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
        header = results[0].keys()
        writer.writerow(header)
        for result in results:
            writer.writerow(result.values())

    print(f"Results saved to {file_path}")
