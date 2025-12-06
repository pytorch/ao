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
        self.torch_compile_mode = params.get("torch_compile_mode", "default")
        self.device = get_default_device(params.get("device", None))
        self.model_type = params.get("model_type", "linear")
        self.output_dir = f"{output_dir}/{self.benchmark_mode}"
        self.name = params.get(
            "name",
            f"benchmark_{self.quantization}_{self.model_type}_m{self.m}_k{self.k}_n{self.n}{'_compile'}",
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
        self.baseline_model_eager_inference_time_in_ms = 0.0
        self.quantized_model_eager_inference_time_in_ms = 0.0
        self.baseline_model_compiled_inference_time_in_ms = 0.0
        self.quantized_model_compiled_inference_time_in_ms = 0.0
        self.eager_speedup_on_baseline = 0.0
        self.compile_speedup_on_baseline = 0.0
        self.compile_speedup_on_eager = 0.0
        self.profiler_json_path: Optional[str] = None
        self.memory_profile_path: Optional[str] = None
        self.memory_visualization_path: Optional[str] = None
        self.memory_stats: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for main function"""
        result_dict = {
            **self.config.to_dict(),
            "baseline_model_eager_inference_time_in_ms": self.baseline_model_eager_inference_time_in_ms,
            "quantized_model_eager_inference_time_in_ms": self.quantized_model_eager_inference_time_in_ms,
            "baseline_model_compiled_inference_time_in_ms": self.baseline_model_compiled_inference_time_in_ms,
            "quantized_model_compiled_inference_time_in_ms": self.quantized_model_compiled_inference_time_in_ms,
            "eager speedup on baseline": self.eager_speedup_on_baseline,
            "compile speedup on baseline": self.compile_speedup_on_baseline,
            "eager vs compile speedup": self.compile_speedup_on_eager,
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
        return Int4WeightOnlyConfig(group_size=group_size, use_hqq=use_hqq, version=1)
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
            from torchao.prototype.dtypes import MarlinQQQLayout

            return Int8DynamicActivationInt4WeightConfig(
                group_size=128,
                mapping_type=MappingType.SYMMETRIC,
                act_mapping_type=MappingType.SYMMETRIC,
                layout=MarlinQQQLayout(),
            )
        elif sparsity is not None and ("semi" in sparsity or "2:4" in sparsity):
            from torchao.dtypes import MarlinSparseLayout

            return Int4WeightOnlyConfig(layout=MarlinSparseLayout(), version=1)
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
            intx_packing_format="opaque_torchao_auto",
        )
    elif "float8wo" in quantization:
        return Float8WeightOnlyConfig()
    elif quantization == "float8_a1x128_w128x128":
        # Blockwise float8 quantization with 1x128 activation and 128x128 weight blocks
        from torchao.quantization import PerBlock

        return Float8DynamicActivationFloat8WeightConfig(
            granularity=(PerBlock([1, 128]), PerBlock([128, 128])),
            activation_value_lb=1e-12,
        )
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
    if "codebook" in quantization:
        # Codebook quantization (prototype)
        # Format: codebook or codebook-<scale_block_size>
        from torchao.prototype.quantization.codebook import codebook_weight_only

        params = quantization.split("-")
        scale_block_size = int(params[1]) if len(params) > 1 else 64
        return codebook_weight_only(dtype=torch.uint4, scale_block_size=scale_block_size)
    return None


def _requires_calibration(quantization: Optional[str]) -> bool:
    """Check if the quantization method requires calibration data."""
    if quantization is None:
        return False
    calibration_methods = ["gptq", "autoround", "awq-uintx", "smoothquant"]
    return any(method in quantization.lower() for method in calibration_methods)


def _apply_spinquant(model: torch.nn.Module) -> torch.nn.Module:
    """Apply SpinQuant rotation transform to the model.

    SpinQuant applies a rotation transform before quantization to reduce
    quantization error. This is a preprocessing step, not quantization itself.

    Args:
        model: The model to apply SpinQuant to

    Returns:
        The model with SpinQuant applied
    """
    from torchao.prototype.spinquant import apply_spinquant

    apply_spinquant(model)
    return model


def _apply_gptq(
    model: torch.nn.Module,
    tokenizer,
    quantization: str,
    calibration_tasks: List[str],
    calibration_limit: int,
    calibration_seq_length: int,
    pad_calibration_inputs: bool,
    device: str,
    input_prep_func=None,
) -> torch.nn.Module:
    """Apply GPTQ quantization with calibration.

    Format: int4wo-<groupsize>-gptq

    Args:
        model: The model to quantize
        tokenizer: Tokenizer for encoding calibration data
        quantization: Quantization string (e.g., "int4wo-128-gptq")
        calibration_tasks: Tasks to use for calibration (e.g., ["wikitext"])
        calibration_limit: Number of calibration samples
        calibration_seq_length: Sequence length for calibration
        pad_calibration_inputs: Whether to pad short sequences
        device: Device to run calibration on
        input_prep_func: Optional function to prepare inputs for the model

    Returns:
        The quantized model
    """
    from torchao._models._eval import LMEvalInputRecorder
    from torchao.quantization.GPTQ import Int4WeightOnlyGPTQQuantizer

    # Parse group size from quantization string: int4wo-<groupsize>-gptq
    parts = quantization.split("-")
    groupsize = int(parts[1])
    assert groupsize in [32, 64, 128, 256], (
        f"int4wo groupsize needs to be one of [32,64,128,256] but got {groupsize}"
    )

    # Default input prep function if not provided
    if input_prep_func is None:
        input_prep_func = lambda x: (x,)

    # Get vocab size from model config
    vocab_size = getattr(model.config, "vocab_size", 32000)

    # Record calibration inputs
    inputs = (
        LMEvalInputRecorder(
            tokenizer,
            calibration_seq_length,
            input_prep_func,
            vocab_size,
            pad_calibration_inputs,
            device="cpu",
        )
        .record_inputs(
            calibration_tasks,
            calibration_limit,
        )
        .get_recorded_inputs()
    )
    print("Obtained calibration inputs, starting GPTQ quantization")

    # Setup caches if model supports it
    if hasattr(model, "setup_caches"):
        model.setup_caches(max_batch_size=1, max_seq_length=calibration_seq_length)

    # Quantize with GPTQ
    quantizer = Int4WeightOnlyGPTQQuantizer(group_size=groupsize, device=device)
    quantizer.quantize(model, *inputs)
    model = model.to(device)

    return model


def _apply_autoround(
    model: torch.nn.Module,
    tokenizer,
    quantization: str,
    device: str,
    quant_lm_head: bool = False,
) -> torch.nn.Module:
    """Apply AutoRound quantization with calibration.

    Format: autoround or autoround-<device>-<quant_lm_head>-<iters>-<groupsize>-<batch_size>-<seqlen>-<nsamples>-<grad_acc_steps>-<compile>

    Args:
        model: The model to quantize
        tokenizer: Tokenizer for encoding calibration data
        quantization: Quantization string with optional parameters
        device: Device to run calibration on
        quant_lm_head: Whether to quantize the lm_head layer

    Returns:
        The quantized model
    """
    from torchao.prototype.autoround.autoround_llm import quantize_model_with_autoround_

    # Parse args from quantization string
    _quant_args = quantization.split("-")
    _default_quant_args = [False, 200, 128, 8, 2048, 128, 1, 0]
    _model_device = _quant_args[1] if len(_quant_args) > 1 else device
    _quant_args = _quant_args[2:]

    (
        quant_lm_head_arg,
        iters,
        groupsize,
        batch_size,
        seqlen,
        nsamples,
        grad_acc_steps,
        compile_optimization_process,
    ) = [int(x) for x in _quant_args] + _default_quant_args[len(_quant_args) :]

    # Override quant_lm_head if explicitly passed
    quant_lm_head = quant_lm_head or bool(quant_lm_head_arg)

    model = model.to(_model_device)
    print(
        f"Quantizing model with AutoRound(iters={iters}, groupsize={groupsize}, "
        f"quant_lm_head={quant_lm_head}, batch_size={batch_size}, seqlen={seqlen}, "
        f"nsamples={nsamples}, gradient_accumulate_steps={grad_acc_steps}, "
        f"compile_optimization_process={compile_optimization_process})"
    )

    # Setup caches if model supports it
    if hasattr(model, "setup_caches"):
        with torch.device(_model_device):
            model.setup_caches(max_batch_size=batch_size, max_seq_length=seqlen, training=True)

    # Determine target modules based on model architecture
    # Try to find the decoder block class dynamically
    decoder_cls = None
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if "Block" in cls_name or "Layer" in cls_name:
            decoder_cls = type(module)
            break

    if decoder_cls is not None:
        if quant_lm_head:
            is_target_module = lambda mod, fqn: isinstance(mod, decoder_cls) or "lm_head" in fqn or "output" in fqn
        else:
            is_target_module = lambda mod, fqn: isinstance(mod, decoder_cls)
    else:
        # Fallback: quantize all Linear layers except embeddings
        if quant_lm_head:
            is_target_module = lambda mod, fqn: isinstance(mod, torch.nn.Linear)
        else:
            is_target_module = lambda mod, fqn: isinstance(mod, torch.nn.Linear) and "lm_head" not in fqn

    quantize_model_with_autoround_(
        model=model,
        tokenizer=tokenizer,
        is_target_module=is_target_module,
        bits=4,
        seqlen=seqlen,
        batch_size=batch_size,
        iters=iters,
        nsamples=nsamples,
        gradient_accumulate_steps=grad_acc_steps,
        compile_optimization_process=compile_optimization_process == 1,
    )

    model.to(device)
    if hasattr(model, "reset_caches"):
        model.reset_caches()

    return model


def _apply_awq(
    model: torch.nn.Module,
    tokenizer,
    quantization: str,
    eval_wrapper_cls,
    max_seq_length: int,
    device: str,
    input_prep_func=None,
) -> torch.nn.Module:
    """Apply AWQ quantization with calibration.

    Format: awq-uintx-<dtype>-<groupsize> or awq-uintx-<dtype>-<groupsize>-hqq

    Args:
        model: The model to quantize
        tokenizer: Tokenizer for encoding calibration data
        quantization: Quantization string (e.g., "awq-uintx-uint4-64")
        eval_wrapper_cls: The evaluation wrapper class to use for calibration
        max_seq_length: Maximum sequence length for calibration
        device: Device to run calibration on
        input_prep_func: Optional function to prepare inputs for the model

    Returns:
        The quantized model
    """
    from torchao.prototype.awq import (
        AWQObservedLinear,
        awq_uintx,
        insert_awq_observer_,
    )
    from torchao.quantization import quantize_

    # Parse quantization string: awq-uintx-<dtype>-<groupsize>[-hqq]
    parts = quantization.split("-")
    quant_dtype_str = parts[2] if len(parts) > 2 else "uint4"
    group_size = int(parts[3]) if len(parts) > 3 else 64
    use_hqq = "hqq" in quantization

    quant_dtype = getattr(torch, quant_dtype_str, torch.uint4)

    model = model.to(device)

    # Insert AWQ observers
    insert_awq_observer_(model, 1, 256, quant_dtype=quant_dtype, group_size=group_size)

    # Run calibration
    eval_wrapper_cls(
        model=model,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        input_prep_func=input_prep_func,
        device=device,
    ).run_eval(
        tasks=["wikitext"],
        limit=1,
    )

    # Convert observed model to quantized model
    is_observed_linear = lambda m, fqn: isinstance(m, AWQObservedLinear)
    quantize_(
        model,
        awq_uintx(quant_dtype=quant_dtype, group_size=group_size, use_hqq=use_hqq),
        is_observed_linear,
    )

    return model


def apply_quantization(
    model: torch.nn.Module,
    quantization: Optional[str],
    sparsity: Optional[str] = None,
    tokenizer=None,
    calibration_tasks: Optional[List[str]] = None,
    calibration_limit: Optional[int] = None,
    calibration_seq_length: Optional[int] = None,
    pad_calibration_inputs: bool = False,
    device: str = "cuda",
    input_prep_func=None,
    eval_wrapper_cls=None,
    max_seq_length: int = 2048,
    **kwargs,
) -> torch.nn.Module:
    """Apply quantization to a model, handling both simple configs and calibration-based methods.

    This is the main entry point for applying quantization. It handles:
    - Simple config-based quantization (int8wo, int4wo, float8, etc.)
    - Calibration-based quantization (GPTQ, AWQ, AutoRound)
    - Preprocessing transforms (SpinQuant)
    - Sparsity (semi-structured, block sparse)

    Args:
        model: The model to quantize
        quantization: Quantization method string. Supported formats:
            - Simple: "int8wo", "int8dq", "int4wo-128", "float8wo", "float8dq-row", etc.
            - GPTQ: "int4wo-128-gptq" (requires tokenizer and calibration params)
            - AWQ: "awq-uintx-uint4-64" (requires tokenizer and eval_wrapper_cls)
            - AutoRound: "autoround" or "autoround-cuda-1-200-128-8-2048-128-1-0"
            - SpinQuant: "spinquant" (preprocessing, can be combined with other methods)
        sparsity: Sparsity method ("semi", "2:4", "block")
        tokenizer: Tokenizer for calibration-based methods
        calibration_tasks: Tasks for calibration (e.g., ["wikitext"])
        calibration_limit: Number of calibration samples
        calibration_seq_length: Sequence length for calibration
        pad_calibration_inputs: Whether to pad short calibration sequences
        device: Device to run on
        input_prep_func: Function to prepare inputs for the model
        eval_wrapper_cls: Evaluation wrapper class for AWQ calibration
        max_seq_length: Maximum sequence length for AWQ calibration
        **kwargs: Additional arguments passed to string_to_config

    Returns:
        The quantized model

    Example:
        >>> # Simple quantization
        >>> model = apply_quantization(model, "int4wo-128")

        >>> # GPTQ quantization
        >>> model = apply_quantization(
        ...     model, "int4wo-128-gptq",
        ...     tokenizer=tokenizer,
        ...     calibration_tasks=["wikitext"],
        ...     calibration_limit=128,
        ...     calibration_seq_length=2048,
        ... )

        >>> # SpinQuant + int4
        >>> model = apply_quantization(model, "spinquant-int4wo-128", tokenizer=tokenizer)
    """
    from torchao.quantization import quantize_
    from torchao.sparsity.sparse_api import sparsify_

    if quantization is None and sparsity is None:
        return model

    # Handle SpinQuant preprocessing (can be combined with other quantization)
    if quantization and "spinquant" in quantization:
        model = _apply_spinquant(model)
        # Remove spinquant from quantization string for further processing
        quantization = quantization.replace("spinquant-", "").replace("spinquant", "")
        if not quantization:
            quantization = None

    # Handle calibration-based methods
    if quantization and "gptq" in quantization:
        if tokenizer is None:
            raise ValueError("GPTQ quantization requires a tokenizer")
        if calibration_tasks is None:
            calibration_tasks = ["wikitext"]
        if calibration_limit is None:
            calibration_limit = 128
        if calibration_seq_length is None:
            calibration_seq_length = 2048

        return _apply_gptq(
            model=model,
            tokenizer=tokenizer,
            quantization=quantization,
            calibration_tasks=calibration_tasks,
            calibration_limit=calibration_limit,
            calibration_seq_length=calibration_seq_length,
            pad_calibration_inputs=pad_calibration_inputs,
            device=device,
            input_prep_func=input_prep_func,
        )

    if quantization and "autoround" in quantization:
        if tokenizer is None:
            raise ValueError("AutoRound quantization requires a tokenizer")

        return _apply_autoround(
            model=model,
            tokenizer=tokenizer,
            quantization=quantization,
            device=device,
        )

    if quantization and quantization.startswith("awq-uintx"):
        if tokenizer is None:
            raise ValueError("AWQ quantization requires a tokenizer")
        if eval_wrapper_cls is None:
            from torchao._models._eval import TransformerEvalWrapper
            eval_wrapper_cls = TransformerEvalWrapper

        return _apply_awq(
            model=model,
            tokenizer=tokenizer,
            quantization=quantization,
            eval_wrapper_cls=eval_wrapper_cls,
            max_seq_length=max_seq_length,
            device=device,
            input_prep_func=input_prep_func,
        )

    # Handle simple config-based quantization and sparsity
    config = string_to_config(quantization, sparsity, **kwargs)

    if config is not None:
        # Check if it's a sparsity-only config
        if quantization is None and sparsity is not None:
            sparsify_(model, config)
        else:
            model = model.to(device)
            quantize_(model, config)

    return model


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
            f"{result.baseline_model_eager_inference_time_in_ms:.2f}",
            f"{result.quantized_model_eager_inference_time_in_ms:.2f}",
            f"{result.eager_speedup_on_baseline:.2f}x",
            f"{result.baseline_model_compiled_inference_time_in_ms:.2f}",
            f"{result.quantized_model_compiled_inference_time_in_ms:.2f}",
            f"{result.compile_speedup_on_baseline:.2f}x",
            f"{result.compile_speedup_on_eager:.2f}x",
            str(result.config.enable_profiler),
        ]

        table_data.append(row)

    # Define headers
    headers = [
        "Name",
        "Quantization",
        "Sparsity",
        "Shape",
        "Eager Baseline Inference Time (ms)",
        "Eager Model Inference Time (ms)",
        "Eager Speedup",
        "Compile Baseline Inference Time (ms)",
        "Compile Model Inference Time (ms)",
        "Compile Speedup",
        "Eager vs Compile Speedup",
        "Profiler Enabled",
    ]

    if table_data:
        print("\nBenchmark Results:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print("\nNo valid results to display")
