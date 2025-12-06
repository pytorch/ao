# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified LLM Evaluation Script

This script provides a unified interface for evaluating language models with various
quantization methods. It supports both:
- gpt-fast format (.pth checkpoints)
- HuggingFace format (model IDs or local paths)

Usage:
    # gpt-fast checkpoint
    python -m benchmarks._models.llm_eval \
        --checkpoint_path /path/to/model.pth \
        --quantization int4wo-128 \
        --tasks wikitext

    # HuggingFace model
    python -m benchmarks._models.llm_eval \
        --model_id meta-llama/Llama-3.1-8B \
        --quantization int8wo \
        --tasks wikitext hellaswag

    # Auto-detect format
    python -m benchmarks._models.llm_eval \
        --model meta-llama/Llama-3.1-8B \
        --quantization int4wo-128 \
        --tasks wikitext
"""

import argparse
import itertools
import time
from pathlib import Path
from typing import List, Optional, Union

import torch

import torchao
from benchmarks.microbenchmarks.utils import apply_quantization


# =============================================================================
# Model Loading
# =============================================================================


def _is_hf_model_id(model_path: str) -> bool:
    """Check if the model path is a HuggingFace model ID."""
    # HF model IDs typically have format: org/model-name or just model-name
    # .pth files are gpt-fast format
    if model_path.endswith(".pth"):
        return False
    path = Path(model_path)
    # If it's a directory with model files, treat as local HF model
    if path.is_dir():
        return (path / "config.json").exists() or (path / "model.safetensors").exists()
    # If it contains a slash and doesn't exist as a file, assume HF model ID
    if "/" in model_path and not path.exists():
        return True
    # If it's a .pth file path (even if doesn't exist yet), it's gpt-fast
    return not str(model_path).endswith(".pth")


def load_model_gptfast(
    checkpoint_path: Path,
    device: str = "cpu",
    precision: torch.dtype = torch.bfloat16,
):
    """Load a gpt-fast format model (.pth checkpoint).

    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load model to
        precision: Model precision (dtype)

    Returns:
        Tuple of (model, tokenizer, input_prep_func)
    """
    # Import from the llama module - these are relative imports when running from the repo
    import sys

    # Add the llama directory to path for relative imports
    llama_dir = Path(__file__).parent.parent.parent / "torchao" / "_models" / "llama"
    if str(llama_dir) not in sys.path:
        sys.path.insert(0, str(llama_dir))

    from torchao._models.llama.generate import _load_model, device_sync
    from torchao._models.llama.model import prepare_inputs_for_model

    # Also need to get the tokenizer
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert checkpoint_path.is_file(), f"Checkpoint not found: {checkpoint_path}"
    assert tokenizer_path.is_file(), f"Tokenizer not found: {tokenizer_path}"

    print(f"Loading gpt-fast model from {checkpoint_path}...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision)
    device_sync(device=device)
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    # Load tokenizer
    from torchao._models.llama.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)

    return model, tokenizer, prepare_inputs_for_model


def load_model_huggingface(
    model_id: str,
    device: str = "cuda",
    precision: torch.dtype = torch.bfloat16,
):
    """Load a HuggingFace format model.

    Args:
        model_id: HuggingFace model ID or local path
        device: Device to load model to
        precision: Model precision (dtype)

    Returns:
        Tuple of (model, tokenizer, input_prep_func)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading HuggingFace model: {model_id}...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=precision,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    # HuggingFace models don't need input preparation
    input_prep_func = None

    return model, tokenizer, input_prep_func


def load_model(
    model: str,
    checkpoint_path: Optional[Path] = None,
    model_id: Optional[str] = None,
    device: str = "cuda",
    precision: torch.dtype = torch.bfloat16,
):
    """Load a model, auto-detecting the format.

    Args:
        model: Model path or ID (used for auto-detection)
        checkpoint_path: Explicit gpt-fast checkpoint path
        model_id: Explicit HuggingFace model ID
        device: Device to load model to
        precision: Model precision

    Returns:
        Tuple of (model, tokenizer, input_prep_func, model_format)
    """
    # Explicit checkpoint path takes precedence
    if checkpoint_path is not None:
        model, tokenizer, input_prep_func = load_model_gptfast(
            checkpoint_path, device="cpu", precision=precision
        )
        return model, tokenizer, input_prep_func, "gptfast"

    # Explicit model_id takes precedence
    if model_id is not None:
        model, tokenizer, input_prep_func = load_model_huggingface(
            model_id, device=device, precision=precision
        )
        return model, tokenizer, input_prep_func, "huggingface"

    # Auto-detect from model argument
    if model is not None:
        if _is_hf_model_id(model):
            model_obj, tokenizer, input_prep_func = load_model_huggingface(
                model, device=device, precision=precision
            )
            return model_obj, tokenizer, input_prep_func, "huggingface"
        else:
            model_obj, tokenizer, input_prep_func = load_model_gptfast(
                Path(model), device="cpu", precision=precision
            )
            return model_obj, tokenizer, input_prep_func, "gptfast"

    raise ValueError("Must provide either --model, --checkpoint_path, or --model_id")


# =============================================================================
# Model Size Calculation
# =============================================================================


def get_model_size_in_bytes(model: torch.nn.Module, ignore_embeddings: bool = False) -> int:
    """Calculate model size in bytes, handling quantized tensors.

    Args:
        model: The model to measure
        ignore_embeddings: Whether to ignore embedding layers

    Returns:
        Model size in bytes
    """

    def flat_size(tensor):
        if hasattr(tensor, "__tensor_flatten__"):
            size = 0
            for attr_name in tensor.__tensor_flatten__()[0]:
                sub_tensor = getattr(tensor, attr_name)
                size += flat_size(sub_tensor)
            return size
        else:
            return tensor.numel() * tensor.element_size()

    model_size = 0
    for _, child in model.named_children():
        if not (isinstance(child, torch.nn.Embedding) and ignore_embeddings):
            for p in itertools.chain(
                child.parameters(recurse=False), child.buffers(recurse=False)
            ):
                model_size += flat_size(p)
            model_size += get_model_size_in_bytes(child, ignore_embeddings)
    return model_size


# =============================================================================
# Evaluation
# =============================================================================


def run_evaluation(
    # Model specification (one of these required)
    model: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
    model_id: Optional[str] = None,
    # Evaluation parameters
    tasks: List[str] = None,
    limit: Optional[int] = None,
    # Model configuration
    device: str = "cuda",
    precision: torch.dtype = torch.bfloat16,
    # Quantization
    quantization: Optional[str] = None,
    sparsity: Optional[str] = None,
    # Compilation
    compile: bool = False,
    compile_mode: str = "max-autotune",
    # Sequence length
    max_length: Optional[int] = None,
    # Calibration (for GPTQ, AWQ, etc.)
    calibration_tasks: Optional[List[str]] = None,
    calibration_limit: Optional[int] = None,
    calibration_seq_length: Optional[int] = None,
    pad_calibration_inputs: bool = False,
    # Output
    print_model: bool = False,
    output_dir: Optional[str] = None,
):
    """Run LLM evaluation with optional quantization.

    This is the main entry point that supports both gpt-fast and HuggingFace models.

    Args:
        model: Model path or HuggingFace model ID (auto-detected)
        checkpoint_path: Explicit gpt-fast checkpoint path (.pth)
        model_id: Explicit HuggingFace model ID
        tasks: List of lm-eval tasks to run
        limit: Number of samples to evaluate (None = all)
        device: Device to run on
        precision: Model precision (torch.bfloat16, torch.float16, etc.)
        quantization: Quantization method string
        sparsity: Sparsity method string
        compile: Whether to torch.compile the model
        compile_mode: Compilation mode (max-autotune, reduce-overhead, etc.)
        max_length: Maximum sequence length for evaluation
        calibration_tasks: Tasks for calibration (GPTQ, AWQ, etc.)
        calibration_limit: Number of calibration samples
        calibration_seq_length: Sequence length for calibration
        pad_calibration_inputs: Whether to pad short calibration sequences
        print_model: Whether to print the model architecture
        output_dir: Directory to save quantized model (HF models only)

    Returns:
        Evaluation results dictionary
    """
    if tasks is None:
        tasks = ["wikitext"]

    print(
        f"\n{'='*60}\n"
        f"LLM Evaluation\n"
        f"{'='*60}\n"
        f"Model: {model or checkpoint_path or model_id}\n"
        f"Tasks: {tasks}\n"
        f"Limit: {limit}\n"
        f"Device: {device}\n"
        f"Precision: {precision}\n"
        f"Quantization: {quantization}\n"
        f"Sparsity: {sparsity}\n"
        f"Compile: {compile}\n"
        f"{'='*60}\n"
    )

    # Set recommended inductor config
    torchao.quantization.utils.recommended_inductor_config_setter()

    # Load model
    model_obj, tokenizer, input_prep_func, model_format = load_model(
        model=model,
        checkpoint_path=checkpoint_path,
        model_id=model_id,
        device=device,
        precision=precision,
    )

    # Set max_length from model config if not provided
    if max_length is None:
        if hasattr(model_obj, "config"):
            if hasattr(model_obj.config, "block_size"):
                max_length = model_obj.config.block_size
            elif hasattr(model_obj.config, "max_position_embeddings"):
                max_length = model_obj.config.max_position_embeddings
            else:
                max_length = 2048
        else:
            max_length = 2048

    # Apply quantization using the unified apply_quantization function
    if quantization or sparsity:
        print(f"Applying quantization: {quantization}, sparsity: {sparsity}")
        model_obj = apply_quantization(
            model=model_obj,
            quantization=quantization,
            sparsity=sparsity,
            tokenizer=tokenizer,
            calibration_tasks=calibration_tasks or ["wikitext"],
            calibration_limit=calibration_limit,
            calibration_seq_length=calibration_seq_length,
            pad_calibration_inputs=pad_calibration_inputs,
            device=device,
            input_prep_func=input_prep_func,
            max_seq_length=max_length,
        )

    # Compile model if requested
    if compile:
        print(f"Compiling model with mode: {compile_mode}")
        if quantization == "float8_a1x128_w128x128":
            model_obj = torch.compile(model_obj)
        else:
            model_obj = torch.compile(model_obj, mode=compile_mode, fullgraph=True)

    # Print model if requested
    if print_model:
        print(model_obj)

    # Calculate and print model size
    model_size = get_model_size_in_bytes(model_obj, ignore_embeddings=True) / 1e9
    print(f"Model size: {model_size:.2f} GB")

    # Move model to device
    model_obj = model_obj.to(device)

    # Run evaluation
    print("\nRunning evaluation...")
    with torch.no_grad():
        from torchao._models._eval import TransformerEvalWrapper

        wrapper = TransformerEvalWrapper(
            model=model_obj,
            tokenizer=tokenizer,
            max_seq_length=max_length,
            input_prep_func=input_prep_func,
            device=device,
        )
        result = wrapper.run_eval(tasks=tasks, limit=limit)

    return result


# =============================================================================
# CLI
# =============================================================================


def main():
    """Main entry point for CLI."""
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        print(
            "lm_eval is required to run this script. "
            "Please install it using: pip install lm-eval"
        )
        return

    parser = argparse.ArgumentParser(
        description="Unified LLM Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate gpt-fast checkpoint with int4 quantization
  python -m benchmarks._models.llm_eval \\
      --checkpoint_path checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth \\
      --quantization int4wo-128 \\
      --tasks wikitext

  # Evaluate HuggingFace model
  python -m benchmarks._models.llm_eval \\
      --model_id meta-llama/Llama-3.1-8B \\
      --quantization int8wo \\
      --tasks wikitext hellaswag

  # GPTQ quantization with calibration
  python -m benchmarks._models.llm_eval \\
      --checkpoint_path checkpoints/model.pth \\
      --quantization int4wo-128-gptq \\
      --calibration_tasks wikitext \\
      --calibration_limit 128 \\
      --calibration_seq_length 2048

Supported quantization methods:
  - int8wo, int8dq: INT8 weight-only / dynamic quantization
  - int4wo-<groupsize>: INT4 weight-only (e.g., int4wo-128)
  - int4wo-<groupsize>-hqq: INT4 with HQQ
  - int4wo-<groupsize>-gptq: INT4 with GPTQ calibration
  - float8wo, float8dq-tensor, float8dq-row: FP8 quantization
  - uintx-<bits>-<groupsize>: UIntX quantization (e.g., uintx-4-64)
  - marlin: Marlin sparse layout
  - spinquant: SpinQuant preprocessing (combinable, e.g., spinquant-int4wo-128)
  - autoround: AutoRound quantization
  - awq-uintx-<dtype>-<groupsize>: AWQ quantization
  - codebook: Codebook quantization
        """,
    )

    # Model specification (mutually exclusive group)
    model_group = parser.add_argument_group("Model Specification")
    model_group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path or HuggingFace model ID (auto-detected)",
    )
    model_group.add_argument(
        "--checkpoint_path",
        type=Path,
        default=None,
        help="Explicit gpt-fast checkpoint path (.pth file)",
    )
    model_group.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Explicit HuggingFace model ID",
    )

    # Evaluation parameters
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=["wikitext"],
        help="List of lm-eval tasks (default: wikitext)",
    )
    eval_group.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    eval_group.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length for evaluation",
    )

    # Device and precision
    device_group = parser.add_argument_group("Device & Precision")
    device_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    device_group.add_argument(
        "--precision",
        type=lambda x: getattr(torch, x.split(".")[-1]),
        default=torch.bfloat16,
        help="Model precision (default: bfloat16)",
    )

    # Quantization
    quant_group = parser.add_argument_group("Quantization")
    quant_group.add_argument(
        "-q",
        "--quantization",
        type=str,
        default=None,
        help="Quantization method (see examples below)",
    )
    quant_group.add_argument(
        "--sparsity",
        type=str,
        default=None,
        help="Sparsity method (semi, 2:4, block)",
    )

    # Compilation
    compile_group = parser.add_argument_group("Compilation")
    compile_group.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile",
    )
    compile_group.add_argument(
        "--compile_mode",
        type=str,
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (default: max-autotune)",
    )

    # Calibration (for GPTQ, AWQ, etc.)
    calib_group = parser.add_argument_group("Calibration (for GPTQ, AWQ, AutoRound)")
    calib_group.add_argument(
        "--calibration_tasks",
        nargs="+",
        type=str,
        default=["wikitext"],
        help="Tasks for calibration data (default: wikitext)",
    )
    calib_group.add_argument(
        "--calibration_limit",
        type=int,
        default=128,
        help="Number of calibration samples (default: 128)",
    )
    calib_group.add_argument(
        "--calibration_seq_length",
        type=int,
        default=2048,
        help="Sequence length for calibration (default: 2048)",
    )
    calib_group.add_argument(
        "--pad_calibration_inputs",
        action="store_true",
        help="Pad short calibration sequences",
    )

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--print_model",
        action="store_true",
        help="Print model architecture",
    )
    output_group.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save quantized model",
    )

    args = parser.parse_args()

    # Validate that at least one model source is provided
    if args.model is None and args.checkpoint_path is None and args.model_id is None:
        parser.error("Must provide one of: --model, --checkpoint_path, or --model_id")

    run_evaluation(
        model=args.model,
        checkpoint_path=args.checkpoint_path,
        model_id=args.model_id,
        tasks=args.tasks,
        limit=args.limit,
        device=args.device,
        precision=args.precision,
        quantization=args.quantization,
        sparsity=args.sparsity,
        compile=args.compile,
        compile_mode=args.compile_mode,
        max_length=args.max_length,
        calibration_tasks=args.calibration_tasks,
        calibration_limit=args.calibration_limit,
        calibration_seq_length=args.calibration_seq_length,
        pad_calibration_inputs=args.pad_calibration_inputs,
        print_model=args.print_model,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

