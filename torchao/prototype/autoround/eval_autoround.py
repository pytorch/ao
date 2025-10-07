# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import logging
import os

import torch

import torchao
import torchao.prototype.autoround.utils as ar_utils
import torchao.quantization

logger = logging.getLogger(__name__)

ar_utils.freeze_random(42)


def _use_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    logger.warning(
        (
            "Reproducibility is enabled with `AO_USE_DETERMINISTIC_ALGORITHMS=1`, which sets "
            "`torch.use_deterministic_algorithms(True, warn_only=False)` and "
            "environment variable `CUBLAS_WORKSPACE_CONFIG` to `:4096:8`.\n"
            "Please note that this may impact performance, or cause crashes if the model includes non-deterministic operations."
        )
    )


AO_USE_DETERMINISTIC_ALGORITHMS = (
    os.environ.get("AO_USE_DETERMINISTIC_ALGORITHMS", "0") == "1"
)
if AO_USE_DETERMINISTIC_ALGORITHMS:
    _use_deterministic()


@ar_utils.dump_elapsed_time()
def run_evaluation(model, tokenizer, tasks, compile=False, batch_size=4):
    try:
        from lm_eval.evaluator import evaluate
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import get_task_dict
    except ImportError:
        print(
            """
    Error: The 'lm_eval' module was not found.
    To install, follow these steps:
    pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
    """
        )
        raise  # Re-raise the ImportError

    with torch.no_grad():
        result = evaluate(
            HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size),
            get_task_dict(tasks),
        )
        torch.cuda.empty_cache()
        from lm_eval.utils import make_table

        print(make_table(result))


def bench_accuracy(model, tokenizer, tasks, msg=""):
    with torch.no_grad():
        print(f"==================== {msg} ====================")
        print(f"tasks: {tasks}")
        from torchao.prototype.autoround.hf_eval_utils import run_evaluation

        torch.cuda.empty_cache()
        run_evaluation(model, tokenizer, tasks=tasks)
        torch.cuda.empty_cache()


def _is_linear_but_not_lm_head(mod, fqn):
    return isinstance(mod, torch.nn.Linear) and "lm_head" not in fqn


def main(args):
    with torch.no_grad():
        model_name_or_path = args.model_name_or_path
        model, tokenizer, decoder_cls = ar_utils.get_float_model_info(
            model_name_or_path, dtype=torch.bfloat16
        )
        model.eval()
        model_device = args.model_device
        # `sorted_logits` does not have a deterministic implementation
        if not AO_USE_DETERMINISTIC_ALGORITHMS:
            ar_utils.gen_text(model, tokenizer, "Float model", max_length=50)
        model = model.to(model_device)
        model.config.use_cache = False
        msg = "Float-model" if args.eval_float_model else "Quantized-model"
        if not args.eval_float_model:
            filter_fn = None if args.quant_lm_head else _is_linear_but_not_lm_head
            # Evaluate the quantized model
            if args.woq_int4:
                msg += " (int4wo)"
                from torchao.quantization import Int4WeightOnlyConfig, quantize_

                quantize_(
                    model,
                    Int4WeightOnlyConfig(group_size=args.group_size, version=1),
                    filter_fn=filter_fn,
                    device=model_device,
                )
            elif args.uintx:
                msg += f" (uintx {args.bits} bits)"
                from torchao.dtypes.uintx.uintx import _BIT_WIDTH_TO_DTYPE
                from torchao.quantization.quant_api import (
                    UIntXWeightOnlyConfig,
                    quantize_,
                )

                bits = args.bits
                assert bits in _BIT_WIDTH_TO_DTYPE, f"Invalid bits: {bits}"
                dtype = _BIT_WIDTH_TO_DTYPE[bits]
                quantize_(
                    model,
                    UIntXWeightOnlyConfig(dtype=dtype, group_size=args.group_size),
                    filter_fn=filter_fn,
                    device=model_device,
                )

            else:
                msg += f" (auto-round {args.bits} bits)"
                torch.cuda.empty_cache()
                from torchao.prototype.autoround.autoround_llm import (
                    quantize_model_with_autoround_,
                )

                # User need to prepare a `is_target_module` function for identifying the target modules that need to be quantized.
                if args.quant_lm_head:
                    is_target_module = (
                        lambda mod, fqn: isinstance(mod, decoder_cls)
                        or "lm_head" in fqn
                    )
                else:
                    is_target_module = lambda mod, fqn: isinstance(mod, decoder_cls)

                model = quantize_model_with_autoround_(
                    model=model,
                    tokenizer=tokenizer,
                    is_target_module=is_target_module,
                    bits=args.bits,
                    group_size=args.group_size,
                    iters=args.iters,
                    seqlen=args.seqlen,
                    batch_size=args.batch_size,
                    nsamples=args.nsamples,
                    use_optimized_layer_output=args.use_optimized_layer_output,
                    gradient_accumulate_steps=args.gradient_accumulate_steps,
                    compile_optimization_process=args.compile_optimization_process,
                )
            quantized_layer_cnt = ar_utils.count_tensor_of_type(
                model, torchao.dtypes.AffineQuantizedTensor
            )
            msg += f" quantized {quantized_layer_cnt} Linear layers "
        if not AO_USE_DETERMINISTIC_ALGORITHMS:
            ar_utils.gen_text(model, tokenizer, msg, max_length=50)

        bench_accuracy(model, tokenizer, tasks=args.tasks, msg=msg)


if __name__ == "__main__" and torch.cuda.is_available():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        default="facebook/opt-125m",
        help="Pretrained model name or path",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="NeelNanda/pile-10k",
        help="Dataset name for calibration",
    )
    parser.add_argument(
        "--iters",
        default=200,
        type=int,
        help="Number of steps for optimizing each block",
    )
    parser.add_argument(
        "--bits", default=4, type=int, help="Number of bits for quantization"
    )
    parser.add_argument(
        "--batch_size", default=8, type=int, help="Batch size for calibration"
    )
    parser.add_argument(
        "--nsamples",
        default=128,
        type=int,
        help="Number of samples for calibration process",
    )
    parser.add_argument(
        "--group_size",
        default=128,
        type=int,
        help="Group size for quantization",
    )
    parser.add_argument(
        "--seqlen",
        default=2048,
        type=int,
        help="Sequence length for each samples",
    )
    parser.add_argument(
        "--gradient_accumulate_steps",
        default=1,
        type=int,
        help=(
            "Number of steps for accumulating gradients before performing"
            "the backward pass when optimizing each target module"
        ),
    )
    parser.add_argument(
        "--quant_lm_head",
        default=False,
        action="store_true",
        help="Whether to quantize the `lm_head`",
    )
    parser.add_argument(
        "--use_optimized_layer_output",
        default=False,
        action="store_true",
        help="Whether to use optimized layer output as input for the next layer",
    )
    parser.add_argument(
        "-c",
        "--compile_optimization_process",
        default=False,
        action="store_true",
        help="Whether to compile the optimization process",
    )
    parser.add_argument(
        "-d",
        "--model_device",
        default="cuda",
        type=str,
        choices=["cpu", "cuda"],
        help="Device for loading the float model",
    )
    parser.add_argument(
        "--eval_float_model",
        default=False,
        action="store_true",
        help="Evaluate the float model",
    )
    parser.add_argument(
        "--woq_int4",
        default=False,
        action="store_true",
        help="Quantize the model with int4 weight only",
    )
    parser.add_argument(
        "--uintx",
        default=False,
        action="store_true",
        help="Quantize the model with int4 weight only",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=["wikitext"],
        help="List of lm-eluther tasks to evaluate usage: --tasks task1 task2",
    )
    args = parser.parse_args()

    main(args)
