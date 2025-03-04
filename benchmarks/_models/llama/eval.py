# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
from pathlib import Path
from typing import List, Optional

import torch
from tokenizer import get_tokenizer

import torchao
from benchmarks._models.llama.model import prepare_inputs_for_model
from benchmarks._models.utils import (
    _load_model,
)
from torchao.quantization import (
    PerRow,
    PerTensor,
    float8_dynamic_activation_float8_weight,
    float8_weight_only,
    fpx_weight_only,
    int4_weight_only,
    int8_dynamic_activation_int8_weight,
    int8_weight_only,
    quantize_,
    uintx_weight_only,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    device_sync,
    unwrap_tensor_subclass,
)


def run_evaluation(
    checkpoint_path: Path,
    tasks: List[str],
    limit: Optional[int] = None,
    device="cuda",
    precision=torch.bfloat16,
    quantization: Optional[str] = None,
    sparsity: Optional[str] = None,
    compile=False,
    max_length=None,
    calibration_tasks: Optional[List[str]] = None,
    calibration_limit: Optional[int] = None,
    calibration_seq_length: Optional[int] = None,
    pad_calibration_inputs: Optional[bool] = False,
):
    """Runs the evaluation of a model using LM Eval."""
    print(
        f"\nEvaluating model {checkpoint_path} on tasks: {tasks}, limit: {limit}, device: {device}, precision: {precision}, "
        + f"quantization: {quantization}, sparsity: {sparsity}, compile: {compile}, max_length: {max_length}, calibration_tasks: {calibration_tasks}, "
        + f"calibration_seq_length: {calibration_seq_length}, pad_calibration_inputs: {pad_calibration_inputs}\n"
    )
    torchao.quantization.utils.recommended_inductor_config_setter()

    assert checkpoint_path.is_file(), checkpoint_path
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)
    # Load Model and Tokenizer
    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, "cpu", precision)

    if max_length is None:
        max_length = model.config.block_size
    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")
    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)

    if quantization:
        if "spinquant" in quantization:
            from torchao.prototype.spinquant import apply_spinquant

            apply_spinquant(model)
        if "int8wo" in quantization:
            quantize_(model, int8_weight_only())
        if "int8dq" in quantization:
            quantize_(model, int8_dynamic_activation_int8_weight())
        if "fp6" in quantization:
            quantize_(model, fpx_weight_only(3, 2))
        if "int4wo" in quantization and not "gptq" in quantization:
            if "hqq" in quantization:
                use_hqq = True
            else:
                use_hqq = False
            groupsize = int(quantization.split("-")[1])
            assert (
                groupsize in [32, 64, 128, 256]
            ), f"int4wo groupsize needs to be one of [32,64,128,256] but got {groupsize}"
            quantize_(
                model.to(device),
                int4_weight_only(group_size=groupsize, use_hqq=use_hqq),
            )
        if "uintx" in quantization:
            # uintx-nbits-groupsize
            # "uintx-2-64"
            if "hqq" in quantization:
                use_hqq = True
            else:
                use_hqq = False
            _quant_args = quantization.split("-")
            nbits = int(_quant_args[1])
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
        if "marlin" in quantization:
            from torchao.dtypes import MarlinSparseLayout

            quantize_(model, int4_weight_only(layout=MarlinSparseLayout()))
        if "int4wo" in quantization and "gptq" in quantization:
            # avoid circular imports
            from benchmarks._models._eval import MultiTensorInputRecorder
            from torchao.quantization.GPTQ_MT import Int4WeightOnlyGPTQQuantizer

            groupsize = int(quantization.split("-")[-2])
            assert (
                groupsize in [32, 64, 128, 256]
            ), f"int4wo groupsize needs to be one of [32,64,128,256] but got {groupsize}"
            assert (
                precision == torch.bfloat16
            ), f"{quantization} requires precision or bfloat16 but got {precision}"
            assert "cuda" in device, "int4 gptq quantization only works on cuda"
            inputs = (
                MultiTensorInputRecorder(
                    tokenizer,
                    calibration_seq_length,
                    prepare_inputs_for_model,
                    pad_calibration_inputs,
                    model.config.vocab_size,
                    device="cpu",
                )
                .record_inputs(
                    calibration_tasks,
                    calibration_limit,
                )
                .get_inputs()
            )

            quantizer = Int4WeightOnlyGPTQQuantizer(group_size=groupsize, device=device)
            model.setup_caches(max_batch_size=1, max_seq_length=calibration_seq_length)
            model = quantizer.quantize(model, inputs).to(device)
        else:
            if not TORCH_VERSION_AT_LEAST_2_5:
                unwrap_tensor_subclass(model)
        if "float8wo" in quantization:
            quantize_(model, float8_weight_only())
        if "float8dq" in quantization:
            granularity = str(quantization.split("-")[-1])
            if granularity == "tensor":
                granularity = PerTensor()
            elif granularity == "row":
                granularity = PerRow()
            else:
                if granularity == "float8dq":
                    granularity = PerTensor()
                else:
                    raise ValueError(f"Unknown granularity {granularity}")
            quantize_(
                model, float8_dynamic_activation_float8_weight(granularity=granularity)
            )
        if "autoround" in quantization:
            from transformers import AutoTokenizer

            from benchmarks._models.llama.model import TransformerBlock
            from torchao.prototype.autoround.autoround_llm import (
                quantize_model_with_autoround_,
            )

            _tokenizer = AutoTokenizer.from_pretrained(checkpoint_path.parent)
            # parse args from quantization string:
            #   autoround-<model_device>-<quant_lm_head>-<iters>-<groupsize>-<batch_size>-<seqlen>-<nsamples>-<grad_acc_steps>-<c>
            _quant_args = quantization.split("-")
            _default_quant_args = [False, 200, 128, 8, 2048, 128, 1, 0]
            _model_devie = _quant_args[1] if len(_quant_args) > 1 else device
            _quant_args = _quant_args[2:]
            (
                quant_lm_head,
                iters,
                groupsize,
                batch_size,
                seqlen,
                nsamples,
                grad_acc_steps,
                compile_optimization_process,
            ) = [int(x) for x in _quant_args] + _default_quant_args[len(_quant_args) :]
            model = model.to(_model_devie)
            print(
                (
                    f"Quantizing model with autoround(iters={iters}, groupsize={groupsize}, "
                    f"quant_lm_head={quant_lm_head}, batch_size={batch_size}, seqlen={seqlen}, nsamples={nsamples}, "
                    f"gradient_accumulate_steps={grad_acc_steps}, "
                    f"compile_optimization_process={compile_optimization_process})"
                )
            )
            with torch.device(_model_devie):
                model.setup_caches(
                    max_batch_size=batch_size, max_seq_length=seqlen, training=True
                )

            if quant_lm_head:
                is_target_module = (
                    lambda mod, fqn: isinstance(mod, TransformerBlock)
                    or "output" in fqn
                )
            else:
                is_target_module = lambda mod, fqn: isinstance(mod, TransformerBlock)
            quantize_model_with_autoround_(
                model=model,
                tokenizer=_tokenizer,
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
            model.reset_caches()
        if "codebook" in quantization:
            from torchao.prototype.quantization.codebook import codebook_weight_only

            model.to(device)
            quantize_(
                model, codebook_weight_only(dtype=torch.uint4, scale_block_size=64)
            )

    if compile:
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
    with torch.no_grad():
        print("Running evaluation ...")
        # avoid circular imports
        from benchmarks._models._eval import TransformerEvalWrapper

        TransformerEvalWrapper(
            model=model.to(device),
            tokenizer=tokenizer,
            max_seq_length=max_length,
            input_prep_func=prepare_inputs_for_model,
            device=device,
        ).run_eval(
            tasks=tasks,
            limit=limit,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run HF Model Evaluation")
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("../../../checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=["wikitext"],
        help="List of lm-eluther tasks to evaluate usage: --tasks task1 task2",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Number of eval samples to evaluate"
    )
    parser.add_argument(
        "--precision",
        type=lambda x: getattr(torch, x.split(".")[-1]),
        default=torch.bfloat16,
        help="dtype precision to use",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for evaluation"
    )
    parser.add_argument(
        "-q",
        "--quantization",
        type=str,
        help=(
            "Which quantization techniques to apply: int8dq, int8wo, fp6, int4wo-<groupsize>, "
            "int4wo-<groupsize>-gptq, autoquant, autoquant-int4, int4wo-<groupsize>-hqq, "
            "uintx-<nbits>-<groupsize>, uintx-<nbits>-<groupsize>-hqq, sparse-marlin, spinquant, "
            "autoround-<model_device>-<quant_lm_head>-<iters>-<groupsize>-<batch_size>-<seqlen>-<nsamples>-<grad_acc_steps>-<c>, "
            "float8wo, float8dq, float8saq"
        ),
    )
    parser.add_argument(
        "--sparsity",
        type=str,
        help=("Which sparsity techniques to apply: semi-structured"),
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Length of text to process at one time",
    )
    parser.add_argument(
        "--calibration_tasks",
        type=str,
        nargs="+",
        default=["wikitext"],
        help="tasks to do gptq calibration on, if doing gptq",
    )
    parser.add_argument(
        "--calibration_limit",
        type=int,
        default=1000,
        help="number of samples to use for gptq calibration",
    )
    parser.add_argument(
        "--calibration_seq_length",
        type=int,
        default=100,
        help="length of sequences to use for gptq calibration",
    )
    parser.add_argument(
        "--pad_calibration_inputs",
        type=bool,
        default=False,
        help="pads sequences shorter than calibration_seq_length to that length, yielding more calibration inputs but running much slower",
    )

    args = parser.parse_args()
    run_evaluation(
        args.checkpoint_path,
        args.tasks,
        args.limit,
        args.device,
        args.precision,
        args.quantization,
        args.sparsity,
        args.compile,
        args.max_length,
        args.calibration_tasks,
        args.calibration_limit,
        args.calibration_seq_length,
        args.pad_calibration_inputs,
    )
