# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torchao
from pathlib import Path
from typing import List, Optional
from generate import (
    _load_model,
    device_sync,

)
from torchao.quantization.quant_api import (
    quantize_,
    int4_weight_only,
    int8_weight_only,
    int8_dynamic_activation_int8_weight,
    fpx_weight_only,
    uintx_weight_only,
    unwrap_tensor_subclass,
)
from torchao._models._eval import TransformerEvalWrapper, InputRecorder

from tokenizer import get_tokenizer
import time
from torchao.quantization.GPTQ import Int4WeightOnlyGPTQQuantizer
from torchao._models.llama.model import prepare_inputs_for_model
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

def run_evaluation(
    checkpoint_path: Path,
    tasks: List[str],
    limit: Optional[int] = None,
    device = "cuda",
    precision = torch.bfloat16,
    quantization: Optional[str] = None,
    compile=False,
    max_length=None,
    calibration_tasks: Optional[List[str]] = None,
    calibration_limit: Optional[int] = None,
    calibration_seq_length: Optional[int] = None,
    pad_calibration_inputs: Optional[bool] = False,
):
    """Runs the evaluation of a model using LM Eval."""

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

    device_sync(device=device) # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")
    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)


    if quantization:
        if "int8wo" in quantization:
            quantize_(model, int8_weight_only())
        if "int8dq" in quantization:
            quantize_(model, int8_dynamic_activation_int8_weight())
        if "fp6" in quantization:
            quantize_(model, fpx_weight_only(3, 2))
        if "int4wo" in quantization and not "gptq" in quantization:
            if "hqq" in quantization:
                quantization = quantization[:-4]
                use_hqq = True
            else:
                use_hqq = False
            groupsize=int(quantization.split("-")[-1])
            assert groupsize in [32,64,128,256], f"int4wo groupsize needs to be one of [32,64,128,256] but got {groupsize}"
            quantize_(model.to(device), int4_weight_only(group_size=groupsize, use_hqq=use_hqq))
        if "uintx" in quantization:
            # uintx-nbits-groupsize
            # "uintx-2-64"
            if "hqq" in quantization:
                use_hqq = True
                quantization = quantization[:-4]
            else:
                use_hqq = False
            _quant_args = quantization.split("-")
            nbits = int(_quant_args[0])
            _NBITS_TO_DTYPE = {1: torch.uint1, 2: torch.uint2, 3: torch.uint3, 4: torch.uint4, 5: torch.uint5, 6: torch.uint6, 7: torch.uint7, 8: torch.uint8}
            dtype = _NBITS_TO_DTYPE[nbits]
            group_size = int(_quant_args[1])
            quantize_(model, uintx_weight_only(dtype, group_size, use_hqq=use_hqq))
        if "int4wo" in quantization and "gptq" in quantization:
            groupsize=int(quantization.split("-")[-2])
            assert groupsize in [32,64,128,256], f"int4wo groupsize needs to be one of [32,64,128,256] but got {groupsize}"
            assert precision==torch.bfloat16, f"{quantization} requires precision or bfloat16 but got {precision}"
            assert "cuda" in device, "int4 gptq quantization only works on cuda"
            inputs = InputRecorder(
                tokenizer,
                calibration_seq_length,
                prepare_inputs_for_model,
                pad_calibration_inputs,
                model.config.vocab_size,
                device="cpu"
            ).record_inputs(
                calibration_tasks,
                calibration_limit,
            ).get_inputs()

            quantizer = Int4WeightOnlyGPTQQuantizer(groupsize=groupsize, device=device)
            model.setup_caches(max_batch_size=1, max_seq_length=calibration_seq_length)
            model = quantizer.quantize(model, inputs).to(device)
        else:
            if not TORCH_VERSION_AT_LEAST_2_5:
                unwrap_tensor_subclass(model)

    if compile:
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
    with torch.no_grad():
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run HF Model Evaluation')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("../../../checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--tasks', nargs='+', type=str, default=["wikitext"], help='List of lm-eluther tasks to evaluate usage: --tasks task1 task2')
    parser.add_argument('--limit', type=int, default=None, help='Number of eval samples to evaluate')
    parser.add_argument('--precision', type=lambda x: getattr(torch, x.split(".")[-1]), default=torch.bfloat16, help='dtype precision to use')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for evaluation')
    parser.add_argument("-q", "--quantization", type=str, help="Which quantization techniques to apply: int8dq, int8wo, int4wo-<groupsize>, int4wo-<groupsize>-gptq, int4wo-<groupsize>-hqq, uintx-<nbits>-<groupsize>, uintx-<nbits>-<groupsize>-hqq")
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--max_length', type=int, default=None, help='Length of text to process at one time')
    parser.add_argument('--calibration_tasks', type=str, nargs='+', default=['wikitext'], help='tasks to do gptq calibration on, if doing gptq')
    parser.add_argument('--calibration_limit', type=int, default=1000, help='number of samples to use for gptq calibration')
    parser.add_argument('--calibration_seq_length', type=int, default=100, help='length of sequences to use for gptq calibration')
    parser.add_argument('--pad_calibration_inputs', type=bool, default=False, help='pads sequences shorter than calibration_seq_length to that length, yielding more calibration inputs but running much slower')

    args = parser.parse_args()
    run_evaluation(
        args.checkpoint_path,
        args.tasks,
        args.limit,
        args.device,
        args.precision,
        args.quantization,
        args.compile,
        args.max_length,
        args.calibration_tasks,
        args.calibration_limit,
        args.calibration_seq_length,
        args.pad_calibration_inputs,
    )
