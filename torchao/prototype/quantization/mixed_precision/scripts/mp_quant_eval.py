import torch
import torch.nn as nn
from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict
from naive_intNwo import intN_weight_only
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchao.quantization import (
    quantize_,
)
from torchao.quantization.quant_api import autoquant

torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.fx_graph_cache = True


def run_evaluation(
    repo_id,
    tasks,
    limit,
    device,
    precision,
    quantization,
    compile,
    batch_size,
    max_length,
    sensi_bit,
    non_sensi_bit,
    quant_sym,
    group_size,
):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id).to(
        device="cpu", dtype=precision
    )

    if quantization == "autoquant":
        model = autoquant(model.to(device=device))

    # naive implementation of uniform precision quantization all layers
    elif quantization in ["2", "3", "4", "5", "6", "8"]:
        quantize_(
            model.to(device=device),
            intN_weight_only(
                n=int(quantization), group_size=group_size, symmetric=quant_sym
            ),
        )

    # mix precision quantization for Llama3
    elif quantization == "MP_llama3":
        # filter for sensitive layers (the first 3 and last 2 layers for Llama3)
        def filter_fn_sen(child: torch.nn.Module, cur_fqn: str) -> bool:
            return isinstance(child, nn.Linear) and any(
                skiplayer in cur_fqn
                for skiplayer in [".0.", ".1.", ".2.", ".30.", ".31."]
            )

        # filter for non-sensitive layers (other 27 layers for Llama3)
        def filter_fn_nonsen(child: torch.nn.Module, cur_fqn: str) -> bool:
            return isinstance(child, nn.Linear) and not (
                any(
                    skiplayer in cur_fqn
                    for skiplayer in [".0.", ".1.", ".2.", ".30.", ".31."]
                )
            )

        # quantize the sensitive layers
        if sensi_bit != 16:
            quantize_(
                model.to(device=device),
                intN_weight_only(
                    n=sensi_bit, group_size=group_size, symmetric=quant_sym
                ),
                filter_fn_sen,
            )

        # quantize the less-sensitive layers
        if sensi_bit == 4:
            quantize_(
                model,
                intN_weight_only(
                    n=non_sensi_bit, group_size=group_size, symmetric=quant_sym
                ),
                filter_fn_nonsen,
            )
        else:
            quantize_(
                model.to(device=device),
                intN_weight_only(
                    n=non_sensi_bit, group_size=group_size, symmetric=quant_sym
                ),
                filter_fn_nonsen,
            )

    if compile:
        model = torch.compile(model, mode="max-autotune", fullgraph=True)

    with torch.no_grad():
        result = evaluate(
            HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=max_length,
            ),
            get_task_dict(tasks),
            limit=limit,
        )

    for task, res in result["results"].items():
        print(f"{task}: {res}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run evaluation for uniform or mixed-precision quantization."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="checkpoints/meta-llama/Meta-Llama-3-8B",
        help="Repository ID to download from HF.",
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
        default="None",
        choices=["2", "3", "4", "5", "6", "8", "MP_llama3", "None"],
        help='Which quantization technique to apply, choose from ["2", "3", "4", "5", "6", "8"] for uniform quantizatoin, choose "MP_llama3" for mixed-precision for Llama3 and need to set corresponding sensi_bit and non_sensi_bit, choose "None" for no quantization',
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size to use for evaluation, note int8wo and int4wo work best with small batchsizes, int8dq works better with large batchsizes",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Length of text to process at one time",
    )
    parser.add_argument(
        "--sensi_bit",
        type=int,
        default=16,
        choices=[16, 8, 6, 5, 4, 3],
        help="Bit setting for sensitive layers",
    )
    parser.add_argument(
        "--non_sensi_bit",
        type=int,
        default=8,
        choices=[8, 6, 5, 4, 3, 2],
        help="Bit setting for non-sensitive layers",
    )
    parser.add_argument(
        "--quant_sym",
        type=bool,
        default=False,
        help="Symmetric or asymmetric quantization, asymmetric by default",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=32,
        help="Group size to perform quantization on",
    )
    args = parser.parse_args()
    run_evaluation(
        args.repo_id,
        args.tasks,
        args.limit,
        args.device,
        args.precision,
        args.quantization,
        args.compile,
        args.batch_size,
        args.max_length,
        args.sensi_bit,
        args.non_sensi_bit,
        args.quant_sym,
        args.group_size,
    )
