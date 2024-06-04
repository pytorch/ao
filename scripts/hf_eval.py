import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import evaluate
from lm_eval.tasks import get_task_dict

from torchao.quantization.quant_api import (
    change_linear_weights_to_int4_woqtensors,
    change_linear_weights_to_int8_dqtensors,
    change_linear_weights_to_int8_woqtensors,
    autoquant,
)

torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.fx_graph_cache = True

def run_evaluation(repo_id, task_list, limit, device, precision, quantization, compile):

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id).to(device="cuda", dtype=precision)
    
    if compile:
        torch.compile(model, mode="max-autotune", fullgraph=True)

    if quantization == "int8dq":
        change_linear_weights_to_int8_dqtensors(model)
    elif quantization == "int8wo":
        change_linear_weights_to_int8_woqtensors(model)
    elif quantization == "int4wo": 
        change_linear_weights_to_int4_woqtensors(model)
    elif quantization == "autoquant":
        model = autoquant(model)

    with torch.no_grad():
        result = evaluate(
            HFLM(pretrained=model, tokenizer=tokenizer),
            get_task_dict(task_list),
            limit = limit
        )
    for task, res in result["results"].items():
        print(f"{task}: {res}")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run HF Model Evaluation')
    parser.add_argument('--repo_id', type=str, default="meta-llama/Meta-Llama-3-8B", help='Repository ID to download from HF.')
    parser.add_argument('--task_list', nargs='+', type=str, default=["wikitext"], help='List of lm-eluther tasks to evaluate usage: --tasks task1 task2')
    parser.add_argument('--limit', type=int, default=None, help='Number of eval samples to evaluate')
    parser.add_argument('--precision', type=lambda x: getattr(torch, x.split(".")[-1]), default=torch.bfloat16, help='dtype precision to use')
    parser.add_argument('--device', type=str, default="cuda", help='Dvice to use for evaluation')
    parser.add_argument('--quantization', default = "None", choices=["int8dq", "int8wo", "int4wo","autoquant", "None"], help='Which quantization technique to apply')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')

    args = parser.parse_args()
    run_evaluation(args.repo_id, args.task_list, args.limit, args.device, args.precision, args.quantization, args.compile)
