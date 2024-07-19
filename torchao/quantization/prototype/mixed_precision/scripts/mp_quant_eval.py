import torch
import torch.nn as nn

from naive_intNwo import intN_weight_only_asym, intN_weight_only_sym
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import evaluate
from lm_eval.tasks import get_task_dict

from torchao.quantization import quantize_, int8_weight_only, int4_weight_only, int8_dynamic_activation_int4_weight
from torchao._models._eval import TransformerEvalWrapper

from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
)

from torchao.quantization.quant_api import autoquant


torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.fx_graph_cache = True


def run_evaluation(repo_id, tasks, limit, device, precision, quantization, compile, batch_size, max_length, sensi_bit, non_sensi_bit, quant_sym, group_size):

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id).to(device="cpu", dtype=precision)

    if quantization == "int8dq":
        quantize_(model.to(device=device), int8_dynamic_activation_int4_weight())

    elif quantization == "int8wo":
        quantize_(model.to(device=device), int8_weight_only())

    elif quantization == "int4wo": 
        quantize_(model.to(device=device), int4_weight_only(group_size=group_size))
        
    elif quantization == "autoquant":
        model = autoquant(model.to(device=device))
        
    # naive implementation of uniform precision quantization all layers    
    elif quantization in ["2","3","4","5","6","8"]:
        if quant_sym == "asym":
            quantize_(model.to(device=device), intN_weight_only_asym(n=int(quantization), group_size=group_size))
        elif quant_sym == "sym":
            quantize_(model.to(device=device), intN_weight_only_sym(n=int(quantization), group_size=group_size))
        
    elif quantization == "MP_llama3":
        
        # filter for sensitive layers
        def filter_fn_sen(child: torch.nn.Module, cur_fqn:str) -> bool:
            return isinstance(child, nn.Linear) and any(skiplayer in cur_fqn for skiplayer in ['.0.', '.1.', '.2.', '.30.', '.31.'])
        
        # filter for non-sensitive layers
        def filter_fn_nonsen(child: torch.nn.Module, cur_fqn:str) -> bool:
            return isinstance(child, nn.Linear) and not(any(skiplayer in cur_fqn for skiplayer in ['.0.', '.1.', '.2.', '.30.', '.31.']))
        
        if sensi_bit != 16:
            # quantize the sensitive layers
            if sensi_bit == 8:
                quantize_(model.to(device=device), int8_weight_only(), filter_fn_sen)
            elif  sensi_bit == 4: 
                quantize_(model.to(device=device), int4_weight_only(group_size=group_size), filter_fn_sen)
            elif sensi_bit in [6,5,3,2]:
                if quant_sym == "asym":
                    quantize_(model.to(device=device), intN_weight_only_asym(n=sensi_bit, group_size=group_size), filter_fn_sen)
                elif quant_sym == "sym":
                    quantize_(model.to(device=device), intN_weight_only_sym(n=sensi_bit, group_size=group_size), filter_fn_sen)

        # quantize the less-sensitive layers
        if non_sensi_bit == 8:
            quantize_(model.to(device=device), int8_weight_only(), filter_fn_nonsen)
        elif  non_sensi_bit == 4: 
            quantize_(model.to(device=device), int4_weight_only(group_size=group_size), filter_fn_nonsen)
        elif non_sensi_bit in [6,5,3,2]:
            if sensi_bit == 4: 
                if quant_sym == "asym":
                    quantize_(model, intN_weight_only_asym(n=non_sensi_bit, group_size=group_size), filter_fn_nonsen)
                elif quant_sym == "sym":
                    quantize_(model, intN_weight_only_sym(n=non_sensi_bit, group_size=group_size), filter_fn_nonsen)     
            else:
                if quant_sym == "asym":
                    quantize_(model.to(device=device), intN_weight_only_asym(n=non_sensi_bit, group_size=group_size), filter_fn_nonsen)
                elif quant_sym == "sym":
                    quantize_(model.to(device=device), intN_weight_only_sym(n=non_sensi_bit, group_size=group_size), filter_fn_nonsen)   
    
    if compile:
        model = torch.compile(model, mode="max-autotune", fullgraph=True)

    with torch.no_grad():
        
        result = evaluate(
            HFLM(
                pretrained=model,
                tokenizer=tokenizer, 
                batch_size=batch_size, 
                max_length=max_length),
            get_task_dict(tasks),
            limit = limit,
        )

    for task, res in result["results"].items():
        print(f"{task}: {res}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run HF Model Evaluation')
    parser.add_argument('--repo_id', type=str, default="checkpoints/meta-llama/Meta-Llama-3-8B", help='Repository ID to download from HF.')
    parser.add_argument('--tasks', nargs='+', type=str, default=["wikitext"], help='List of lm-eluther tasks to evaluate usage: --tasks task1 task2')
    parser.add_argument('--limit', type=int, default=None, help='Number of eval samples to evaluate')
    parser.add_argument('--precision', type=lambda x: getattr(torch, x.split(".")[-1]), default=torch.bfloat16, help='dtype precision to use')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for evaluation')
    parser.add_argument('-q', '--quantization', default = "None", help='Which quantization technique to apply')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to use for evaluation, note int8wo and int4wo work best with small batchsizes, int8dq works better with large batchsizes')
    parser.add_argument('--max_length', type=int, default=None, help='Length of text to process at one time')
    parser.add_argument('--sensi_bit', type=int, default=16, help='Bit setting for sensitive layers')
    parser.add_argument('--non_sensi_bit', type=int, default=16, help='Bit setting for non-sensitive layers')
    parser.add_argument('--quant_sym', type=str, default="asym", help='symmetric or asymmetric quantization')
    parser.add_argument('--group_size', type=int, default=32, help='group size to perform quantization on')
    args = parser.parse_args()
    run_evaluation(args.repo_id, args.tasks, args.limit, args.device, args.precision, args.quantization, args.compile, args.batch_size, args.max_length, args.sensi_bit, args.non_sensi_bit, args.quant_sym, args.group_size)
