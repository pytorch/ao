import torch
import torch.nn as nn

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

from torchao.quantization.quant_api import (
    change_linear_weights_to_int4_woqtensors,
    change_linear_weights_to_int8_dqtensors,
    change_linear_weights_to_int8_woqtensors,
    autoquant,
)

torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.fx_graph_cache = True

def intN_weight_only(group_size=32, n=8):
    def apply_intN_weight_only_quant(weight):
        # avoid circular dep
        from torchao.dtypes import to_affine_quantized

        mapping_type = MappingType.ASYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.uint8
        quant_min = 0
        quant_max = 2**n-1
        eps = 1e-6
        preserve_zero = False
        zero_point_dtype = torch.bfloat16
        zero_point_domain = ZeroPointDomain.FLOAT
        return to_affine_quantized(weight, mapping_type, block_size, target_dtype, quant_min, quant_max, eps, zero_point_dtype=zero_point_dtype, preserve_zero=preserve_zero,zero_point_domain=zero_point_domain)

    return apply_intN_weight_only_quant



def run_evaluation(repo_id, tasks, limit, device, precision, quantization, compile, batch_size, max_length, layer, linear_type):

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id).to(device="cpu", dtype=precision)
 
    def filter_fn_sen(child: torch.nn.Module, cur_fqn:str) -> bool:
        #return isinstance(child, nn.Linear) and "."+str(layer)+"." in cur_fqn
        return isinstance(child, nn.Linear) and linear_type in cur_fqn and (".0." not in cur_fqn) and (".1." not in cur_fqn) and (".2." not in cur_fqn) and (".30." not in cur_fqn) and (".31." not in cur_fqn)

    if quantization in ["2","3","4","5","6","8"]:
        quantize_(model.to(device=device), intN_weight_only(n=int(quantization)), filter_fn_sen)

    if compile:
        model = torch.compile(model, mode="max-autotune", fullgraph=True)

    with torch.no_grad():
        
        result = evaluate(
            HFLM(
                pretrained=model,#.to(device), 
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
    parser.add_argument('--repo_id', type=str, default="meta-llama/Meta-Llama-3-8B", help='Repository ID to download from HF.')
    parser.add_argument('--tasks', nargs='+', type=str, default=["wikitext"], help='List of lm-eluther tasks to evaluate usage: --tasks task1 task2')
    parser.add_argument('--limit', type=int, default=None, help='Number of eval samples to evaluate')
    parser.add_argument('--precision', type=lambda x: getattr(torch, x.split(".")[-1]), default=torch.bfloat16, help='dtype precision to use')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for evaluation')
    parser.add_argument('-q', '--quantization', default = "None", choices=["2","3","4","5","6","8","MP_llama3", "None"], help='Which quantization technique to apply')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to use for evaluation, note int8wo and int4wo work best with small batchsizes, int8dq works better with large batchsizes')
    parser.add_argument('--max_length', type=int, default=None, help='Length of text to process at one time')
    parser.add_argument('--layer', type=int, default=0, help='The layer to quantize')
    parser.add_argument('--linear_type', type=str, default=0, choices=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], help='The linear type to quantize')

    args = parser.parse_args()
    run_evaluation(args.repo_id, args.tasks, args.limit, args.device, args.precision, args.quantization, args.compile, args.batch_size, args.max_length, args.layer, args.linear_type)
