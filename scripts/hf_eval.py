import torch
from tabulate import tabulate

from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from lm_eval.models.huggingface import HFLM
    from lm_eval.evaluator import evaluate
    from lm_eval.tasks import get_task_dict
except ImportError as e:
    print("""
Error: The 'lm_eval' module was not found.
To install, follow these steps:
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
""")
    raise  # Re-raise the ImportError

from torchao.quantization import (
    int4_weight_only,
    int8_weight_only,
    int8_dynamic_activation_int8_weight,
    quantize_,
    autoquant,
    fpx_weight_only,
)
from torchao.sparsity import (
    sparsify_,
    semi_sparse_weight,
)

torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.fx_graph_cache = True

def pretty_print_nested_results(results, precision: int = 6):
    def format_value(value):
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        return value

    main_table = []
    for task, metrics in results["results"].items():
        subtable = [[k, format_value(v)] for k, v in metrics.items() if k != 'alias']
        subtable.sort(key=lambda x: x[0])  # Sort metrics alphabetically
        formatted_subtable = tabulate(subtable, tablefmt='grid')
        main_table.append([task, formatted_subtable])

    print(tabulate(main_table, headers=['Task', 'Metrics'], tablefmt='grid'))

def run_evaluation(repo_id, tasks, limit, device, precision, quantization, sparsity, compile, save, batch_size, max_length):

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=precision).to(device)

    if quantization == "autoquant" and compile:
        model = torch.compile(model, mode="max-autotune", fullgraph=True)

    if quantization == "int8dq":
        quantize_(model, int8_dynamic_activation_int8_weight())
    elif quantization == "int8wo":
        quantize_(model, int8_weight_only())
    elif quantization == "int4wo":
        # note cannot quantize this model on cpu and run it on cuda at this time
        quantize_(model.to(device=device), int4_weight_only())
    elif quantization == "fp6":
        quantize_(model, fpx_weight_only(3, 2))
    elif quantization == "autoquant":
        model = autoquant(model.to(device=device))
    elif quantization == "awq":
        from torchao.utils import TORCH_VERSION_AT_LEAST_2_3
        from torchao.prototype.awq.example import get_calib_dataset
        if not TORCH_VERSION_AT_LEAST_2_3:
            print("AWQ quantization requires torch2.3+")
            exit()
        from torchao.prototype.awq import insert_awq_observer_, awq_uintx, AWQObservedLinear
        quant_dtype = torch.uint4
        group_size = 64
        calibration_limit = 10
        calibration_seq_length = 1024
        model=model.to(device)
        insert_awq_observer_(model,calibration_limit, calibration_seq_length, quant_dtype=quant_dtype, group_size=group_size)
        with torch.no_grad():
            calibration_data = get_calib_dataset(tokenizer=tokenizer, n_samples=calibration_limit, block_size=calibration_seq_length)
            for batch in calibration_data:
                model(batch.to(device))
                del batch
        is_observed_linear = lambda m, fqn: isinstance(m, AWQObservedLinear)
        quantize_(model, awq_uintx(quant_dtype=quant_dtype, group_size = group_size), is_observed_linear)

    if quantization != "autoquant" and compile:
        model = torch.compile(model, mode= "max-autotune", fullgraph=True)

    if sparsity == "semi_sparse":
        def all_linear(mod, name):
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
                return True
            return False
        torch.sparse.semi_structured._FORCE_CUTLASS = False
        sparsify_(model, semi_sparse_weight(), filter_fn=all_linear)
    elif sparsity == "semi_sparse_mlp_only":
        def all_linear(mod, name):
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in name and "mlp" in name:
                return True
            return False
        torch.sparse.semi_structured._FORCE_CUTLASS = False
        sparsify_(model, semi_sparse_weight(), filter_fn=all_linear)

    if sparsity and compile:
        model = torch.compile(model, mode="max-autotune", fullgraph=True)

    with torch.no_grad():
        result = evaluate(
            HFLM(
                pretrained=model.to(device),
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=max_length),
            get_task_dict(tasks),
            limit = limit,
        )

        pretty_print_nested_results(result)

    if save:
        # This doesn't work yet: https://github.com/huggingface/transformers/issues/32364
        # model.save_pretrained("quantized_model_test", safe_serialization=False)
        file_name = repo_id.split("/")[-1] + "-" + quantization + ".pt"
        torch.save(model.state_dict(), file_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run HF Model Evaluation')
    parser.add_argument('--repo_id', type=str, default="meta-llama/Meta-Llama-3-8B", help='Repository ID to download from HF.')
    parser.add_argument('--tasks', nargs='+', type=str, default=["wikitext"], help='List of lm-eluther tasks to evaluate usage: --tasks task1 task2')
    parser.add_argument('--limit', type=int, default=None, help='Number of eval samples to evaluate')
    parser.add_argument('--precision', type=lambda x: getattr(torch, x.split(".")[-1]), default=torch.bfloat16, help='dtype precision to use')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for evaluation')
    parser.add_argument('-q', '--quantization', default = "None", choices=["int8dq", "int8wo", "int4wo","autoquant", "awq", "None"], help='Which quantization technique to apply')
    parser.add_argument('-s', '--sparsity', default = "None", choices=["semi_sparse", "semi_sparse_mlp_only", "None"], help='Which sparsity technique to apply')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--save', action='store_true', help='Whether to save the model.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to use for evaluation, note int8wo and int4wo work best with small batchsizes, int8dq works better with large batchsizes')
    parser.add_argument('--max_length', type=int, default=None, help='Length of text to process at one time')

    args = parser.parse_args()
    run_evaluation(args.repo_id, args.tasks, args.limit, args.device, args.precision, args.quantization, args.sparsity, args.compile, args.save, args.batch_size, args.max_length)
