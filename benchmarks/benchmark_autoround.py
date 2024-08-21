import argparse
import logging

import torch
import torchao.prototype.autoround.utils as ar_utils
from torchao.prototype.autoround.autoround_demo import quantize_model_with_autoround
from torchao.prototype.autoround.core import auto_round_config
from torchao.utils import benchmark_model, TORCH_VERSION_AT_LEAST_2_4


def main(args):
    with torch.no_grad():
        model_name_or_path = args.model_name_or_path
        torch_dtype = torch.bfloat16
        model, tokenizer, decoder_cls = ar_utils.get_float_model_info(
            model_name_or_path, torch_dtype=torch_dtype
        )
        torch_dtype = model.config.torch_dtype
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        # Workaround for disabling the `kv_cache`, which cause the OOM.
        model.config.use_cache = False

        example_inputs = ar_utils.gen_example_inputs(tokenizer, device, max_length=1024)
        # Note: as the real model is quite large, we run the benchmark two times for the float model and the quantized model seperately
        if not args.bench_float_model:
            auto_round_config.iters = args.iters
            auto_round_config.nsamples = args.nsamples
            auto_round_config.seqlen = args.seqlen
            auto_round_config.quant_lm_head = args.quant_lm_head
            if args.woq_int4:
                from torchao.quantization import int4_weight_only, quantize_

                quantize_(model, int4_weight_only(group_size=32))
            else:
                model = quantize_model_with_autoround(
                    model, tokenizer, decoder_cls, auto_round_config, device=device
                )
        if args.skip_compile:
            logging.warning("Skip compiling the model!")
        else:
            model = torch.compile(model, mode="max-autotune")

        torch._dynamo.reset()
        msg = "Quantized-model" if not args.bench_float_model else "Float-model"
        if args.woq_int4:
            msg += " with int4 weight only "
        time = benchmark_model(model, args.num_runs, example_inputs)

        print(f"{msg} mean time of {args.num_runs} runs: {time}")


if __name__ == "__main__" and TORCH_VERSION_AT_LEAST_2_4 and torch.cuda.is_available():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        default="facebook/opt-125m",
        help="Model name or path",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed for torch")
    parser.add_argument(
        "--iters", default=20, type=int, help="Number of iterations for optimization"
    )
    parser.add_argument(
        "--num_runs", default=100, type=int, help="Number of runs for benchmarking"
    )
    parser.add_argument(
        "--nsamples", default=128, type=int, help="Number of samples for optimization"
    )
    parser.add_argument(
        "--seqlen", default=2048, type=int, help="Sequence length for optimization"
    )
    parser.add_argument(
        "--quant_lm_head",
        default=False,
        action="store_true",
        help="Quantize the `lm_head` or not",
    )
    parser.add_argument(
        "--bench_float_model",
        default=False,
        action="store_true",
        help="Benchmark the float model",
    )
    parser.add_argument(
        "--skip_compile",
        default=False,
        action="store_true",
        help="Skip compile the model",
    )
    parser.add_argument(
        "--woq_int4",
        default=False,
        action="store_true",
        help="Quantize the model with int4 weight only",
    )
    args = parser.parse_args()
    main(args)
