import argparse

import torch

import torchao
import torchao.prototype.autoround.utils as ar_utils

from torchao.prototype.autoround.core import (
    apply_auto_round,
    prepare_model_for_applying_auto_round_,
)
from torchao.prototype.autoround.multi_tensor import multi_tensor_config, MultiTensor
from torchao.quantization import quantize_

ar_utils.freeze_random(42)


@torch.no_grad()
def quantize_model_with_autoround_(
    model,
    tokenizer,
    decoder_cls,
    bits: int = 4,
    group_size: int = 128,
    iters: int = 200,
    quant_lm_head: bool = False,
    seqlen: int = 2048,
    bs: int = 4,
    nsamples: int = 128,
    offload: bool = False,
):
    # 1. Prepare the model for applying auto-round
    # User need to prepare a `is_target_module` function for identifying the target modules that need to be quantized.
    if quant_lm_head:
        is_target_module = (
            lambda mod, fqn: isinstance(mod, decoder_cls) or "lm_head" in fqn
        )
    else:
        is_target_module = lambda mod, fqn: isinstance(mod, decoder_cls)

    prepare_model_for_applying_auto_round_(
        model, is_target_module, bits, group_size, iters
    )

    # Step 2. Caliration and optimization
    dataloader = ar_utils.get_dataloader(
        tokenizer,
        seqlen=seqlen,
        bs=bs,
        nsamples=nsamples,
    )

    model_device = next(model.parameters()).device
    input_ids_lst = []
    attn_mask_lst = []
    for i, data in enumerate(dataloader):
        input_ids_lst.append(data["input_ids"].to(model_device))
        attn_mask_lst.append(data["attention_mask"].to(model_device))
    print(
        f"Number of batches: {len(input_ids_lst)}, shape of all batches: {[inp.shape for inp in input_ids_lst]}"
    )

    multi_t_input_ids = MultiTensor(input_ids_lst)
    multi_t_attn_mask = MultiTensor(attn_mask_lst)

    if offload:
        multi_tensor_config.enable_offload = "cpu"

    # The optimization is applied during the forward pass
    out = model(multi_t_input_ids, multi_t_attn_mask)

    # Step 3. Apply the quantization
    quantize_(model, apply_auto_round(), is_target_module)

    num_quantized_weight = ar_utils.count_tensor_of_type(
        model, torchao.dtypes.AffineQuantizedTensor
    )
    print(f"Quantized {num_quantized_weight} Linear layers.")

    # 4(Optional). Generate text using the optimized model
    ar_utils.gen_text(model, tokenizer, "Quantized model", max_length=50)
    return model


def main(args):
    # Get the model, tokenizer, and decoder_cls
    model_name_or_path = args.model_name_or_path
    # Use `torch.bfloat16` as the default dtype for better speed performance
    torch_dtype = torch.bfloat16
    model, tokenizer, decoder_cls = ar_utils.get_float_model_info(
        model_name_or_path, torch_dtype=torch_dtype
    )
    # Disabling the `kv_cache`, which cause the OOM.
    model.config.use_cache = False
    ar_utils.gen_text(model, tokenizer, "Float model", max_length=50)

    model = model.to(args.device)
    quantize_model_with_autoround_(
        model,
        tokenizer,
        decoder_cls,
        bits=args.bits,
        iters=args.iters,
        quant_lm_head=args.quant_lm_head,
        seqlen=args.seqlen,
        bs=args.train_bs,
        nsamples=args.nsamples,
        offload=args.enable_offload,
    )
    # Revert the `kv_cache` value
    model.config.use_cache = True


if __name__ == "__main__":
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
        "--iters", default=200, type=int, help="Number of iterations for optimization"
    )
    parser.add_argument(
        "--bits", default=4, type=int, help="Number of bits for quantization"
    )
    parser.add_argument(
        "--train_bs", default=4, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--nsamples",
        default=128,
        type=int,
        help="Number of samples for calibration dataset",
    )
    parser.add_argument(
        "--seqlen",
        default=2048,
        type=int,
        help="Sequence length for calibration dataset",
    )
    parser.add_argument(
        "--quant_lm_head",
        default=False,
        action="store_true",
        help="Quantize the `lm_head` or not",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda",
        type=str,
        choices=["cpu", "cuda"],
        help="Device for running the model",
    )
    parser.add_argument(
        "-offload",
        "--enable_offload",
        default=False,
        action="store_true",
        help="Enable the offload for `MultiTensor`",
    )
    args = parser.parse_args()
    main(args)


# p autoround_demo.py -m /models//models/Llama-2-7b-chat-hf/  --iters 20 --device cpu
# p autoround_demo.py -m /models//models/Llama-2-7b-chat-hf/  --iters 20 --device cuda
# p autoround_demo.py -m /models/Meta-Llama-3.1-8B-Instruct/  --iters 20 --device cpu --enable_offload
