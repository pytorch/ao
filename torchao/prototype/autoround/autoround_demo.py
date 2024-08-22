import argparse

import torch

import torchao
import torchao.prototype.autoround.utils as ar_utils

from torchao.prototype.autoround.core import (
    apply_auto_round,
    prepare_model_for_applying_auto_round_,
)
from torchao.prototype.autoround.multi_tensor import MultiTensor
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
    speedup_optimization: bool = True,
    seqlen: int = 2048,
    dataset_name: str = "NeelNanda/pile-10k",
    bs: int = 4,
    nsamples: int = 128,
):
    # Step 1. Prepare the model for applying auto-round
    # User need to prepare a `is_target_module` function for identifying the target modules that need to be quantized.
    if quant_lm_head:
        is_target_module = (
            lambda mod, fqn: isinstance(mod, decoder_cls) or "lm_head" in fqn
        )
    else:
        is_target_module = lambda mod, fqn: isinstance(mod, decoder_cls)
    model_device = next(model.parameters()).device
    device = (
        "cuda"
        if (model_device.type == "cuda")
        or (speedup_optimization and torch.cuda.is_available())
        else "cpu"
    )

    prepare_model_for_applying_auto_round_(
        model,
        is_target_module,
        bits,
        group_size,
        iters,
        device=device,
    )

    # Step 2. Caliration and optimization
    dataloader = ar_utils.get_dataloader(
        tokenizer,
        seqlen=seqlen,
        dataset_name=dataset_name,
        bs=bs,
        nsamples=nsamples,
    )
    input_ids_lst = []
    attn_mask_lst = []
    for data in dataloader:
        input_ids_lst.append(data["input_ids"].to(model_device))
        attn_mask_lst.append(data["attention_mask"].to(model_device))
    print(
        f"Number of batches: {len(input_ids_lst)}, shape of all batches: {[inp.shape for inp in input_ids_lst]}"
    )

    multi_t_input_ids = MultiTensor(input_ids_lst)
    multi_t_attn_mask = MultiTensor(attn_mask_lst)

    # The optimization is applied during the forward pass
    out = model(multi_t_input_ids, multi_t_attn_mask)

    # Step 3. Apply the quantization
    quantize_(model, apply_auto_round(), is_target_module)

    num_quantized_weight = ar_utils.count_tensor_of_type(
        model, torchao.dtypes.AffineQuantizedTensor
    )
    print(f"Quantized {num_quantized_weight} Linear layers.")

    return model


def main(args):
    # Get the model, tokenizer, and decoder_cls
    model_name_or_path = args.model_name_or_path
    model, tokenizer, decoder_cls = ar_utils.get_float_model_info(
        model_name_or_path, torch_dtype=torch.bfloat16
    )
    # Disable the `use_cache` for calibration process, which cause the OOM.
    model.config.use_cache = False
    ar_utils.gen_text(model, tokenizer, "Float model", max_length=50)

    model = model.to(args.model_device)

    quantize_model_with_autoround_(
        model=model,
        tokenizer=tokenizer,
        decoder_cls=decoder_cls,
        bits=args.bits,
        iters=args.iters,
        quant_lm_head=args.quant_lm_head,
        speedup_optimization=args.speedup_optimization,
        seqlen=args.seqlen,
        dataset_name=args.dataset_name,
        bs=args.train_bs,
        nsamples=args.nsamples,
    )
    # Revert the `use_cache`
    model.config.use_cache = True

    # Generate text using the quantized model
    ar_utils.gen_text(model, tokenizer, "Quantized model", max_length=50)


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
        help="Number of iterations for auto-round optimization",
    )
    parser.add_argument(
        "--bits", default=4, type=int, help="Number of bits for quantization"
    )
    parser.add_argument(
        "--train_bs", default=4, type=int, help="Batch size for auto-round optimization"
    )
    parser.add_argument(
        "--nsamples",
        default=128,
        type=int,
        help="Number of samples for calibration process",
    )
    parser.add_argument(
        "--seqlen",
        default=2048,
        type=int,
        help="Sequence length for calibration process",
    )
    parser.add_argument(
        "--quant_lm_head",
        default=False,
        action="store_true",
        help="Quantize the `lm_head` or not",
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
        "-s",
        "--speedup_optimization",
        default=False,
        action="store_true",
        help="Load the compute-intensive operations to GPU for acceleration",
    )
    args = parser.parse_args()
    main(args)
