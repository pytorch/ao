import argparse

import torchao

from torchao.prototype.autoround.core import (
    auto_round_config,
    MultiTensor,
    post_process_model_after_applying_auto_round_,
    prepare_model_for_applying_auto_round_,
)


def main(args):
    # 0. Get the model, tokenizer, and decoder_cls
    import torchao.prototype.autoround.utils as ar_utils

    model_name_or_path = args.model_name_or_path
    model, tokenizer, decoder_cls = ar_utils.get_float_model_info(model_name_or_path)
    # Workaround for disabling the `kv_cache`, which cause the OOM.
    model.config.use_cache = False
    ar_utils.gen_text(model, tokenizer, "Float model", device="cuda", max_length=50)

    auto_round_config.iters = args.iters
    auto_round_config.nsamples = args.nsamples
    auto_round_config.seqlen = args.seqlen

    # 1. Prepare the model for applying auto-round
    # User should provide the `is_decoder` function for identifying the decoder block
    # It can be extended to other modules, such as `lm_head`, the function like:
    #   is_target_module = lambda mod, fqn: isinstance(mod, decoder_cls) or "lm_head" in fqn
    if args.quant_lm_head:
        # is_decoder = lambda mod, fqn:  "lm_head" in fqn
        is_decoder = lambda mod, fqn: isinstance(mod, decoder_cls) or "lm_head" in fqn
    else:
        is_decoder = lambda mod, fqn: isinstance(mod, decoder_cls)
    prepare_model_for_applying_auto_round_(model, is_decoder)

    # 2. Caliration and optimization
    dataloader = ar_utils.get_dataloader(
        tokenizer,
        auto_round_config.seqlen,
        seed=auto_round_config.seed,
        bs=auto_round_config.train_bs,
        nsamples=auto_round_config.nsamples,
    )

    input_ids_lst = []
    attn_mask_lst = []
    for i, data in enumerate(dataloader):
        input_ids_lst.append(data["input_ids"])
        attn_mask_lst.append(data["attention_mask"])

    mul_t_input_ids = MultiTensor(input_ids_lst)
    mul_t_attn_mask = MultiTensor(attn_mask_lst)

    # The optimization is applied during the forward pass
    out = model(mul_t_input_ids, mul_t_attn_mask)

    # 3. Post-process the model after applying auto-round
    post_process_model_after_applying_auto_round_(model)
    assert ar_utils.has_tensor_of_type(model, torchao.dtypes.AffineQuantizedTensor)

    # 4(Optional). Generate text using the optimized model
    ar_utils.gen_text(model, tokenizer, "Quantized model", device="cuda", max_length=50)


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
        "--iters", default=20, type=int, help="Number of iterations for optimization"
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
    args = parser.parse_args()
    main(args)
