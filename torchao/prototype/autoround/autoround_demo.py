import torch
import torchao.quantization as ao_quant
import torchao.prototype.autoround.utils as ar_utils
from torchao.prototype.autoround.core import insert_observers_for_block_, apply_auto_round, BlockObserver, ObservedBlock


# ==------------------------------------------------------------------------------------------==
# The Modeling User API
# ==------------------------------------------------------------------------------------------==
with torch.no_grad():
    # Step 0. Load the float model
    device = torch.device("cuda")

    model_name_or_path = "facebook/opt-125m"
    # model_name_or_path = "/models/Llama-2-7b-chat-hf/"
    model, tokenizer, decoder_cls = ar_utils.get_float_model_info(model_name_or_path)
    model = model.to(device)

    ar_utils.gen_text(model, tokenizer, "Float model")

    # Step 1. replace the block with an observed block
    # Similar with the `insert_observers_`, but for block
    insert_observers_for_block_(model, BlockObserver.is_decoder_block(decoder_cls))

    print(f"Model with observer (before calibration): \n{model}")

    # Step 2. calibrating / training
    # For capturing the input of block
    for example_inputs in ar_utils.get_dataloader(tokenizer, seqlen=128, split="train[0:32]"):
        if example_inputs is not None:
            model(**ar_utils.move_input_to_device(example_inputs, device))

    print(f"Model with observer (after calibration): \n{model}")

# Step 3. quantize the block
is_observed_block = lambda model, fqn: isinstance(model, ObservedBlock)
ao_quant.quantize_(model, apply_auto_round, is_observed_block)

ar_utils.gen_text(model, tokenizer, "Quantized model")
