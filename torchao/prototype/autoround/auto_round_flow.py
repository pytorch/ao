# ==------------------------------------------------------------------------------------------==
# TorchAO
# ==------------------------------------------------------------------------------------------==
from typing import Optional, Callable, Any, List
import torch
import torchao.quantization as ao_quant


def create_qmodel_from_qdq_model(qdq_model):
    # TODO: simplify this process by creating a new class at unwrapper stage
    def _is_quantized_linear(model, fqn):
        return hasattr(model, "scale")

    def create_qlinear(linear):
        def _get_qdq_data(linear):
            # TODO: below is a fake implementation
            int_data = linear.weight()
            scales_and_zeros = [(linear.scale, linear.zero_point)]
            return int_data, scales_and_zeros

        int_data, scales_and_zeros = _get_qdq_data(linear)
        # TODO: below is a fake implementation, need more quantization info to dispatch the right quantization class
        woq_linear = ao_quant.Int4WeightOnlyQuantizedLinearWeight(
            int_data,
            scales_and_zeros,
            transposed=False,
            shape=linear.weight.shape,
            groupsize=128,
            inner_k_tiles=32,
            dtype=linear.weight.dtype,
        )
        return woq_linear

    qmodel = ao_quant.quant_api._replace_with_custom_fn_if_matches_filter(
        qdq_model, create_qlinear, _is_quantized_linear
    )
    return qmodel


class ModuleInputCapture(torch.nn.Module):
    """Capture the input of the given module."""

    def __init__(self):
        super().__init__()
        self.inputs: List[Any] = []

    def forward(self, *args, **kwarsg):
        self.inputs.append((args, kwarsg))


class ObservedBlock(torch.nn.Module):
    def __init__(self, float_block: torch.nn.Module, block_observer: ModuleInputCapture):
        super().__init__()
        # e.g., replace `transformers.models.llama.modeling_llama.LlamaDecoderLayer`
        self.float_block = float_block
        self.block_observer = block_observer

    def forward(self, *args, **kwarsg):
        self.block_observer(*args, **kwarsg)
        # Here we not really run the forward of float_block, but just capture the input of the block.
        # We run the forward of the float_block in the `apply_auto_round` function,as we may only
        # sample partial of the inputs.

    @classmethod
    def from_float_block(cls, float_block):
        # TODO: should we pass the float_block to `ModuleInputCapture`?
        block_observer = ModuleInputCapture()
        return cls(float_block, block_observer)

    def get_module_inputs(self):
        # TODO: concat all inputs
        inputs = self.block_observer.inputs
        return inputs


def insert_observers_for_block_(
    model: torch.nn.Module,
    block_observer: ModuleInputCapture,
    filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
) -> ObservedBlock:
    replacement_fn = lambda m: ObservedBlock.from_float(m, block_observer)
    return ao_quant.quant_api._replace_with_custom_fn_if_matches_filter(model, replacement_fn, filter_fn)


def apply_auto_round(observed_block: ObservedBlock):
    # Call the autoround to execute the optimization process
    import auto_round

    # Start the training process to update the v and alpha and betta.
    rounder = auto_round.AutoRound(
        model=observed_block,
        tokenizer=None,
        bits=4,
        iters=2,
        use_quant_input=False,  # disable it for now
        n_samples=2,  # double-check it
        amp=False,
    )
    inputs = observed_block.get_module_inputs()
    # TODO: rename the `quant_block` to `quant_block_`
    rounder.quant_block(observed_block, input_ids=inputs["input_ids"], input_others=inputs["input_others"])
    return create_qmodel_from_qdq_model(observed_block)


# ==------------------------------------------------------------------------------------------==
# The Modeling User API
# ==------------------------------------------------------------------------------------------==

# Step 0. Load the float model
import transformers
pretrained_model_name_or_path = "facebook/opt-125m"
model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)

# Step 1. replace the block with an observed block
# Similar with the `insert_observers_`, but for block

is_block = lambda model, fqn: isinstance(model, transformers.models.opt.modeling_opt.OPTDecoderLayer)
block_observer = ModuleInputCapture()
insert_observers_for_block_(model, block_observer, is_block)

# Step 2. calibrating / training
# For capturing the input of block
batch_size, seq_len, hidden_size = 2, 10, 768
example_inputs = torch.rannd((batch_size, seq_len, hidden_size))
for _ in range(10):
    model(*example_inputs)

# Step 3. quantize the block
is_observed_block = lambda model, fqn: isinstance(model, ObservedBlock)
ao_quant.quantize_(model, apply_auto_round, is_observed_block)
