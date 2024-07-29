from torchao.prototype.autoround.utils import freeze_random, assert_same, get_dataloader
from typing import Optional, Callable, Any, List, Tuple, Dict

freeze_random()

# ==------------------------------------------------------------------------------------------==
# TorchAO
# ==------------------------------------------------------------------------------------------==

import torch
import torchao.quantization as ao_quant
from functools import partial
import transformers
import os


def create_qmodel_from_qdq_model(qdq_model):
    # TODO: simplify this process by creating a new class at unwrapper stage
    def _is_quantized_linear(model, fqn):
        return hasattr(model, "scale")

    @torch.no_grad()
    def create_qlinear(linear):
        def _get_qinfo(linear):
            # qdq_weight shape: (oc, ic)
            qdq_weight = linear.weight.clone()
            device = qdq_weight.device
            # scales, zeros shape: (oc, n_groups)
            scales = linear.scale.to(device)
            zeros = linear.zp.to(device)

            # Requantize the qdqweight to get the int_data
            orig_shape = qdq_weight.shape
            oc, ic = orig_shape
            groupsize = linear.group_size
            assert ic % groupsize == 0, f"expect k % groupsize == 0, but got {ic % groupsize}"
            n_groups = ic // groupsize

            # Check the shapes of scales and zeros with int_data
            scales_zeros_expected_shape = torch.Size([oc, n_groups])
            assert (
                scales.shape == scales_zeros_expected_shape
            ), f"expect scales shape {scales_zeros_expected_shape}, but got {scales.shape}"

            assert (
                zeros.shape == scales_zeros_expected_shape
            ), f"expect zeros shape {scales_zeros_expected_shape}, but got {zeros.shape}"

            flatten_scales = scales.reshape(-1, 1)
            flatten_zeros = zeros.reshape(-1, 1)
            gs_shape = (-1, groupsize)
            int_data = (
                qdq_weight.reshape(gs_shape)
                .div(flatten_scales)
                .add(flatten_zeros)
                .round()
                .reshape(orig_shape)
                .to(torch.int32)
            )

            # Shift the zeros to align with tinygemm
            # TODO: more notes or discard this step
            zeros = (8 - zeros) * scales

            # Pack to tinygemm reqiured format
            # Hard code inner_k_tiles = 2
            inner_k_tiles = 2

            packed_int_data = torch.ops.aten._convert_weight_to_int4pack(int_data, inner_k_tiles)
            scales_and_zeros = ao_quant.utils.pack_tinygemm_scales_and_zeros(
                scales.to(torch.bfloat16), zeros.to(torch.bfloat16)
            )
            return packed_int_data, scales_and_zeros

        int_data, scales_and_zeros = _get_qinfo(linear)
        woq_weight = ao_quant.Int4WeightOnlyQuantizedLinearWeight(
            int_data,
            scales_and_zeros,
            transposed=False,
            shape=linear.weight.shape,
            groupsize=128,
            inner_k_tiles=32,
            dtype=torch.bfloat16,
        )
        linear.weight = torch.nn.Parameter(woq_weight, requires_grad=False)
        del linear.scale
        del linear.zp
        return linear

    qmodel = ao_quant.quant_api._replace_with_custom_fn_if_matches_filter(
        qdq_model, create_qlinear, _is_quantized_linear
    )
    return qmodel


class ModuleInputCapture(torch.nn.Module):
    """Capture the input of the given module."""

    def __init__(self):
        super().__init__()
        # [(args, kwargs), ...]
        self.inputs: List[Tuple[Tuple[Any], Dict[str, Any]]] = []
        self.outputs = []

    def forward(self, *args, **kwarsg):
        self.inputs.append((args, kwarsg))

    def __repr__(self):
        return f"ModuleInputCapture(inputs: {len(self.inputs)})"


class ObservedBlock(torch.nn.Module):
    def __init__(
        self,
        float_block: torch.nn.Module,
        block_observer: ModuleInputCapture,
        input_hook_handle=None,
        output_hook_handle=None,
    ):
        super().__init__()
        # e.g., replace `transformers.models.llama.modeling_llama.LlamaDecoderLayer`
        self.float_block = float_block
        self.block_observer = block_observer
        self.input_hook_handle = input_hook_handle
        self.output_hook_handle = output_hook_handle

    def remove_hook_handles(self):
        self.input_hook_handle.remove()
        self.output_hook_handle.remove()

    def forward(self, *args, **kwarsg):
        return self.float_block(*args, **kwarsg)

    @classmethod
    def from_float(cls, float_block: torch.nn.Module, block_observer: ModuleInputCapture = None):
        # TODO: remove `block_observer`?
        def capture_inputs_hook(
            block_observer: ModuleInputCapture,
            module: torch.nn.Module,
            args: Tuple[torch.Tensor],
            kwargs: Dict[str, Any],
        ) -> Tuple[Any, Any]:
            block_observer.inputs.append((args, kwargs))
            return args, kwargs

        def capture_outputs_hook(
            block_observer: ModuleInputCapture,
            module: torch.nn.Module,
            inputs,
            outputs,
        ):
            block_observer.outputs.append(outputs)
            return outputs

        if block_observer is None:
            block_observer = ModuleInputCapture()
        pre_forward_hook_handle = float_block.register_forward_pre_hook(
            partial(capture_inputs_hook, block_observer), with_kwargs=True
        )
        forward_hook_handle = float_block.register_forward_hook(partial(capture_outputs_hook, block_observer))
        return cls(float_block, block_observer, pre_forward_hook_handle, forward_hook_handle)

    def get_module_inputs_outputs(self):
        self.remove_hook_handles()
        inputs = self.block_observer.inputs
        outputs = self.block_observer.outputs
        return inputs, outputs


def insert_observers_for_block_(
    model: torch.nn.Module,
    filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
) -> ObservedBlock:
    replacement_fn = lambda m: ObservedBlock.from_float(m)
    return ao_quant.quant_api._replace_with_custom_fn_if_matches_filter(model, replacement_fn, filter_fn)


def apply_auto_round(observed_block: ObservedBlock):
    block_inputs, block_outputs = observed_block.get_module_inputs_outputs()
    # Call the autoround to execute the optimization process
    import auto_round

    block = observed_block.float_block

    # Start the training process to update the v, alpha and betta.
    # TODO: refactor the `quant_block_new` to a static method
    rounder = auto_round.AutoRound(
        model=block,
        tokenizer=None,
        sym=False,  # Both True and False are OK
        bits=4,
        iters=2,
        use_quant_input=False,  # disable it for now
        amp=False,
        low_gpu_mem_usage=False,
        model_dtype=next(block.parameters()).dtype,
    )
    rounder.quant_block_new(block, block_inputs, block_outputs)
    return create_qmodel_from_qdq_model(observed_block)


# ==------------------------------------------------------------------------------------------==
# Tests
# ==------------------------------------------------------------------------------------------==


class TestFlow:
    def test_with_opt(self):
        # ==------------------------------------------------------------------------------------------==
        # The Modeling User API
        # ==------------------------------------------------------------------------------------------==
        with torch.no_grad():
            # Step 0. Load the float model
            device = torch.device("cuda")
            import transformers

            # pretrained_model_name_or_path = "hf-internal-testing/tiny-random-GPTJForCausalLM"
            pretrained_model_name_or_path = "facebook/opt-125m"
            model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path).to(device)
            tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

            # Step 1. replace the block with an observed block
            # Similar with the `insert_observers_`, but for block
            is_block = lambda model, fqn: isinstance(model, transformers.models.opt.modeling_opt.OPTDecoderLayer)
            # is_block = lambda model, fqn: isinstance(model, transformers.models.gptj.modeling_gptj.GPTJBlock)
            insert_observers_for_block_(model, is_block)

            print(f"Model with observer (before calibration): \n{model}")

            # Step 2. calibrating / training
            # For capturing the input of block
            # TODO: replace it with a real calibration dataset
            iters = 4
            prompt = "The meaning of life is"
            example_inputs = tokenizer(prompt, return_tensors="pt")
            for _ in range(iters):
                model(**example_inputs.to(device))

            print(f"Model with observer (after calibration): \n{model}")

        # Step 3. quantize the block
        is_observed_block = lambda model, fqn: isinstance(model, ObservedBlock)
        ao_quant.quantize_(model, apply_auto_round, is_observed_block)
