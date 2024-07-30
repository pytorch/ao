from typing import Optional, Callable, Any, List, Tuple, Dict
import torchao.prototype.autoround.utils as ar_utils

ar_utils.freeze_random()

# ==------------------------------------------------------------------------------------------==
# TorchAO
# ==------------------------------------------------------------------------------------------==

import torch
import torchao.quantization as ao_quant
from functools import partial
import os


@torch.no_grad()
def create_qmodel_from_qdq_model(qdq_model):
    # TODO: simplify this process by creating a new class at unwrapper stage
    def _is_quantized_linear(model, fqn):
        """The linear from auto-round includes qdq weight, scale, zp."""
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

            # Shift the zeros to align with tiny gemm.
            # The dequantization process in tiny gemm:
            #   tiny_dequant = (tinny_quant - 8) * scale + tinny_zp
            # The dequantization porcess in auto-round
            #   dequant = (quant - zp) * scale
            # To align with tiny gemm:
            #   dequant = (quant - 8 + 8 - zp) * scale
            #           = (quant - 8) * scale + (8 - zp) * scale
            #              \___/                \______________/
            #            tiny_quant                 tiny_zp
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


class BlockObserver(torch.nn.Module):
    """Capture the inputs and outputs of the given module.

    Define the hooks to capture the inputs and outputs of the given module. The hooks may model-specific.
    """

    def __init__(self):
        super().__init__()
        # [(args, kwargs), ...]
        self.inputs: List[Tuple[Tuple[Any], Dict[str, Any]]] = []
        self.outputs: List[torch.Tensor] = []

    def forward(self, *args, **kwarsg):
        self.inputs.append((args, kwarsg))

    def __repr__(self):
        return f"BlockObserver(inputs: {len(self.inputs)}, outputs: {len(self.outputs)})"

    @staticmethod
    def is_decoder_block(decoder_cls):
        def _is_decoder_block(block, fqn):
            return isinstance(block, decoder_cls)

        return _is_decoder_block

    def block_input_hook(self, block: torch.nn.Module, args: Tuple[torch.Tensor], kwargs: Optional[Dict[str, Any]]):
        partial_kwargs = {k: v for k, v in kwargs.items() if k in ['position_ids', 'attention_mask']}
        self.inputs.append((args, partial_kwargs))
        return args, kwargs

    def block_output_hook(self, block: torch.nn.Module, inputs, outputs):
        if isinstance(outputs, torch.Tensor):
            self.outputs.append(outputs)
        elif isinstance(outputs, (list, tuple)):
            self.outputs.append(outputs[0])
        else:
            raise NotImplementedError(f"Unsupported type: {type(outputs)}")

    def register_hooks(self, block):
        pre_forward_hook_handle = block.register_forward_pre_hook(self.block_input_hook, with_kwargs=True)
        forward_hook_handle = block.register_forward_hook(self.block_output_hook)
        return pre_forward_hook_handle, forward_hook_handle


class ObservedBlock(torch.nn.Module):
    def __init__(
        self,
        float_block: torch.nn.Module,
        block_observer: BlockObserver,
        input_hook_handle=None,
        output_hook_handle=None,
    ):
        super().__init__()
        # e.g., replace `transformers.models.llama.modeling_llama.LlamaDecoderLayer`
        # TODO: Use function is enough?
        self.float_block = float_block
        self.block_observer = block_observer
        self.input_hook_handle = input_hook_handle
        self.output_hook_handle = output_hook_handle

    def forward(self, *args, **kwarsg):
        return self.float_block(*args, **kwarsg)

    @classmethod
    def from_float(cls, float_block: torch.nn.Module):
        block_observer = BlockObserver()
        pre_forward_hook_handle, forward_hook_handle = block_observer.register_hooks(float_block)
        return cls(float_block, block_observer, pre_forward_hook_handle, forward_hook_handle)

    def remove_hook_handles(self):
        self.input_hook_handle.remove()
        self.output_hook_handle.remove()

    def get_module_inputs_outputs(self):
        self.remove_hook_handles()
        return self.block_observer.inputs, self.block_observer.outputs


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
    # TODO: refactor the `quant_block_` to a static method
    rounder = auto_round.AutoRound(
        model=block,
        tokenizer=None,
        sym=False,  # Both True and False are OK
        bits=4,
        iters=100,
        use_quant_input=False,  # disable it for now
        amp=False,
        low_gpu_mem_usage=False,
        model_dtype=next(block.parameters()).dtype,
    )
    rounder.quant_block_(block, block_inputs, block_outputs)
    return create_qmodel_from_qdq_model(observed_block)


# ==------------------------------------------------------------------------------------------==
# Tests
# ==------------------------------------------------------------------------------------------==


class TestAutoRound:
    def test_with_opt(self):
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
