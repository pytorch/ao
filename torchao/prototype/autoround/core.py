from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

import torchao

import torchao.prototype.autoround.utils as ar_utils
import torchao.quantization as ao_quant
from torchao.dtypes import to_affine_quantized_static

# TODO: remove it before merge
ar_utils.freeze_random()


@torch.no_grad()
def create_qmodel_from_qdq_model(qdq_model: torch.nn.Module):
    """Create a quantized model from a qdq model.

    The qdq_model includes Linear quantized by auto-round, which includes qdq weight, scale, zp.
    """

    @torch.no_grad()
    def apply_static_quant(observed_linear):
        device = observed_linear.weight.device
        weight_scale = observed_linear.scale.to(device)
        weight_zero_point = observed_linear.zp.to(device)

        def weight_quant_func(weight):
            block_size = (1, observed_linear.group_size)
            # TODO: shift the zero and prepack the weight to use tinygemm?
            return to_affine_quantized_static(
                input_float=weight,
                scale=weight_scale,
                zero_point=weight_zero_point,
                block_size=block_size,
                target_dtype=torch.uint8,
                quant_min=0,
                quant_max=15,
                zero_point_domain=ao_quant.quant_primitives.ZeroPointDomain.INT,
            )

        observed_linear.weight = torch.nn.Parameter(
            weight_quant_func(observed_linear.weight), requires_grad=False
        )
        del observed_linear.scale
        del observed_linear.zp
        return observed_linear

    def _is_observed_linear(mod: torch.nn.Module, fqn: str):
        return hasattr(mod, "scale")

    qmodel = ao_quant.quant_api._replace_with_custom_fn_if_matches_filter(
        qdq_model, apply_static_quant, _is_observed_linear
    )
    return qmodel.float_block


class BlockObserver(torch.nn.Module):
    """Capture the inputs and outputs of the given module.

    Define the hooks to capture the inputs and outputs of the given module. The hooks may model-specific.
    """

    def __init__(self):
        super().__init__()
        # [(args, kwargs), ...]
        self.inputs: List[Tuple[Tuple[Any], Dict[str, Any]]] = []
        self.outputs: List[torch.Tensor] = []

    def __repr__(self):
        return (
            f"BlockObserver(inputs: {len(self.inputs)}, outputs: {len(self.outputs)})"
        )

    @staticmethod
    def is_decoder_block(decoder_cls):
        def _is_decoder_block(block, fqn):
            return isinstance(block, decoder_cls)

        return _is_decoder_block

    def block_input_hook(
        self,
        block: torch.nn.Module,
        args: Tuple[torch.Tensor],
        kwargs: Optional[Dict[str, Any]],
    ):
        """Capture the input of the block for perform infrence on qdq block."""
        partial_kwargs = {k: v for k, v in kwargs.items() if k in ["attention_mask"]}
        self.inputs.append((args, partial_kwargs))
        return args, kwargs

    def block_output_hook(self, block: torch.nn.Module, inputs, outputs):
        """Capture the output of the block for computing the reconstruction error.

        The output of the block may be a tuple, e.g., (hidden_states, present_key_value, ...),
        we only take the hidden_states.
        """
        if isinstance(outputs, torch.Tensor):
            self.outputs.append(outputs)
        elif isinstance(outputs, (list, tuple)):
            self.outputs.append(outputs[0])
        else:
            raise NotImplementedError(f"Unsupported type: {type(outputs)}")

    def register_hooks(self, block):
        pre_forward_hook_handle = block.register_forward_pre_hook(
            self.block_input_hook, with_kwargs=True
        )
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
        self.float_block = float_block
        self.block_observer = block_observer
        self.input_hook_handle = input_hook_handle
        self.output_hook_handle = output_hook_handle

    def forward(self, *args, **kwarsg):
        return self.float_block(*args, **kwarsg)

    @classmethod
    def from_float(cls, float_block: torch.nn.Module, block_observer_cls):
        block_observer = block_observer_cls()
        pre_forward_hook_handle, forward_hook_handle = block_observer.register_hooks(
            float_block
        )
        return cls(
            float_block, block_observer, pre_forward_hook_handle, forward_hook_handle
        )

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
    replacement_fn = lambda m: ObservedBlock.from_float(m, BlockObserver)
    return ao_quant.quant_api._replace_with_custom_fn_if_matches_filter(
        model, replacement_fn, filter_fn
    )


def apply_auto_round(observed_block: ObservedBlock):
    block_inputs, block_outputs = observed_block.get_module_inputs_outputs()
    # Call the auto-round to execute the optimization process
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
