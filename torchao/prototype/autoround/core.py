from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

import torchao.prototype.autoround.utils as ar_utils
import torchao.quantization as ao_quant
from torchao.dtypes import to_affine_quantized_static
from torchao.prototype.autoround.multi_tensor import MultiTensor

# TODO: remove it before merge
ar_utils.freeze_random()


@dataclass
class AutoRoundConfig:
    bits: int = 4
    sym: bool = False
    iters: int = 200
    group_size: int = 128
    train_bs: int = 4
    eval_bs: int = 4
    seed: int = 42
    amp: bool = True
    nsamples: int = 128
    seqlen: int = 2048
    quant_lm_head: bool = False


auto_round_config = AutoRoundConfig()


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
    return qmodel


@ar_utils.dump_elapsed_time()
@torch.no_grad()
def apply_auto_round(block, grouped_args, spec, block_outputs):
    # Call the auto-round to execute the optimization process
    import auto_round

    ar_utils.see_memory_usage("Before apply auto-round")

    global auto_round_config

    # Start the training process to update the v, alpha and betta.
    rounder = auto_round.AutoRound(
        model=block,
        tokenizer=None,
        sym=auto_round_config.sym,  # Both True and False are OK
        bits=auto_round_config.bits,
        iters=auto_round_config.iters,
        use_quant_input=False,  # disable it for now
        amp=auto_round_config.amp,
        low_gpu_mem_usage=False,
        model_dtype=next(block.parameters()).dtype,
    )

    @torch.no_grad()
    def _unflatten_grouped_args(grouped_args, spec):
        inputs = []
        for i, inp in enumerate(grouped_args):
            cur_args, cur_kwargs = tree_unflatten(inp, spec)
            inputs.append((cur_args, cur_kwargs))
        return inputs

    block_inputs = _unflatten_grouped_args(grouped_args, spec)
    with torch.enable_grad():
        rounder.quant_block_v2_(
            block, inputs=block_inputs, outputs=block_outputs, device="cuda"
        )
    block.to("cpu")
    qmodel = create_qmodel_from_qdq_model(block)
    ar_utils.see_memory_usage("After apply auto-round.")
    return qmodel


@torch.no_grad()
def optimize_module(
    module: torch.nn.Module,
    args: Tuple[MultiTensor],
    kwargs: Dict[str, MultiTensor],
    output: Tuple[MultiTensor],
):
    # Remove the hook before otpimization process to avoid the recursive call
    module._forward_hook_handle_for_autoround.remove()
    flat_args, spec = tree_flatten((args, kwargs))
    grouped_args = MultiTensor.flat_to_grouped(flat_args)
    output_flat_args, output_spec = tree_flatten((output, {}))
    output_grouped_args = MultiTensor.flat_to_grouped(output_flat_args)
    apply_auto_round(module, grouped_args, spec, output_grouped_args)
    torch.cuda.empty_cache()


@torch.no_grad()
def prepare_model_for_applying_auto_round_(model, is_decoder):

    def forward_hook(
        module,
        args: Tuple[MultiTensor],
        kwargs: Dict[str, MultiTensor],
        output: Tuple[MultiTensor],
    ):
        optimize_module(module, args, kwargs, output)
        return output

    def _register_forward_hook(module: torch.nn.Module):
        forward_hook_handle = module.register_forward_hook(
            forward_hook, with_kwargs=True
        )
        module._forward_hook_handle_for_autoround = forward_hook_handle
        return module

    ao_quant.quant_api._replace_with_custom_fn_if_matches_filter(
        model, _register_forward_hook, is_decoder
    )
