# ==------------------------------------------------------------------------------------------==
# Utils
# ==------------------------------------------------------------------------------------------==
import os

seed = 0
import random

random.seed(seed)
import torch

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import numpy as np

np.random.seed(seed)
from typing import Optional, Callable, Any, List, Tuple, Dict


def assert_same(
    a: Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]],
    b: Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]],
):
    assert len(a) == len(b), f"len: {len(a)} != {len(b)}"
    for i, _ in enumerate(a):
        assert type(a[i]) == type(b[i]), f"type: {type(a[i])} != {type(b[i])}"
        if isinstance(a[i], torch.Tensor):
            torch.testing.assert_allclose(a[i], b[i])
        elif isinstance(a[i], tuple):
            assert_same(a[i], b[i])
        elif isinstance(a[i], dict):
            for k in a[i].keys():
                assert k in b[i], f"key: {k} not in {b[i]}"
                assert_same(a[i][k], b[i].get(k))
        elif a[i] is None:
            assert b[i] is None
        else:
            raise ValueError(f"Unsupported type: {type(a[i])}")
    print("Same!")


def inspect_module_inputs(inputs, indent=""):
    if isinstance(inputs, torch.Tensor):
        print(f"{indent}Tensor: {inputs.shape}")
    elif isinstance(inputs, tuple) or isinstance(inputs, list):
        for i in inputs:
            inspect_module_inputs(i, indent + "  ")
    elif isinstance(inputs, dict):
        for k, v in inputs.items():
            print(f"{indent}{k}:")
            inspect_module_inputs(v, indent + "  ")
    elif inputs is None:
        print(f"{indent}None")
    else:
        print(f"{indent}{type(inputs)}")


# ==------------------------------------------------------------------------------------------==
# TorchAO
# ==------------------------------------------------------------------------------------------==

import torch
import torchao.quantization as ao_quant
from functools import partial
import transformers


def create_qmodel_from_qdq_model(qdq_model):
    # TODO: simplify this process by creating a new class at unwrapper stage
    def _is_quantized_linear(model, fqn):
        return hasattr(model, "scale")

    @torch.no_grad()
    def create_qlinear(linear):
        def _get_qdq_data(linear):
            # TODO: refine the impl, refer: https://github.com/yiliu30/auto-round/pull/2
            # qdq_weight shape: (m, k)
            qdq_weight = linear.weight.clone()
            device = qdq_weight.device
            # scales, zeros shape: (m, n_groups)
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

            # Shift the zeros to align with tinnygemm
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

        # For Debug
        int_data, scales_and_zeros = _get_qdq_data(linear)
        float_out = None
        random_input = torch.randn((4, linear.weight.shape[1]), dtype=torch.bfloat16).to(int_data.device)
        if os.environ.get("DEBUG", "0") == "1":
            float_out = linear(random_input.to(linear.weight.dtype))

        # TODO: Double check below args
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
        if os.environ.get("DEBUG", "0") == "1":
            new_lin_int_out = linear(random_input)
            # _max = (float_out - new_lin_int_out).abs().max()
            # print(f"Max diff between float and int: {_max}")
            assert torch.allclose(
                float_out.to(new_lin_int_out.dtype), new_lin_int_out, atol=2e-1
            ), f"The max diff between float and int is too large: {(float_out - new_lin_int_out).abs().max()}"
        return linear

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

    def __repr__(self):
        return f"ModuleInputCapture(inputs: {len(self.inputs)})"


class ObservedBlock(torch.nn.Module):
    def __init__(self, float_block: torch.nn.Module, block_observer: ModuleInputCapture, input_hook_handle=None):
        super().__init__()
        # e.g., replace `transformers.models.llama.modeling_llama.LlamaDecoderLayer`
        self.float_block = float_block
        self.block_observer = block_observer
        self.input_hook_handle = input_hook_handle

    def remove_input_hook_handle(self):
        self.input_hook_handle.remove()

    def forward(self, *args, **kwarsg):
        return self.float_block(*args, **kwarsg)

    @classmethod
    def from_float(cls, float_block: torch.nn.Module, block_observer: ModuleInputCapture = None):
        # TODO: only insert hook to the float_block to capture the input and save it to the block_observer
        # TODO: look like no need new module for it?
        def capture_inputs_hook(
            block_observer: ModuleInputCapture,
            module: torch.nn.Module,
            args: Tuple[torch.Tensor],
            kwargs: Dict[str, Any],
        ) -> Tuple[Any, Any]:
            block_observer.inputs.append((args, kwargs))
            return args, kwargs

        if block_observer is None:
            block_observer = ModuleInputCapture()
        hook_handle = float_block.register_forward_pre_hook(
            partial(capture_inputs_hook, block_observer), with_kwargs=True
        )
        return cls(float_block, block_observer, hook_handle)

    def get_module_inputs(self):
        inputs = self.block_observer.inputs
        return inputs


def insert_observers_for_block_(
    model: torch.nn.Module,
    filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
) -> ObservedBlock:
    replacement_fn = lambda m: ObservedBlock.from_float(m)
    return ao_quant.quant_api._replace_with_custom_fn_if_matches_filter(model, replacement_fn, filter_fn)


def apply_auto_round(observed_block: ObservedBlock):
    block_inputs = observed_block.block_observer.inputs
    # first_inputs = block_inputs[0][1]
    # hidden_states = first_inputs["hidden_states"]
    # position_ids = first_inputs["position_ids"]
    # attention_mask = first_inputs["attention_mask"]

    # # WA for now
    # _input_ids = hidden_states
    # _input_others = {"positional_inputs": position_ids, "attention_mask": attention_mask}

    _input_ids = block_inputs[0][0][0].detach()
    position_ids = []
    attention_mask = block_inputs[0][1]["attention_mask"].detach()
    _input_others = {"positional_inputs": position_ids, "attention_mask": attention_mask}

    seq_len = _input_ids.shape[1]
    # Call the autoround to execute the optimization process
    import auto_round

    block = observed_block.float_block
    block.dtype = next(block.parameters()).dtype

    # Start the training process to update the v and alpha and betta.
    rounder = auto_round.AutoRound(
        model=block,
        tokenizer=None,
        sym=False,  # Both True and False are OK
        bits=4,
        iters=2,
        use_quant_input=False,  # disable it for now
        n_samples=1,  # double-check it
        amp=False,
        seqlen=seq_len,
        batch_size=1,
        low_gpu_mem_usage=False,
    )
    # TODO: rename the `quant_block` to `quant_block_`
    rounder.quant_block(block, input_ids=_input_ids, input_others=_input_others, device=torch.device("cuda"))
    return create_qmodel_from_qdq_model(observed_block)


# ==------------------------------------------------------------------------------------------==
# Tests
# ==------------------------------------------------------------------------------------------==


import pytest


class TestFlow:
    @torch.no_grad()
    def test_obseverblock(self):
        model = transformers.AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-GPTJForCausalLM")
        block = model.transformer.h[0]
        observed_block = ObservedBlock.from_float(block)
        bs, seq_len, hidden_size = 2, 3, 32
        hidden_states = torch.randn((bs, seq_len, hidden_size))
        position_ids = torch.randint(0, seq_len, (bs, seq_len))
        # Record the input and output of the block
        origin_output = []
        out1 = observed_block(hidden_states, position_ids=position_ids)
        origin_output.append(out1)
        print(observed_block.block_observer.inputs)
        attention_mask = torch.randn(bs, 4, seq_len, seq_len)
        out2 = observed_block(hidden_states, None, attention_mask, position_ids=position_ids)
        origin_output.append(out2)
        print(observed_block.block_observer.inputs)
        observed_block.remove_input_hook_handle()
        # Replay
        new_output = []
        for args, kwargs in observed_block.block_observer.inputs:
            out = observed_block(*args, **kwargs)
            new_output.append(out)
        assert_same(origin_output, new_output)

    def test_with_opt(self):
        with torch.no_grad():
            device = torch.device("cuda")

            # Step 0. Load the float model
            import transformers

            # pretrained_model_name_or_path = "hf-internal-testing/tiny-random-GPTJForCausalLM"
            pretrained_model_name_or_path = "facebook/opt-125m"
            model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path).to(device)
            tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

            # Step 1. replace the block with an observed block
            # Similar with the `insert_observers_`, but for block
            is_block = lambda model, fqn: isinstance(model, transformers.models.opt.modeling_opt.OPTDecoderLayer)
            # is_block = lambda model, fqn: isinstance(model, transformers.models.gptj.modeling_gptj.GPTJBlock)
            block_observer = ModuleInputCapture()
            insert_observers_for_block_(model, is_block)

            print(f"model with observer (before calibration): \n{model}")

            # Step 2. calibrating / training
            # For capturing the input of block
            # batch_size, seq_len, hidden_size = 2, 5, 32
            iters = 4
            prompt = "The meaning of life is"
            # "input_ids", "attention_mask"
            example_inputs = tokenizer(prompt, return_tensors="pt")
            for _ in range(iters):
                model(**example_inputs.to(device))

            print(f"model with observer (after calibration): \n{model}")

        # Step 3. quantize the block
        is_observed_block = lambda model, fqn: isinstance(model, ObservedBlock)
        ao_quant.quantize_(model, apply_auto_round, is_observed_block)


# ==------------------------------------------------------------------------------------------==
# The Modeling User API
# ==------------------------------------------------------------------------------------------==


def test_user_api():
    # Step 0. Load the float model
    import transformers

    pretrained_model_name_or_path = "facebook/opt-125m"
    model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)

    # Step 1. replace the block with an observed block
    # Similar with the `insert_observers_`, but for block

    is_block = lambda model, fqn: isinstance(model, transformers.models.opt.modeling_opt.OPTDecoderLayer)
    insert_observers_for_block_(model, is_block)

    # Step 2. calibrating / training
    # For capturing the input of block
    batch_size, seq_len, hidden_size = 2, 10, 768
    example_inputs = torch.rannd((batch_size, seq_len, hidden_size))

    for _ in range(10):
        model(*example_inputs)

    # Step 3. quantize the block
    is_observed_block = lambda model, fqn: isinstance(model, ObservedBlock)
    ao_quant.quantize_(model, apply_auto_round, is_observed_block)
