# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config

import torchao
from torchao._models.utils import (
    get_arch_name,
    write_json_result_local,
    write_json_result_ossci,
)
from torchao.quantization.quant_primitives import MappingType
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TORCH_VERSION_AT_LEAST_2_6,
    get_model_size_in_bytes,
)

torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = False
torch.backends.cuda.enable_cudnn_sdp(True)


class HostEvent:
    def __init__(self):
        self.event_time = None

    def record(self):
        self.event_time = time.perf_counter()

    def elapsed_time(self, other_event):
        if self.event_time is None:
            raise ValueError("Event not recorded!")
        # return ms to match cuda event
        return abs(other_event.event_time - self.event_time) * 1000


def device_timer(device):
    if "cuda" in device:
        return torch.cuda.Event(enable_timing=True)
    elif ("cpu" in device) or ("mps" in device):
        return HostEvent()
    else:
        print(f"device={device} is not yet suppported")


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif "xpu" in device:
        torch.xpu.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


default_device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "cpu"
)

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from torchao._models.llama.model import Transformer, prepare_inputs_for_model
from torchao._models.llama.tokenizer import get_tokenizer


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(
    model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(
    model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    callback=lambda _: _,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            next_token, next_prob = next_token.clone(), next_prob.clone()
            input_pos += 1
            # in some instances not having this causes weird issues with the stored tokens when you run the next decode_one_token step
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob)
            cur_token = next_token

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    *,
    interactive: bool,
    callback=lambda x: x,
    kv_cache_quantization: bool = False,
    cache_size: Optional[int] = None,
    linear_causal_mask: bool = False,
    prefill_start_event: Optional[torch.cuda.Event] = None,
    prefill_end_event: Optional[torch.cuda.Event] = None,
    decode_start_event: Optional[torch.cuda.Event] = None,
    decode_end_event: Optional[torch.cuda.Event] = None,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    device = prompt.device
    T = prompt.size(-1)

    # calculate how many tokens to generate based on max_new_tokens and model's upper bound (block_size)
    max_seq_length = (
        min(T + max_new_tokens, model.config.block_size) if not interactive else 350
    )
    new_tokens = max_seq_length - T

    # format model input
    prompt, input_pos = prepare_inputs_for_model(prompt)
    prompt = prompt.repeat(batch_size, 1)  # expand prompt based on batchsize

    # full prompt+output will be stored in seq
    seq = torch.empty(batch_size, max_seq_length, dtype=prompt.dtype, device=device)
    seq[:, :T] = prompt

    # setup model caches
    with torch.device(device):
        if cache_size is None:
            cache_size = max_seq_length
        assert cache_size >= max_seq_length, (
            "need cache_size to be greater than max_new_tokens + size-of-prompt"
        )
        model.setup_caches(
            max_batch_size=batch_size,
            max_seq_length=cache_size,
            kv_cache_quantization=kv_cache_quantization,
            linear_causal_mask=linear_causal_mask,
            prompt_length=T,
        )

    # execute prefill
    if prefill_start_event is not None:
        prefill_start_event.record()
    next_token = prefill(
        model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs
    ).clone()
    seq[:, T] = next_token.squeeze()
    if prefill_end_event is not None:
        prefill_end_event.record()

    # execute token generation
    if decode_start_event is not None:
        decode_start_event.record()
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(
        model,
        next_token.view(batch_size, -1),
        input_pos,
        new_tokens - 1,
        callback=callback,
        **sampling_kwargs,
    )
    seq = torch.cat((seq[:, : T + 1], *generated_tokens), dim=-1)
    if decode_end_event is not None:
        decode_end_event.record()

    return seq


def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def _load_model(checkpoint_path, device, precision):
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    with torch.device("meta"):
        model = Transformer.from_name(checkpoint_path.parent.name)
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(device=device, dtype=precision)

    return model.eval()


B_INST, E_INST = "[INST]", "[/INST]"


def main(
    prefill_size: Optional[int] = None,
    prompt: str = "Hello, my name is",
    demo_summarize_prompt: Optional[str] = None,
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    batch_size: int = 1,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path(
        "checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"
    ),
    quantization: Optional[str] = None,
    min_sqnr: Optional[float] = None,
    sparsity: Optional[str] = None,
    kv_cache_quantization: bool = False,
    cache_size: Optional[int] = None,
    linear_causal_mask: bool = False,
    save: bool = False,
    compile: bool = True,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    memory_profile: Optional[Path] = None,
    device=default_device,
    precision=torch.bfloat16,
    write_result: Optional[Path] = None,
    output_json_path: Optional[Path] = None,
    output_json_local: bool = False,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""

    if prefill_size is not None and prefill_size > 0:
        # create prompt of prefill size
        if demo_summarize_prompt is None:
            prompt = "prompt " * (int(prefill_size) - 2)
        else:
            with open(demo_summarize_prompt, "r") as f:
                prompt = f.read()

    torchao.quantization.utils.recommended_inductor_config_setter()

    assert checkpoint_path.is_file(), checkpoint_path
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    print(f"Using device={device}")
    is_chat = "chat" in str(checkpoint_path)

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision)

    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)

    encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

    if demo_summarize_prompt is not None:
        end_tag = encode_tokens(tokenizer, "\n <END_TEXT>", bos=False, device=device)
        encoded = encoded[: prefill_size - end_tag.size(0)]
        encoded = torch.cat((encoded, end_tag), dim=0)

    prompt_length = encoded.size(0)

    torch.manual_seed(1234)

    def ffn_only(mod, fqn):
        return isinstance(mod, torch.nn.Linear) and "feed_forward" in fqn

    def not_ffn_only(mod, fqn):
        return isinstance(mod, torch.nn.Linear) and not ffn_only(mod, fqn)

    def ffn_or_attn_only(mod, fqn):
        return isinstance(mod, torch.nn.Linear) and (
            "feed_forward" in fqn or "attention" in fqn
        )

    if quantization:
        from torchao.quantization import (
            Float8DynamicActivationFloat8SemiSparseWeightConfig,
            autoquant,
            float8_dynamic_activation_float8_weight,
            float8_weight_only,
            fpx_weight_only,
            gemlite_uintx_weight_only,
            int4_dynamic_activation_int4_weight,
            int4_weight_only,
            int8_dynamic_activation_int4_weight,
            int8_dynamic_activation_int8_weight,
            int8_weight_only,
            quantize_,
            uintx_weight_only,
        )
        from torchao.quantization.granularity import PerRow, PerTensor
        from torchao.utils import unwrap_tensor_subclass

        if "spinquant" in quantization:
            from torchao.prototype.spinquant import apply_spinquant

            apply_spinquant(model)
        if quantization.startswith("gemlite"):
            import os
            import pwd

            from gemlite.core import GemLiteLinearTriton

            _quant_args = quantization.split("-")
            bit_width = int(_quant_args[-2])
            group_size = None if _quant_args[-1] == "None" else int(_quant_args[-1])
            try:
                packing_bitwidth = int(_quant_args[-3])
            except:
                # if only 2 inputs found, use default value
                packing_bitwidth = 32

            quantize_(
                model,
                gemlite_uintx_weight_only(group_size, bit_width, packing_bitwidth),
            )

            # try to load gemlite kernel config
            try:
                GemLiteLinearTriton.load_config(
                    f"/tmp/{pwd.getpwuid(os.getuid()).pw_gecos}_gemlite.json"
                )
                print(
                    f"loaded gemlite kernel cache /tmp/{pwd.getpwuid(os.getuid()).pw_gecos}_gemlite.json"
                )
            except:
                print(
                    f"unable to load gemlite kernel cache /tmp/{pwd.getpwuid(os.getuid()).pw_gecos}_gemlite.json"
                )

            print("running gemlite warmup")
            generate(
                model,
                encode_tokens(tokenizer, prompt, bos=True, device=device),
                max_new_tokens,
                batch_size,
                interactive=False,
                temperature=temperature,
                top_k=top_k,
            )
            GemLiteLinearTriton.cache_config(
                f"/tmp/{pwd.getpwuid(os.getuid()).pw_gecos}_gemlite.json"
            )
        if "int8wo" in quantization:
            quantize_(model, int8_weight_only())
        if "int8dq" in quantization:
            if sparsity and "semi" in sparsity:
                from torchao.dtypes import SemiSparseLayout

                quantize_(
                    model,
                    int8_dynamic_activation_int8_weight(layout=SemiSparseLayout()),
                    filter_fn=ffn_only,
                )
                quantize_(
                    model, int8_dynamic_activation_int8_weight(), filter_fn=not_ffn_only
                )
            elif "int8dq_prefill_wo_decode" in quantization:
                quantize_(
                    model, int8_dynamic_activation_int8_weight(weight_only_decode=True)
                )
            else:
                quantize_(model, int8_dynamic_activation_int8_weight())
        if "int4wo" in quantization:
            use_hqq = False
            if "hqq" in quantization:
                use_hqq = True
            group_size = int(quantization.split("-")[1])
            assert group_size in [
                32,
                64,
                128,
                256,
            ], (
                f"int4wo group_size needs to be one of [32,64,128,256] but got {group_size}"
            )
            quantize_(model, int4_weight_only(group_size=group_size, use_hqq=use_hqq))
        elif "int4dq-" in quantization:
            from torchao.dtypes import CutlassInt4PackedLayout

            nbits = int(quantization.removeprefix("int4dq-"))
            assert nbits == 4 or nbits == 8
            if nbits == 4:
                quantize_(
                    model,
                    int4_dynamic_activation_int4_weight(
                        mapping_type=MappingType.SYMMETRIC,
                        act_mapping_type=MappingType.SYMMETRIC,
                        layout=CutlassInt4PackedLayout(),
                    ),
                )
            elif nbits == 8:
                quantize_(
                    model,
                    int8_dynamic_activation_int4_weight(
                        group_size=None,
                        mapping_type=MappingType.SYMMETRIC,
                        act_mapping_type=MappingType.SYMMETRIC,
                        layout=CutlassInt4PackedLayout(),
                    ),
                )
        if "marlin" in quantization:
            if "qqq" in quantization:
                from torchao.dtypes import MarlinQQQLayout

                quantize_(
                    model,
                    int8_dynamic_activation_int4_weight(
                        group_size=128,
                        mapping_type=MappingType.SYMMETRIC,
                        act_mapping_type=MappingType.SYMMETRIC,
                        layout=MarlinQQQLayout(),
                    ),
                )
            elif "semi" in sparsity:
                from torchao.dtypes import MarlinSparseLayout

                quantize_(
                    model,
                    int4_weight_only(layout=MarlinSparseLayout()),
                    filter_fn=ffn_or_attn_only,
                )
        if "fp6" in quantization:
            quantize_(model, fpx_weight_only(3, 2))
        elif "embed-int8wo" in quantization:
            quantize_(
                model,
                int8_weight_only(group_size=64),
                filter_fn=lambda x, *args: isinstance(x, torch.nn.Embedding),
            )
        elif quantization.startswith("awq"):
            from torchao._models._eval import TransformerEvalWrapper
            from torchao.utils import TORCH_VERSION_AT_LEAST_2_3

            if not TORCH_VERSION_AT_LEAST_2_3:
                print("Awq requires torch2.3+")
                exit()
            from torchao.prototype.awq import (
                AWQObservedLinear,
                awq_uintx,
                insert_awq_observer_,
            )

            quant_dtype = quantization.split("-")[1]
            group_size = int(quantization.split("-")[2])
            quant_dtype = getattr(torch, quant_dtype, torch.uint8)
            model = model.to(device)
            # get calibration data
            insert_awq_observer_(
                model, 1, 256, quant_dtype=quant_dtype, group_size=group_size
            )
            TransformerEvalWrapper(
                model=model.to(device),
                tokenizer=tokenizer,
                max_seq_length=256,
                input_prep_func=prepare_inputs_for_model,
                device=device,
            ).run_eval(
                tasks=["wikitext"],
                limit=1,
            )
            is_observed_linear = lambda m, fqn: isinstance(m, AWQObservedLinear)
            use_hqq = "hqq" in quantization
            quantize_(
                model,
                awq_uintx(
                    quant_dtype=quant_dtype, group_size=group_size, use_hqq=use_hqq
                ),
                is_observed_linear,
            )
        elif "uintx" in quantization:
            # uintx-nbits-group_size, e.g. "uintx-2-64"
            if "hqq" in quantization:
                # uintx-nbits-group_size-hqq
                use_hqq = True
            else:
                use_hqq = False
            _quant_args = quantization.split("-")
            nbits = int(_quant_args[1])
            assert nbits >= 1 and nbits <= 8, "nbits must be 1 to 8"
            _NBITS_TO_DTYPE = {
                1: torch.uint1,
                2: torch.uint2,
                3: torch.uint3,
                4: torch.uint4,
                5: torch.uint5,
                6: torch.uint6,
                7: torch.uint7,
                8: torch.uint8,
            }
            dtype = _NBITS_TO_DTYPE[nbits]
            group_size = int(_quant_args[2])
            quantize_(model, uintx_weight_only(dtype, group_size, use_hqq=use_hqq))
        elif "int8_dynamic_activation_intx_weight" in quantization:
            assert TORCH_VERSION_AT_LEAST_2_6, (
                "int8_dynamic_activation_intx_weight requires torch2.6+"
            )
            assert precision == torch.float32, (
                "int8_dynamic_activation_intx_weight requires using precision=torch.float32"
            )

            from torchao.dtypes import PackedLinearInt8DynamicActivationIntxWeightLayout
            from torchao.quantization.granularity import PerAxis, PerGroup
            from torchao.quantization.quant_api import (
                Int8DynamicActivationIntxWeightConfig,
            )

            # Quantize model
            _quant_args = quantization.split("-")
            weight_dtype = getattr(torch, f"int{_quant_args[1]}")
            group_size = int(_quant_args[2])
            granularity = PerGroup(group_size) if group_size > 0 else PerAxis(0)
            is_asymmetric = bool(_quant_args[3])
            quantize_(
                model,
                Int8DynamicActivationIntxWeightConfig(
                    weight_dtype=weight_dtype,
                    weight_granularity=granularity,
                    weight_mapping_type=MappingType.ASYMMETRIC
                    if is_asymmetric
                    else MappingType.SYMMETRIC,
                    weight_scale_dtype=torch.bfloat16,
                    layout=PackedLinearInt8DynamicActivationIntxWeightLayout(),
                ),
            )
        elif "float8wo" in quantization:
            quantize_(model, float8_weight_only())
        elif "float8dq" in quantization:
            if sparsity and "semi" in sparsity:
                quantize_(
                    model,
                    Float8DynamicActivationFloat8SemiSparseWeightConfig(),
                    filter_fn=ffn_only,
                )
            else:
                granularity = str(quantization.split("-")[-1])
                if granularity == "tensor":
                    granularity = PerTensor()
                elif granularity == "row":
                    granularity = PerRow()
                else:
                    granularity = PerTensor()
                quantize_(
                    model,
                    float8_dynamic_activation_float8_weight(granularity=granularity),
                )
        elif "autoquant_v2" in quantization:
            from torchao._models._eval import InputRecorder
            from torchao._models.llama.model import prepare_inputs_for_model
            from torchao.prototype.quantization.autoquant_v2 import autoquant_v2

            calibration_seq_length = 256
            inputs = (
                InputRecorder(
                    tokenizer,
                    calibration_seq_length,
                    prepare_inputs_for_model,
                    False,  # pad_calibration_inputs
                    model.config.vocab_size,
                    device="cuda",
                )
                .record_inputs(
                    ["wikitext"],
                    1,
                )
                .get_inputs()[0]
                .values[0]
            )
            inputs = prepare_inputs_for_model(inputs)
            with torch.device("cuda"):
                model.setup_caches(
                    max_batch_size=1, max_seq_length=calibration_seq_length
                )

            if "autoquant_v2-int4" == quantization:
                model = autoquant_v2(
                    model,
                    manual=True,
                    qtensor_class_list=torchao.prototype.quantization.autoquant_v2.DEFAULT_INT4_AUTOQUANT_CLASS_LIST,
                    example_input=inputs,
                    batch_size=calibration_seq_length,
                )
            elif "autoquant_v2-float8" == quantization:
                model = autoquant_v2(
                    model,
                    manual=True,
                    qtensor_class_list=torchao.prototype.quantization.autoquant_v2.OTHER_AUTOQUANT_CLASS_LIST,
                    example_input=inputs,
                    batch_size=calibration_seq_length,
                )
            elif "autoquant_v2-fp" == quantization:
                model = autoquant_v2(
                    model,
                    manual=True,
                    qtensor_class_list=torchao.prototype.quantization.autoquant_v2.DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST,
                    example_input=inputs,
                    batch_size=calibration_seq_length,
                )
            elif "autoquant_v2-all" == quantization:
                all_qtensor_classes = (
                    torchao.prototype.quantization.autoquant_v2.DEFAULT_AUTOQUANT_CLASS_LIST
                    + torchao.prototype.quantization.autoquant_v2.DEFAULT_INT4_AUTOQUANT_CLASS_LIST
                    + torchao.prototype.quantization.autoquant_v2.DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST
                )
                if torchao.utils.is_sm_89():
                    # this is fp8 related subclasses, should rename
                    all_qtensor_classes += torchao.prototype.quantization.autoquant_v2.OTHER_AUTOQUANT_CLASS_LIST
                model = autoquant_v2(
                    model,
                    manual=True,
                    qtensor_class_list=all_qtensor_classes,
                    example_input=inputs,
                    batch_size=calibration_seq_length,
                )
            else:
                model = autoquant_v2(
                    model,
                    manual=True,
                    example_input=inputs,
                    batch_size=calibration_seq_length,
                )

            print("running generate")
            generate(
                model,
                encode_tokens(tokenizer, prompt, bos=True, device=device),
                max_new_tokens,
                batch_size,
                interactive=False,
                temperature=temperature,
                top_k=top_k,
            )

            print("running finalize autoquant")
            # do autoquantization
            model.finalize_autoquant()
        elif "autoquant" in quantization:
            from torchao._models._eval import InputRecorder
            from torchao._models.llama.model import prepare_inputs_for_model

            calibration_seq_length = 256
            inputs = (
                InputRecorder(
                    tokenizer,
                    calibration_seq_length,
                    prepare_inputs_for_model,
                    False,  # pad_calibration_inputs
                    model.config.vocab_size,
                    device="cuda",
                )
                .record_inputs(
                    ["wikitext"],
                    1,
                )
                .get_inputs()[0]
                .values[0]
            )
            inputs = prepare_inputs_for_model(inputs)
            with torch.device("cuda"):
                model.setup_caches(
                    max_batch_size=1, max_seq_length=calibration_seq_length
                )

            if "autoquant-int4" == quantization:
                model = autoquant(
                    model,
                    manual=True,
                    qtensor_class_list=torchao.quantization.DEFAULT_INT4_AUTOQUANT_CLASS_LIST,
                    example_input=inputs,
                    min_sqnr=min_sqnr,
                )
            elif "autoquant-float8" == quantization:
                model = autoquant(
                    model,
                    manual=True,
                    qtensor_class_list=torchao.quantization.OTHER_AUTOQUANT_CLASS_LIST,
                    example_input=inputs,
                    min_sqnr=min_sqnr,
                )
            elif "autoquant-fp" == quantization:
                model = autoquant(
                    model,
                    manual=True,
                    qtensor_class_list=torchao.quantization.DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST,
                    example_input=inputs,
                    min_sqnr=min_sqnr,
                )
            elif "autoquant-sparse" == quantization:
                model = autoquant(
                    model,
                    manual=True,
                    qtensor_class_list=torchao.quantization.DEFAULT_SPARSE_AUTOQUANT_CLASS_LIST,
                    example_input=inputs,
                    min_sqnr=min_sqnr,
                )
            elif "autoquant-gemlite-int4" == quantization:
                import os
                import pwd

                from gemlite.core import GemLiteLinearTriton

                GemLiteLinearTriton.load_config(
                    f"/tmp/{pwd.getpwuid(os.getuid()).pw_gecos}_gemlite.json"
                )
                model = autoquant(
                    model,
                    manual=True,
                    qtensor_class_list=torchao.quantization.GEMLITE_INT4_AUTOQUANT_CLASS_LIST,
                    example_input=inputs,
                    min_sqnr=min_sqnr,
                )
            elif "autoquant-all" == quantization:
                try:
                    import os
                    import pwd

                    from gemlite.core import GemLiteLinearTriton

                    GemLiteLinearTriton.load_config(
                        f"/tmp/{pwd.getpwuid(os.getuid()).pw_gecos}_gemlite.json"
                    )
                except:
                    pass

                model = autoquant(
                    model,
                    manual=True,
                    qtensor_class_list=torchao.quantization.ALL_AUTOQUANT_CLASS_LIST,
                    example_input=inputs,
                    min_sqnr=min_sqnr,
                )
            else:
                model = autoquant(
                    model, manual=True, example_input=inputs, min_sqnr=min_sqnr
                )

            generate(
                model,
                encode_tokens(tokenizer, prompt, bos=True, device=device),
                max_new_tokens,
                batch_size,
                interactive=False,
                temperature=temperature,
                top_k=top_k,
            )

            # do autoquantization
            model.finalize_autoquant()
        elif "codebook" in quantization:
            from torchao.prototype.quantization.codebook import codebook_weight_only

            model.to(device)
            quantize_(
                model, codebook_weight_only(dtype=torch.uint4, scale_block_size=64)
            )

        else:
            if not TORCH_VERSION_AT_LEAST_2_5:
                unwrap_tensor_subclass(model)

    # standalone sparsity
    elif sparsity:
        from torchao.sparsity import semi_sparse_weight, sparsify_

        if "semi" in sparsity:
            # Fixed sparsity level for 2:4
            sparsify_(model.to(device), semi_sparse_weight(), filter_fn=ffn_only)

        if "bsr" in sparsity:
            from torchao.sparsity import SupermaskLinear, block_sparse_weight

            # parse "bsr-0.9-64"
            _, sparsity_level, blocksize = sparsity.split("-")
            sparsity_level, blocksize = float(sparsity_level), int(blocksize)
            sparsify_(
                model,
                lambda x: SupermaskLinear.from_linear(
                    x,
                    sparsity_level=sparsity_level,
                    blocksize=blocksize,
                ),
                filter_fn=ffn_only,
            )
            print(model)
            sparsify_(
                model,
                SupermaskLinear.to_linear,
                filter_fn=ffn_only,
            )
            print(model)

            # Accelerate with triton bsr kernels
            sparsify_(
                model, block_sparse_weight(blocksize=blocksize), filter_fn=ffn_only
            )

    model_size = get_model_size_in_bytes(model, ignore_embeddings=True) / 1e9

    if save:
        output_dir = str(checkpoint_path.cwd())
        filename = str(checkpoint_path.name).split(".")[0]
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, filename + f"-{quantization}.pt"),
        )

    if compile:
        print("Compiling Model")
        global decode_one_token, prefill
        decode_one_token = torch.compile(
            decode_one_token,
            mode="reduce-overhead",
            fullgraph=True,
        )

        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    if memory_profile:
        if device == "cuda":
            torch.cuda.memory._record_memory_history(
                True, trace_alloc_max_entries=250000, trace_alloc_record_context=True
            )
        elif device == "xpu":
            torch.xpu.memory._record_memory_history(
                True, trace_alloc_max_entries=250000, trace_alloc_record_context=True
            )
        else:
            print("Memory profiling only works on CUDA or XPU devices")

    aggregate_metrics = {
        "tokens_per_sec": [],
        "time": [],
        "decode_tokens_per_sec": [],
        "prefill_time": [],
    }
    start = -1 if compile else 0

    for i in range(start, num_samples):
        if i == 0:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()  # MKG
            elif device == "xpu":
                torch.xpu.reset_peak_memory_stats()  # MKG
        device_sync(device=device)  # MKG
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            if is_chat:
                prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if interactive and i >= 0 and prefill_size is None:
            buffer = []
            period_id = tokenizer.encode(".")[0]
            done_generating = False

            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                buffer.append(tokenizer.decode([period_id] + x.squeeze(0).tolist())[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print("".join(buffer), end="", flush=True)
                    buffer.clear()
                # print(, end="", flush=True)

        elif demo_summarize_prompt is not None and i >= 0:
            buffer = []
            period_id = tokenizer.encode(".")[0]

            def callback(x):
                buffer.append(tokenizer.decode([period_id] + x.squeeze(0).tolist())[1:])
                if len(buffer) == 4:
                    print("".join(buffer), end="", flush=True)
                    buffer.clear()
        else:
            callback = lambda x: x
        t0 = time.perf_counter()
        prefill_start_event, prefill_end_event = (
            device_timer(device),
            device_timer(device),
        )
        decode_start_event, decode_end_event = (
            device_timer(device),
            device_timer(device),
        )
        import contextlib

        if i != num_samples - 1 or not profile:
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            y = generate(
                model,
                encoded,
                max_new_tokens,
                batch_size,
                interactive=interactive,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
                kv_cache_quantization=kv_cache_quantization,
                cache_size=cache_size,
                linear_causal_mask=linear_causal_mask,
                prefill_start_event=prefill_start_event,
                prefill_end_event=prefill_end_event,
                decode_start_event=decode_start_event,
                decode_end_event=decode_end_event,
            )
        if i < 0:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        if hasattr(prof, "export_chrome_trace"):
            prof.export_chrome_trace(f"{profile}.json")
        device_sync(device=device)  # MKG
        t = time.perf_counter() - t0

        if not interactive and demo_summarize_prompt is None and prefill_size is None:
            tok_list = y[0].tolist()
            # truncate text after end of string token
            tokens = (
                tok_list
                if tokenizer.eos_id() not in tok_list
                else tok_list[: tok_list.index(tokenizer.eos_id())]
            )
            print(tokenizer.decode(tokens))
        else:
            print("\n")
        tokens_generated = y.size(-1) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics["tokens_per_sec"].append(tokens_sec)
        aggregate_metrics["time"].append(t)
        decode_time = decode_start_event.elapsed_time(decode_end_event) / 1000
        decode_tokens_sec = tokens_generated / decode_time
        aggregate_metrics["decode_tokens_per_sec"].append(decode_tokens_sec)
        prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000
        aggregate_metrics["prefill_time"].append(prefill_time)
        print(
            f"Sample {i + 1} | overall time {t:.04f} s {tokens_sec:.02f} tokens/sec",
            f"| prefill time {prefill_time:.04f} s decode {decode_tokens_sec:.02f} tokens/sec",
        )
        print(f"Bandwidth achieved: {model_size * tokens_sec:.02f} GB/s")

        if memory_profile and i == 0:
            if device == "cuda":
                snapshot = torch.cuda.memory._snapshot()
            elif device == "xpu":
                snapshot = torch.xpu.memory._snapshot()
            else:
                print("Memory profiling only works on CUDA or XPU devices")

            with open(f"{memory_profile}.pickle", "wb") as f:
                from pickle import dump

                dump(snapshot, f)
            print(
                f"\nmemory profile {memory_profile}.pickle saved, to convert that to a usable file, use",
                "python pytorch/torch/cuda/_memory_viz.py trace_plot <pickle file> -o <desired output name>.html",
            )
            break
    print("==========")

    # ignore first sample for warmup
    tokpersec = torch.mean(torch.tensor(aggregate_metrics["tokens_per_sec"])).item()
    ttft = torch.mean(torch.tensor(aggregate_metrics["prefill_time"])).item()
    decode_tokpersec = torch.mean(
        torch.tensor(aggregate_metrics["decode_tokens_per_sec"])
    ).item()
    bandwidth = model_size * tokpersec
    mem = torch.cuda.max_memory_reserved() / 1e9
    print(f"Average overall tokens/sec: {tokpersec:.2f}")
    print(f"Average decode tokens/sec: {decode_tokpersec:.04f} s")
    print(f"Average TTFT: {ttft:.04f} s")
    if device == "cuda":
        mem = torch.cuda.max_memory_reserved() / 1e9
    elif device == "xpu":
        mem = torch.xpu.max_memory_reserved() / 1e9
    print(f"Average tokens/sec: {tokpersec:.2f}")
    if batch_size > 1:
        print(f"Average tokens/sec including batches {batch_size * tokpersec:.2f}")
    print(f"Average Bandwidth: {bandwidth:.02f} GB/s")
    print(f"Peak Memory Usage: {mem:.02f} GB")
    print(f"Model Size: {model_size:.02f} GB")
    if write_result:
        result_txt = f"\n{datetime.today().strftime('%Y%m%d%H%M%S')}, tok/s={tokpersec:6.2f}, tok/s_decode={decode_tokpersec:6.2f}, ttft={ttft:5.4f}, mem/s={bandwidth:7.2f} GB/s, peak_mem={mem:5.2f} GB, model_size={model_size:5.2f} GB "
        result_txt += f"quant: {quantization}, sparse: {sparsity}, mod: {checkpoint_path.parent.name}, kv_quant: {kv_cache_quantization}, compile: {compile}, compile_prefill: {compile_prefill}, dtype: {precision}, device: {device} "
        result_txt += "repro: python generate.py "
        result_txt += f"--quantization {quantization} " if quantization else ""
        result_txt += f"--sparsity {sparsity} " if sparsity else ""
        result_txt += f"--checkpoint_path {checkpoint_path} "
        result_txt += f"--device {device} "
        result_txt += f"--precision {precision} "
        result_txt += "--compile " if compile else ""
        result_txt += "--compile_prefill " if compile_prefill else ""
        result_txt += f"--prefill_size {prefill_size}" if prefill_size else ""
        result_txt += f"--profile {profile} " if profile else ""
        result_txt += f"--profile {memory_profile} " if memory_profile else ""
        result_txt += "--interactive " if interactive else ""
        result_txt += f"--num_samples {num_samples} "
        result_txt += f"--max_new_tokens {max_new_tokens} "
        result_txt += f"--batch_size {batch_size} "
        result_txt += f"--top_k {top_k} "
        result_txt += f"--temperature {temperature} "
        result_txt += f"--cache_size {cache_size}" if cache_size else ""
        result_txt += "--kv_cache_quantization " if kv_cache_quantization else ""
        result_txt += "--linear_causal_mask " if linear_causal_mask else ""

        f = open(write_result, "a")
        f.write(result_txt)
        f.close()

    if output_json_path:
        headers = [
            "name",
            "dtype",
            "min_sqnr",
            "compile",
            "device",
            "arch",
            "metric",
            "actual",
            "target",
        ]
        name = checkpoint_path.parent.name
        arch = get_arch_name()
        dtype = quantization or "noquant"
        memory_result = [
            name,
            dtype,
            min_sqnr,
            compile,
            device,
            arch,
            "mem/s",
            bandwidth,
            None,
        ]
        performance_result = [
            name,
            dtype,
            min_sqnr,
            compile,
            device,
            arch,
            "tok/s",
            tokpersec,
            None,
        ]
        write_json_result = (
            write_json_result_local if output_json_local else write_json_result_ossci
        )
        write_json_result(output_json_path, headers, memory_result)
        write_json_result(output_json_path, headers, performance_result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")
    parser.add_argument(
        "--prefill_size", type=int, default=None, help="Whether to run in ttft mode"
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello, my name is", help="Input prompt."
    )
    parser.add_argument(
        "--demo_summarize_prompt", type=str, help="Read prompt from text file"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to launch in interactive mode",
    )
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="Maximum number of new tokens."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size to benchmark with"
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("../../../checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "-q",
        "--quantization",
        type=str,
        help=(
            "Which quantization techniques to apply: int8dq, int8wo, fp6, int4wo-<groupsize>, int4wo-<groupsize>-hqq, autoquant, "
            + "autoquant-int4, autoquant-gemlite-int4, autoquant-float8, autoquant-sparse, autoquant-all, uintx-<nbits>-<groupsize>, uintx-<nbits>-<groupsize>-hqq, sparse-marlin, spinquant, "
            + "embed-int8wo, marlin_qqq, gemlite-<pack_bitwidth>-<nbits>-<groupsize>, float8dq, int4dq-<nbits>"
        ),
    )
    parser.add_argument(
        "--min_sqnr",
        type=float,
        default=None,
        help=(
            "min sqnr for quantizing v.s. not quantizing a layer, used in autoquant options",
        ),
    )
    parser.add_argument(
        "-s",
        "--sparsity",
        type=str,
        help=("Which sparsity techniques to apply: semi-structured"),
    )
    parser.add_argument(
        "--kv_cache_quantization",
        action="store_true",
        help="Whether to quantize the KV cache",
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=None,
        help="Force size of cache to be a certain number of tokens, if not set, will use max_new_tokens+prompt_size",
    )
    parser.add_argument(
        "--linear_causal_mask",
        action="store_true",
        help="Whether to use the memory efficient, but slightly less fast, linear causal mask (important for long context lengths)",
    )
    parser.add_argument(
        "--save", action="store_true", help="Whether to save the quantized model."
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--compile_prefill",
        action="store_true",
        help="Whether to compile the prefill (improves prefill perf, but higher compile times)",
    )
    parser.add_argument("--profile", type=Path, default=None, help="Profile path.")
    parser.add_argument(
        "--memory_profile", type=Path, default=None, help="filename for memory profile."
    )
    parser.add_argument(
        "--device", type=str, default=default_device, help="Device to use"
    )
    parser.add_argument(
        "--precision",
        type=lambda x: getattr(torch, x.split(".")[-1]),
        default=torch.bfloat16,
        help="dtype precision to use",
    )
    parser.add_argument(
        "--write_result", type=Path, default=None, help="Path where to write the result"
    )
    parser.add_argument(
        "--output_json_path",
        type=Path,
        default=None,
        help="Path where to write the json result for dashboard",
    )
    parser.add_argument(
        "--output_json_local",
        action="store_true",
        help="Whether to output json result for local machine or for CI machine, local option will fill in some dummy fields",
    )

    args = parser.parse_args()
    print(args)
    main(
        args.prefill_size,
        args.prompt,
        args.demo_summarize_prompt,
        args.interactive,
        args.num_samples,
        args.max_new_tokens,
        args.batch_size,
        args.top_k,
        args.temperature,
        args.checkpoint_path,
        args.quantization,
        args.min_sqnr,
        args.sparsity,
        args.kv_cache_quantization,
        args.cache_size,
        args.linear_causal_mask,
        args.save,
        args.compile,
        args.compile_prefill,
        args.profile,
        args.memory_profile,
        args.device,
        args.precision,
        args.write_result,
        args.output_json_path,
        args.output_json_local,
    )
