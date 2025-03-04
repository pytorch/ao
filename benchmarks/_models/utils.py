import datetime
import hashlib
import json
import os
import platform
import time
from typing import Optional, Tuple

import torch

from benchmarks._models.llama.model import Transformer
from torchao.utils import default_device


def get_arch_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    else:
        # This returns x86_64 or arm64 (for aarch64)
        return platform.machine()


def write_json_result_ossci(output_json_path, headers, row):
    """
    Write the result into JSON format, so that it can be uploaded to the benchmark database
    to be displayed on OSS dashboard. The JSON format is defined at
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database

    OSS CI version, that will leave many fields to be filled in by CI
    """
    mapping_headers = {headers[i]: v for i, v in enumerate(row)}
    record = {
        "benchmark": {
            "name": "TorchAO benchmark",
            "mode": "inference",
            "dtype": mapping_headers["dtype"],
            "extra_info": {
                "device": mapping_headers["device"],
                "arch": mapping_headers["arch"],
                "min_sqnr": mapping_headers["min_sqnr"],
                # True means compile is enabled, False means eager mode
                "compile": mapping_headers["compile"],
            },
        },
        "model": {
            "name": mapping_headers["name"],
            "type": "model",
            "origins": ["torchao"],
        },
        "metric": {
            "name": mapping_headers["metric"],
            "benchmark_values": [mapping_headers["actual"]],
            "target_value": mapping_headers["target"],
        },
    }

    with open(f"{os.path.splitext(output_json_path)[0]}.json", "a") as f:
        print(json.dumps(record), file=f)


def write_json_result_local(output_json_path, headers, row):
    """
    Write the result into JSON format, so that it can be uploaded to the benchmark database
    to be displayed on OSS dashboard. The JSON format is defined at
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database

    Local version (filling in dummy values for fields that should be populated by CI)
    """
    mapping_headers = {headers[i]: v for i, v in enumerate(row)}
    today = datetime.date.today()
    sha_hash = hashlib.sha256(str(today).encode("utf-8")).hexdigest()
    first_second = datetime.datetime.combine(today, datetime.time.min)
    workflow_id = int(first_second.timestamp())
    job_id = workflow_id + 1
    record = {
        "timestamp": int(time.time()),
        "schema_version": "v3",
        "name": "devvm local benchmark",
        "repo": "pytorch/ao",
        "head_branch": "main",
        "head_sha": sha_hash,
        "workflow_id": workflow_id,
        "run_attempt": 1,
        "job_id": job_id,
        "benchmark": {
            "name": "TorchAO benchmark",
            "mode": "inference",
            "dtype": mapping_headers["dtype"],
            "extra_info": {
                "device": mapping_headers["device"],
                "arch": mapping_headers["arch"],
                "min_sqnr": mapping_headers["min_sqnr"],
                # True means compile is enabled, False means eager mode
                "compile": mapping_headers["compile"],
            },
        },
        "model": {
            "name": mapping_headers["name"],
            "type": "model",
            "origins": ["torchao"],
        },
        "metric": {
            "name": mapping_headers["metric"],
            "benchmark_values": [mapping_headers["actual"]],
            "target_value": mapping_headers["target"],
        },
    }

    with open(f"{os.path.splitext(output_json_path)[0]}.json", "a") as f:
        print(json.dumps(record), file=f)


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
