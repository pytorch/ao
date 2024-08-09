from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import torch

from torchao.profiler import PerformanceTimer


@contextmanager
def patch_device(device_name):
    with patch("torch.cuda.get_device_name", return_value=device_name):
        yield


@dataclass(frozen=True)
class PerfCounterTestConfig:
    name: str
    batch_size: int
    seqlen: int
    dtype: torch.dtype
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    vocab_size: int


def get_leaf_nodes(count_keys, module_name):
    return [k for k in count_keys if k.endswith(module_name)]


def qkv_proj_io_check(model_config, batch_size, seqlen, element_size):
    input_size = batch_size * seqlen * model_config.hidden_size * element_size
    weight_size = model_config.hidden_size * model_config.hidden_size * element_size
    output_size = batch_size * seqlen * model_config.hidden_size * element_size
    return input_size + weight_size + output_size


def attn_io_check(model_config, batch_size, seqlen, element_size):
    # queries, keys, values -> factor of 3
    input_size = (batch_size * seqlen * model_config.hidden_size * 3) * element_size
    output_size = (batch_size * seqlen * model_config.hidden_size) * element_size
    return input_size + output_size


def ffn_io_check(model_config, batch_size, seqlen, element_size, module_name):
    assert module_name in ["up_proj", "gate_proj", "down_proj"]

    if module_name == "down_proj":
        input_size = batch_size * seqlen * model_config.intermediate_size * element_size
    else:
        input_size = batch_size * seqlen * model_config.hidden_size * element_size
    weight_size = (
        model_config.hidden_size * model_config.intermediate_size * element_size
    )
    if module_name == "down_proj":
        output_size = batch_size * seqlen * model_config.hidden_size * element_size
    else:
        output_size = (
            batch_size * seqlen * model_config.intermediate_size * element_size
        )

    return input_size + weight_size + output_size


@dataclass(frozen=True)
class PerfStatsTestConfig:
    label: str
    num_tokens: int
    latency: float
    total_flops: float
    total_io: float
    flops_summary: dict
    io_summary: dict
    flop_counts: dict
    io_counts: dict
    device_bandwidth: Optional[float] = None
    device_flops_per_s: Optional[float] = None


def get_test_name(cls, num, params_dict):
    return f"{cls.__name__}_{num}_{params_dict['name']}"


@dataclass(frozen=True)
class PerfCounterResult:
    name: str
    latency: float
    flops: float
    io: float
    total_flops: float
    total_io: float


@dataclass
class PerfCounterManagerTestConfig:
    name: str
    shape: tuple[int]
    timer_cls: PerformanceTimer
    dtype: torch.dtype
    device_spec: tuple[Optional[str], int]
