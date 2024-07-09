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

class PerfStatsTestConfig:
    label: str
    num_tokens: int
    duration: float
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
    duration: float
    flops: float
    io: float
    total_flops: float
    total_io: float

@dataclass
class PerfCounterTestConfig:
    name: str
    shape: tuple[int]
    timer_cls: PerformanceTimer
    dtype: torch.dtype
    device_spec: tuple[Optional[str], int]
