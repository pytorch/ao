import pytest

# Skip if transformers is not installed
transformers = pytest.importorskip("transformers")
LlamaConfig = transformers.models.llama.modeling_llama.LlamaConfig
LlamaForCausalLM = transformers.models.llama.modeling_llama.LlamaForCausalLM
import time
from contextlib import contextmanager
from unittest.mock import patch

import torch

from torchao.profiler.device_spec import CUDADeviceSpec
from torchao.profiler.performance_counter import (
    CUDAPerformanceTimer,
    PerformanceCounterManager,
    PerformanceCounterMode,
    PerformanceStats,
    PerformanceTimer,
)
from torchao.utils import TORCH_VERSION_AFTER_2_5


def get_leaf_nodes(count_keys, module_name):
    return [k for k in count_keys if k.endswith(module_name)]

def attn_proj_io_check(model_config, batch_size, seqlen, element_size):
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
    weight_size = model_config.hidden_size * model_config.intermediate_size * element_size 
    if module_name == "down_proj":
        output_size = batch_size * seqlen * model_config.hidden_size * element_size
    else:
        output_size = batch_size * seqlen * model_config.intermediate_size * element_size 

    return input_size + weight_size + output_size


CONFIG_7B = (32, 4096, 11008, 32, 32000)
MEDIUM_CONFIG = [p // 2 for p in CONFIG_7B]
SMALL_CONFIG = [p // 4 for p in CONFIG_7B]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not TORCH_VERSION_AFTER_2_5, reason="requires torch >= 2.5")
@pytest.mark.parametrize("num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, vocab_size", [MEDIUM_CONFIG, SMALL_CONFIG])
@pytest.mark.parametrize("batch_size, seqlen", [(1, 128),])
@pytest.mark.parametrize("dtype", [torch.float16], ids=lambda p: str(p))
def test_performance_counter(num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, vocab_size, batch_size, seqlen, dtype):

    cfg = LlamaConfig(num_hidden_layers=num_hidden_layers, 
                      hidden_size=hidden_size,
                      intermediate_size=intermediate_size, 
                      num_attention_heads=num_attention_heads, 
                      vocab_size=vocab_size)

    # Note we set some options manually since the model doesn't seem to be initialized correctly
    # when these options are set in LlamaConfig
    cfg._attn_implementation = "sdpa"
    model = LlamaForCausalLM(cfg).to(dtype).to("cuda")
    model_config = model.config
    element_size = dtype.itemsize
    
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seqlen), device="cuda")
    with torch.no_grad():
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
            with PerformanceCounterMode() as perf_counter:
                _ = model(input_ids)
    
    summary_flops = perf_counter.get_summary_flop_counts()
    summary_io = perf_counter.get_summary_io_counts()
    flops_by_op = perf_counter.get_flop_counts()
    io_by_op = perf_counter.get_io_counts()
    assert len(summary_flops) == len(summary_io)
    assert summary_flops.keys() == summary_io.keys()

    # Attn Projections
    for k in ["q_proj", "k_proj", "v_proj"]:
        # Flops check
        proj_keys = get_leaf_nodes(summary_flops.keys(), k)
        assert len(proj_keys) == model.config.num_hidden_layers
        expected_flops = 2 * batch_size * seqlen * model_config.hidden_size * model_config.hidden_size
        assert expected_flops == summary_flops[proj_keys[0]]
        
        # io check
        expected_size = attn_proj_io_check(model_config, batch_size, seqlen, element_size)
        assert expected_size == summary_io[proj_keys[0]]

    # Attention
    attention_keys = get_leaf_nodes(summary_flops.keys(), "self_attn")
    for k in attention_keys:
        flops = flops_by_op[k]
        io_movement = io_by_op[k]
        for op, count in flops.items():
            if "attention" in op.__name__: 
                expected_flops = 2 * 2 * batch_size * seqlen * seqlen * model_config.hidden_size
                assert expected_flops == count
        for op, count in io_movement.items():
            if "attention" in op.__name__:
                # Check approx equal due to other small artifacts returned by sdpa.mem_efficient_attention
                # See #https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/cuda/attention.cu#L867
                # Check within 100 bytes
                expected_size = attn_io_check(model_config, batch_size, seqlen, element_size)
                assert abs(expected_size - count) < 100
    # FFN
    for k in ["up_proj", "gate_proj", "down_proj"]:
        proj_keys = get_leaf_nodes(summary_flops.keys(), k)
        assert len(proj_keys) == model.config.num_hidden_layers
        expected_flops = 2 * batch_size * seqlen * model_config.hidden_size * model_config.intermediate_size
        assert expected_flops == summary_flops[proj_keys[0]]
        
        # io check
        expected_size = ffn_io_check(model_config, batch_size, seqlen, element_size, k)
        assert expected_size == summary_io[proj_keys[0]]

@contextmanager
def patch_device(device_name):
    with patch("torch.cuda.get_device_name", return_value=device_name):
        yield

TEST_STATS = [("with_device", 128, 0.1, 123e9, 123e6, {"a": 234e12, "b": 345e9}, 1, {"a": 1, "b": 2}, {"a": 1, "b": 2},  1e9, 23e9),
              ("no_device", 128, 0.1, 123e9, 123e6, {"a": 234e12, "b": 345e9}, 1, {"a": 1, "b": 2}, {"a": 1, "b": 2}, None, None)]
@pytest.mark.parametrize("label, num_tokens, duration, total_flops, total_io, flops_summary, io_summary, flop_counts, io_counts, device_bandwidth, device_flops_per_s", TEST_STATS)
def test_performance_stats(label, num_tokens, duration, total_flops, total_io, flops_summary, io_summary, flop_counts, io_counts, device_bandwidth, device_flops_per_s):
    stats = PerformanceStats(label=label,
                             num_tokens=num_tokens,
                             duration=duration,
                             total_flops=total_flops,
                             total_io=total_io,
                             flops_summary=flops_summary,
                             io_summary=io_summary,
                             flop_counts=flop_counts,
                             io_counts=io_counts,
                             device_bandwidth=device_bandwidth,
                             device_flops_per_s=device_flops_per_s)
    
    # Test derived metrics
    assert stats.token_throughput == num_tokens / duration
    assert stats.io_throughput == total_io / duration
    assert stats.flops_throughput == total_flops / duration
    if device_bandwidth is not None:
        assert stats.bandwidth_utilization == stats.io_throughput / device_bandwidth
    else:
        assert stats.bandwidth_utilization is None
    if device_flops_per_s is not None:
        assert stats.flops_utilization == stats.flops_throughput / device_flops_per_s
    else:
        assert stats.flops_utilization is None
        
    # Test str - stats should be formatted to closest power of 10 ** 3 with 2 decimal places of precision
    stats_str = str(stats)
    print(stats_str)
    # Base Stats
    expected_io_str = ".12 GB"
    expected_flops_str = ".12 TFLOPs"
    assert expected_io_str in stats_str
    assert expected_flops_str in stats_str

    # Derived Stats
    expected_io_throughput_str = "1.23 GB/s"
    expected_flops_throughput_str = "1.23 TFLOPs/s"
    assert expected_io_throughput_str in stats_str
    assert expected_flops_throughput_str in stats_str
    
    # Utilization Stats
    if device_bandwidth is not None:
        expected_bandwidth_utilization_str = f"{stats.io_throughput / device_bandwidth:.2f}%"
        assert expected_bandwidth_utilization_str in stats_str
    if device_flops_per_s is not None:
        expected_flops_utilization_str = f"{stats.flops_throughput / device_flops_per_s:.2f}%"
        assert expected_flops_utilization_str in stats_str
        

@pytest.fixture
def device_spec(request):
    device_name, bandwidth = request.param
    if device_name is not None:
        with patch_device(device_name):
            return CUDADeviceSpec(dtype=torch.bfloat16, bandwidth=bandwidth)
    else:
        return None
@pytest.fixture
def performance_counter_manager(device_spec, request):
    shape, timer_cls, dtype = request.param
    batch_size, query_len, in_features, out_features = shape
    num_tokens = batch_size * query_len
    element_size = dtype.itemsize
    a = torch.randn(num_tokens, in_features, dtype=dtype, device="cuda")
    b = torch.randn(in_features, out_features, dtype=dtype, device="cuda")
       
    cm = PerformanceCounterManager(timer_cls=timer_cls, device_spec=device_spec)
    
    # Start count
    start = time.perf_counter()
    with cm.count("a", num_tokens=num_tokens):
        _ = torch.matmul(a, b)
    end = time.perf_counter()
    
    duration_a = (end - start)
    expected_flops = 2 * num_tokens * in_features * out_features
    expected_io = (num_tokens * in_features + in_features * out_features + num_tokens * out_features) * element_size 
    
    start = time.perf_counter()
    with cm.count("b", num_tokens=num_tokens):
        _ = torch.matmul(a, b)
    end = time.perf_counter()
    duration_b = end - start 

@pytest.mark.parametrize("shape", [(1, 1024, 4096, 4096), (128, 1, 1024, 4096)], ids=lambda p: ",".join(map(str, p)))
@pytest.mark.parametrize("timer_cls", [PerformanceTimer, CUDAPerformanceTimer], ids=lambda p: p.__name__)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("device_spec", [(None, 0), ("A100", 2e12)], indirect=True, ids=lambda p: p[0])
def test_performance_counter_manager(shape, timer_cls, dtype, device_spec):
    FLOAT_TOL = 1e-5
    # Set up inputs
    batch_size, query_len, in_features, out_features = shape
    num_tokens = batch_size * query_len
    element_size = dtype.itemsize
    a = torch.randn(num_tokens, in_features, dtype=dtype, device="cuda")
    b = torch.randn(in_features, out_features, dtype=dtype, device="cuda")
       
    cm = PerformanceCounterManager(timer_cls=timer_cls, device_spec=device_spec)
    
    # Start count
    start = time.perf_counter()
    with cm.count("a", num_tokens=num_tokens):
        _ = torch.matmul(a, b)
    end = time.perf_counter()
    
    duration = (end - start)
    expected_flops = 2 * num_tokens * in_features * out_features
    expected_io = (num_tokens * in_features + in_features * out_features + num_tokens * out_features) * element_size 
    assert cm.total_flops == expected_flops
    assert cm.total_io == expected_io
    counts = cm.get_counts()
    assert "a" in counts
    # Check captured performance stats
    psa: PerformanceStats = counts["a"]
    # Raw metrics
    assert abs(psa.duration - duration) < 1e-1 # +/- 100ms
    assert psa.total_flops == expected_flops
    assert psa.total_io == expected_io
    # Derived metrics
    assert psa.token_throughput == psa.num_tokens / psa.duration
    assert psa.achieved_flops_per_s == psa.total_flops / psa.duration
    assert psa.achieved_bandwidth == psa.total_io / psa.duration
    
    start = time.perf_counter()
    with cm.count("b", num_tokens=num_tokens):
        _ = torch.matmul(a, b)
    end = time.perf_counter()
    duration = end - start 
    assert "b" in cm.counts
    psb = cm.counts["b"]
    assert abs(psb.duration - duration) < 1e-1 # +/- 100ms
    assert psb.total_flops == expected_flops
    assert psb.total_io == expected_io
    
    # Test that total properties account for both a and b
    assert cm.total_time == psa.duration + psb.duration
    assert cm.total_flops == 2 * expected_flops
    assert cm.total_io == 2 * expected_io
    
    # Test stats_summary property, which returns a new PerformanceStats object with accumulated stats 
    summary: PerformanceStats = cm.stats_summary
    # Raw stats
    assert summary.num_tokens == psa.num_tokens + psb.num_tokens
    assert summary.total_io == psa.total_io + psb.total_io
    assert summary.total_flops == psa.total_flops + psb.total_flops
    assert summary.duration == psa.duration + psb.duration

    # Derived stats
    expected_token_throughput = (psa.num_tokens + psb.num_tokens) / (psa.duration + psb.duration)
    expected_io_throughput = (psa.total_io + psb.total_io) / (psa.duration + psb.duration)
    expected_flops_throughput = (psa.total_flops + psb.total_flops) / (psa.duration + psb.duration)
    assert abs(summary.token_throughput - expected_token_throughput) < FLOAT_TOL
    assert abs(summary.achieved_bandwidth - expected_io_throughput) < FLOAT_TOL
    assert abs(summary.achieved_flops_per_s - expected_flops_throughput) < FLOAT_TOL
    
    if device_spec is not None:
        expected_bandwidth_utilization = expected_io_throughput / device_spec.bandwidth
        expected_flops_utilization = expected_flops_throughput / device_spec.flops_per_s
        assert abs(summary.bandwidth_utilization - expected_bandwidth_utilization) < FLOAT_TOL
        assert abs(summary.flops_utilization - expected_flops_utilization) < FLOAT_TOL