import pytest

# Skip if transformers is not installed
transformers = pytest.importorskip("transformers")
LlamaConfig = transformers.models.llama.modeling_llama.LlamaConfig
LlamaForCausalLM = transformers.models.llama.modeling_llama.LlamaForCausalLM

import json
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Union

import torch
from parameterized import parameterized_class
from utils import (
    PerfCounterManagerTestConfig,
    PerfCounterResult,
    PerfCounterTestConfig,
    PerfStatsTestConfig,
    attn_io_check,
    ffn_io_check,
    get_leaf_nodes,
    get_test_name,
    patch_device,
    qkv_proj_io_check,
)

from torchao.prototype.profiler.device_spec import CUDADeviceSpec, DeviceSpec
from torchao.prototype.profiler.performance_counter import (
    CUDAPerformanceTimer,
    PerformanceCounterMode,
    PerformanceStats,
    PerformanceTimer,
    TransformerPerformanceCounter,
)
from torchao.utils import TORCH_VERSION_AFTER_2_5

# ------------------- PerformanceCounter Tests ------------------- #

PERFCOUNTER_TEST_CONFIGS = [
    PerfCounterTestConfig(
        name="3.5B",
        batch_size=1,
        seqlen=128,
        dtype=torch.float16,
        num_hidden_layers=32 // 2,
        hidden_size=4096 // 2,
        intermediate_size=11008 // 2,
        num_attention_heads=32 // 2,
        vocab_size=32000 // 2,
    ),
    PerfCounterTestConfig(
        name="1.25B",
        batch_size=1,
        seqlen=128,
        dtype=torch.float16,
        num_hidden_layers=32 // 4,
        hidden_size=4096 // 4,
        intermediate_size=11008 // 4,
        num_attention_heads=32 // 4,
        vocab_size=32000 // 4,
    ),
    PerfCounterTestConfig(
        name="tiny",
        batch_size=1,
        seqlen=128,
        dtype=torch.float16,
        num_hidden_layers=1,
        hidden_size=4096 // 4,
        intermediate_size=11008 // 4,
        num_attention_heads=32 // 4,
        vocab_size=32000 // 4,
    ),
]


@unittest.skipIf(
    not TORCH_VERSION_AFTER_2_5, "PerformanceCounter requires torch >= 2.5+."
)
@unittest.skipIf(not torch.cuda.is_available(), "PerformanceCounter requires CUDA")
@parameterized_class(
    [asdict(cfg) for cfg in PERFCOUNTER_TEST_CONFIGS], class_name_func=get_test_name
)
class PerformanceCounterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_cfg = LlamaConfig(
            num_hidden_layers=cls.num_hidden_layers,
            hidden_size=cls.hidden_size,
            intermediate_size=cls.intermediate_size,
            num_attention_heads=cls.num_attention_heads,
            vocab_size=cls.vocab_size,
        )

        # Note we set some options manually since the model doesn't seem to be initialized correctly
        # when these options are set in LlamaConfig
        model_cfg._attn_implementation = "sdpa"
        cls.model = model = LlamaForCausalLM(model_cfg).to(cls.dtype).to("cuda")
        cls.model_config = model.config
        cls.element_size = cls.dtype.itemsize

        input_ids = torch.randint(
            0, model.config.vocab_size, (cls.batch_size, cls.seqlen), device="cuda"
        )
        with torch.no_grad():
            with torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
            ):
                with PerformanceCounterMode() as perf_counter:
                    _ = model(input_ids)
                    cls.perf_counter = perf_counter
        cls.summary_flops = perf_counter.get_summary_flop_counts()
        cls.summary_io = perf_counter.get_summary_io_counts()
        cls.flops_by_op = perf_counter.get_flop_counts()
        cls.io_by_op = perf_counter.get_io_counts()

    def test_qkv_proj(self):
        batch_size, seqlen = self.batch_size, self.seqlen
        element_size = self.element_size

        assert len(self.summary_flops) == len(self.summary_io)
        assert self.summary_flops.keys() == self.summary_io.keys()

        # Attn Projections
        for k in ["q_proj", "k_proj", "v_proj"]:
            # Flops check
            proj_keys = get_leaf_nodes(self.summary_flops.keys(), k)
            assert len(proj_keys) == self.model.config.num_hidden_layers
            expected_flops = (
                2
                * batch_size
                * seqlen
                * self.model_config.hidden_size
                * self.model_config.hidden_size
            )
            assert expected_flops == self.summary_flops[proj_keys[0]]

            # io check
            expected_size = qkv_proj_io_check(
                self.model_config, batch_size, seqlen, element_size
            )
            assert expected_size == self.summary_io[proj_keys[0]]

    def test_attn(self):
        batch_size, seqlen = self.batch_size, self.seqlen
        element_size = self.element_size
        model_config = self.model.config

        attention_keys = get_leaf_nodes(self.summary_flops.keys(), "self_attn")
        for k in attention_keys:
            flops = self.flops_by_op[k]
            io_movement = self.io_by_op[k]
            for op, count in flops.items():
                if "attention" in op.__name__:
                    expected_flops = (
                        2 * 2 * batch_size * seqlen * seqlen * model_config.hidden_size
                    )
                    assert expected_flops == count
            for op, count in io_movement.items():
                if "attention" in op.__name__:
                    # Check approx equal due to other small artifacts returned by sdpa.mem_efficient_attention
                    # See #https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/cuda/attention.cu#L867
                    # Check within 100 bytes
                    expected_size = attn_io_check(
                        model_config, batch_size, seqlen, element_size
                    )
                    assert abs(expected_size - count) < 100

    def test_ffn(self):
        batch_size, seqlen = self.batch_size, self.seqlen
        element_size = self.element_size

        for k in ["up_proj", "gate_proj", "down_proj"]:
            proj_keys = get_leaf_nodes(self.summary_flops.keys(), k)
            assert len(proj_keys) == self.model.config.num_hidden_layers
            expected_flops = (
                2
                * batch_size
                * seqlen
                * self.model_config.hidden_size
                * self.model_config.intermediate_size
            )
            assert expected_flops == self.summary_flops[proj_keys[0]]

            # io check
            expected_size = ffn_io_check(
                self.model_config, batch_size, seqlen, element_size, k
            )
            assert expected_size == self.summary_io[proj_keys[0]]


# ------------------- PerformanceStats Tests ------------------- #

PERFSTATS_TEST_CONFIGS = [
    PerfStatsTestConfig(
        label="with_device",
        num_tokens=128,
        latency=0.1,
        total_flops=123e9,
        total_io=123e6,
        flops_summary={"a": 234e12, "b": 345e9},
        io_summary={"a": 1, "b": 2},
        flop_counts={"a": 234e12, "b": 345e9},
        io_counts={"a": 1, "b": 2},
        device_bandwidth=1e9,
        device_flops_per_s=23e9,
    ),
    PerfStatsTestConfig(
        label="no_device",
        num_tokens=128,
        latency=0.1,
        total_flops=123e9,
        total_io=123e6,
        flops_summary={"a": 234e12, "b": 345e9},
        io_summary={"a": 1, "b": 2},
        flop_counts={"a": 234e12, "b": 345e9},
        io_counts={"a": 1, "b": 2},
        device_bandwidth=None,
        device_flops_per_s=None,
    ),
]


@pytest.mark.parametrize("cfg", PERFSTATS_TEST_CONFIGS, ids=lambda cfg: cfg.label)
def test_performance_stats(cfg: PerfStatsTestConfig):
    stats = PerformanceStats(**asdict(cfg))
    num_tokens = cfg.num_tokens
    latency = cfg.latency
    total_flops = cfg.total_flops
    total_io = cfg.total_io
    device_bandwidth = cfg.device_bandwidth
    device_flops_per_s = cfg.device_flops_per_s

    # Test derived metrics
    assert stats.token_throughput == num_tokens / latency
    assert stats.achieved_bandwidth == total_io / latency
    assert stats.achieved_flops_per_s == total_flops / latency
    if device_bandwidth is not None:
        assert (
            stats.bandwidth_utilization == stats.achieved_bandwidth / device_bandwidth
        )
        assert stats.theoretical_io_latency == total_io / device_bandwidth
    else:
        assert stats.bandwidth_utilization is None
        assert stats.theoretical_io_latency is None
    if device_flops_per_s is not None:
        assert (
            stats.flops_utilization == stats.achieved_flops_per_s / device_flops_per_s
        )
        assert stats.theoretical_compute_latency == total_flops / device_flops_per_s
    else:
        assert stats.flops_utilization is None
        assert stats.theoretical_compute_latency is None

    # Test str - stats should be formatted to closest power of 10 ** 3 with 2 decimal places of precision
    stats_str = str(stats)

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
        expected_bandwidth_utilization_str = (
            f"{stats.achieved_bandwidth / device_bandwidth:.4f}"
        )
        expected_io_latency_str = f"{stats.theoretical_io_latency:.2f} s"
        assert expected_bandwidth_utilization_str in stats_str
        assert expected_io_latency_str in stats_str

    if device_flops_per_s is not None:
        expected_flops_utilization_str = (
            f"{stats.achieved_flops_per_s / device_flops_per_s:.4f}"
        )
        expected_compute_latency_str = f"{stats.theoretical_compute_latency:.2f} s"
        assert expected_flops_utilization_str in stats_str
        assert expected_compute_latency_str in stats_str


# ------------------- TransformerPerformanceCounter Tests ------------------- #

PERFCOUNTERMANAGER_TEST_CONFIGS = [
    PerfCounterManagerTestConfig(
        "no_device", (1, 1024, 4096, 4096), PerformanceTimer, torch.bfloat16, (None, 0)
    ),
    PerfCounterManagerTestConfig(
        "a100",
        (1, 1024, 4096, 4096),
        CUDAPerformanceTimer,
        torch.bfloat16,
        ("A100", 2e12),
    ),
]


@unittest.skipIf(
    not TORCH_VERSION_AFTER_2_5, "TransformerPerformanceCounter requires torch >= 2.5+."
)
@unittest.skipIf(
    not torch.cuda.is_available(), "TransformerPerformanceCounter requires CUDA"
)
@parameterized_class(
    [asdict(cfg) for cfg in PERFCOUNTERMANAGER_TEST_CONFIGS],
    class_name_func=get_test_name,
)
class TestTransformerPerformanceCounter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shape, timer_cls, dtype = cls.shape, cls.timer_cls, cls.dtype
        batch_size, query_len, in_features, out_features = shape
        num_tokens = batch_size * query_len
        element_size = dtype.itemsize
        a = torch.randn(num_tokens, in_features, dtype=dtype, device="cuda")
        b = torch.randn(in_features, out_features, dtype=dtype, device="cuda")
        # Set up device spec
        device_name, bandwidth = cls.device_spec
        if device_name is not None:
            with patch_device(device_name):
                device_spec = CUDADeviceSpec(dtype=torch.bfloat16, bandwidth=bandwidth)

        else:
            device_spec = None

        # Stateful class level objects, which will be used in individual tests
        cls.cm = cm = TransformerPerformanceCounter(
            timer_cls=timer_cls, device_spec=device_spec
        )
        cls.FLOAT_TOL = 1e-5
        cls.expected = expected = {}

        # Start count for a
        start = time.perf_counter()
        with cm.count("a", num_tokens=num_tokens):
            _ = torch.matmul(a, b)
        end = time.perf_counter()

        latency = end - start
        expected_flops = 2 * num_tokens * in_features * out_features
        expected_io = (
            num_tokens * in_features
            + in_features * out_features
            + num_tokens * out_features
        ) * element_size

        expected["a"] = PerfCounterResult(
            name="a",
            latency=latency,
            flops=expected_flops,
            io=expected_io,
            total_flops=expected_flops,
            total_io=expected_io,
        )

        # Start count for b
        start = time.perf_counter()
        with cm.count("b", num_tokens=num_tokens):
            _ = torch.matmul(a, b)
        end = time.perf_counter()
        latency = end - start

        expected["b"] = PerfCounterResult(
            name="b",
            latency=latency,
            flops=expected_flops,
            io=expected_io,
            total_flops=cm.total_flops,
            total_io=cm.total_io,
        )

    def test_perf_stats_a(self):
        cm: TransformerPerformanceCounter = self.cm
        expected = self.expected["a"]

        counts = cm.get_counts()
        assert "a" in counts

        # Check captured performance stats
        psa: PerformanceStats = counts["a"]
        # Raw metrics
        # Latency won't be exact since timing external to the profiler
        assert abs(psa.latency - expected.latency) < 1e-1  # +/- 100ms
        assert psa.total_flops == expected.flops
        assert psa.total_io == expected.io

        # Derived metrics
        assert psa.token_throughput == psa.num_tokens / psa.latency
        assert psa.achieved_flops_per_s == psa.total_flops / psa.latency
        assert psa.achieved_bandwidth == psa.total_io / psa.latency

    def test_perf_stats_b(self):
        cm: TransformerPerformanceCounter = self.cm
        assert "a" in cm.counts
        assert "b" in cm.counts
        psa = cm.counts["a"]
        psb = cm.counts["b"]
        expected = self.expected["b"]
        assert abs(psb.latency - expected.latency) < 1e-1  # +/- 100ms
        assert psb.total_flops == expected.flops
        assert psb.total_io == expected.io

        # check that **total** flops and io after matmul `b` has run accounts for both matmuls
        # also check that these global properties are updated correctly in the manager object
        assert (
            expected.total_flops == psa.total_flops + psb.total_flops == cm.total_flops
        )
        assert expected.total_io == psa.total_io + psb.total_io == cm.total_io
        assert cm.total_time == psa.latency + psb.latency

    def test_stats_summary(self):
        cm: TransformerPerformanceCounter = self.cm
        FLOAT_TOL = self.FLOAT_TOL
        psa = cm.counts["a"]
        psb = cm.counts["b"]
        summary: PerformanceStats = cm.stats_summary

        # Raw stats
        assert summary.num_tokens == psa.num_tokens + psb.num_tokens
        assert summary.total_io == psa.total_io + psb.total_io
        assert summary.total_flops == psa.total_flops + psb.total_flops
        assert summary.latency == psa.latency + psb.latency

        # Derived stats
        expected_token_throughput = (psa.num_tokens + psb.num_tokens) / (
            psa.latency + psb.latency
        )
        expected_io_throughput = (psa.total_io + psb.total_io) / (
            psa.latency + psb.latency
        )
        expected_flops_throughput = (psa.total_flops + psb.total_flops) / (
            psa.latency + psb.latency
        )
        assert abs(summary.token_throughput - expected_token_throughput) < FLOAT_TOL
        assert abs(summary.achieved_bandwidth - expected_io_throughput) < FLOAT_TOL
        assert abs(summary.achieved_flops_per_s - expected_flops_throughput) < FLOAT_TOL

        device_spec = cm.device_spec
        if device_spec is not None:
            expected_bandwidth_utilization = (
                expected_io_throughput / device_spec.bandwidth
            )
            expected_flops_utilization = (
                expected_flops_throughput / device_spec.flops_per_s
            )
            assert (
                abs(summary.bandwidth_utilization - expected_bandwidth_utilization)
                < FLOAT_TOL
            )
            assert (
                abs(summary.flops_utilization - expected_flops_utilization) < FLOAT_TOL
            )
        else:
            assert summary.bandwidth_utilization is None
            assert summary.flops_utilization is None

    def test_json(self):
        cm: TransformerPerformanceCounter = self.cm
        psa: PerformanceStats = cm.counts["a"]
        psb: PerformanceStats = cm.counts["b"]
        device_spec: Union[DeviceSpec, None] = cm.device_spec

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "test.json"
            cm.to_json(json_path)

            with open(json_path, "r") as f:
                perf_dict = json.load(f)

            assert "a" in perf_dict
            assert "b" in perf_dict

            # Test basic stats are recorded properly
            assert perf_dict["a"]["num_tokens"] == psa.num_tokens
            assert perf_dict["a"]["total_io"] == psa.total_io
            assert perf_dict["a"]["total_flops"] == psa.total_flops
            assert perf_dict["a"]["latency"] == psa.latency

            assert perf_dict["b"]["num_tokens"] == psb.num_tokens
            assert perf_dict["b"]["total_io"] == psb.total_io
            assert perf_dict["b"]["total_flops"] == psb.total_flops
            assert perf_dict["b"]["latency"] == psb.latency

            # Test derived properties are present
            perf_dict["a"]["achieved_flops_per_s"] == psa.achieved_flops_per_s
            perf_dict["a"]["achieved_bandwidth"] == psa.achieved_bandwidth
            perf_dict["b"]["achieved_flops_per_s"] == psb.achieved_flops_per_s
            perf_dict["b"]["achieved_bandwidth"] == psb.achieved_bandwidth

            if device_spec is not None:
                assert perf_dict["a"]["device_flops_per_s"] == device_spec.flops_per_s
                assert perf_dict["a"]["device_bandwidth"] == device_spec.bandwidth
                assert (
                    perf_dict["a"]["theoretical_io_latency"]
                    == psa.theoretical_io_latency
                )
                assert (
                    perf_dict["a"]["theoretical_compute_latency"]
                    == psa.theoretical_compute_latency
                )
                assert (
                    perf_dict["a"]["bandwidth_utilization"] == psa.bandwidth_utilization
                )
                assert perf_dict["a"]["flops_utilization"] == psa.flops_utilization

                assert perf_dict["b"]["device_flops_per_s"] == device_spec.flops_per_s
                assert perf_dict["b"]["device_bandwidth"] == device_spec.bandwidth
                assert (
                    perf_dict["b"]["theoretical_io_latency"]
                    == psb.theoretical_io_latency
                )
                assert (
                    perf_dict["b"]["theoretical_compute_latency"]
                    == psb.theoretical_compute_latency
                )
                assert (
                    perf_dict["b"]["bandwidth_utilization"] == psb.bandwidth_utilization
                )
                assert perf_dict["b"]["flops_utilization"] == psb.flops_utilization
