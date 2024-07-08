import json
import math
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Dict, Optional

import torch
from torch.utils._pytree import tree_map
from torch.utils.flop_counter import FlopCounterMode

from .device_spec import DeviceSpec

aten = torch.ops.aten

class PerformanceCounterMode(FlopCounterMode):
    def __init__(self, display=False, depth=10, debug=False):
        self.debug = debug
        self.io_counts = defaultdict(lambda: defaultdict(int))
        super().__init__(display=display, depth=depth)
    
    def get_io_counts(self):
        return {k: dict(v) for k,v in self.io_counts.items()}
    
    def get_total_io(self):
        return sum(self.io_counts['Global'].values())

    def _get_io_sizes(self, args):
        sizes = tree_map(lambda x: x.numel() * x.element_size() if isinstance(x, torch.Tensor) else 0, args)
        if not hasattr(sizes, "__len__"):
            sizes = [sizes]
        return sizes
    
    def get_summary_flop_counts(self):
        flop_counts = self.get_flop_counts()
        return {k: sum(v.values()) for k,v in flop_counts.items()}
    
    def get_summary_io_counts(self):
        io_counts = self.get_io_counts()
        return {k: sum(v.values()) for k,v in io_counts.items()}
    
    def _nearest_power_of_10(self, x):
        if x == 0:
            return x, 0
        
        power = int(math.floor(math.log10(abs(x)) / 3))
        scaled_value = x / (10 ** (3 * power))
    
        return scaled_value, power
    
    def pretty_summary_counts(self, type="flops", precision=2, depth=None):
        assert type in ["flops", "io"]
        metric_units = {0: '', 1: 'k', 2: 'M', 3: 'G', 4: 'T', 5: 'P', 6: 'E', 7: 'Z', 8: 'Y'}

        if depth is None:
            depth = self.depth
        summary_counts = self.get_summary_flop_counts() if type == "flops" else self.get_summary_io_counts()
        keys_to_print = [k for k in summary_counts.keys() if len(k.split(".")) <= depth]
        units = "FLOPs" if type == "flops" else "B"
        summary_str = []
        for k in sorted(keys_to_print, key=lambda x: len(x.split("."))):
            if k == "Global" or k is None:
                continue
            spaces = " " * (len(k.split(".")) - 1)
            scaled_val, power = self._nearest_power_of_10(summary_counts[k])
            formatted_val = f"{scaled_val:.{precision}f}{metric_units[power]}{units}"      
            summary_str.append(f"{spaces}{k}: {formatted_val}")
        
        return "\n".join(summary_str)
    
    def _count_io(self, func_packet, out, args, kwargs):
        arg_sizes = self._get_io_sizes(args)
        kwargs_sizes = self._get_io_sizes(kwargs.values())
        out_sizes = self._get_io_sizes(out)
        arg_size, kwargs_size, out_size = sum(arg_sizes), sum(kwargs_sizes), sum(out_sizes)
        return arg_size, kwargs_size, out_size
    
    def _count_flops(self, func_packet, out, args, kwargs):
        if func_packet in self.flop_registry:
            flop_count_func = self.flop_registry[func_packet]
            flop_count = flop_count_func(*args, **kwargs, out_val=out)  # type: ignore[operator]
            arg_size, kwarg_size, out_size = self._count_io(func_packet, out, args, kwargs)
            total_size = arg_size + kwarg_size + out_size

            for par in set(self.mod_tracker.parents):
                if self.debug:
                    print(f"Counting flops for {par}, {func_packet}: {flop_count}")
                    print(f"Counting io for {par}, {func_packet}: {sum([arg_size, kwarg_size, out_size])} = {arg_size} + {kwarg_size} + {out_size}")
                self.flop_counts[par][func_packet] += flop_count
                self.io_counts[par][func_packet] += total_size
        
        return out

class PerformanceTimer:
    def __init__(self, name, precision=1, display=False, depth=10):
        self.name = name
        self.precision = precision
        self.display = display 
        self.depth = depth
        self.perf_counter = PerformanceCounterMode(display=display, depth=depth)
        
    def __enter__(self):
        self.start = time.perf_counter()
        self.perf_counter.__enter__()
        return self

    def _print_exit_msg(self):
        gflops = round(self.total_flops / 1e9, self.precision)
        ms = round(self.duration * 1e3, self.precision)
        if self.display: 
            print(f"{self.name.upper()}:  duration = {ms} ms, FLOPS = {gflops} GFLOPs")

    def __exit__(self, type, value, traceback):
        self.end = time.perf_counter()
        #Convert to ms
        self.duration = (self.end - self.start)
        self.perf_counter.__exit__(type, value, traceback)
        if self.display:
            self._print_exit_msg()        

    @property
    def total_flops(self):
        return self.perf_counter.get_total_flops()
    
    @property
    def total_io(self):
        return self.perf_counter.get_total_io()
    
    @property
    def flops_table(self):
        return self.perf_counter.get_table()
    
    def get_summary_flop_counts(self):
        return self.perf_counter.get_summary_flop_counts()
    
    def get_summary_io_counts(self):
        return self.perf_counter.get_summary_io_counts()
    
    @property
    def flop_counts(self):
        return self.perf_counter.get_flop_counts()
    
    @property
    def io_counts(self):
        return self.perf_counter.get_io_counts()
    
    def get_pretty_summary(self, depth):
        return self.perf_counter.pretty_summary_counts(depth=depth if depth is not None else self.depth)
class CUDAPerformanceTimer(PerformanceTimer):
        
    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        self.perf_counter = PerformanceCounterMode(display=self.display, depth=self.depth)
        self.perf_counter.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        # Convert from ms to s
        self.duration = self.start.elapsed_time(self.end) * 1e-3
        self.perf_counter.__exit__(type, value, traceback)

        if self.display:
            self._print_exit_msg()        

def to_nearest_power_of_10(x, precision=2):
    
    # Dictionary mapping powers of 10 to their metric abbreviations
    metric_units = {
        0: '',
        -6: 'µ',
        -3: 'm',
        6: 'M',
        9: 'G',
        12: 'T'
    }
    
    # Determine the closest power of 10
    if x == 0:
        return f"{x:.{precision}f}"
    
    power = int(math.floor(math.log10(abs(x))))
    # Adjust power to fit within the given metric units
    powers = sorted(metric_units.keys())
    closest_power = min(powers, key=lambda p: abs(p - power))
    
    # Calculate the value formatted to the closest power of 10
    value = x / 10**closest_power
    
    # Map the power to the metric unit
    unit = metric_units.get(closest_power, f"e{closest_power}")
    
    return f"{value:,.{precision}f} {unit}"

@dataclass
class PerformanceStats:
    label: str
    num_tokens: int
    duration: float
    total_flops: int
    total_io: int
    flops_summary: Dict[str, int]
    io_summary: Dict[str, int]
    flop_counts: Dict[str, Dict[Any, int]]
    io_counts: Dict[str, Dict[Any, int]]
    device_bandwidth: Optional[float] = None
    device_flops_per_s: Optional[float] = None    
    @property
    def token_throughput(self):
        return self.num_tokens / self.duration
    
    @property
    def achieved_flops_per_s(self):
        return self.total_flops / self.duration
    
    @property
    def achieved_bandwidth(self):
        return self.total_io / self.duration
    
    @property
    def bandwidth_utilization(self):
        if self.device_bandwidth is not None:
            return self.achieved_bandwidth / self.device_bandwidth
        else:
            print("Device bandwidth is not specified. Please specify the device bandwidth to enable bandwidth utilization calculation")
            return None
    @property
    def flops_utilization(self):
        if self.device_flops_per_s is not None:
            return self.achieved_flops_per_s / self.device_flops_per_s
        else:
            print("Device flops_per_s is not specified. Please specify the device throughput to enable flops utilization calculation")
            return None
    def _format(self, value, suffix):
        return to_nearest_power_of_10(value) + suffix
    def __str__(self):
        txt = textwrap.dedent(f"""\
            {self.label}:
              Duration = {self._format(self.duration, "s")}
              Tokens
                Total: {self.num_tokens} tokens
                Throughput: {self.token_throughput:,.0f} tokens/s
              IO
                Total: {self._format(self.total_io, "B")}
                Throughput: {self._format(self.io_throughput, "B/s")}
              FLOPs 
                Total: {self._format(self.total_flops, "FLOPs")}
                Throughput: {self._format(self.flops_throughput, "FLOPs/s")}""")
        
        indent_2 = " " * 2
        indent_4 = " " * 4
        if self.bandwidth_utilization is not None:
            txt += "\n" + textwrap.indent("""Utilization:\n""", indent_2)
            txt += textwrap.indent(f"""Bandwidth: {self.bandwidth_utilization:.2f}%""", indent_4)
        
        if self.flops_utilization is not None:
            txt +=  "\n" + textwrap.indent(f"""FLOPs: {self.flops_utilization:.2f}%""", indent_4)
        
        return txt

    def to_dict(self):
        return asdict(self)
    
class PerformanceCounterManager:
    def __init__(self, depth=10, timer_cls: PerformanceTimer=PerformanceTimer, device_spec: DeviceSpec=None, verbose=False):
        super().__init__()
        self._counts: Dict[str, PerformanceStats] = {}
        self._depth = depth
        self.timer_cls = timer_cls
        self.device_spec = device_spec
        self.verbose = verbose

    @contextmanager
    def count(self, label: str, num_tokens: int):
        perf_timer = self.timer_cls(name=label, depth=self._depth)
        perf_timer.__enter__()
        try:
            yield self
        finally:
            perf_timer.__exit__(None, None, None)
            stats = PerformanceStats(label=label, 
                                     num_tokens=num_tokens, 
                                     duration=perf_timer.duration,
                                     total_flops=perf_timer.total_flops,
                                     total_io=perf_timer.total_io,
                                     flops_summary=perf_timer.get_summary_flop_counts(),
                                     io_summary=perf_timer.get_summary_io_counts(),
                                     flop_counts=perf_timer.flop_counts,
                                     io_counts=perf_timer.io_counts,
                                     device_bandwidth=self.device_spec.bandwidth if self.device_spec.bandwidth is not None else None,
                                     device_flops_per_s=self.device_spec.flops_per_s if self.device_spec.flops_per_s is not None else None)
            self._counts[label] = stats
    @property
    def counts(self):
        return self._counts
    def get_counts(self):
        return self._counts            

    @property
    def total_flops(self):
        return sum(count["total_flops"] for count in self._counts.values())
    
    @property
    def total_io(self):
        return sum(count["total_io"] for count in self._counts.values())
    @property
    def total_tokens(self):
        return sum(count["num_tokens"] for count in self._counts.values())
    
    @property
    def total_time(self):
        return sum(count["duration"] for count in self._counts.values())
    
    def to_dict(self):
        # Convert flop_counts from OpOverloadPackets to str
        counts = deepcopy(self._counts)
        for label,label_counts in counts.items():
            counts[label]['flop_counts'] = {mod: {str(op): count for op, count in op_count.items()} for mod, op_count in label_counts['flop_counts'].items()}
            counts[label]['io_counts'] = {mod: {str(op): count for op, count in op_count.items()} for mod, op_count in label_counts['io_counts'].items()}

        return counts
    
    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)
    
    def _summarize_stat(self, key):
        return {label: getattr(self._counts[label], key) for label in self._counts.keys()}
    
    @property
    def flops_summary(self):
        return self._summarize_stat(key="summary_flops")
    
    @property
    def io_summary(self):
        return self._summarize_stat(key="summary_io")
    
    @property
    def flop_counts_summary(self):
        return self._summarize_stat(key="flop_counts")
    
    @property
    def io_counts_summary(self):
        return self._summarize_stat(key="io_counts")
    @property
    def stats_summary(self):
        stats = PerformanceStats(label="Performance Summary",
                                 num_tokens=self.total_tokens,
                                 duration=self.total_time,
                                 total_flops=self.total_flops,
                                 total_io=self.total_io,
                                 flops_summary=self.flops_summary,
                                 io_summary=self.io_summary,
                                 flop_counts=self.flop_counts_summary,
                                 io_counts=self.io_counts_summary,
                                 device_bandwidth=self.device_spec.bandwidth if self.device_spec.bandwidth is not None else None,
                                 device_flops_per_s=self.device_spec.flops_per_s if self.device_spec.flops_per_s is not None else None)
    
        return stats
    
    
    def _format_totals(self, precision=2):
        ms = round(self.total_time * 1e3, precision)
        token_throughput = round(self.total_tokens / self.total_time, precision)
        gflops = round(self.total_flops / 1e9, precision)
        gb = round(self.total_io / 1e9, precision)
        flop_throughput = round(gflops / self.total_time, precision)
        io_throughput = round(gb / self.total_time, precision)
        text = textwrap.dedent(f"""\
            FlopCounter Summary:
              Total time = {ms:,} ms
              Tokens:
                Total {self.total_tokens}
                Throughput {token_throughput:,} tokens/s
              IO:
                Total {gb:,} GB
                Throughput {io_throughput:,} GB/s
              FLOPs:
                Total {gflops:,} GFLOPs
                Throughput {flop_throughput:,} GFLOP/s""")
        return text
      
    def print_summary(self, labels: list[str] = None, precision=2, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        _print = partial(print, flush=True, end='\n')
        if labels is None:
            text = self._format_totals(precision=precision)
            _print(text)
        else:
            for label in labels:
                _print(self._count[label])
                # text = self._format_single(label, self._counts[label], precision=precision, verbose=verbose)
                # _print(text)