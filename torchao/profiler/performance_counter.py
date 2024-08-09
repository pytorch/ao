import inspect
import json
import math
import textwrap
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.utils._pytree import tree_map
from torch.utils.flop_counter import FlopCounterMode

from .device_spec import DeviceSpec

aten = torch.ops.aten


class DeviceInfoMissing(UserWarning):
    pass


# Prevent excessive output
warnings.simplefilter("once", DeviceInfoMissing)


class PerformanceCounterMode(FlopCounterMode):
    """
    ``PerformanceCounterMode`` extends FlopCounterMode to track IO in addition to flops.

    It does this using a ``TorchDispatchMode`` per `FlopCounterMode` and tracks the
    inputs and outputs of each operator, organized by module.

    In addition to the methods exposed by FlopCounterMode, the following methods are
    available:
    - ``get_io_counts``: returns a dictionary of module names and their associated IO counts by aten operator
    - ``get_total_io``: returns the total number of IO operations across all modules
    - ``get_summary_io_counts``: returns a summary of the IO counts for each module (totals by operator)
    - ``get_summary_flop_counts``: returns a summary of the flop counts for each module (totals by operator)
    """

    def __init__(self, display=False, depth=10, debug=False):
        self.debug = debug
        self.io_counts = defaultdict(lambda: defaultdict(int))
        super().__init__(display=display, depth=depth)

    def get_io_counts(self):
        return {k: dict(v) for k, v in self.io_counts.items()}

    def get_total_io(self):
        return sum(self.io_counts["Global"].values())

    def _get_io_sizes(self, args):
        sizes = tree_map(
            lambda x: x.numel() * x.element_size()
            if isinstance(x, torch.Tensor)
            else 0,
            args,
        )
        if not hasattr(sizes, "__len__"):
            sizes = [sizes]
        return sizes

    def get_summary_flop_counts(self):
        flop_counts = self.get_flop_counts()
        return {k: sum(v.values()) for k, v in flop_counts.items()}

    def get_summary_io_counts(self):
        io_counts = self.get_io_counts()
        return {k: sum(v.values()) for k, v in io_counts.items()}

    def _nearest_power_of_10(self, x):
        if x == 0:
            return x, 0

        power = int(math.floor(math.log10(abs(x)) / 3))
        scaled_value = x / (10 ** (3 * power))

        return scaled_value, power

    def pretty_summary_counts(self, type="flops", precision=2, depth=None):
        assert type in ["flops", "io"]
        metric_units = {
            0: "",
            1: "k",
            2: "M",
            3: "G",
            4: "T",
            5: "P",
            6: "E",
            7: "Z",
            8: "Y",
        }

        if depth is None:
            depth = self.depth
        summary_counts = (
            self.get_summary_flop_counts()
            if type == "flops"
            else self.get_summary_io_counts()
        )
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
        arg_size, kwargs_size, out_size = (
            sum(arg_sizes),
            sum(kwargs_sizes),
            sum(out_sizes),
        )
        return arg_size, kwargs_size, out_size

    def _count_flops(self, func_packet, out, args, kwargs):
        if func_packet in self.flop_registry:
            flop_count_func = self.flop_registry[func_packet]
            flop_count = flop_count_func(*args, **kwargs, out_val=out)  # type: ignore[operator]
            arg_size, kwarg_size, out_size = self._count_io(
                func_packet, out, args, kwargs
            )
            total_size = arg_size + kwarg_size + out_size

            for par in set(self.mod_tracker.parents):
                if self.debug:
                    print(f"Counting flops for {par}, {func_packet}: {flop_count}")
                    print(
                        f"Counting io for {par}, {func_packet}: {sum([arg_size, kwarg_size, out_size])} = {arg_size} + {kwarg_size} + {out_size}"
                    )
                self.flop_counts[par][func_packet] += flop_count
                self.io_counts[par][func_packet] += total_size

        return out


class PerformanceTimer:
    """
    Context manager that records the latency, io, and flops of a torch operator / module.

    Timing is done using `time.perf_counter` and can be overridden to use a different
    timer (see `CUDAPerformanceTimer`).

    IO and FLOPs are recorded using `PerformanceCounterMode`.

    Available attributes:
        name: str
        precision: int
        display: bool
        depth (int): passed to `PerformanceCounterMode` if displaying and determines depth of module tree to display.
    **Note**: these attributes are primarily used for debugging when using the `PerformanceTimer` standalone.
    The TransformerPerformanceCounter class is a higher-level API that should be used instead.

    """

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
        ms = round(self.latency * 1e3, self.precision)
        if self.display:
            print(f"{self.name.upper()}:  latency = {ms} ms, FLOPS = {gflops} GFLOPs")

    def __exit__(self, type, value, traceback):
        self.end = time.perf_counter()
        # Convert to ms
        self.latency = self.end - self.start
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
        return self.perf_counter.pretty_summary_counts(
            depth=depth if depth is not None else self.depth
        )


class CUDAPerformanceTimer(PerformanceTimer):
    """
    `PerformanceTimer` that uses `cudaEvents` to record latency.
    """

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        self.perf_counter = PerformanceCounterMode(
            display=self.display, depth=self.depth
        )
        self.perf_counter.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        # Convert from ms to s
        self.latency = self.start.elapsed_time(self.end) * 1e-3
        self.perf_counter.__exit__(type, value, traceback)

        if self.display:
            self._print_exit_msg()


def to_nearest_power_of_10(x, precision=2):
    # Dictionary mapping powers of 10 to their metric abbreviations
    metric_units = {0: "", -6: "µ", -3: "m", 6: "M", 9: "G", 12: "T"}

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


class DictMixin:
    """
    Enables dict-like interface to dataclasses.
    """

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def __iter__(self):
        for key in self.__dict__:
            yield key


def _get_property_methods(cls):
    return [
        name for name, _ in inspect.getmembers(cls, lambda m: isinstance(m, property))
    ]


@dataclass
class PerformanceStats(DictMixin):
    """
    Data struct that stores performance statistics.

    Attrs:
        num_tokens (int): number of tokens processed
        latency (float): latency in seconds
        total_flops (int): total FLOPs
        total_io (int): total data movement in bytes
        flops_summary (Dict[str, int]): summary of FLOPs by module
        io_summary (Dict[str, int]): summary of data movement in bytes by module
        flop_counts (Dict[str, Dict[Any, int]]): FLOP counts by module and operation
        io_counts (Dict[str, Dict[Any, int]]): data movement by module and operation
        device_bandwidth (Optional[float]): device bandwidth in bytes per second
        device_flops_per_s (Optional[float]): device FLOPs per second

    Additionally, the following derived properties are available:
        token_throughput (float): number of tokens processed per second
        achieved_flops_per_s (float): achieved FLOPs per second
        achieved_bandwidth (float): achieved data movement in bytes per second
        theoretical_io_latency (Optional[float]): theoretical I/O latency in seconds, set to None if
        no device bandwidth is available.
        theoretical_compute_latency (Optional[float]): theoretical compute latency in seconds, set to None if
        no device FLOPs are available.
    """

    label: str
    num_tokens: int
    latency: float
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
        return self.num_tokens / self.latency

    @property
    def achieved_flops_per_s(self):
        return self.total_flops / self.latency

    @property
    def achieved_bandwidth(self):
        return self.total_io / self.latency

    @property
    def theoretical_io_latency(self):
        if self.device_bandwidth is not None:
            return self.total_io / self.device_bandwidth
        else:
            warnings.warn(
                "Device bandwidth is not specified. Please specify the device bandwidth to enable io latency calculation"
            )
            return None

    @property
    def theoretical_compute_latency(self):
        if self.device_flops_per_s is not None:
            return self.total_flops / self.device_flops_per_s
        else:
            warnings.warn(
                "Device flops_per_s is not specified. Please specify the device throughput to enable compute latency calculation"
            )
            return None

    @property
    def bandwidth_utilization(self):
        if self.device_bandwidth is not None:
            return self.achieved_bandwidth / self.device_bandwidth
        else:
            warnings.warn(
                "Device bandwidth is not specified. Please specify the device bandwidth to enable bandwidth utilization calculation"
            )
            return None

    @property
    def flops_utilization(self):
        if self.device_flops_per_s is not None:
            return self.achieved_flops_per_s / self.device_flops_per_s
        else:
            warnings.warn(
                "Device flops_per_s is not specified. Please specify the device throughput to enable flops utilization calculation"
            )
            return None

    def _format(self, value, suffix, precision=2, round=True):
        if round:
            return to_nearest_power_of_10(value, precision=precision) + suffix
        return f"{value:.{precision}f} " + suffix

    def __str__(self):
        txt = textwrap.dedent(f"""\
            {self.label}:
              Latency = {self._format(self.latency, "s")}
              Tokens
                Total: {self.num_tokens} tokens
                Throughput: {self.token_throughput:,.0f} tokens/s
              IO
                Total: {self._format(self.total_io, "B")}
                Throughput: {self._format(self.achieved_bandwidth, "B/s")}
                Theoretical Latency: {self._format(self.theoretical_io_latency, "s") if self.theoretical_io_latency is not None else "N/A"}
              FLOPs 
                Total: {self._format(self.total_flops, "FLOPs")}
                Throughput: {self._format(self.achieved_flops_per_s, "FLOPs/s")}
                Theoretical Latency: {self._format(self.theoretical_compute_latency, "s") if self.theoretical_compute_latency is not None else "N/A"}
              Utilization
                Bandwidth: {self._format(self.bandwidth_utilization, round=False, precision=4, suffix="%") if self.bandwidth_utilization is not None else "N/A"}
                FLOPs: {self._format(self.flops_utilization, round=False, precision=4, suffix="%") if self.flops_utilization is not None else "N/A"}""")

        return txt

    def to_dict(self):
        d = asdict(self)
        # Update dict with properties
        props = _get_property_methods(self.__class__)
        d.update({prop: getattr(self, prop) for prop in props})

        return d


class TransformerPerformanceCounter:
    """
    Context manager-like class for tracking performance across multiple calls
    to a Transformer model.

    Provides properties for accessing performance stats for data movement and FLOPs for each context as well as
    summary stats across all contexts.
    Additionally, if a device_spec is provided, theoretical peak bandwidth / FLOPs stats will be available.

    See `PerformanceStats` struct for description of tracked metrics.

    Example:
        >>> manager = TransformerPerformanceCounter(device_spec=device_spec)
        >>> with manager.count(label="prefill", num_tokens=x.numel()):
        >>>     out = model(encoded_prompt)
        >>> manager.print_summary(labels=["prefill"]) # prints recorded stats for "prefill" context
        >>> with manager.count(label="decode", num_tokens=1):
        >>>     out = model(out[-1])
        >>> manager.print_summary(labels=["decode"]) # prints recorded stats for "decode" context
        >>> print(manager.print_summary) # prints accumulated stats across all contexts
    """

    def __init__(
        self,
        depth=10,
        timer_cls: PerformanceTimer = PerformanceTimer,
        device_spec: DeviceSpec = None,
    ):
        super().__init__()
        self._counts: Dict[str, PerformanceStats] = {}
        self._depth = depth
        self.timer_cls = timer_cls
        self.device_spec = device_spec

    @contextmanager
    def count(self, label: str, num_tokens: int):
        perf_timer = self.timer_cls(name=label, depth=self._depth)
        perf_timer.__enter__()
        try:
            yield self
        finally:
            perf_timer.__exit__(None, None, None)
            stats = PerformanceStats(
                label=label,
                num_tokens=num_tokens,
                latency=perf_timer.latency,
                total_flops=perf_timer.total_flops,
                total_io=perf_timer.total_io,
                flops_summary=perf_timer.get_summary_flop_counts(),
                io_summary=perf_timer.get_summary_io_counts(),
                flop_counts=perf_timer.flop_counts,
                io_counts=perf_timer.io_counts,
                device_bandwidth=self.device_spec.bandwidth
                if self.device_spec is not None
                else None,
                device_flops_per_s=self.device_spec.flops_per_s
                if self.device_spec is not None
                else None,
            )
            self._counts[label] = stats

    @property
    def counts(self):
        return self._counts

    def get_counts(self):
        return self._counts

    @property
    def total_flops(self):
        return sum(count.total_flops for count in self._counts.values())

    @property
    def total_io(self):
        return sum(count.total_io for count in self._counts.values())

    @property
    def total_tokens(self):
        return sum(count.num_tokens for count in self._counts.values())

    @property
    def total_time(self):
        return sum(count.latency for count in self._counts.values())

    def _summarize_stat(self, key):
        return {
            label: getattr(self._counts[label], key) for label in self._counts.keys()
        }

    @property
    def flops_summary(self):
        return self._summarize_stat(key="flops_summary")

    @property
    def io_summary(self):
        return self._summarize_stat(key="io_summary")

    @property
    def flop_counts_summary(self):
        return self._summarize_stat(key="flop_counts")

    @property
    def io_counts_summary(self):
        return self._summarize_stat(key="io_counts")

    @property
    def stats_summary(self):
        stats = PerformanceStats(
            label="Performance Summary",
            num_tokens=self.total_tokens,
            latency=self.total_time,
            total_flops=self.total_flops,
            total_io=self.total_io,
            flops_summary=self.flops_summary,
            io_summary=self.io_summary,
            flop_counts=self.flop_counts_summary,
            io_counts=self.io_counts_summary,
            device_bandwidth=self.device_spec.bandwidth
            if self.device_spec is not None
            else None,
            device_flops_per_s=self.device_spec.flops_per_s
            if self.device_spec is not None
            else None,
        )

        return stats

    def print_summary(self, labels: list[str] = None, show: bool = False):
        _print = partial(print, flush=True, end="\n")
        # Delegate to __str__ of PerformanceStats for pretty printing
        if labels is None:
            text = str(self.stats_summary)
            if show:
                _print(text)
            return text
        else:
            txts = []
            for label in labels:
                text = str(self._counts[label])
                if show:
                    _print(text)
                txts.append(text)
            return "\n".join(txts)

    def to_dict(self):
        # Convert flop_counts from OpOverloadPackets to str
        # Then delegate to PerformanceStats `to_dict`, which updates with derived metrics (property methods)
        counts = deepcopy(self._counts)
        for label, label_counts in counts.items():
            counts[label]["flop_counts"] = {
                mod: {str(op): count for op, count in op_count.items()}
                for mod, op_count in label_counts["flop_counts"].items()
            }
            counts[label]["io_counts"] = {
                mod: {str(op): count for op, count in op_count.items()}
                for mod, op_count in label_counts["io_counts"].items()
            }
            counts[label] = counts[label].to_dict()

        return counts

    def to_json(self, path: Union[str, Path] = None):
        d = self.to_dict()
        if path:
            with open(path, "w") as f:
                f.write(json.dumps(d, indent=2))
        return d
