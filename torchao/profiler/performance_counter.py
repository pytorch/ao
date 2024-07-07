# mypy: allow-untyped-defs
import math
from collections import defaultdict

import torch
from torch.utils._pytree import tree_map
from torch.utils.flop_counter import FlopCounterMode

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

if __name__ == "__main__":
    import torch
    from transformers import AutoModelForCausalLM, LlamaForCausalLM
    
    model_id = "/home/ubuntu/gpt-fast-dev/checkpoints/7B"
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 16), dtype=torch.int64, device="cuda")

    with PerformanceCounterMode(display=False, depth=10) as perf_counter:
        _ = model(input_ids)
        
    print(perf_counter.pretty_summary_counts(type="flops", depth=3))
    print(perf_counter.pretty_summary_counts(type="io", depth=3))