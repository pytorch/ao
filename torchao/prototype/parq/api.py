from dataclasses import dataclass
from typing import Optional

from torchao.prototype.parq.optim import QuantOptimizer
from torchao.prototype.parq.quant import (
    Quantizer,
    StretchedUnifTorchaoQuantizer,
    UnifTorchaoQuantizer,
)


@dataclass(frozen=True, slots=True)
class QuantConfig:
    bitwidth: int
    group_size: Optional[int] = None
    quantizer: Optional[Quantizer] = None

    def __post_init__(self):
        if self.bitwidth < 2:
            raise ValueError("bitwidth must be >= 2")
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("group_size must be positive")

        if self.quantizer is None:
            if self.bitwidth in [2, 3]:
                q = StretchedUnifTorchaoQuantizer(b=self.bitwidth)
            else:
                q = UnifTorchaoQuantizer()
            object.__setattr__(self, "quantizer", q)


def create_param_groups_and_group_quantizer_map(model, quant_configs_and_filter_fns):
    param_groups = []
    group_quantizer_map = {}
    for idx, (config, _) in enumerate(quant_configs_and_filter_fns):
        params_quant = []
        param_group = {
            "params": params_quant,
            "quant_bits": config.bitwidth,
        }
        if config.group_size is not None:
            param_group["quant_block_size"] = config.group_size
        param_group["_quantizer"] = config.quantizer
        param_groups.append(param_group)

    # Non-quantized group at end so that index in param_groups
    # is the index in the subset of quantized param groups, which is
    # used in defining group_quantizer_map
    params_no_quant = []
    param_groups.append({"params": params_no_quant, "weight_decay": 0.0})

    seen_data_ptrs = {}
    for param_name, param in model.named_parameters():
        module_name, _, param_basename = param_name.rpartition(".")
        owning_module = model.get_submodule(module_name) if module_name else model

        data_ptr = param.data_ptr()
        if data_ptr in seen_data_ptrs:
            print(
                f"Not considering {param} because it shares a data_ptr with {seen_data_ptrs[data_ptr]}, which was previously considered"
            )
            continue
        seen_data_ptrs[data_ptr] = param_name

        print(
            "param_name",
            param_name,
            "module_type",
            type(owning_module),
            "matching_config:",
            end="",
        )
        matching_config = None
        for idx, (config, filter_fn) in enumerate(quant_configs_and_filter_fns):
            if filter_fn(owning_module, param_name):
                param_groups[idx]["params"].append(param)
                if matching_config is None:
                    matching_config = config
                    print(f"{config.bitwidth},{config.group_size}")
                else:
                    raise ValueError(
                        f"Found multiple matching configs for {param_name}. Previous match={matching_config}, new match={config}."
                    )

        # If no match, add to no-quant group at last idx
        if matching_config is None:
            print("NONE")
            param_groups[-1]["params"].append(param)

    # Filter out empty param groups
    param_groups = [pg for pg in param_groups if len(pg["params"]) > 0]

    # After filter define group_quantizer_map
    # The index in group_quantizer_map must correspond to index in
    # quantized params
    group_quantizer_map = {}
    for idx, param_group in enumerate(param_groups):
        if "_quantizer" in param_group:
            group_quantizer_map[idx] = param_group.pop("_quantizer")

    expected_n_params = sum(1 for p in model.parameters())
    n_found_params = sum(len(pg["params"]) for pg in param_groups)
    assert n_found_params == expected_n_params, (
        f"{n_found_params} != {expected_n_params=}"
    )

    return param_groups, group_quantizer_map


from torchao.prototype.parq import ProxHardQuant


def create_optimizer(
    model,
    quant_configs_and_filter_fns,
    base_optimizer_cls,
    base_optimizer_kwargs,
    *,
    warmup_steps=0,
    quant_period=1,
    quant_per_channel=True,
):
    param_groups, group_quantizer_map = create_param_groups_and_group_quantizer_map(
        model, quant_configs_and_filter_fns
    )
    base_optimizer = base_optimizer_cls(param_groups, **base_optimizer_kwargs)
    optimizer = QuantOptimizer(
        base_optimizer,
        quantizer=UnifTorchaoQuantizer(),
        prox_map=ProxHardQuant(),
        warmup_steps=warmup_steps,
        quant_period=quant_period,
        quant_per_channel=quant_per_channel,
        group_quantizer_map=group_quantizer_map,
    )
    return optimizer
