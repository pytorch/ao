from typing import Callable, Optional

from torch import nn

from torchao.core.config import AOBaseConfig
from torchao.prototype.moe_training.tensor import ScaledGroupedMMTensor
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)


class MoETrainingConfig(AOBaseConfig):
    """
    The MoETrainingConfig is specifically designed to be used on MoE models using
    `torch._grouped_mm` to implement expert computation in token-choice routing,
    where expert weights are implemented as 3D nn.Parameters wit `num_experts` as
    the leading dim.

    MoETrainingConfig has a module handler registered to it which will
    find all nn.Parameters whose parent module matches the module filter function,
    and swap their data tensor with a ScaledGroupedMMTensor.

    The ScaledGroupedMMTensor is a tensor subclass which overrides the
    `torch._grouped_mm` op by dispatching to a differentiable scaled grouped mm,
    which performs dynamic float8 rowwise quantization on scaled grouped GEMM
    operands in both the forward and backward pass.

    For all other ops, ScaledGroupedMMTensor behaves like a regular torch.Tensor.
    """

    use_triton_for_per_group_scales: bool = True


@register_quantize_module_handler(MoETrainingConfig)
def _moe_training_transform(
    module: nn.Module,
    config: MoETrainingConfig,
) -> nn.Module:
    """
    Swaps `torch.nn.Parameter` data tensor with a ScaledGroupedMMTensor.

    Args:
        module: Module to modify.
        config: MoETrainingConfig which defines how to perform the MoE training transform.

    Returns:
     nn.Module: The modified module with swapped parameters.
    """
    out = _swap_params(module, config=config)
    return out


def _swap_params(
    module: nn.Module,
    *,
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
    config: Optional[MoETrainingConfig] = None,
) -> nn.Module:
    """
    Recurses through the nn.Module, recursively swapping the data tensor of
    each nn.Parameter with a ScaledGroupedMMTensor. Only applies if the module
    passed the module_filter_fn, if specified.

    Args:
        module: Module to modify.
        module_filter_fn: If specified, only the `torch.nn.Parameter` subclasses that
            that pass the filter function will be swapped. The inputs to the
            filter function are the module instance, and the FQN.

    Returns:
     nn.Module: The modified module with swapped linear layers.
    """
    use_triton = config.use_triton_for_per_group_scales if config is not None else False
    if isinstance(module, nn.Parameter) and (
        module_filter_fn is None or module_filter_fn(module, "")
    ):
        if len(list(module.children())) > 0:
            raise AssertionError(
                f"Does not support a root nn.Parameter with children: {module}"
            )
        if not isinstance(module.data, ScaledGroupedMMTensor):
            new_data = ScaledGroupedMMTensor(
                module.data, use_triton_for_per_group_scales=use_triton
            )
            return nn.Parameter(new_data, requires_grad=module.requires_grad)
        return module

    root_module = module

    def post_order_traversal(
        module: nn.Module,
        cur_fqn: Optional[str] = None,
        parent_module: Optional[nn.Module] = None,
    ):
        if cur_fqn is None:
            cur_fqn = ""

        for child_module_name, child_module in module.named_children():
            if cur_fqn == "":
                new_fqn = child_module_name
            else:
                new_fqn = f"{cur_fqn}.{child_module_name}"

            post_order_traversal(child_module, new_fqn, module)

        if module_filter_fn is None or module_filter_fn(module, cur_fqn):
            for param_name, param in module.named_parameters(recurse=False):
                if not isinstance(param.data, ScaledGroupedMMTensor):
                    new_param = nn.Parameter(
                        ScaledGroupedMMTensor(param), requires_grad=param.requires_grad
                    )
                    setattr(module, param_name, new_param)
                    print(f"Swapped {cur_fqn}.{param_name} to ScaledGroupedMMTensor")

    post_order_traversal(root_module)
    return root_module
