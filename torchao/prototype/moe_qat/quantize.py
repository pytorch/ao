from typing import Any, Callable, Tuple

from torch import nn

from torchao.prototype.moe_qat.tensor import FakeQuantizedWeightWrapperBaseTensor


def _is_parameter(param: nn.Parameter, fqn: str) -> bool:
    """
    The default filter for the parameter-level recursion in _replace_params_with_custom_fn_if_matches_filter,
    returning True for all nn.Parameter not wrapped by FakeQuantizedWeightWrapperBaseTensor
    """
    return isinstance(param, nn.Parameter) and not isinstance(
        param.data, FakeQuantizedWeightWrapperBaseTensor
    )


def _is_parameter_with_wrapped_data(param: nn.Parameter, fqn: str) -> bool:
    """
    The filter for the convert step of MoEQATConfig, identifying nn.Parameters with wrapped data.
    """
    return isinstance(param, nn.Parameter) and isinstance(
        param.data, FakeQuantizedWeightWrapperBaseTensor
    )


def unwrap_param(
    module: nn.Module,
    param_fqn: str,
    param: nn.Parameter,
    extra_args: Tuple[Any, ...] = (),
):
    return nn.Parameter(
        param.data.to_tensor(),
        requires_grad=param.requires_grad,
    )


def _replace_params_with_custom_fn_if_matches_filter(
    module: nn.Module,
    params_replacement_fn,
    params_filter_fn: Callable[[nn.Parameter, str], bool],
    cur_fqn: str = "",
    extra_args: Tuple[Any, ...] = (),
):
    """Recursively replace matching parameters in a module and its submodules with custom replacements"""

    params_filter_fn = _is_parameter if params_filter_fn is None else params_filter_fn

    for child_name, child in module.named_children():
        child_fqn = f"{cur_fqn}.{child_name}" if cur_fqn else child_name
        new_child = _replace_params_with_custom_fn_if_matches_filter(
            child,
            params_replacement_fn,
            params_filter_fn,
            cur_fqn=child_fqn,
            extra_args=extra_args,
        )
        if new_child is not child and new_child is not None:
            # _replace_params_with_custom_fn_if_matches_filter modify child in-place,
            # so this branch normally is never executed.
            setattr(module, child_name, new_child)

    for param_name, param in module.named_parameters(recurse=False):
        param_fqn = f"{cur_fqn}.{param_name}" if cur_fqn else param_name
        if not params_filter_fn(param, param_fqn):
            continue
        new_param = params_replacement_fn(module, param_fqn, param, extra_args)
        if new_param is not param and new_param is not None:
            setattr(module, param_name, new_param)

    return module
