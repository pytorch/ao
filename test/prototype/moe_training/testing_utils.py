from torch import nn

from torchao.prototype.moe_training.tensor import ScaledGroupedMMTensor


def _validate_model_conversion(
    root_module: nn.Module,
    target_fqns: list[str],
):
    def _recursive_validate(
        module: nn.Module,
        cur_fqn: str,
    ):
        is_allowed_module = any([target_fqn in cur_fqn for target_fqn in target_fqns])

        # check current module params
        for param_name, param in module.named_parameters(recurse=False):
            is_converted_type = isinstance(param, ScaledGroupedMMTensor)
            if is_converted_type:
                assert is_allowed_module, (
                    f"Module {cur_fqn} is not in target_fqns, but has converted param {param_name}."
                )
            if not is_allowed_module:
                assert not is_converted_type, (
                    f"Module {cur_fqn} is not in target_fqns, but has converted param {param_name}."
                )

        # recursively check child modules
        for child_name, child_module in module.named_children():
            child_fqn = f"{cur_fqn}.{child_name}" if cur_fqn else child_name
            _recursive_validate(child_module, child_fqn)

    _recursive_validate(root_module, "")
