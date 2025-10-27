import torch
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


def generate_split_sizes(K: int, N: int, device: str = "cpu") -> torch.Tensor:
    """
    Generates a tensor of K random non-negative integers that sum to N.
    Used for testing mxfp8_all_to_all_v implementation.
    """
    if K <= 0:
        raise ValueError("K must be a positive integer.")
    if N < 0:
        raise ValueError("N must be a non-negative integer.")

    if K == 1:
        return torch.tensor([N], dtype=torch.long, device=device)

    # Generate K-1 random "dividers" in the range [0, N].
    dividers = torch.randint(0, N + 1, (K - 1,), device=device)

    # Add 0 and N to the set of dividers to form the boundaries.
    boundaries = torch.cat(
        [torch.tensor([0], device=device), dividers, torch.tensor([N], device=device)]
    )

    # Sort the boundaries to ensure they are in order
    sorted_boundaries = torch.sort(boundaries).values

    # The K integers are the differences between consecutive boundaries (will sum to N)
    result = sorted_boundaries[1:] - sorted_boundaries[:-1]

    return result.to(dtype=torch.int64)
