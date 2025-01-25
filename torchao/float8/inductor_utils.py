import functools
import inspect
import traceback
from collections import deque

import torch


def amax_with_scaling_pattern(tensor_x_inp, scale_x, fp8_dtype, fp8_max):
    tensor_x = tensor_x_inp.to(torch.float32) * scale_x
    tensor_x = tensor_x.clamp(min=-1 * fp8_max, max=fp8_max)
    tensor_x = tensor_x.to(fp8_dtype)
    amax = torch.max(torch.abs(tensor_x_inp))
    return (tensor_x, amax)


def amax_with_scaling_tiled_replacement(tensor_x_inp, scale_x, fp8_dtype, fp8_max):
    tensor_x = tensor_x_inp.to(torch.float32) * scale_x
    tensor_x = tensor_x.clamp(min=-1 * fp8_max, max=fp8_max)
    tensor_x = tensor_x.to(fp8_dtype)
    amax_1 = torch.max(torch.abs(tensor_x_inp), dim=-1).values
    amax = torch.max(amax_1)
    return (tensor_x, amax)


# The amax_with_scaling_pattern will also match dynamic scaling cases, we want to avoid that.
# `scale_x` of delayed scaling comes from the previous iteration, instead of from `tensor_x_inp`.
# We check that `scale_x` is not a dependency of `tensor_x_inp`
def fp8_delayed_scaling_extra_check(match):
    scale_x_inputs = deque([match.kwargs["scale_x"]])
    max_num_node_to_check = 20  # Don't traverse too many nodes
    current_num_node = 0
    while len(scale_x_inputs) > 0 and current_num_node < max_num_node_to_check:
        current_node = scale_x_inputs.popleft()
        for n in current_node.all_input_nodes:
            if n == match.kwargs["tensor_x_inp"]:
                return False
            scale_x_inputs.append(n)
            current_num_node += 1
    return True


def partialize_and_update_signature(func, **kwargs):
    """
    Equivalent to functools.partial but also updates the signature on returned function
    """
    original_sig = inspect.signature(func)
    parameters = original_sig.parameters

    new_parameters = {
        key: value for key, value in parameters.items() if key not in kwargs
    }
    new_sig = inspect.Signature(parameters=list(new_parameters.values()))

    partial_func = functools.partial(func, **kwargs)

    def wrapper(*args, **kwargs):
        return partial_func(*args, **kwargs)

    wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
    wrapper.__name__ = func.__name__

    return wrapper


def register_fp8_delayed_scaling_patterns_inner():
    from torch._inductor.fx_passes.post_grad import (
        pass_patterns as post_grad_patterns_all,
    )
    from torch._inductor.pattern_matcher import fwd_only, register_replacement

    post_grad_patterns = post_grad_patterns_all[1]  # medium priority

    if torch.cuda.is_available():
        for fp8_dtype in [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        ]:
            # torch.float16 has the same pattern as torch.bfloat16, because they both needs `tensor_x_inp.to(torch.float32)`
            for dtype in [torch.float32, torch.bfloat16]:
                device = "cuda"
                register_replacement(
                    partialize_and_update_signature(
                        amax_with_scaling_pattern,
                        fp8_dtype=fp8_dtype,
                        fp8_max=torch.finfo(fp8_dtype).max,
                    ),
                    partialize_and_update_signature(
                        amax_with_scaling_tiled_replacement,
                        fp8_dtype=fp8_dtype,
                        fp8_max=torch.finfo(fp8_dtype).max,
                    ),
                    [
                        torch.tensor((16, 16), device=device, dtype=dtype),
                        torch.tensor(2.0, device=device, dtype=torch.float32),
                    ],
                    fwd_only,
                    post_grad_patterns,
                    extra_check=fp8_delayed_scaling_extra_check,
                )


"""
This a short-term workaround of the delayed scaling performance issue.
It explicitly replaces `max(x)` with `max(max(x, dim=-1))`, enabling the fusion of amax scaling factor calculation and fp8 casting.

Usage:
    To use this solution, add the following line at the beginning of your user code:
    torchao.float8._prototype_register_float8_delayed_scaling_inductor_passes()
"""


def _prototype_register_float8_delayed_scaling_inductor_passes() -> None:
    # To make the fp8 delayed scaling pattern work, we need a fix pr from inductor, https://github.com/pytorch/pytorch/pull/139321
    # Will throw the error if the pattern registration did not work, up to user to decide what to do with it
    try:
        register_fp8_delayed_scaling_patterns_inner()
    except AssertionError as e:
        if "assert pattern_repr not in _seen_patterns" in traceback.format_exc():
            print(
                f"Caught duplicated patterns in register_fp8_delayed_scaling_patterns: {traceback.format_exc()}",
                "\nPlease update your pytorch dependency to the latest main branch to fix it.\n",
            )
        raise e
