# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os

import torch

counter = [0]


def reset_counter():
    """Resets the global logging counter to zero."""
    counter[0] = 0


def get_default_stats(x: torch.Tensor):
    """Computes summary statistics for a tensor.

    Args:
        x: Input tensor to compute statistics for.

    Returns:
        A tuple of (max_abs, avg, std) as Python floats.
    """
    max_abs = torch.max(torch.abs(x)).item()
    avg = torch.mean(x).item()
    std = torch.std(x, correction=0).item()
    return max_abs, avg, std


@torch.library.custom_op("quant_logger::log_tensor", mutates_args=("x",))
def log_tensor(
    x: torch.Tensor, fqn: str, op: str, tag: str, extra: str | None = None
) -> None:
    """Logs summary statistics for a tensor. This is the default implementation
    that prints to stdout; users can redefine this custom op to customize
    logging behavior (e.g. write to file, log custom stats).

    Note: do not use this function directly. Instead, you can override it
    (see code sample below) to modify what gets logged and/or where it gets
    logged to.

    Args:
        x: The tensor to log statistics for.
        fqn: Fully qualified name of the module parameter associated with this tensor.
        op: The operation being logged (e.g. ``"linear"``), or ``""`` for parameters.
        tag: A tag categorizing the log entry (e.g. ``"act"`` for activations,
            ``"param"`` for parameters).
        extra: Optional extra metadata string to include in the log line.

    Example:

    .. literalinclude:: ../../examples/prototype/quant_logger/log_tensor_example.py
       :language: python
    """
    counter_val = counter[0]
    counter[0] += 1
    max_abs, avg, std = get_default_stats(x)
    extra_str = ""
    if extra is not None:
        extra_str = f"{extra=}, "
    print(
        f"t={tag}, c={counter_val}, {fqn=}, {op=}, {extra_str}max={max_abs:.2f}, avg={avg:.2f}, std={std:.2f}"
    )


@log_tensor.register_fake
def _(x: torch.Tensor, fqn: str, op: str, tag: str, extra: str | None = None) -> None:
    pass


# convenience overrides


def enable_log_tensor_save_tensors_to_disk(save_dir):
    """Redefines ``quant_logger::log_tensor`` to save full tensors to disk.

    Each logged tensor is cloned and saved as ``{fqn}_{op}_{tag}.pt``
    under *save_dir*. The directory is created if it does not exist.

    Args:
        save_dir: Path to the directory where tensor files will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)

    @torch.library.custom_op("quant_logger::log_tensor", mutates_args=("x",))
    def log_tensor(
        x: torch.Tensor, fqn: str, op: str, tag: str, extra: str | None = None
    ) -> None:
        filename = f"{fqn}_{op}_{tag}.pt"
        # Replace invalid path characters
        filename = filename.replace("/", "_").replace(":", "_")
        filepath = os.path.join(save_dir, filename)
        torch.save(x.clone(), filepath)


def enable_log_stats_to_file(filename):
    """Redefines ``quant_logger::log_tensor`` to append summary statistics to a CSV file.

    The file is initialized with a header row. Each subsequent call to
    ``log_tensor`` appends a row with columns: tag, counter_val, fqn, op,
    max_abs, avg, std.

    Args:
        filename: Path to the CSV file to write.
    """
    headers = ["tag", "counter_val", "fqn", "op", "max_abs", "avg", "std"]
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    @torch.library.custom_op("quant_logger::log_tensor", mutates_args=("x",))
    def log_tensor(
        x: torch.Tensor, fqn: str, op: str, tag: str, extra: str | None = None
    ) -> None:
        counter_val = counter[0]
        counter[0] += 1
        max_abs, avg, std = get_default_stats(x)
        data = [tag, counter_val, fqn, op, max_abs, avg, std]
        with open(filename, "a") as f:
            writer = csv.writer(f)
            writer.writerow(data)


class ActivationLoggingTensor(torch.Tensor):
    """A wrapper tensor subclass that intercepts ``F.linear`` calls to log
    input activations via the ``quant_logger::log_tensor`` custom op.

    When ``F.linear`` is called with an ``ActivationLoggingTensor`` as the
    weight, the input activation tensor is logged (with GEMM shape metadata)
    before the linear op is executed with the unwrapped weight. All other ops
    fall through to the underlying tensor.

    Args:
        original_weight_tensor: The real weight tensor to wrap.
        fqn: Fully qualified name of the parameter in the model (e.g.
            ``"layers.0.self_attn.q_proj.weight"``).
    """

    @staticmethod
    def __new__(
        cls,
        original_weight_tensor: torch.Tensor,
        fqn: str,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            original_weight_tensor.shape,
            dtype=original_weight_tensor.dtype,
            device=original_weight_tensor.device,
            requires_grad=False,
        )

    def __init__(
        self,
        original_weight_tensor: torch.Tensor,
        fqn: str,
    ):
        self.original_weight_tensor = original_weight_tensor
        self.fqn = fqn

    def __repr__(self):
        return f"ActivationLoggingTensor(fqn={self.fqn}, shape={self.shape})"

    def __tensor_flatten__(self):
        return ["original_weight_tensor"], {"fqn": self.fqn}

    @staticmethod
    def __tensor_unflatten__(tensor_data, metadata, outer_size, outer_stride):
        return ActivationLoggingTensor(
            tensor_data["original_weight_tensor"],
            metadata["fqn"],
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        if func is torch.nn.functional.linear:
            # F.linear signature: linear(input, weight, bias=None)
            input_tensor = args[0]
            weight = args[1]
            bias = args[2] if len(args) > 2 else kwargs.get("bias", None)

            # set the extra argument to the gemm shape (MKN)
            M, K = input_tensor.reshape(-1, input_tensor.shape[-1]).shape
            N, K2 = weight.shape
            assert K == K2
            extra = f"MKN={M}|{K}|{N}"

            # Log the activation
            torch.ops.quant_logger.log_tensor(
                input_tensor, weight.fqn, "linear", "act", extra
            )

            # Call F.linear with the unwrapped weight
            return func(input_tensor, weight.original_weight_tensor, bias)

        # TODO(future PR): add more ops here (grouped_mm, bmm, etc)

        # Fallback: disable torch function and call normally
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # Required by _make_wrapper_subclass, but we handle everything in __torch_function__
        kwargs = kwargs or {}

        # Handle detach specially to preserve the tensor type (needed for nn.Parameter)
        if func is torch.ops.aten.detach.default:
            return ActivationLoggingTensor(
                args[0].original_weight_tensor.detach(),
                args[0].fqn,
            )

        # Fallback: unwrap and call
        def unwrap(t):
            if isinstance(t, ActivationLoggingTensor):
                return t.original_weight_tensor
            return t

        unwrapped_args = torch.utils._pytree.tree_map(unwrap, args)
        unwrapped_kwargs = torch.utils._pytree.tree_map(unwrap, kwargs)
        return func(*unwrapped_args, **unwrapped_kwargs)


def add_activation_loggers(model: torch.nn.Module):
    """Wraps ``nn.Linear`` weights with :class:`ActivationLoggingTensor` so that
    input activations are logged each time ``F.linear`` is called.

    The logging behavior is user-configurable by redefining the
    ``quant_logger::log_tensor`` custom op (see
    :func:`enable_log_tensor_save_tensors_to_disk` and
    :func:`enable_log_stats_to_file` for built-in alternatives).

    Args:
        model: The model whose ``nn.Linear`` weights will be wrapped.

    Example:

    .. literalinclude:: ../../examples/prototype/quant_logger/add_activation_loggers_example.py
       :language: python
    """

    fqn_to_module = dict(model.named_modules())
    for fqn, parameter in model.named_parameters():
        parent_fqn = ".".join(fqn.split(".")[:-1])
        child_fqn = fqn.split(".")[-1]
        parent_module = fqn_to_module[parent_fqn]
        if not isinstance(parent_module, torch.nn.Linear):
            # TODO(future PR): handle wrapping parameter weights for bmm, grouped_mm, etc
            continue
        if child_fqn == "bias":
            # don't need to wrap bias, as weight is already wrapped
            continue
        new_parameter = torch.nn.Parameter(ActivationLoggingTensor(parameter, fqn))
        setattr(parent_module, child_fqn, new_parameter)


def log_parameter_info(model: torch.nn.Module):
    """Logs summary statistics for every parameter in the model.

    Each parameter is passed to ``quant_logger::log_tensor`` with
    ``tag="param"``, so the output format depends on the current
    ``log_tensor`` implementation.

    Args:
        model: The model whose parameters will be logged.

    Example:

    .. literalinclude:: ../../examples/prototype/quant_logger/log_parameter_info_example.py
       :language: python
    """
    for fqn, parameter in model.named_parameters():
        torch.ops.quant_logger.log_tensor(parameter, fqn, "", "param", None)
