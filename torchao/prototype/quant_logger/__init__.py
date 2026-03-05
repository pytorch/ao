# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from torchao.prototype.quant_logger.quant_logger import (
    ActivationLoggingTensor,
    add_activation_loggers,
    enable_log_stats_to_file,
    enable_log_tensor_save_tensors_to_disk,
    log_parameter_info,
    log_tensor,
    reset_counter,
)

# CustomOpDef.__doc__ returns the class docstring, not our function's
# docstring. Copy it so Sphinx autodoc picks up the right one.
log_tensor.__doc__ = log_tensor._init_fn.__doc__

__all__ = [
    "ActivationLoggingTensor",
    "add_activation_loggers",
    "enable_log_stats_to_file",
    "enable_log_tensor_save_tensors_to_disk",
    "log_parameter_info",
    "log_tensor",
    "reset_counter",
]
