# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


# can switch to StrEnum (https://docs.python.org/3/library/enum.html#enum.StrEnum)
# after python 3.10 is end of life (https://devguide.python.org/versions/)
class QuantizationStep(str, Enum):
    """
    The step enum for observer-based quantization algorithms like AWQ/SmoothQuant/GPTQ.

    Example:
        # Step 1: Prepare model for calibration
        quantize_(model, SmoothQuantConfig(step="prepare"))

        # Step 2: Run calibration data
        for batch in calibration_data:
            model(batch)

        # Step 3: Convert to quantized model
        quantize_(model, SmoothQuantConfig(step="convert"))

        # Prepare model to load pre-quantized weights
        config = SmoothQuantConfig(step="prepare_for_loading")

        quantize_(model, config)

        # Now the model can load pre-quantized weights
        model.load_state_dict(quantized_state_dict)
    """

    """Insert observers before running calibration flow."""

    PREPARE = "prepare"

    """Convert the observed linear modules to quantized modules using activation scale factors computed during calibration."""

    CONVERT = "convert"

    """Convert the floating point model to a dummy quantized model to load pre-quantized weights at inference stage."""

    PREPARE_FOR_LOADING = "prepare_for_loading"

    """Compute smoothing factor for SmoothQuant two-pass calibration.
    After this step, run calibration data through the model again to collect
    smoothed activation statistics for accurate activation scale computation.
    """

    PREPARE_FOR_SMOOTHQUANT_SMOOTHING_FACTOR = (
        "prepare_for_smoothquant_smoothing_factor"
    )

    """Compute activation scales from smoothed activation statistics collected
    during the second calibration pass (after PREPARE_FOR_SMOOTHQUANT_SMOOTHING_FACTOR).
    """

    PREPARE_FOR_SMOOTHQUANT_ACTIVATION_SCALES = (
        "prepare_for_smoothquant_activation_scales"
    )
