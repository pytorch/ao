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
    The step enum backend for observer based algorithms process like AWQ/SmoothQuant/GPTQ
        PREPARE: insert observers to linear layers
        CONVERT: convert the observed linear modules to quantized modules
        PREPARE_FOR_LOADING: convert the floating point model to a dummy quantized model,
        so we can load the quantized weights through copy_ later

    Example:
        # Stage 1: PREPARE - insert observers
        quantize_(model, Float8StaticActivationFloat8WeightConfig(step="prepare"))

        # Stage 2: CALIBRATE - collect statistics
        for batch in calibration_data:
            model(batch)

        # Stage 3: CONVERT - apply quantization and remove observers
        quantize_(model, Float8StaticActivationFloat8WeightConfig(step="convert"))

        # Stage 4 (optional): PREPARE_FOR_LOADING - for saving/loading quantized models
        # This creates a dummy quantized model structure so weights can be loaded via copy_
        from torchao.prototype.smoothquant import SmoothQuantConfig

        config = SmoothQuantConfig(base_config, step="prepare_for_loading")
        quantize_(model, config)

        # Now the model can load pre-quantized weights
        model.load_state_dict(quantized_state_dict)

    """

    PREPARE = "prepare"
    CONVERT = "convert"
    PREPARE_FOR_LOADING = "prepare_for_loading"
