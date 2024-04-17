.. _api_quantization:

====================
torchao.quantization
====================

.. currentmodule:: torchao.quantization

.. autosummary::
    :toctree: generated/
    :nosignatures:

    apply_weight_only_int8_quant
    apply_dynamic_quant
    change_linear_weights_to_int8_dqtensors
    change_linear_weights_to_int8_woqtensors
    change_linear_weights_to_int4_woqtensors
    SmoothFakeDynQuantMixin
    SmoothFakeDynamicallyQuantizedLinear
    swap_linear_with_smooth_fq_linear
    smooth_fq_linear_to_inference
    Int4WeightOnlyGPTQQuantizer
    Int4WeightOnlyQuantizer
