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
    swap_conv2d_1x1_to_linear
    safe_int_mm
    dynamically_quantize_per_tensor
    quantize_activation_per_token_absmax
    dynamically_quantize_per_channel
    dequantize_per_tensor
    dequantize_per_channel
    quant_int8_dynamic_linear
    quant_int8_matmul
    quant_int8_dynamic_per_token_linear
    quant_int8_per_token_matmul
    get_scale
    SmoothFakeDynQuantMixin
    SmoothFakeDynamicallyQuantizedLinear
    swap_linear_with_smooth_fq_linear
    smooth_fq_linear_to_inference
    set_smooth_fq_attribute
    Int8DynamicallyQuantizedLinearWeight
    Int8WeightOnlyQuantizedLinearWeight
    Int4WeightOnlyQuantizedLinearWeight
    compute_error
    get_model_size_in_bytes
    WeightOnlyInt8QuantLinear
    Int4WeightOnlyGPTQQuantizer
    Int4WeightOnlyQuantizer
