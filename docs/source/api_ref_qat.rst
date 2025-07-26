.. _api_qat:

========================
torchao.quantization.qat
========================

.. currentmodule:: torchao.quantization.qat

QAT Configs for quantize_
---------------------------------------
For a full example of how to use QAT with our main `quantize_` API,
please refer to the `QAT README <https://github.com/pytorch/ao/blob/main/torchao/quantization/qat/README.md#quantize_-api-recommended>`__.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    QuantizationAwareTrainingConfig
    FromQuantizationAwareTrainingConfig

Custom QAT APIs
---------------
.. autosummary::
    :toctree: generated/
    :nosignatures:

    IntxFakeQuantizeConfig
    FakeQuantizedLinear
    FakeQuantizedEmbedding
    FakeQuantizer
    linear.enable_linear_fake_quant
    linear.disable_linear_fake_quant

Legacy QAT Quantizers
---------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Int4WeightOnlyQATQuantizer
    linear.Int4WeightOnlyQATLinear
    Int8DynActInt4WeightQATQuantizer
    linear.Int8DynActInt4WeightQATLinear
    Int4WeightOnlyEmbeddingQATQuantizer
    embedding.Int4WeightOnlyQATEmbedding
    embedding.Int4WeightOnlyEmbedding
    Float8ActInt4WeightQATQuantizer
    ComposableQATQuantizer

Prototype
---------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    initialize_fake_quantizers
