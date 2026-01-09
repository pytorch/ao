.. _api_dtypes:

================
torchao.dtypes
================

.. currentmodule:: torchao.dtypes

Layouts and Tensor Subclasses
-----------------------------
.. autosummary::
    :toctree: generated/
    :nosignatures:

    NF4Tensor
    AffineQuantizedTensor
    Layout
    PlainLayout
    SemiSparseLayout
    TensorCoreTiledLayout
    Float8Layout
    FloatxTensor
    Int4CPULayout
    CutlassSemiSparseLayout

Quantization techniques
-----------------------
.. autosummary::
    :toctree: generated/
    :nosignatures:

    to_affine_quantized_intx
    to_affine_quantized_intx_static
    to_affine_quantized_floatx
    to_affine_quantized_floatx_static
    to_marlinqqq_quantized_intx
    to_nf4

Prototype
---------
.. currentmodule:: torchao.prototype.dtypes

.. autosummary::
    :toctree: generated/
    :nosignatures:

    BlockSparseLayout
    CutlassInt4PackedLayout
    Int8DynamicActInt4WeightCPULayout
    MarlinQQQTensor
    MarlinQQQLayout
    UintxLayout

..
  _NF4Tensor - add after fixing torchao/dtypes/nf4tensor.py:docstring
  of torchao.dtypes.nf4tensor.NF4Tensor.dequantize_scalers:6:Unexpected indentation.
