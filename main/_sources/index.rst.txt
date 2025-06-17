Welcome to the torchao Documentation
====================================

`torchao <https://github.com/pytorch/ao>`__ is a library for custom data types and optimizations.
Quantize and sparsify weights, gradients, optimizers, and activations for inference and training
using native PyTorch. Please checkout torchao `README <https://github.com/pytorch/ao#torchao-pytorch-architecture-optimization>`__
for an overall introduction to the library and recent highlight and updates.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Getting Started

   quick_start
   pt2e_quant

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Developer Notes

   quantization
   sparsity
   contributor_guide

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Reference

   api_ref_dtypes
   api_ref_quantization
   api_ref_sparsity
   api_ref_float8

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Eager Quantization Tutorials

   serialization
   subclass_basic
   subclass_advanced
   static_quantization
   pretraining
   torchao_vllm_integration

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: PT2E Quantization Tutorials

   tutorials_source/pt2e_quant_ptq
   tutorials_source/pt2e_quant_qat
   tutorials_source/pt2e_quant_x86_inductor
   tutorials_source/pt2e_quant_xpu_inductor
   tutorials_source/pt2e_quantizer
   tutorials_source/openvino_quantizer
