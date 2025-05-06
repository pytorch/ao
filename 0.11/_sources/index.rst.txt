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

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials

   serialization
   subclass_basic
   subclass_advanced
