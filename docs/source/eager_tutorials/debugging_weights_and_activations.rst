Debugging Weights and Activations with `quant_logger`
=====================================================

`torchao.prototype.quant_logger` is a lightweight logging framework designed to
help extract information about a model's weights and activations such as activation
shapes and numerical statistics about tensor values. 

See the :ref:`API reference <api_prototype_quant_logger>` for full details.

An e2e example of using the logging to extract the activation shapes flowing
through the `torch.nn.Linear` layers of the `FLUX-1.schnell` model is below.
The activation shapes can be further used to select which layers to quantize,
as quantizing layers with small activation shapes is usually not beneficial.

Example
-------

.. literalinclude:: ../examples/prototype/quant_logger/e2e_example.py
   :language: python
