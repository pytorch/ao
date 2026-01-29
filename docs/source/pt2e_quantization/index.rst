PT2E Quantization Tutorials
===========================

Tutorials for quantization using PyTorch 2 Export.

Quick Start
-----------
PyTorch 2 Export Quantization is a full graph quantization workflow mostly for static quantization. It targets hardwares that requires both input and output activation and weight to be quantized and relies of recognizing an operator pattern to make quantization decisions (such as linear - relu). PT2E quantization produces a pattern with quantize and dequantize ops inserted around the operators and during lowering quantized operator patterns will be fused into real quantized ops. Currently there are two typical lowering paths, 1. torch.compile through inductor lowering 2. ExecuTorch through delegation

Here we show an example with X86InductorQuantizer

API Example::

  import torch
  from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e
  from torch.export import export
  from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
      X86InductorQuantizer,
      get_default_x86_inductor_quantization_config,
  )

  class M(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.linear = torch.nn.Linear(5, 10)

     def forward(self, x):
         return self.linear(x)

  # initialize a floating point model
  float_model = M().eval()

  # define calibration function
  def calibrate(model, data_loader):
      model.eval()
      with torch.no_grad():
          for image, target in data_loader:
              model(image)

  # Step 1. program capture
  m = export(m, *example_inputs).module()
  # we get a model with aten ops

  # Step 2. quantization
  # backend developer will write their own Quantizer and expose methods to allow
  # users to express how they
  # want the model to be quantized
  quantizer = X86InductorQuantizer()
  quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

  # or prepare_qat_pt2e for Quantization Aware Training
  m = prepare_pt2e(m, quantizer)

  # run calibration
  # calibrate(m, sample_inference_data)
  m = convert_pt2e(m)

  # Step 3. lowering
  # lower to target backend

  # Optional: using the C++ wrapper instead of default Python wrapper
  import torch._inductor.config as config
  config.cpp_wrapper = True

  with torch.no_grad():
      optimized_model = torch.compile(converted_model)

      # Running some benchmark
      optimized_model(*example_inputs)


Please follow these tutorials to get started on PyTorch 2 Export Quantization:

Modeling Users:

- `PyTorch 2 Export Post Training Quantization <pt2e_quant_ptq.html>`__
- `PyTorch 2 Export Quantization Aware Training <pt2e_quant_qat.html>`__
- `PyTorch 2 Export Post Training Quantization with X86 Backend through Inductor <pt2e_quant_x86_inductor.html>`__
- `PyTorch 2 Export Post Training Quantization with XPU Backend through Inductor <pt2e_quant_xpu_inductor.html>`__
- `PyTorch 2 Export Quantization for OpenVINO torch.compile Backend <pt2e_quant_openvino_inductor.html>`__


Backend Developers (please check out all Modeling Users docs as well):

- `How to Write a Quantizer for PyTorch 2 Export Quantization <pt2e_quantizer.html>`_

.. toctree::
   :maxdepth: 1
   :hidden:

   pt2e_quant_ptq
   pt2e_quant_qat
   pt2e_quant_x86_inductor
   pt2e_quant_xpu_inductor
   pt2e_quant_openvino_inductor
   pt2e_quantizer
