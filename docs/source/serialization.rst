Serialization
-------------

Serialization and deserialization is an important question that people care about especially when we integrate torchao with other libraries. Here we want to describe how serialization and deserialization works for torchao optimized (quantized or sparsified) models.

Serialization and deserialization flow
======================================

Here is the serialization and deserialization flow::

  import copy
  import tempfile
  import torch
  from torchao.utils import get_model_size_in_bytes
  from torchao.quantization.quant_api import (
      quantize_,
      Int4WeightOnlyConfig,
  )
  from torchao.testing.model_architectures import ToyTwoLinearModel

  dtype = torch.bfloat16
  m = ToyTwoLinearModel(1024, 1024, 1024).eval().to(dtype).to("cuda")
  print(f"original model size: {get_model_size_in_bytes(m) / 1024 / 1024} MB")

  example_inputs = m.example_inputs(dtype=dtype, device="cuda")
  quantize_(m, Int4WeightOnlyConfig())
  print(f"quantized model size: {get_model_size_in_bytes(m) / 1024 / 1024} MB")

  ref = m(*example_inputs)
  with tempfile.NamedTemporaryFile() as f:
      torch.save(m.state_dict(), f)
      f.seek(0)
      state_dict = torch.load(f)

  with torch.device("meta"):
      m_loaded = ToyTwoLinearModel(1024, 1024, 1024).eval().to(dtype)

  # `linear.weight` is nn.Parameter, so we check the type of `linear.weight.data`
  print(f"type of weight before loading: {type(m_loaded.linear1.weight.data), type(m_loaded.linear2.weight.data)}")
  m_loaded.load_state_dict(state_dict, assign=True)
  print(f"type of weight after loading: {type(m_loaded.linear1.weight), type(m_loaded.linear2.weight)}")

  res = m_loaded(*example_inputs)
  assert torch.equal(res, ref)


What happens when serializing an optimized model?
=================================================
To serialize an optimized model, we just need to call ``torch.save(m.state_dict(), f)``, because in torchao, we use tensor subclass to represent different dtypes or support different optimization techniques like quantization and sparsity. So after optimization, the only thing change is the weight Tensor is changed to an optimized weight Tensor, and the model structure is not changed at all. For example:

original floating point model ``state_dict``::

  {"linear1.weight": float_weight1, "linear2.weight": float_weight2}

quantized model ``state_dict``::

  {"linear1.weight": quantized_weight1, "linear2.weight": quantized_weight2, ...}


The size of the quantized model is typically going to be smaller to the original floating point model, but it also depends on the specific techinque and implementation you are using. You can print the model size with ``torchao.utils.get_model_size_in_bytes`` utility function, specifically for the above example using Int4WeightOnlyConfig quantization, we can see the size reduction is around 4x::

  original model size: 4.0 MB
  quantized model size: 1.0625 MB


What happens when deserializing an optimized model?
===================================================
To deserialize an optimized model, we can initialize the floating point model in `meta <https://pytorch.org/docs/stable/meta.html>`__ device and then load the optimized ``state_dict`` with ``assign=True`` using `model.load_state_dict <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict>`__::


  with torch.device("meta"):
      m_loaded = ToyTwoLinearModel(1024, 1024, 1024).eval().to(dtype)

  print(f"type of weight before loading: {type(m_loaded.linear1.weight), type(m_loaded.linear2.weight)}")
  m_loaded.load_state_dict(state_dict, assign=True)
  print(f"type of weight after loading: {type(m_loaded.linear1.weight), type(m_loaded.linear2.weight)}")


The reason we initialize the model in ``meta`` device is to avoid initializing the original floating point model since original floating point model may not fit into the device that we want to use for inference.

What happens in ``m_loaded.load_state_dict(state_dict, assign=True)`` is that the corresponding weights (e.g. m_loaded.linear1.weight) are updated with the Tensors in ``state_dict``, which is an optimized tensor subclass instance (e.g. int4 ``AffineQuantizedTensor``). No dependency on torchao is needed for this to work.

We can also verify that the weight is properly loaded by checking the type of weight tensor::

  type of weight before loading: (<class 'torch.Tensor'>, <class 'torch.Tensor'>)
  type of weight after loading: (<class 'torchao.dtypes.affine_quantized_tensor.AffineQuantizedTensor'>, <class 'torchao.dtypes.affine_quantized_tensor.AffineQuantizedTensor'>)
