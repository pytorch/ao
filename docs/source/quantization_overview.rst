Quantization Overview
---------------------

First we want to lay out the torchao stack::

  Quantization Algorithms/Flows: weight only/dynamic/static quantization, hqq, awq, gptq etc.
  ---------------------------------------------------------------------------------------------
      Quantized Tensors (derived dtypes): Int4Tensor, Int4PreshuffledTensor, Float8Tensor
  ---------------------------------------------------------------------------------------------
    Quantization Primitive Ops/Efficient Kernels: matmul, quantize, dequantize
  ---------------------------------------------------------------------------------------------
              Basic dtypes: uint1-uint7, int1-int8, float3-float8


Any quantization algorithm will be using some components from the above stack, for example per row float8 dynamic activation and float8 weight quantization (with default preference) uses:

* dynamic quantization flow
* `Float8Tensor <https://github.com/pytorch/ao/blob/main/torchao/quantization/quantize_/workflows/float8/float8_tensor.py>`__
* `float8 activation + float8 weight fbgemm kernel <https://github.com/pytorch/ao/blob/6cfa47705f60ea614695b52b4b120ac5fd84d1cb/torchao/quantization/quantize_/workflows/float8/float8_tensor.py#L280>`__ and `triton quant primitive ops from fbgemm library <https://github.com/pytorch/ao/blob/6cfa47705f60ea614695b52b4b120ac5fd84d1cb/torchao/quantization/quantize_/workflows/float8/float8_tensor.py#L198>`__
* ``torch.float8_e4m3fn`` dtype

Basic DTypes
~~~~~~~~~~~~
`dtype <https://en.wikipedia.org/wiki/Data_type>`__ is a bit of overloaded term, by basic dtype, we mean the dtypes that makes sense without any extra metadata (e.g. makes sense when people call ``torch.empty(.., dtype)``), for more details please check out `this post <dev-discuss.pytorch.org/t/supporting-new-dtypes-in-pytorch/1833>`__.

No matter what quantization we are doing, in the end we will be using some low precision dtypes to represent the quantized data or quantization parameters, the low precision dtypes relevant for torchao are:

* ``torch.uint1`` to ``torch.uint7`` available in pytorch 2.3 and later
* ``torch.int1`` to ``torch.int7`` available in pytorch 2.6 and later
* ``torch.float4_e2m1fn_x2``, ``torch.float8_e4m3fn``, ``torch.float8_e4m3fnuz``, ``torch.float8_e5m2``, ``torch.float8_e5m2fnuz``, ``torch.float8_e8m0fnu``

In terms of actual implementation, ``uint1`` to ``uint7`` and ``int1`` to ``int7`` are just placeholders that does not have real implementations (i.e. the ops does not work for the PyTorch Tensor with these dtypes). Example PR added these dtypes can be found `here <https://github.com/pytorch/pytorch/pull/117208>`__. Floating point dtypes are what we call shell dtypes that have limited op support.

For more details please check out the `official PyTorch dtype doc <https://docs.pytorch.org/docs/main/tensor_attributes.html>`__.

.. note::
   Dervied dtypes like mxfp8, mxfp4, nvfp4 are implemented with these basic dtypes, e.g. mxfp4 uses ``torch.float8_e8m0fnu`` for scale and ``torch.float4_e2m1fn_x2`` for 4 bit data.

Quantization Primitive Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~
Quantization primitive ops means the operators used to convert between low preicison quantized tensors and high precision tensors. We will mainly have the following quantization primitive operators:

* choose_qparams ops: that chooses quantization parameter based on the original Tensor, typically used in dynamic quantization, e.g. scale and zero_point for affine quantization
* quantize op: quantizes the original high precision tensor to the low precision tensor with the dtypes mentioned in previous section based on the quantization parameters
* dequantize op: dequantizes the low precision tensor into the high precision tensor based on quantization parameters

There could be variations of the above to accommodate specific use cases, for example for static quantization we may have ``choose_qparams_affine_with_min_max`` that will choose quantization parameters based on min/max values derived from the observation process.

There could be multiple versions of the op that is different by different kernel libraries that we can use in torchao, for example, for quantizing a bfloat16 Tensor to a raw float8 Tensor and scale: `_choose_scale_float8 <https://github.com/pytorch/ao/blob/6cfa47705f60ea614695b52b4b120ac5fd84d1cb/torchao/quantization/quant_primitives.py#L2183>`__ and `_quantize_affine_float8 <https://github.com/pytorch/ao/blob/6cfa47705f60ea614695b52b4b120ac5fd84d1cb/torchao/quantization/quant_primitives.py#L2282>`__ for torchao implementation, and `torch.ops.triton.quantize_fp8_row <https://github.com/pytorch/ao/blob/6cfa47705f60ea614695b52b4b120ac5fd84d1cb/torchao/quantization/quantize_/workflows/float8/float8_tensor.py#L198C27-L198C60>`__ from fbgemm library.

Efficient kernels
~~~~~~~~~~~~~~~~~
We'll also have efficient kernels that works with the low precision tensors, for example:

* `torch.ops.fbgemm.f8f8bf16_rowwise <https://github.com/pytorch/ao/blob/6cfa47705f60ea614695b52b4b120ac5fd84d1cb/torchao/quantization/quantize_/workflows/float8/float8_tensor.py#L280>`__ (rowwise float8 activation and float8 weight matrix multiplication kernel in fbgemm library)
* `torch._scaled_mm <https://github.com/pytorch/ao/blob/6cfa47705f60ea614695b52b4b120ac5fd84d1cb/torchao/float8/inference.py#L116>`__ (float8 activation and float8 weight matrix multiplication kernel in PyTorch for both rowwise and tensorwise)
* `int_matmul <https://github.com/pytorch/ao/blob/3e9746cf636e39e3c1ec0de6e0ef2e31f75c4c02/torchao/kernel/intmm.py#L90>`__ that takes two int8 tensors and outputs an int32 tensor
* `int_scaled_matmul <https://github.com/pytorch/ao/blob/3e9746cf636e39e3c1ec0de6e0ef2e31f75c4c02/torchao/kernel/intmm.py#L107>`__ that does matmul and also applies a scale to the result.

.. note::
   We can also rely on torch.compile to generate kernels (through triton), for example the current int8 weight only quantization `kernel <https://github.com/pytorch/ao/blob/e283743b3cc4612bb641b88dca3670231724d396/torchao/dtypes/affine_quantized_tensor.py#L1292-L1309>`__ just relies on torch.compile to get speedup. In this case there is no custom handwritten "efficient kernel" that's corresponding to the type of quantization.

Quantized Tensors (derived dtypes and packing format)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
On top of the basic dtypes, quantization primitive operators and efficient kernels, we can glue everything together and build out a Quantized (low precision) Tensor by subclassing torch.Tensor that can be constructed from a high precision Tensor and some parameters that can configure the specific quantization user wants, we can also call this derived dtypes since it can be represented with Tensors of basic dtypes and some extra metadata like scale.

Another dimension for quantized Tensor is packing format, meaning how the quantized raw data is laid out in memory. For example, for int4, we can pack two elements together side by side in a uint8 value, or people can do some preshuffling/swizzling to make the format more efficient for memory operations (loading from memory to register) and computation.

So in general we structure Tensor subclasses by dervied dtpype and packing format:

.. list-table:: Tensor Subclasses in TorchAO
   :widths: 20 10 30 40
   :header-rows: 1

   * - Tensor
     - Derived Dtype
     - Packing Format
     - Support
   * - Float8Tensor
     - scaled float8
     - plain (no packing needed)
     - float8 act + float8 weight dynamic quantization and float8 weight only quantization
   * - Int4Tensor
     - scaled int4
     - plain (pack 2 adjacent int4 to a single int8 value)
     - int4 weight only quantization
   * - Int4PreshuffledTensor
     - scaled int4
     - preshuffled (special format to optimize for loading)
     - float8 act + int4 weight dynamic quantization and int4 weight only quantization

.. note::
   We don't have granularity specific tensor subclasses, i.e. no Float8RowwiseTensor or Float8BlockwiseTensor, all granularities are implemented in the same Tensor, we typically use a general `block_size` attribute to distinguish between different granularities, and each Tensor is allowed to support only a subset of all possible granularity options.

.. note::
   We also don't use dynamic activation in the name, since we are talking about the weight tensor object, including information about activation in the tensor subclass name will be confusing, but
   we do implement both weight only and dynamic activation quantization in the same linear function implementation, without relying on additional abstractions, this keeps relevant quantization operations close
   to each other (quantization of activation and weight) in the same tensor subclass.

In terms of how we quantize a Tensor, most of Tensors are using affine quantization, meaning the low precision Tensor is quantized from the high precision Tensor by an affine mapping, that is: ``low_precision_val = high_precision_val / scale + zero_point``, where ``scale`` and ``zero_point`` are the quantization parameters that can be calculated by quantization primitive ops or through some optimization procedure. Another common type of quantization, especially for lower bitwidths (e.g. lower than 4 bit) is codebook / look up table based quantization where the raw quantized data is the index we can use to look up a ``codebook`` that stores the values or vectors each index corresponds to. A common way to get the codebook and the raw quantized data for codebook quantization is kmeans clustering.

Quantization Algorithms/Flows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
On the top of the stack will be the final quantization algorithms and quantization flows. Traditionally we have weight only quantization, dynamic quantization and static quantization, but now we are also seeing more types of quantization coming up.

For demonstration purposes, let's say after previous step we have ``Float8Tensor`` defined. ``Float8Tensor.from_hp`` takes a high precision floating point Tensor and a target_dtype (e.g ``torch.float8_e4m3fn``) and converts it to a ``Float8Tensor``

Note: below are all for explaining the concepts, more detailed introduction for utils and examples we provide can be found in `Contributor Guide <contributor_guide.html>`__.

Weight Only Quantization
########################
This is the simplest form of quantization and it's easy to apply weight only quantization to the model, especially since we have Quantized Tensor. all we need to do is::

  linear_module.weight = torch.nn.Parameter(Float8Tensor.from_hp(linear_module.weight, ...), requires_grad=False))

apply the above to all linear modules in the model and we'll get a weight only quantized model.

Dynamic Activation and Weight Quantization
##########################################

This is called "dynamic quantization" before but it means we quantize activation dynamically at runtime, and also quantize the weights as well. Compared to the weight only quantization, the main question is how do we apply the quantization to activation. In torchao we pass around the quantization keyword args for activation and the keyword args will be applied to activation when needed (e.g. in linear)::

  activation_dtype = torch.float8_e4m3fn
  activation_granularity = PerRow()
  # define kwargs for float8 activation quantization
  act_quant_kwargs = QuantizeTensorToFloat8Kwargs(
    activation_dtype,
    activation_granularity,
  )
  weight_dtype = torch.float8_e4m3fn
  weight_granularity = PerRow()
  quantized_weight = Float8Tensor.from_hp(linear_module.weight, float8_dtype=weight_dtype, granularity=weight_granularity, act_quant_kwargs=act_quant_kwargs)
  linear_module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False))

Static Activation Quantization and Weight Quantization
######################################################
We'll skip the instruction for now since we haven't seen many use cases for static quantization with tensor subclass based flow, we recommend to look into the `PT2 export quantization flow <quick_start.html#pytorch-2-export-quantization>`__ for static quantization.

Other Quantization Flows
########################

For other quantization flow/algorithms that does not fit into any of the above, we also intend to provide examples for common patterns. For example, `GPTQ like quantization flow <https://github.com/pytorch/ao/blob/e283743b3cc4612bb641b88dca3670231724d396/tutorials/calibration_flow/gptq_like.py>`__ that is adopted by `Autoround <https://github.com/pytorch/ao/blob/e283743b3cc4612bb641b88dca3670231724d396/torchao/prototype/autoround/README.md>`__, it uses `MultiTensor <https://gist.github.com/HDCharles/a1b575bbf8875f994af8a01b225e1227>`__ and module hooks to optimize the module.

If you are working on a new quantization algorithm/flow and not sure how to implement it in a PyTorch native way, please feel free to open an issue to describe how your algorithm works and we can help advise on the implementation details.

Training
########
The above flow are mainly focused on inference, but low bit dtype Tensors can be used in training as well.

User facing docs for float8 training can be found `here <https://docs.pytorch.org/ao/main/pretraining.html>`__ and docs for finetuning can be found `here <https://docs.pytorch.org/ao/main/finetuning.html>`__

Quantization Aware Training
***************************
TorchAO supports `quantization aware training <https://github.com/pytorch/ao/tree/main/torchao/quantization/qat>`__ through the `quantize_` API as well.


Low Bit Optimizers
******************
We support `low bit optimizers <https://github.com/pytorch/ao/tree/main/torchao/optim>`__ that implements a specific type of 4 bit, 8 bit and float8, and is also composable with FSDP (with look up table quantization).

Quantized Training
******************
We have quantized training prototype in `main/torchao/prototype/quantized_training <https://github.com/pytorch/ao/tree/main/torchao/prototype/quantized_training>`__, and we could extend existing tensor subclasses to support training as well, initial enablement is in progress, but there will be a lot of follow up work needed including making it work for different kernels etc.

You can also checkout the tutorial for `Quantized Training <https://github.com/pytorch/ao/blob/main/tutorials/developer_api_guide/my_trainable_tensor_subclass.py>`__ that talks about how to make a dtype tensor subclass trainable.

Case Study: How float8 dynamic activation and float8 weight quantization works in torchao?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To connect everything together, here is a more detailed walk through for float8 dynamic activation and float8 weight quantization in torchao (DEFAULT kernel preference, in H100, when fbgemm_gpu_genai library is installed):

Quantization Flow: ``quantize_(model, Float8DynamicActivationFloat8WeightConfig())``
    * What happens: ``linear.weight = torch.nn.Parameter(Float8Tensor.from_hp(linear.weight), requires_grad=False)``
    * quantization primitive ops: ``torch.ops.triton.quantize_fp8_row``
    * quantized Tensor will be ``Float8Tensor``, a quantized tensor with derived dtype of scaled float8

During Model Execution: model(input)
    * ``torch.ops.fbgemm.f8f8bf16_rowwise`` is called on input, raw float8 weight and scale

During Quantization
###################
First we start with the API call: ``quantize_(model, Float8DynamicActivationFloat8WeightConfig())`` what this does is it converts the weights of nn.Linear modules in the model to ``Float8Tensor``, with plain packing format, no packing is required, since we have ``torch.float8_e4m3fn`` that can represent quantized float8 raw data directly without additional operations.

* `quantize_ <https://docs.pytorch.org/ao/main/generated/torchao.quantization.quantize_.html#torchao.quantization.quantize_>`__: the model level API that quantizes the weight of linear by applying the config from user (second argument)
* `Float8DynamicActivationFloat8WeightConfig <https://docs.pytorch.org/ao/main/generated/torchao.quantization.Float8DynamicActivationFloat8WeightConfig.html#torchao.quantization.Float8DynamicActivationFloat8WeightConfig>`__: the config for float8 dynamic activation and float8 weight quantization
  * Calls quantization primitives ops ``torch.ops.triton.quantize_fp8_row`` to quantize a bfloat16 Tensor to float8 raw Tensor and get a scale


During Model Execution
######################

When we run the quantized model ``model(inputs)``, we'll run through the functional linear operator in nn.Linear::

  return F.linear(input, weight, bias)

where input is a ``bfloat16`` Tensor, weight is a ``Float8Tensor``, it calls into a ``__torch_function__`` of the ``Float8Tensor`` subclass, which will end up in an implementation for ``F.linear`` when one of the `input <https://github.com/pytorch/ao/blob/6cfa47705f60ea614695b52b4b120ac5fd84d1cb\/torchao/quantization/quantize_/workflows/float8/float8_tensor.py#L233>`__ is ``Float8Tensor``::

  @implements([torch.nn.functional.linear, aten.linear.default])
  def _(func, types, args, kwargs):
      input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
      )
      # quantizing activation, if `act_quant_kwargs` is specified
      if act_quant_kwargs is not None:
        input_tensor = _choose_quant_func_and_quantize_tensor(
            input_tensor, act_quant_kwargs
        )

      # omitting kernel_preference related code
      # granularity checks, let's say we are doing rowwise quant
      # both input_tensor and weight_tensor will now be Float8Tensor
      xq = input_tensor.qdata.reshape(-1, input_tensor.qdata.shape[-1])
      wq = weight_tensor.qdata.contiguous()
      x_scale = input_tensor.scale
      w_scale = weight_tensor.scale
      res = torch.ops.fbgemm.f8f8bf16_rowwise(
         xq,
         wq,
         x_scale,
         w_scale,
      ).reshape(out_shape)
      return res

The function first quantizes the input to be ``Float8Tensor``, then get the raw float Tensor and scale from both the input and weight Tensor: ``t.qdata``, ``t.scale``, and calls the fbgemm kernel to do the matrix multiplication for float8 dynamic quantization: ``torch.ops.fbgemm.f8f8bf16_rowwise``.

During Save/Load
################

Since ``Float8Tensor`` weight is still a ``torch.Tensor``, save/load works the same way as the original high precision floating point model. See the `serialization doc <serialization.html>`__ for more details.
