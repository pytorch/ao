Quantization Overview
---------------------

First we want to lay out the torchao stack::

  Quantization Algorithms/Flows: weight only/dynamic/static quantization, hqq, awq, gptq etc.
  ---------------------------------------------------------------------------------------------
          Quantized Tensors (derived dtypes): AffineQuantizedTensor, CodebookQuantizedTensor
  ---------------------------------------------------------------------------------------------
    Quantization Primitive Ops/Efficient Kernels: matmul, quantize, dequantize
  ---------------------------------------------------------------------------------------------
              Basic dtypes: uint1-uint7, int1-int8, float3-float8


Any quantization algorithm will be using some components from the above stack, for example int4_weight_only quantization uses:
(1) weight only quantization flow
(2) `tinygemm bf16 activation + int4 weight kernel <https://github.com/pytorch/pytorch/blob/136e28f616140fdc9fb78bb0390aeba16791f1e3/aten/src/ATen/native/native_functions.yaml#L4148>`__ and `quant primitive ops <https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_primitives.py>`__
(3) `AffineQuantizedTensor <https://github.com/pytorch/ao/blob/main/torchao/dtypes/affine_quantized_tensor.py>`__ tensor subclass with `TensorCoreTiledLayout <https://github.com/pytorch/ao/blob/e41ca4ee41f5f1fe16c59e00cffb4dd33d25e56d/torchao/dtypes/affine_quantized_tensor.py#L573>`__
(4) torch.uint4 dtype (simulated with quant_min/quant_max right now)

Note: we'll also talk about how to compose sparsity with quantization in the Quantized Tensors section

Basic DTypes
~~~~~~~~~~~~
`dtype <https://en.wikipedia.org/wiki/Data_type>`__ is a bit of overloaded term, by basic dtype, we mean the dtypes that makes sense without any extra metadata (e.g. makes sense when people call ``torch.empty(.., dtype)``), for more details please check out: dev-discuss.pytorch.org/t/supporting-new-dtypes-in-pytorch/1833

No matter what quantization we are doing, in the end we will be using some low precision dtypes to represent the quantized data, the dtypes we aim to support in torchao are:

* ``torch.uint1`` to ``torch.uint8`` available in pytorch 2.3 and later
* ``torch.int1`` to ``torch.int8`` available in pytorch 2.6 and later
* ``torch.float3_e2_m0``, ``torch.float4_e2_m1``, ``torch.float4_e3_m0``, ``torch.float5_e2_m2``, ``torch.float5_e3_m1``, ``torch.float6_e2_m3``, ``torch.float6_e3_m2``, ``torch.float8_e4m3fn``, ``torch.float8_e5m2``, ``torch.float8_e4m3fnuz``, ``torch.float8_e5m2fnuz`` (float8 is added to torch, we also plan to add float4 and float6 to torch if they become popular)

Note some of the above are prototype only for now. We'll consider adding then to pytorch core when they become popular and have hardware support.

Current Support
###############
In terms of actual implementation, there are two parts:
1). In PyTorch, we need to add the dtype to torch.dtype, e.g. torch.uint2, example: pytorch/pytorch#117208, but these are just placeholders so that we can use torch.uint2.
2). Outside of PyTorch (e.g. in torchao), we implement the tensor operations for these dtypes with tensor subclasses, also a standard packing format is needed.

Adding placeholder dtype in PyTorch
***********************************

As mentioned in dev-discuss.pytorch.org/t/supporting-new-dtypes-in-pytorch/1833, the criteria for adding dtype in PyTorch is that it shows wide adoption. For the above mentioned fundamental dtypes, the ones that are supported in PyTorch are:

* ``torch.uint1`` to ``torch.uint8``, ``torch.int1`` to ``torch.int8``, ``torch.float8_e4m3fn``, ``torch.float8_e5m2``, ``torch.float8_e4m3fnuz``, ``torch.float8_e5m2fnuz``

For the other types we plan to wait until there is more evidence of wide adoption and hardware support.

Implementing tensor operations for these dtypes with Tensor subclasses
**********************************************************************
For this, the requirement is we decide on a "standard" packing format, and hopefully one that is amenable to efficient implementation, but for both uintx and floatx we haven't integrate enough kernels to decide on this. So current `packing implementations <https://github.com/pytorch/ao/blob/d2bce6a56eae5701cb72eb0cf6359626e7bd0190/torchao/dtypes/uintx/uintx.py#L36>`__ are ont final. We can revisit after there are more uintx, intx and floatx kernels being integrated into torchao.

Integrate Tensor subclass to pytorch native factory functions
*************************************************************
After that we can connect the factory function with the tensor subclass, for example: ``torch.empty(..., dtype=torch.int4, ...)`` can create a ``Int4Tensor`` tensor subclass with the packing format decided in the previous step.

Quantization Primitive Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~
Quantization primitive ops means the operators used to convert between low preicison quantized tensors and high precision tensors. We will mainly have the following quantization primitive operators:
choose_qparams ops: that chooses quantization parameter based on the original Tensor, typically used in dynamic quantization, e.g. scale and zero_point for affine quantization
quantize op: quantizes the original high precision tensor to the low precision tensor with the dtypes mentioned in previous section based on the quantization parameters
dequantize op: dequantizes the low precision tensor into the high precision tensor based on quantization parameters

There could be variations of the above to accommodate specific use cases, for example for static quantization we may have ``choose_qparams_affine_with_min_max`` that will choose quantization parameters based on min/max values derived from the observation process.

Efficient kernels
~~~~~~~~~~~~~~~~~
We'll also have efficient kernels that works with the low precision tensors, for example

`_weight_int4pack_mm <https://github.com/pytorch/pytorch/blob/136e28f616140fdc9fb78bb0390aeba16791f1e3/aten/src/ATen/native/native_functions.yaml#L4148>`__ the tinygemm int4 kernel (bf16 activation + int4 weight)
`int_matmul <https://github.com/pytorch/ao/blob/3e9746cf636e39e3c1ec0de6e0ef2e31f75c4c02/torchao/kernel/intmm.py#L90>`__ that takes two int8 tensors and outputs an int32 tensor
`int_scaled_matmul <https://github.com/pytorch/ao/blob/3e9746cf636e39e3c1ec0de6e0ef2e31f75c4c02/torchao/kernel/intmm.py#L107>`__ that does matmul and also applies a scale to the result.

Note: We can also rely on torch.compile to generate kernels (through triton), for example the current int8 weight only quantization `kernel <https://github.com/pytorch/ao/blob/e283743b3cc4612bb641b88dca3670231724d396/torchao/dtypes/affine_quantized_tensor.py#L1292-L1309>`__ just relies on torch.compile to get speedup. In this case there is no specific "efficient kernel" that's corresponding to the type of quantization.

Quantized Tensors (derived dtypes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
On top of the basic dtypes, quantization primitive operators and efficient kernels, we can glue everything together and build out a Quantized (low precision) Tensor by subclassing torch.Tensor that can be constructed from a high precision Tensor and some parameters that can configure the specific quantization user wants, we can also call this derived dtypes since it can be represented with Tensors of basic dtypes and some extra metadata like scale.

Existing example in torchao is ``AffineQuantizedTensor``, meaning the low precision Tensor is quantized from the high precision Tensor by an affine mapping, that is: ``low_precision_val = high_precision_val / scale + zero_point``, where ``scale``/``zero_point`` are the quantization parameters that can be calculated by quantization primitive ops or through some optimization procedure. Affine quantization is a very common type of quantization, since it's straightforward that when we try to map from higher precision values to lower precision values, we do an affine transformation (``high_preicsion_val / scale + zero_point``). Another common type of quantization, especially for lower bitwidths (e.g. lower than 4 bit) is codebook / look up table based quantization.

Layout and TensorImpl
#####################
Native tensors have a hardcoded list of selections of `layout <pytorch/pytorch@6478150/c10/core/Layout.h#L10>`__, most common one is strided layout, it provides a strided, multi-dimensional view of storage, we also have some sparse and mkldnn layout.

Take `sparse COO tensor <https://pytorch.org/docs/stable/sparse.html#sparse-coo-tensors>`__ as an example, it has `torch.sparse_coo` layout, and `SparseTensorImpl <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/SparseTensorImpl.h>`__ which changes how the tensor is stored.

The idea of packing the tensor into different formats fits nicely with the layout concept, that’s why we want to reuse this for packing. We can use `Layout` for different type of packing format and `TensorImpl` for different storage format implementations. And new TensorImpl that stores the Tensor in a packed format can be added at python level tensor subclasses without modifying C++ pytorch core code.

For example, for ``_weight_int4pack_mm`` we need to pack the weight to an format that is friendly for Tensor Core, we call it `TensorCoreTiledLayout <https://github.com/pytorch/ao/blob/e41ca4ee41f5f1fe\16c59e00cffb4dd33d25e56d/torchao/dtypes/affine_quantized_tensor.py#L573>`__. We add a ``tensor_impl`` for the quantized tensor to store the packed (or unpacked) weight, and we use ``layout`` to store different parameters that's relevant for packing::

  class AffineQuantizedTensor(...):
    # tensor_impl is also implemented with tensor subclass    
    tensor_impl: torch.Tensor

    # to not conflict with existing layout property, we use `_layout`
    @property
    def _layout(self) -> Layout:
        return self.tensor_impl._layout

Note that layout is an abstraction not only for custom data representation, it is also used for how the
`TensorImpl` interacts with different operators, e.g. the same data representation can have different
implementations when running the same operator, e.g. transpose, quantized_linear, but the operator semantics should stay the same.

Quantize + Sparse Tensor can also be supported through the Layout abstraction, for example, `int4 weight only quantization + sparse <https://github.com/pytorch/ao/pull/621>`__. We also provide some common utils that helps people to add different layouts to a quantized tensor, please check out the developer guide below for code examples.

Quantization Algorithms/Flows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
On the top of the stack will be the final quantization algorithms and quantization flows. Traditionally we have weight only quantization, dynamic quantization and static quantization, but now we are also seeing more types of quantization coming up.

For demonstration purposes, let's say after previous step we have ``AffineQuantizedTensor`` and ``to_affine_quantized`` factory function defined. For simplicity, let's say ``to_affine_quantized`` takes a high precision floating point Tensor and a target_dtype (e.g. torch.int8) and converts it to an ``AffineQuantizedTensor`` with corresponding dtype.

Note: below are all for explaining the concepts, more detailed introduction for utils and examples we provide can be found in ``Tensor Subclass Developer Guide`` section.

Weight Only Quantization
########################
This is the simplest form of quantization and it's easy to apply weight only quantization to the model, especially since we have Quantized Tensor. all we need to do is::
  linear_module.weight = torch.nn.Parameter(to_affine_quantized_intx(linear_module.weight, ...), requires_grad=False))

apply the above to all linear modules in the model and we'll get a weight only quantized model.

Dynamic Activation and Weight Quantization
##########################################

This is called "dynamic quantization" before but it means we quantize activation dynamically at runtime, and also quantize the weights as well. Compared to the weight only quantization, the main question is how do we apply the quantization to activation. In torchao, the common pattern we use is by applying ``to_linear_activation_quantized`` on top of quantized weight::
  quantized_weight = to_affine_quantized(linear_module.weight)
  activation_and_weight_quantized = to_linear_activation_quantized(quantized_weight)
  linear_module.weight = torch.nn.Parameter(activation_and_weight_quantized, requires_grad=False))

``to_linear_activation_quantized`` is used to apply quantization to activation, it takes a ``input_quant_func`` that will quantize the activation and the original weight, and during runtime when it encounters a ``F.linear`` op, it will apply the stored input_qunat_func to activation and redispatch to ``F.linear`` with quantized activation and weight.

If the above does not work, user can also do module swaps, or use ``torch.fx.symbolic_trace()`` to get a traced module that you can `modify <https://pytorch.org/docs/stable/fx.html#direct-graph-manipulation>`__.

But using tensor subclass is preferred because it is easier for serialization/deserialization, if we use tensor subclasses to support dynamic quantization, then we can load the quantized weights directly without further preparation for the model. Otherwise, we'd need to do module swap or other modifications to the model first before loading the quantized weights.

Static Activation Quantization and Weight Quantization
######################################################
Static quantization means activation is statically quantized instead of dynamically quantized at runtime. In terms of flow, static quantization requires calibration with sample data in order that we can figure out the appropriate quantization parameters.

At the high level there are three steps for static quantization: (1) insert observers (2) calibration (3) quantize the model


Insert Observers
****************
In insert observers step, we need to add observer modules to input (and output) activation and weight of the operator to collect statistics of the Tensor. So there are two things we need to address, how to define observer module? how to add observer module to the model.

How to define observer module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Observers are specific to: (1) type of quantization (e.g. affine quantization, look up table based quantization) (2) type of stats we want to track, e.g. min max observer, moving average observer.

Generally an observer module should define `forward <https://github.com/pytorch/ao/blob/e283743b3cc4612bb641b88dca3670231724d396/torchao/quantization/observer.py#L165>`__ and `calculate_qparams <https://github.com/pytorch/ao/blob/e283743b3cc4612bb641b88dca3670231724d396/torchao/quantization/observer.py#L172>`__

For affine quantization, we defined `AffineQuantizedMinMaxObserver <https://github.com/pytorch/ao/blob/e283743b3cc4612bb641b88dca3670231724d396/torchao/quantization/observer.py#L179>`__ that records min_val/max_val based on the granularity of affine quantization, and also defines how to calculate_qparams based on the recorded stats.

How to add observer module to the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Use Tensor Subclasses
   If the only operator you are interested in quantizing is linear, you can use `linear activation weight observer <https://github.com/pytorch/ao/blob/e283743b3cc4612bb641b88dca3670231724d396/torchao/quantization/linear_activation_weight_observer.py>`__, we also have a corresponding `insert_observer_ <https://github.com/pytorch/ao/blob/e283743b3cc4612bb641b88dca3670231724d396/torchao/quantization/quant_api.py#L291>`__ API that handles modifying the weight of linear.

2. Module Swap
   Alternatively, you could also define and `ObservedLinear <https://github.com/pytorch/ao/blob/e283743b3cc4612bb641b88dca3670231724d396/tutorials/calibration_flow/static_quant.py#L29>`__ module (or other module types) and swap the non observed with the observed module

Calibration
^^^^^^^^^^^
Calibration step is typically straightforward, typically we just need to run the model through the calibration dataset. For more complicated calibration (e.g. where we record all inputs and do optimizations based on all inputs), we'll cover some of them in next section.

Quantize
^^^^^^^^
We can reuse the ``quantize_`` API but provide a different ``apply_tensor_subclass`` function that converts the observed linear module to a linear module with quantized weight and statically quantized input activation, this can be done in the same manner as the dynamic quantization (with ``to_linear_activation_quantized``), see `example <https://github.com/pytorch/ao/blob/e283743b3cc4612bb641b88dca3670231724d396/tutorials/calibration_flow/static_quant.py#L59>`__.

Alternatively, user can do `module swap <https://github.com/pytorch/ao/blob/e283743b3cc4612bb641b88dca3670231724d396/tutorials/calibration_flow/static_quant.py#L130>`__ as well.

Other Quantization Flows
########################

For other quantization flow/algorithms that does not fit into any of the above, we also intend to provide examples for common patterns. For example, `GPTQ like quantization flow <https://github.com/pytorch/ao/blob/e283743b3cc4612bb641b88dca3670231724d396/tutorials/calibration_flow/gptq_like.py>`__ that is adopted by `Autoround <https://github.com/pytorch/ao/blob/e283743b3cc4612bb641b88dca3670231724d396/torchao/prototype/autoround/README.md>`__, it uses `MultiTensor <https://gist.github.com/HDCharles/a1b575bbf8875f994af8a01b225e1227>`__ and module hooks to optimize the module.

If you are working on a new quantization algorithm/flow and not sure how to implement it in a PyTorch native way, please feel free to open an issue to describe how your algorithm works and we can help advise on the implementation details.

Training
########
The above flow are mainly focused on inference, but low bit dtype Tensors can be used in training as well.

Quantization Aware Training
***************************
TODO


Low Bit Optimizers
******************
Today we have some prototype low bit optimizers: `main/torchao/prototype/low_bit_optim <https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim>`__ that implements a specific type of 4 bit, 8 bit and float8, and is also composable with FSDP (with look up table quantization).

Quantized Training
******************
Similar to low bit optimizers, we have quantized training prototype in `main/torchao/prototype/quantized_training <https://github.com/pytorch/ao/tree/main/torchao/prototype/quantized_training>`__, and we could extend AffineQuantizedTensor to support training as well, initial enablement is in progress, but there will be a lot of follow up work needed including making it work for different kernels etc.

You can also checkout the tutorial for `Quantized Training <https://github.com/pytorch/ao/blob/main/tutorials/developer_api_guide/my_trainable_tensor_subclass.py>`__ that talks about how to make a dtype tensor subclass trainable.

Case Study: How int4 weight only quantization works in torchao?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To connect everything together, here is a more detailed walk through for how int4 weight only quantization is implemented in torchao.

Quantization Flow: quantize_(model, int4_weight_only())
    * What happens: linear.weight = torch.nn.Parameter(to_affine_quantized_intx(linear.weight), requires_grad=False)
    * quantization primitive ops: choose_qparams and quantize_affine are called to quantize the Tensor
    * quantized Tensor will be `AffineQuantizedTensor`, a quantized tensor with derived dtype (e.g. int4 with scale and zero_point)
    * packing op `_convert_weight_to_int4pack` to pack the quantized weight for efficient execution

During Model Execution: model(input)
    * `torch.ops.aten._weight_int4pack_mm` is called on input and the packed weight

During Quantization
###################
First we start with the API call: ``quantize_(model, int4_weight_only())`` what this does is it converts the weights of nn.Linear modules in the model to int4 quantized tensor (``AffineQuantizedTensor`` that is int4 dtype, asymmetric, per group quantized), using the layout for tinygemm kernel: ``tensor_core_tiled`` layout.

* `quantize_ <https://github.com/pytorch/ao/blob/4865ee61340cc63a1469f437388067b853c9289e/torchao/quantization/quant_api.py#L403>`__: the model level API that quantizes the weight of linear by applying the conversion function from user (second argument)
* `int4_weight_only <https://github.com/pytorch/ao/blob/242f181fe59e233b458740b06464ad42da8df6af/torchao/quantization/quant_api.py#L522>`__: the function that returns a function that converts weight of linear to int4 weight only quantized weight
  * Calls quantization primitives ops like choose_qparams_affine and quantize_affine to quantize the Tensor
* `TensorCoreTiledLayout <https://github.com/pytorch/ao/blob/242f181fe59e233b458740b06464ad42da8df6af/torchao/dtypes/affine_quantized_tensor.py#L573>`__: the tensor core tiled layout type, storing parameters for the packing format
* `TensorCoreTiledAQTTensorImpl <https://github.com/pytorch/ao/blob/242f181fe59e233b458740b06464ad42da8df6af/torchao/dtypes/affine_quantized_tensor.py#L1376>`__: the tensor core tiled TensorImpl, stores the packed weight for efficient int4 weight only kernel (tinygemm kernel)

During Model Execution
######################

When we run the quantized model ``model(inputs)``, we'll run through the functional linear operator in nn.Linear::

  return F.linear(input, weight, bias)

where input is a ``bfloat16`` Tensor, weight is an int4 ``AffineQuantizedTensor``, it calls into a ``__torch_function__`` of the ``AffineQuantizedTensor`` subclass, which will end up in an implementation for ``F.linear`` when one of the input is ``AffineQuantizedTensor``, so it calls::
  return weight_tensor._quantized_linear_op(input_tensor, weight_tensor, bias)

The ``_quantized_linear_op`` goes through the ``_AQT_QLINEAR_DISPATCH_TABLE`` and checks each dispatch conditions, if the dispatch condition passes, it will call the implementation with ``input``/``weight``/``bias``. Please check out `this doc <https://github.com/pytorch/ao/blob/4865ee61340cc63a1469f437388067b853c9289e/torchao/dtypes/affine_quantized_tensor.py#L97>`__ for the explanation of ``dispatch_condition`` and ``impl``.

int4 weight only `dispatch_condition <https://github.com/pytorch/ao/blob/242f181fe59e233b458740b06464ad42da8df6af/torchao/dtypes/affine_quantized_tensor.py#L1784>`__ checks if the input is ``bfloat16`` Tensor and weight is a uint4 ``AffineQuantizedTensor``
wint4 weight only quantization `kernel implementation <https://github.com/pytorch/ao/blob/242f181fe59e233b458740b06464ad42da8df6af/torchao/dtypes/affine_quantized_tensor.py#L1800>`__ takes an bfloat16 input Tensor and an int4 AffineQuantizedTensor, and call ``torch.ops.aten._weight_int4pack_mm`` with the input Tensor and the packed weight that's stored in ``weight_tensor.tensor_impl``.

During Save/Load
################

Since ``AffineQuantizedTensor`` weight is still a ``torch.Tensor``, save/load works the same way as the original high precision floating point model. See the `serialization doc <https://pytorch.org/ao/stable/serialization.html>`__ for more details.


