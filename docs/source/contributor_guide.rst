torchao Contributor Guide
-------------------------

.. toctree::
 :maxdepth: 3

Objective
=========
In this doc we’ll talk about
(1). How different optimization techniques are structured in torchao
(2). How to contribute to torchao

Note: the doc is heavily focused on inference right now, but we plan to expand to cover training techniques in the future as well.

torchao Stack Overview
======================

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

High Level Summary
##################

::
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

Tensor Subclass Developer Guide
===============================

We have covered high level overview and how everything is connected together in the previous section, this section will focus on Tensor Subclasses, which is the main extension point we rely on to provide flexibility of supporting inference, training and fine tuning with low precision Tensors and composability with torch.compile, autograd, distributed primitives in these scenarios.

Prerequisites
~~~~~~~~~~~~~
Some externally available resources for tensor subclasses:

* `tensor subclass doc <pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor>`__
* `Edward's podcast about tensor subclasses <https://podcasts.apple.com/us/podcast/tensor-subclasses-and-pt2/id1566080008?i=1000646728968>`__
* `Tensor subclass zoo <https://github.com/albanD/subclass_zoo>`__

Why Tensor Subclass?
~~~~~~~~~~~~~~~~~~~~
There are multiple ways people can implement quantization techniques or new dtypes, main motivation for us to recommend the tensor subclass based approach are three things:
(1). It’s natural for quantization to be modeled as a dtype conversion, so implementing it with tensor subclass means we are not introducing new concepts but reusing existing concepts like dtype, layout that already exists in pytorch core
(2). Since tensor subclass intercepts computation at torch function or aten ops level, as long as the same function/operator is used, we will be able to quantize the model. This allows the model that’s using variants of native modules (e.g. a slightly modified version of nn.Linear) to still be compatible with quantization
(3). Tensor subclass is also the approach adopted by other techniques like sparsity and distributed, so implementing quantization or dtype conversion with tensor subclass would make it easier for it to be composable with these techniques

Example Code for a new DType
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Please feel free to start with `tutorial <https://github.com/pytorch/ao/blob/main/tutorials/developer_api_guide/my_dtype_tensor_subclass.py>`__ for a end to end working example that combines everything we talked about together and come back to the doc for clarifications and documentations.

Basic Structure
~~~~~~~~~~~~~~~
A tensor subclass needs to define a few basic methods: ``__new__``, ``__init__``, ``__tensor_flatten__``, ``__tensor_unflatten__``
and also dispatch functions for torch functions ``__torch_function__`` and aten ops ``__torch_dispatch__``.

Here is an example of basic structure::
  # check out docs in https://github.com/pytorch/ao/blob/e283743b3cc4612bb641b88dca3670231724d396/torchao/utils.py#L437
  from torchao.utils import TorchAOBaseTensor

  class MyDTypeLayout(TorchAOBaseTensor):
      # see tutorial code for details
      pass

  class MyDtypeTensor(TorchAOBaseTensor):
      """We need to define `__new__` for constructing a new tensor subclass instance and `__init__` for initialize
      the instance. There is no requirement on what the argument list should look like here, only requirement is
      that `__new__` must return a Tensor instance with `torch.Tensor._make_wrapper_subclass(cls, shape, ...)` call
      """
      @staticmethod
      def __new__(
          cls,
          tensor_impl: MyDTypeLayout,
          shape: torch.Size,
          dtype: Optional[torch.dtype] = None,
      ):
          ...
          return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

      def __init__(
          self,
          tensor_impl: MyDTypeLayout,
          shape: torch.Size, ...
      ):
          self.tensor_impl = tensor_impl


      """`__tensor_flatten__` and `__tensor_unflatten__` are used to desugar the tensor into native Tensors/attributes and
      reconstruct the tensor subclass instance from the desugared tensor and attributes, these are required to define
      a Tensor subclass for torch.compile support
      """
      def __tensor_flatten__(self):
          return ["tensor_impl"], [self.shape]

      """see https://github.com/pytorch/pytorch/blob/3bc2004f9123a32f381ef64202252d59109507f3/torch/utils/_python_dispatch.py#L289 for documentations for outer_size and outer_stride
      """
      @classmethod
      def __tensor_unflatten__(
          cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
      ):
         tensor_impl = tensor_data_dict["tensor_impl"]
          shape, = tensor_attributes
          return cls(
              tensor_impl,
              shape if outer_size is None else outer_size,
          )


      """classmethod that converts from a floating point Tensor (fp32/fp16/bf16) to the current dtype
      """
     @classmethod
      def from_float(
          cls,
          input_float: torch.Tensor,
      ):
          mapping_type = MappingType.SYMMETRIC
          block_size = input_float.shape
          dtype = torch.int16
          scale, _ = choose_qparams_affine(input_float, mapping_type, block_size, dtype)
          int_data = (input_float / scale).to(torch.int8)
          tensor_impl = MyDTypeLayout.from_plain(int_data, scale)
          return cls(tensor_impl, input_float.shape)


      """[Optional] see docs for `Layout/Packing` under `Quantized Tensors` section to understand what layout_type is
      """
      @property
      def _layout(self) -> LayoutType:
          return self.tensor_impl._layout

      """There are two entry points that we can modify the behavior of a pytorch op: torch_function and torch_dispatch:

      __torch_function__: will be called whenever a torch level function is called on the Tensor object, for example: torch.nn.functional.linear,
      tensor.detach, tensor.reshape, tensor.t etc.

      __torch_dispatch__: will be called in the C++ dispatcher, when an aten operator is called on the Tensor object, for example:
      aten.mm, aten.addmm, aten.detach.default, aten.t.default etc.
      you can checkout https://github.com/pytorch/ao/blob/e283743b3cc4612bb641b88dca3670231724d396/torchao/utils.py#L361-L389 to understand what `__torch_function__` and `__torch_dispatch__` are doing, but with `TorchAoBaseTensor` user can use
      some helper functions directly (see next section)

Operator Support
~~~~~~~~~~~~~~~~
There are two types of operator support, torch function and aten ops. For torch functions (e.g. ``torch.nn.functional.linear``), we’ll need to overwrite ``__torch_function__`` callback in the Tensor subclass, for aten ops (e.g. ``torch.ops.aten.mm``), we’ll need to overwrite ``__torch_dispatch__`` callback function.

For a new dtype, we’d like people to define the following decorator::
  if your dtype class is inherited from `torchao.utils.TorchAoBaseTensor`, you can do:

  implements = my_dtype_tensor_cls.implements

And we can implement the operator dispatch with the following::
  # Example for torch_function dispatch for torch.nn.functional.linear
  def _quantized_linear_op(input_tensor, weight_tensor, bias):
      if isinstance(input_tensor, MyDtypeTensor):
          input_tensor = input_tensor.dequantize()
      if isinstance(weight_tensor, MyDtypeTensor):
          weight_tensor = weight_tensor.dequantize()
      return torch.nn.functional.linear(input_tensor, weight_tensor, bias)


  @implements(torch.nn.functional.linear)
  def _(*args, **kwargs):
      input_tensor, weight_tensor, bias = (
          args[0],
          args[1],
          args[2] if len(args) > 2 else None,
      )
      # using try/except here so that we can have a general fallback when input_tensor/weight_tensor
      # is not picked up by any of the dispatch paths in `_quantized_linear_op`, this allows us to
      # make the branches easier to understand in `_quantized_linear_op`
      try:
          return _quantized_linear_op(input_tensor, weight_tensor, bias)
      except NotImplementedError:
          if isinstance(input_tensor, MyDtypeTensor):
              input_tensor = input_tensor.dequantize()
          if isinstance(weight_tensor, MyDtypeTensor):
              weight_tensor = weight_tensor.dequantize()
          return torch.nn.functional.linear(input_tensor, weight_tensor, bias)

  # Example for aten op dispatch for aten.detach.default
  @implements(aten.detach.default)
  def _(func, *args, **kwargs):
      # `return_and_correct_aliasing` should be used by wrapper tensor ``__torch_dispatch__`` subclasses that would like to 
      # work with torch.compile. It ensures that the subclass properly implements the aliasing behavior of every op, 
      # which is needed for correctness in AOTAutograd.

      # `_apply_fn_to_data` just applies the function to the tensor data in `args[0]`, `args[0]` is a tensor subclass
      # of `my_dtype`
      return return_and_correct_aliasing(
          func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
      )

What ops do we need to overwrite? This depends on the model we are trying to quantize, commonly overwritten ops are:
``__torch_function__``: ``torch.nn.functional.linear``
``__torch_dispatch__``: ``torch.ops.aten.addmm.default``, ``torch.ops.aten.mm.default``, ``torch.ops.aten.detach.default``, ``torch.ops.aten.t.default``

You can also find the ops that can be overwritten in ``__torch_function__`` or ``__torch_dispatch__`` with the following code, and you can start with a model that you want to optimize, start with just overwriting the important ops like linear, and gradually expand the coverage until the test runs and you get the expected optimized generated code (see Optimized Operators section for more details)::
  class M(torch.nn.Module): 
    def __init__(self) -> None: 
        super().__init__() 
        self.linear = torch.nn.Linear(10, 10)
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.linear(x) + x

  from torch.overrides import TorchFunctionMode
  class TorchFunctionLoggingMode(TorchFunctionMode):
      def __torch_function__(cls, func, types, args=(), kwargs=None):
          if kwargs is None:
              kwargs = {}
          print(f"TORCH_FUNC={str(func)}")
          return func(*args, **kwargs)

  with TorchFunctionLoggingMode():
       m(*example_inputs)

  ## Example output
  # TORCH_FUNC=<built-in function linear>
  # TORCH_FUNC=<method 'add' of 'torch._C.TensorBase' objects>


  from torch.utils._python_dispatch import TorchDispatchMode
  class TorchDispatchLoggingMode(TorchDispatchMode):
      def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
          if kwargs is None:
              kwargs = {}
          print(f"ATEN_FUNC={str(func)}")
          return func(*args, **kwargs)

  with TorchDispatchLoggingMode():
       m(*example_inputs)

  ## Example output
  # ATEN_FUNC=aten.t.default
  # ATEN_FUNC=aten.addmm.default
  # ATEN_FUNC=aten.add.Tensor

  # or a more polished logging for torch_dispatch (aten) ops: https://github.com/albanD/subclass_zoo/blob/main/logging_mode.py

Alternatively, you can run a test example (e.g. use your quantized model with tensor parallelism, FSDP etc.) and discover the missing ops and add them until the test passes.

We are still working on a table that talks about for each feature what are the operators that need to be supported.

Adding Efficient Kernels
~~~~~~~~~~~~~~~~~~~~~~~~

Custom triton kernels
#####################
Custom triton kernels can be implemented and registered in `torchao/kernel <https://github.com/pytorch/ao/tree/main/torchao/kernel>`__

* `Implementation Example <https://github.com/pytorch/ao/blob/0bdde92114b470823aa24725bf3b0811e980c8ce/torchao/kernel/intmm_triton.py#L270-L302>`__
* `Register as a custom op <https://github.com/pytorch/ao/blob/0bdde92114b470823aa24725bf3b0811e980c8ce/torchao/kernel/intmm_triton.py#L337-L364>`__

You may need to define you own `autotuner <https://github.com/pytorch/ao/blob/main/torchao/kernel/autotuner.py>`__ as well.

Custom hand written kernels
###########################
Custom kernels (implementations) for cpu/cuda/mps can be implemented through `torchao/csrc <https://github.com/pytorch/ao/tree/main/torchao/csrc>`__ e.g. int4 cuda, and accessible through torch.ops.my_custom_op

Dispatches
~~~~~~~~~~

For dispatching to optimized kernels for cpu/cuda/mps devices, we can have checks for the dispatch conditions in ``__torch_function__`` or ``__torch_dispatch__`` and dispatch to target operators, for example, condition for bfloat16 activation and uint4 weight kernel can be found `here <https://github.com/pytorch/ao/blob/242f181fe59e233b458740b06464ad42da8df6af/torchao/dtypes/affine_quantized_tensor.py#L1784-L1797>`__.

Specifically for ``AffineQuantizedTensor``, we also allow people to extend the quantized linear to use a new efficient kernel or implement by defining two functions:
``dispatch_condition`` (defines the condition to dispatch to the kernel) and impl (actual implementation that takes activation, (quantized) weight, bias Tensor and runs the efficient kernel), both taking ``input_tensor``, ``weight_tensor``, ``bias`` as argument, and can be registered into dispatch of quantized linear in ``AffineQuantizedTensor`` with ``register_aqt_quantized_linear_dispatch``. `Here <https://github.com/pytorch/ao/blob/e283743b3cc4612bb641b88dca3670231724d396/test/dtypes/test_affine_quantized.py#L92-L113>`__ is an example showing how it works.

Layout/TensorImpl
~~~~~~~~~~~~~~~~~

Sometimes the quantized weights has to be packed in order to yield optimal performance. And this can be abstracted with ``layout``. See `here <https://github.com/pytorch/ao/blob/17a0a96d24ebfc154a23342b84e788d9ed6776f4/tutorials/developer_api_guide/my_dtype_tensor_subclass.py#L215-L317>`__ for full example.

Flow
~~~~

After the tensor subclass is implemented, we can also wrap that into factory functions, e.g.::
  # convert from floating point tensor to my dtype tensor subclass
  to_my_dtype = MyDTypeTensor.from_float

For model level API, people can reuse ``torchao.quantize_`` that allows people to apply a tensor subclass conversion to weight of linear, and allows `filtering function <https://github.com/pytorch/ao/blob/17a0a96d24ebfc154a23342b84e788d9ed6776f4/torchao/quantization/quant_api.py#L421>`__ to choose which module the tensor subclass conversion should be applied to.

See Quantization Algorithms/Flows section for examples of weight only/dynamic quant/static quant and other types of model level APIs based on the factory function.

Using torch.compile for Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note: for pytorch 2.4 and below, we need to use the following::
  from torchao.utils import unwrap_tensor_subclass
  m_unwrapped = unwrap_tensor_subclass(m)

In order to be compatible with ``torch.compile``, to aim for performance optimization, we should run through ``torch.compile`` with ``fullgraph=True`` first, and remove any unnecessary graph breaks. You can add ``TORCH_LOGS="output_code"`` when you run the script in order to see the inductor generated code. e.g. ``TORCH_LOGS="output_code" python example.py``::
  model = torch.compile(model, mode="max-autotune", fullgraph=True)

Serialization
~~~~~~~~~~~~~

Please checkout the `serialization doc <https://pytorch.org/ao/stable/serialization.html>`__ for more details.

.. note::
   We are integrated with huggingface transformer and supports serialization/deserialization through the huggingface save_pretrained/push_to_hub/from_pretrained APIs: https://huggingface.co/docs/transformers/main/en/quantization/torchao

.. note::
   Another example can be found in integration with diffuser: https://github.com/sayakpaul/diffusers-torchao/blob/main/inference/serialization_and_loading.md


Other Feature Support
~~~~~~~~~~~~~~~~~~~~~

The above just talks about basic feature support, we also provide examples on how to add supports for training, tensor parallel, FSDP by extending the `MyDTypeTensor <https://github.com/pytorch/ao/blob/main/tutorials/developer_api_guide/my_dtype_tensor_subclass.py>`__, we'll put more examples in `developer_api_guide <https://github.com/pytorch/ao/tree/main/tutorials/developer_api_guide>`__ folder covering the following use cases.

* `Quantized Training <https://github.com/pytorch/ao/blob/main/tutorials/developer_api_guide/my_trainable_tensor_subclass.py>`__
* `Tensor Parallel Support for Quantized Tensor <https://github.com/pytorch/ao/blob/main/tutorials/developer_api_guide/tensor_parallel.py>`__
* `Compatibility with executorch / torchchat <https://github.com/pytorch/ao/blob/main/tutorials/developer_api_guide/export_to_executorch.py>`__
* [TODO] FSDP
* [TODO] QAT


General Guide on Extending torchao
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a new use case, for example, a training dtype (like fp4 training), it's fine to start with adding a new tensor subclass in prototype folder `torchao/prototype <https://github.com/pytorch/ao/tree/main/torchao/prototype>`__, but you could also take a look at ``AffineQuantizedTensor`` if what you want to do is mostly supported there, e.g. adding int3 kernel for the exact same affine quantization. Please feel free to open an issue and if you have questions on what to do for a specific new use case.

To contribute to existing code base:

* Adding features to AffineQuantizedTensor, e.g. making it trainable, add tensor parallelism support etc.: `torchao/dtypes/affine_quantized_tensor.py <https://github.com/pytorch/ao/blob/main/torchao/dtypes/affine_quantized_tensor.py>`__
* Adding new quantization APIs: `torchao/quantization/quant_api.py <https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py>`__
* Adding new quantization primitive ops, e.g. slight variations of existing quantization primitive ops: `torchao/quantization/quant_primitives.py <https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_primitives.py>`__
* Adding new autotuned triton kernels: `torchao/kernel <https://github.com/pytorch/ao/tree/main/torchao/kernel>`__
* Adding new custom cpu/cuda/mps kernels: `torchao/csrc <https://github.com/pytorch/ao/tree/main/torchao/csrc>`__
* Integrating custom kernel with AffineQuantizedTensor (maybe a new layout as well): Add sparse marlin AQT layout `#621 <https://github.com/pytorch/ao/pull/621>`__ as an example. We are still not decided if we want to split ``AffineQuantizedTensor`` to more tensor subclasses or not.

Tensor Subclass Functionality/Composability Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We are also working on test suites to test out the functionalities of tensor subclass and the composability with different systems like torch.compile, DTensor etc. (we recommend to copy paste the tests and adapt to test your own tensor subclass for now):

* `Basic Test <https://github.com/pytorch/ao/blob/17a0a96d24ebfc154a23342b84e788d9ed6776f4/torchao/testing/utils.py#L74>`__
* `Compile Test <https://github.com/pytorch/ao/blob/17a0a96d24ebfc154a23342b84e788d9ed6776f4/torchao/testing/utils.py#L147>`__
* `Tensor Parallel Test <https://github.com/pytorch/ao/blob/17a0a96d24ebfc154a23342b84e788d9ed6776f4/torchao/testing/utils.py#L227>`__

Kernel Microbenchmarks
~~~~~~~~~~~~~~~~~~~~~~
Before we test performance on models, we can also do some microbenchmarks on single linear operator (or other compute intensive/memory intensive) operators with different input dimensions to get a sense of speedup. For a specific kernel that you'd like to benchmark, you can create a benchmark file like `benchmarks/benchmark_aq.py <https://github.com/pytorch/ao/blob/main/benchmarks/benchmark_aq.py>`__ and run benchmark with different shapes that's important for target model. A quick way to get the relevant shape for linear op and other ops is by running the example with `this <https://github.com/pytorch/ao/blob/main/tutorials/developer_api_guide/print_op_and_shapes.py>`__.

Change the model with the model you are interested in optimizing, and run the following::
  python tutorials/developer_api_guide/print_op_and_shapes.py

Example output::
  TORCH_FUNC=<built-in function linear> (M, K, N): 10 10 10
  TORCH_FUNC=<method 'add' of 'torch._C.TensorBase' objects> args[0] shape: torch.Size([10, 10])

  all linear shapes (M, K, N): [(10, 10, 10)]


The output of all linear shapes can be copy pasted to microbenchmarking script code under ``benchmarks/benchmark_your_kernel.py`` for benchmarking.

For benchmark helper functions, right now we have `1 <https://github.com/pytorch/ao/blob/0bdde92114b470823aa24725bf3b0811e980c8ce/torchao/utils.py#L55>`__ and `2 <https://github.com/pytorch/ao/blob/0bdde92114b470823aa24725bf3b0811e980c8ce/torchao/utils.py#L139>`__, feel free to use either one for now, but we'll probably keep one in the future.

Model Benchmarks and Eval
~~~~~~~~~~~~~~~~~~~~~~~~~

After you have the quantization flow implemented, you can run benchmark and eval on llama (llama2/llama3) or sam models that are already modified to be friendly to torch.compile, and compare with existing techniques in torchao.

Note: llama model (llama2/llama3) is our representative model for memory bound models and sam is our representative model for compute bound models.

* `llama <https://github.com/pytorch/ao/tree/main/torchao/_models/llama>`__
  * `benchmark <https://github.com/pytorch/ao/blob/main/torchao/_models/llama/generate.py>`__
  * `eval <https://github.com/pytorch/ao/blob/main/torchao/_models/llama/eval.py>`__
* `sam <https://github.com/pytorch/ao/tree/main/torchao/_models/sam>`__
  * `benchmark and eval <https://github.com/pytorch/ao/blob/main/torchao/_models/sam/eval_combo.py>`__

Please checkout the ``--help`` option for each of the script to understand the supported options, e.g. you can use ``--profile=profile_path`` to get the chrome trace of the run to understand detailed `chrome trace <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-tracing-functionality>`__.

Please let us know if there are any new important models that makes sense to be added to torchao model benchmark/eval folder.
