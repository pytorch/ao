Contributor Guide
-------------------------

General Guide on Extending torchao
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please start by reading our `quantization overview page <quantization_overview.html>`__ first.

To contribute to existing code base:

* Adding a new Tensor: `torchao/quantization/quantize_/workflows <https://github.com/pytorch/ao/tree/main/torchao/quantization/quantize_/workflows>`__
* Adding new quantization APIs: `torchao/quantization/quant_api.py <https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py>`__
* Adding features to existing Tensor subclasses like ``Float8Tensor``, e.g. adding new operator support, making it trainable, add tensor parallelism support etc., `tensor subclasses <https://github.com/pytorch/ao/tree/main/torchao/quantization/quantize_/workflows>`__, `tests <https://github.com/pytorch/ao/tree/main/test/quantization/quantize_/workflows>`__
* Adding new quantization primitive ops, e.g. slight variations of existing quantization primitive ops: `torchao/quantization/quant_primitives.py <https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_primitives.py>`__
* Adding new autotuned triton kernels: `torchao/kernel <https://github.com/pytorch/ao/tree/main/torchao/kernel>`__
* Adding new custom cpu/cuda/mps kernels: `torchao/csrc <https://github.com/pytorch/ao/tree/main/torchao/csrc>`__

Adding New Tensor Subclasses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torchao Tensor subclasses are structured by ``derived dtype`` and ``packing format``, please check out the `quantization overview page <quantization_overview.html>`__ to understand these concepts. If a new tensor subclass is needed for your use case, i.e. a new dtype, or a new packing format that does not already exist, we could define a new Tensor.

To understand how to use tensor subclass in the context of quantization, please also check `Writing Your Own Quantized Tensor <https://docs.pytorch.org/ao/main/subclass_basic.html>`__.

We have utility base class: ``torchao.utils.TorchAOBaseTensor`` that can help define common util functions and methods for you, if you specified the names of Tensor and non-Tensor attributes of the tensor subclass. for example::

  class MyTensor(TorchAOBaseTensor):
      tensor_data_names = ["qdata", "scale"]
      tensor_attribute_names = ["device", "dtype"]


With the above, we'll have multiple methods and functions available to use for this Tensor, for more details please check the docs for `TorchAOBaseTensor <https://docs.pytorch.org/ao/main/generated/torchao.utils.TorchAOBaseTensor.html#torchao.utils.TorchAOBaseTensor>`__

.. note::
   Many of the existing use cases in torchao still uses AffineQuantizedTensor, but we plan to move away from it to reduce the abstractions and make it easier for people to contribute to torchao.

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

Using hand written kernels in Tensor Subclasses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For calling optimized kernels, we have ``implements`` from the tensor subclass, for example, if we want to call into a new custom op: ``torch.ops.torchao.my_mm_for_mps``::

  class Float8Tensor(TorchAOBaseTensor):
      ...

  implements = Float8Tensor.implements

  @implements([torch.nn.functional.linear, aten.linear.default])
  def _(func, types, args, kwargs):
      ...
      # call into the custom op
      res = torch.ops.torchao.my_mm_for_mps(input_tensor.qdata, weight_tensor.qdata, input_tensor.scale, weight_tensor.scale)
      return res

KernelPreference
################

For some tensor subclasses, there could be multiple kernel choices for quantize and mm etc. The recommended way to handle this in torchao tensor subclasses is through ``KernelPreference``, that represents which group of kernels we want to use for quantize, mm, group_mm etc. We can use use ``KernelPreference.AUTO`` as default option, as the option for developers to choose whatever we think is the fastest under different conditions for user, so user don't need to worry about the details, and we can have other more specific kernel options for debugging purposes.

``Float8Tensor`` for example, has:

* ``KernelPreference.AUTO`` that will choose the most performant quantize and mm kernel based on hardware (H100 SM89 or SM90+), availability of libraries (whether ``fbgemm_gpu_genai`` is installed), granularity (per row or per tensor)
* ``KernelPreference.TORCH`` will use torchao quantize op (``_choose_scale_float8`` and ``_quantize_affine_float8``) and ``_scaled_mm``
* ``Kerenel.FBGEMM`` uses fbgemm quantize and mm op (``torch.ops.fbgemm.f8f8bf16_rowwise``)


Flow
~~~~

For model level API, people can reuse ``torchao.quantize_`` that allows people to apply a tensor subclass conversion to weight of linear, and allows `filtering function <https://docs.pytorch.org/ao/main/generated/torchao.quantization.quantize_.html#torchao.quantization.quantize_>`__ to choose which module the tensor subclass conversion should be applied to.

See Quantization Algorithms/Flows section for examples of weight only/dynamic quant and other types of model level APIs.

Using torch.compile for Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to be compatible with ``torch.compile``, to aim for performance optimization, we should run through ``torch.compile`` with ``fullgraph=True`` first, and remove any unnecessary graph breaks. You can add ``TORCH_LOGS="output_code"`` when you run the script in order to see the inductor generated code. e.g. ``TORCH_LOGS="output_code" python example.py``::

  model = torch.compile(model, mode="max-autotune", fullgraph=True)

Serialization
~~~~~~~~~~~~~

To enable support for serialization (torch.save and torch.load with tensor subclasses as weights), we need to add the tensor subclass and the relevant object to safe globals (available after torch 2.5), e.g.::
  torch.serialization.add_safe_globals([Float8Tensor, QuantizeTensorToFloat8Kwargs])

Please checkout the `serialization doc <serialization.html>`__ for more details.

.. note::
   We are `integrated <https://huggingface.co/docs/transformers/main/en/quantization/torchao>`__ with huggingface transformer and supports serialization and deserialization through the huggingface ``save_pretrained``, ``push_to_hub`` and ``from_pretrained`` APIs. We also have `serialization examples <https://github.com/sayakpaul/diffusers-torchao/blob/main/inference/serialization_and_loading.md>`__ with diffuser models.


Other Feature Support
~~~~~~~~~~~~~~~~~~~~~

The above just talks about basic feature support, we also provide examples on how to add supports for training, tensor parallel, FSDP by extending the `MyDTypeTensor <https://github.com/pytorch/ao/blob/main/tutorials/developer_api_guide/my_dtype_tensor_subclass.py>`__, we'll put more examples in `developer_api_guide <https://github.com/pytorch/ao/tree/main/tutorials/developer_api_guide>`__ folder covering the following use cases.

* `Quantized Training <https://github.com/pytorch/ao/blob/main/tutorials/developer_api_guide/my_trainable_tensor_subclass.py>`__
* `Tensor Parallel Support for Quantized Tensor <https://github.com/pytorch/ao/blob/main/tutorials/developer_api_guide/tensor_parallel.py>`__
* `Compatibility with executorch / torchchat <https://github.com/pytorch/ao/blob/main/tutorials/developer_api_guide/export_to_executorch.py>`__


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

Please also check out `Benchmarking User Guide <https://docs.pytorch.org/ao/main/benchmarking_user_guide.html>`__ and `Benchmarking API Guide <https://docs.pytorch.org/ao/main/benchmarking_api_guide.html>`__ to understand how to use our benchmarking framework.
