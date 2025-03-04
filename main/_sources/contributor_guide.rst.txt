Contributor Guide
-------------------------

General Guide on Extending torchao
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a new use case, for example, a training dtype (like fp4 training), it's fine to start with adding a new tensor subclass in prototype folder `torchao/prototype <https://github.com/pytorch/ao/tree/main/torchao/prototype>`__, but you could also take a look at ``AffineQuantizedTensor`` if what you want to do is mostly supported there, e.g. adding int3 kernel for the exact same affine quantization. Please feel free to open an issue and if you have questions on what to do for a specific new use case. For more details, please refer to our `quantization overview page <quantization.html>`__.

To contribute to existing code base:

* Adding features to AffineQuantizedTensor, e.g. making it trainable, add tensor parallelism support etc.: `torchao/dtypes/affine_quantized_tensor.py <https://github.com/pytorch/ao/blob/main/torchao/dtypes/affine_quantized_tensor.py>`__
* Adding new quantization APIs: `torchao/quantization/quant_api.py <https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py>`__
* Adding new quantization primitive ops, e.g. slight variations of existing quantization primitive ops: `torchao/quantization/quant_primitives.py <https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_primitives.py>`__
* Adding new autotuned triton kernels: `torchao/kernel <https://github.com/pytorch/ao/tree/main/torchao/kernel>`__
* Adding new custom cpu/cuda/mps kernels: `torchao/csrc <https://github.com/pytorch/ao/tree/main/torchao/csrc>`__
* Integrating custom kernel with AffineQuantizedTensor (maybe a new layout as well): Add sparse marlin AQT layout `#621 <https://github.com/pytorch/ao/pull/621>`__ as an example. We are still not decided if we want to split ``AffineQuantizedTensor`` to more tensor subclasses or not.

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

* `llama <https://github.com/pytorch/ao/tree/main/benchmarks/_models/llama>`__
  * `benchmark <https://github.com/pytorch/ao/blob/main/benchmarks/_models/llama/generate.py>`__
  * `eval <https://github.com/pytorch/ao/blob/main/benchmarks/_models/llama/eval.py>`__
* `sam <https://github.com/pytorch/ao/tree/main/benchmarks/_models/sam>`__
  * `benchmark and eval <https://github.com/pytorch/ao/blob/main/benchmarks/_models/sam/eval_combo.py>`__

Please checkout the ``--help`` option for each of the script to understand the supported options, e.g. you can use ``--profile=profile_path`` to get the chrome trace of the run to understand detailed `chrome trace <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-tracing-functionality>`__.

Please let us know if there are any new important models that makes sense to be added to torchao model benchmark/eval folder.
