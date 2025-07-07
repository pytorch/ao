Microbenchmarking Tutorial
==========================

This tutorial will guide you through using the TorchAO microbenchmarking framework. The tutorial contains different use cases for benchmarking your API and integrating with the dashboard.

1. Add an API to benchmarking recipes
2. Add a model to benchmarking recipes
3. Benchmark your API locally
4. Add an API to benchmarking CI dashboard

1. Add an API to Benchmarking Recipes
--------------------------------------

The framework currently supports quantization and sparsity recipes, which can be run using the quantize_() or sparsity_() functions:

To add a new recipe, add the corresponding string configuration to the function ``string_to_config()`` in ``benchmarks/microbenchmarks/utils.py``.

.. code-block:: python

  def string_to_config(
    quantization: Optional[str], sparsity: Optional[str], **kwargs
  ) -> AOBaseConfig:

  # ... existing code ...

  elif quantization == "my_new_quantization":
    # If additional information needs to be passed as kwargs, process it here
    return MyNewQuantizationConfig(**kwargs)
  elif sparsity == "my_new_sparsity":
    return MyNewSparsityConfig(**kwargs)

  # ... rest of existing code ...

Now we can use this recipe throughout the benchmarking framework.

.. note::

  If the ``AOBaseConfig`` uses input parameters, like bit-width, group-size etc, you can pass them appended to the string config in input
  For example, for ``GemliteUIntXWeightOnlyConfig`` we can pass it-width and group-size as ``gemlitewo-<bit_width>-<group_size>``

2. Add a Model to Benchmarking Recipes
---------------------------------------

To add a new model architecture to the benchmarking system, you need to modify ``torchao/testing/model_architectures.py``.

1. To add a new model type, define your model class in ``torchao/testing/model_architectures.py``:

.. code-block:: python

    class MyCustomModel(torch.nn.Module):
        def __init__(self, input_dim, output_dim, dtype=torch.bfloat16):
            super().__init__()
            # Define your model architecture
            self.layer1 = torch.nn.Linear(input_dim, 512, bias=False).to(dtype)
            self.activation = torch.nn.ReLU()
            self.layer2 = torch.nn.Linear(512, output_dim, bias=False).to(dtype)

        def forward(self, x):
            x = self.layer1(x)
            x = self.activation(x)
            x = self.layer2(x)
            return x

2. Update the ``create_model_and_input_data`` function to handle your new model type:

.. code-block:: python

    def create_model_and_input_data(
        model_type: str,
        m: int,
        k: int,
        n: int,
        high_precision_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        activation: str = "relu",
    ):
        # ... existing code ...

        elif model_type == "my_custom_model":
            model = MyCustomModel(k, n, high_precision_dtype).to(device)
            input_data = torch.randn(m, k, device=device, dtype=high_precision_dtype)

        # ... rest of existing code ...

**Model Design Considerations**

When adding new models:

- **Input/Output Dimensions**: Ensure your model handles the (m, k, n) dimension convention where:

  - ``m``: Batch size or sequence length
  - ``k``: Input feature dimension
  - ``n``: Output feature dimension

- **Data Types**: Support the ``high_precision_dtype`` parameter (typically ``torch.bfloat16``)

- **Device Compatibility**: Ensure your model works on CUDA, CPU, and other target devices

- **Quantization Compatibility**: Design your model to work with TorchAO quantization methods

3. Benchmark Your API Locally
------------------------------

For local development and testing:

3.1 Quick Start
~~~~~~~~~~~~~~~

Create a minimal configuration for local testing:

.. code-block:: yaml

    # local_test.yml
    benchmark_mode: "inference"
    quantization_config_recipe_names:
      - "baseline"
      - "int8wo"
      # Add your recipe here

    output_dir: "local_results" # Add your output directory here

    model_params:
      # Add your model configurations here
      - name: "quick_test"
        matrix_shapes:
          # Define a custom shape, or use one of the predefined shape generators
          - name: "custom"
            shapes: [[1024, 1024, 1024]]
        high_precision_dtype: "torch.bfloat16"
        use_torch_compile: true
        device: "cuda"
        model_type: "linear"

.. note::
  - For a list of latest supported config recipes for quantization or sparsity, please refer to ``benchmarks/microbenchmarks/README.md``.
  - For a list of all model types, please refer to ``torchao/testing/model_architectures.py``.

3.2 Run Local Benchmark
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -m benchmarks.microbenchmarks.benchmark_runner --config local_test.yml

3.3 Analysing the Output
~~~~~~~~~~~~~~~~~~~~~~~~

The output generated after running the benchmarking script, is the form of a csv. It'll contain the following:
 - time for inference for running baseline model and quantized model
 - speedup in inference time in quantized model
 - compile or eager mode
 - if enabled, memory snapshot and gpu chrome trace

4. Add an API to Benchmarking CI Dashboard
------------------------------------------

To integrate your API with the CI `dashboard <https://hud.pytorch.org/benchmark/llms?repoName=pytorch%2Fao&benchmarkName=micro-benchmark+api>`_:

4.1 Modify Existing CI Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add your quantization method to the existing CI configuration file at ``benchmarks/dashboard/microbenchmark_quantization_config.yml``:

.. code-block:: yaml

    # benchmarks/dashboard/microbenchmark_quantization_config.yml
    benchmark_mode: "inference"
    quantization_config_recipe_names:
      - "int8wo"
      - "int8dq"
      - "float8dq-tensor"
      - "float8dq-row"
      - "float8wo"
      - "my_new_quantization"  # Add your method here

    output_dir: "benchmarks/microbenchmarks/results"

    model_params:
      - name: "small_bf16_linear"
        matrix_shapes:
          - name: "small_sweep"
            min_power: 10
            max_power: 15
        high_precision_dtype: "torch.bfloat16"
        use_torch_compile: true
        torch_compile_mode: "max-autotune"
        device: "cuda"
        model_type: "linear"

4.2 Run CI Benchmarks
~~~~~~~~~~~~~~~~~~~~~

Use the CI runner to generate results in PyTorch OSS benchmark database format:

.. code-block:: bash

    python benchmarks/dashboard/ci_microbenchmark_runner.py \
        --config benchmarks/dashboard/microbenchmark_quantization_config.yml \
        --output benchmark_results.json

4.3 CI Output Format
~~~~~~~~~~~~~~~~~~~~

The CI runner outputs results in a specific JSON format required by the PyTorch OSS benchmark database:

.. code-block:: json

    [
      {
        "benchmark": {
          "name": "micro-benchmark api",
          "mode": "inference",
          "dtype": "int8wo",
          "extra_info": {
            "device": "cuda",
            "arch": "NVIDIA A100-SXM4-80GB"
          }
        },
        "model": {
          "name": "1024-1024-1024",
          "type": "micro-benchmark custom layer",
          "origins": ["torchao"]
        },
        "metric": {
          "name": "speedup(wrt bf16)",
          "benchmark_values": [1.25],
          "target_value": 0.0
        },
        "runners": [],
        "dependencies": {}
      }
    ]

4.4 Integration with CI Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To integrate with your CI pipeline, add the benchmark step to your workflow:

.. code-block:: yaml

    # Example GitHub Actions step
    - name: Run Microbenchmarks
      run: |
        python benchmarks/dashboard/ci_microbenchmark_runner.py \
          --config benchmarks/dashboard/microbenchmark_quantization_config.yml \
          --output benchmark_results.json

    - name: Upload Results
      # Upload benchmark_results.json to your dashboard system

Advanced Usage
--------------

Multiple Model Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can benchmark multiple model configurations in a single run:

.. code-block:: yaml

    model_params:
      - name: "small_models"
        matrix_shapes:
          - name: "pow2"
            min_power: 10
            max_power: 12
        model_type: "linear"
        device: "cuda"

      - name: "transformer_models"
        matrix_shapes:
          - name: "llama"
        model_type: "transformer_block"
        device: "cuda"

      - name: "cpu_models"
        matrix_shapes:
          - name: "custom"
            shapes: [[512, 512, 512]]
        model_type: "linear"
        device: "cpu"

Running Tests
~~~~~~~~~~~~~

To verify your setup and run the test suite:

.. code-block:: bash

    python -m unittest discover benchmarks/microbenchmarks/test

Interpreting Results
~~~~~~~~~~~~~~~~~~~~

The benchmark results include:

- **Speedup**: Performance improvement compared to baseline (bfloat16)
- **Memory Usage**: Peak memory consumption during inference
- **Latency**: Time taken for inference operations
- **Profiling Data**: Detailed performance traces (when enabled)

Results are saved in CSV format with columns for:

- Model configuration
- Quantization method
- Shape dimensions (M, K, N)
- Performance metrics
- Device information

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **CUDA Out of Memory**: Reduce batch size or matrix dimensions
2. **Compilation Errors**: Set ``use_torch_compile: false`` for debugging
3. **Missing Quantization Methods**: Ensure TorchAO is properly installed
4. **Device Not Available**: Check device availability and drivers

Best Practices
~~~~~~~~~~~~~~

1. Always include a baseline configuration for comparison
2. Use ``small_sweep`` for initial testing, ``sweep`` for comprehensive analysis
3. Enable profiling only when needed (adds overhead)
4. Test on multiple devices when possible
5. Use consistent naming conventions for reproducibility

For more detailed information about the framework components, see the README files in the ``benchmarks/microbenchmarks/`` directory.
