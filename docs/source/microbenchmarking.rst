Microbenchmarking Tutorial
==========================

This tutorial will guide you through using the TorchAO microbenchmarking framework. The tutorial contains different use cases for benchmarking your API and integrating with the dashboard.

1. Add an API to benchmarking recipes
2. Add a model to benchmarking recipes
3. Benchmark your API locally
4. Add an API to benchmarking CI dashboard

1. Add an API to Benchmarking Recipes
--------------------------------------

To add a new quantization API to the benchmarking system, you need to ensure your quantization method is available in the TorchAO quantization recipes.

1.1 Supported Quantization Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework currently supports these quantization types:

- ``baseline``: No quantization (bfloat16 reference)
- ``int8wo``: 8-bit weight-only quantization
- ``int8dq``: 8-bit dynamic quantization
- ``int4wo-{group_size}``: 4-bit weight-only quantization with specified group size
- ``int4wo-{group_size}-hqq``: 4-bit weight-only quantization with HQQ
- ``float8wo``: Float8 weight-only quantization
- ``float8dq-tensor``: Float8 dynamic quantization (tensor-wise)
- ``float8dq-row``: Float8 dynamic quantization (row-wise)
- ``gemlitewo-{bit_width}-{group_size}``: 4 or 8 bit integer quantization with gemlite triton kernel

1.2 Adding a New Quantization Recipe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add a new quantization method:

1. **Implement your quantization function** in the appropriate TorchAO module (e.g., ``torchao/quantization/``)

2. **Add the recipe to the quantization system** by ensuring it can be called with the same interface as existing methods

3. **Test your quantization method** with a simple benchmark configuration:

.. code-block:: yaml

    # test_my_quantization.yml
    benchmark_mode: "inference"
    quantization_config_recipe_names:
      - "baseline"
      - "my_new_quantization"  # Your new method

    output_dir: "test_results"

    model_params:
      - name: "test_linear"
        matrix_shapes:
          - name: "custom"
            shapes: [[1024, 1024, 1024]]
        high_precision_dtype: "torch.bfloat16"
        use_torch_compile: false
        device: "cuda"
        model_type: "linear"

4. **Verify the integration** by running:

.. code-block:: bash

    python -m benchmarks.microbenchmarks.benchmark_runner --config test_my_quantization.yml

2. Add a Model to Benchmarking Recipes
---------------------------------------

To add a new model architecture to the benchmarking system, you need to modify ``torchao/testing/model_architectures.py``.

2.1 Current Model Types
~~~~~~~~~~~~~~~~~~~~~~~

The framework supports these model types:

- ``linear``: Simple linear layer (``ToyLinearModel``)
- ``ln_linear_<activation>``: LayerNorm + Linear + Activation (``LNLinearActivationModel``)

  - ``ln_linear_sigmoid``: LayerNorm + Linear + Sigmoid
  - ``ln_linear_relu``: LayerNorm + Linear + ReLU
  - ``ln_linear_gelu``: LayerNorm + Linear + GELU
  - ``ln_linear_silu``: LayerNorm + Linear + SiLU
  - ``ln_linear_leakyrelu``: LayerNorm + Linear + LeakyReLU
  - ``ln_linear_relu6``: LayerNorm + Linear + ReLU6
  - ``ln_linear_hardswish``: LayerNorm + Linear + Hardswish

- ``transformer_block``: Transformer block with self-attention and MLP (``TransformerBlock``)

2.2 Adding a New Model Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add a new model type:

1. **Define your model class** in ``torchao/testing/model_architectures.py``:

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

2. **Update the** ``create_model_and_input_data`` **function** to handle your new model type:

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

3. **Test your new model** with a benchmark configuration:

.. code-block:: yaml

    # test_my_model.yml
    benchmark_mode: "inference"
    quantization_config_recipe_names:
      - "baseline"
      - "int8wo"

    output_dir: "test_results"

    model_params:
      - name: "test_my_custom_model"
        matrix_shapes:
          - name: "custom"
            shapes: [[1024, 1024, 1024]]
        high_precision_dtype: "torch.bfloat16"
        use_torch_compile: false
        device: "cuda"
        model_type: "my_custom_model"  # Your new model type

4. **Verify the integration**:

.. code-block:: bash

    python -m benchmarks.microbenchmarks.benchmark_runner --config test_my_model.yml

2.3 Model Design Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    output_dir: "local_results"

    model_params:
      - name: "quick_test"
        matrix_shapes:
          - name: "custom"
            shapes: [[1024, 1024, 1024]]
        high_precision_dtype: "torch.bfloat16"
        use_torch_compile: false  # Disable for faster iteration
        device: "cuda"
        model_type: "linear"

3.2 Run Local Benchmark
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -m benchmarks.microbenchmarks.benchmark_runner --config local_test.yml

3.3 Shape Generation Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use different shape generation strategies:

**Custom Shapes:**

.. code-block:: yaml

    matrix_shapes:
      - name: "custom"
        shapes: [
          [1024, 1024, 1024],  # [m, k, n]
          [2048, 4096, 1024]
        ]

**LLaMa Model Shapes:**

.. code-block:: yaml

    matrix_shapes:
      - name: "llama"  # Uses LLaMa 2 70B single-node weight shapes

**Power of 2 Shapes:**

.. code-block:: yaml

    matrix_shapes:
      - name: "pow2"
        min_power: 10  # 2^10 = 1024
        max_power: 12  # 2^12 = 4096

**Extended Power of 2 Shapes:**

.. code-block:: yaml

    matrix_shapes:
      - name: "pow2_extended"
        min_power: 10  # Generates: 1024, 1536, 2048, 3072, etc.
        max_power: 11

**Small Sweep (for heatmaps):**

.. code-block:: yaml

    matrix_shapes:
      - name: "small_sweep"
        min_power: 10
        max_power: 15

**Full Sweep:**

.. code-block:: yaml

    matrix_shapes:
      - name: "sweep"
        min_power: 8
        max_power: 9

3.4 Enable Profiling for Debugging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For detailed performance analysis, enable profiling:

.. code-block:: yaml

    model_params:
      - name: "debug_model"
        # ... other parameters ...
        enable_profiler: true        # Enable standard profiling
        enable_memory_profiler: true # Enable CUDA memory profiling

This will generate:

- Standard PyTorch profiler traces
- CUDA memory snapshots and visualizations
- Memory usage analysis in the ``memory_profiler`` subdirectory

3.5 Device Options
~~~~~~~~~~~~~~~~~~

Test on different devices:

.. code-block:: yaml

    device: "cuda"  # NVIDIA GPU
    # device: "xpu"   # Intel GPU
    # device: "mps"   # Apple Silicon GPU
    # device: "cpu"   # CPU fallback

3.6 Compilation Options
~~~~~~~~~~~~~~~~~~~~~~

Control PyTorch compilation for performance tuning:

.. code-block:: yaml

    use_torch_compile: true
    torch_compile_mode: "max-autotune"  # Options: "default", "max-autotune", "false"

4. Add an API to Benchmarking CI Dashboard
------------------------------------------

To integrate your API with the continuous integration dashboard:

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
