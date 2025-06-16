Inference Tutorial: From Quantization to Deployment
===================================================

This tutorial demonstrates how to perform post-training quantization and deploy models for inference using torchao's integration with popular frameworks. All quantization techniques shown here use torchao as the underlying optimization engine, seamlessly integrated through HuggingFace Transformers, vLLM, and ExecuTorch.

.. contents::
   :local:
   :depth: 2

Overview
--------

This tutorial covers the complete inference pipeline:

1. **Post-training Quantization**: Using float8 dynamic quantization with HuggingFace integration
2. **Sparsity**: Combining sparsity with quantization for additional speedups
3. **High-throughput Serving**: Deploying quantized models with vLLM
4. **Mobile Deployment**: Lowering to ExecuTorch for on-device inference

All these workflows leverage torchao's optimized kernels and quantization algorithms under the hood.

Post-training Quantization with HuggingFace
############################################

HuggingFace Transformers provides seamless integration with torchao quantization. The ``TorchAoConfig`` automatically applies torchao's optimized quantization algorithms during model loading.

Float8 Dynamic Quantization
------------------------------

Float8 dynamic quantization shows 36% reduction in model size with minimal accuracy loss:

.. code-block:: python

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    torch.random.manual_seed(0)

    model_path = "pytorch/Phi-4-mini-instruct-float8dq"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
        {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
    ]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    output = pipe(messages, **generation_args)
    print(output[0]['generated_text'])


Advanced: Per-Layer Quantization Control
----------------------------------------

For models where you need different quantization strategies for different layers:

.. code-block:: python

    from torchao.quantization import (
        IntxWeightOnlyConfig,
        Int8DynamicActivationIntxWeightConfig,
        ModuleFqnToConfig
    )
    from torchao.quantization.granularity import PerAxis, PerGroup

    # Different configs for different layer types
    embedding_config = IntxWeightOnlyConfig(
        weight_dtype=torch.int8,
        granularity=PerAxis(0)
    )

    linear_config = Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int4,
        weight_granularity=PerGroup(32),
        weight_scale_dtype=torch.bfloat16
    )

    # Map specific layers to configs - torchao applies optimizations per layer
    quant_config = ModuleFqnToConfig({
        "_default": linear_config,
        "model.embed_tokens": embedding_config,
        "lm_head": embedding_config
    })

    quantization_config = TorchAoConfig(
        quant_type=quant_config,
        include_embedding=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.float32,
        device_map="auto"
    )

Sparsity Integration
####################

Torchao's sparsity support can be combined with quantization for additional performance gains. The Marlin sparse layout provides optimized kernels for 2:4 structured sparsity.

Sparse + Quantized Models
-------------------------

.. code-block:: python

    from torchao.quantization import Int4WeightOnlyConfig
    from torchao.dtypes import MarlinSparseLayout

    # Combine sparsity with int4 quantization - both optimized by torchao
    quant_config = Int4WeightOnlyConfig(layout=MarlinSparseLayout())
    quantization_config = TorchAoConfig(quant_type=quant_config)

    # Load a pre-sparsified checkpoint
    model = AutoModelForCausalLM.from_pretrained(
        "nm-testing/Meta-Llama-3.1-8B-Instruct-W4A16-G128-2of4",  # 2:4 sparse model
        torch_dtype=torch.float16,
        device_map="cuda",
        quantization_config=quantization_config
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # Use static KV cache for best performance with torchao optimizations
    messages = [{"role": "user", "content": "What are the benefits of sparse neural networks?"}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

    outputs = model.generate(
        inputs,
        max_new_tokens=150,
        cache_implementation="static",  # Optimized for torchao
        do_sample=False
    )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(response)

High-throughput Serving with vLLM
##################################

vLLM automatically leverages torchao's optimized kernels when serving quantized models, providing significant throughput improvements.

Setting up vLLM with Quantized Models
--------------------------------------

First, install vLLM with torchao support:

.. code-block:: bash

    pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
    pip install torchao

Inference with vLLM
-------------------

.. code-block:: python

    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


    if __name__ == '__main__':
        # Create an LLM.
        llm = LLM(model="pytorch/Phi-4-mini-instruct-float8dq")
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = llm.generate(prompts, sampling_params)
        # Print the outputs.
        print("\nGenerated Outputs:\n" + "-" * 60)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt:    {prompt!r}")
            print(f"Output:    {generated_text!r}")
            print("-" * 60)


Serving Quantized Models
-----------------------------

.. code-block:: bash

    vllm serve pytorch/Phi-4-mini-instruct-float8dq --tokenizer microsoft/Phi-4-mini-instruct -O3


Performance Optimization Notes
------------------------------

When using vLLM with torchao:

- **Float8 dynamic quantization**: Provides 36% memory reduction with torchao's optimized kernels
- **Sparse models**: Additional ---- speedup speedup when combined with quantization
- **KV cache**:
- **Compile optimizations**:

Mobile Deployment with ExecuTorch
##################################

ExecuTorch enables on-device inference using torchao's mobile-optimized quantization schemes. The 8da4w (8-bit dynamic activation, 4-bit weight) configuration is specifically designed for mobile deployment.

Preparing Models for Mobile
----------------------------

**Step 1: Create Mobile-Optimized Quantization**

.. code-block:: python

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
    from torchao.quantization import (
        IntxWeightOnlyConfig,
        Int8DynamicActivationIntxWeightConfig,
        ModuleFqnToConfig
    )
    from torchao.quantization.granularity import PerAxis, PerGroup

    model_id = "microsoft/Phi-4-mini-instruct"

    # Mobile-optimized quantization scheme using torchao
    embedding_config = IntxWeightOnlyConfig(
        weight_dtype=torch.int8,
        granularity=PerAxis(0)
    )

    linear_config = Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int4,
        weight_granularity=PerGroup(32),
        weight_scale_dtype=torch.bfloat16
    )

    # 8da4w configuration optimized by torchao for mobile
    quant_config = ModuleFqnToConfig({
        "_default": linear_config,
        "model.embed_tokens": embedding_config
    })

    quantization_config = TorchAoConfig(
        quant_type=quant_config,
        include_embedding=True,
        untie_embedding_weights=True
    )

    # Load with mobile-optimized settings
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Required for mobile export
        quantization_config=quantization_config,
        device_map="cpu"  # Export from CPU
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Save quantized model
    model.save_pretrained("./phi4-mini-8da4w-mobile")
    tokenizer.save_pretrained("./phi4-mini-8da4w-mobile")

**Step 2: Export to ExecuTorch**

.. code-block:: bash

    # Install ExecuTorch
    git clone https://github.com/pytorch/executorch.git
    cd executorch
    ./install_requirements.sh

    # Convert checkpoint format for ExecuTorch
    .. Add code here..

    # Export to PTE format with torchao optimizations preserved
    python -m executorch.examples.models.llama.export_llama \
        --model "phi_4_mini" \
        --checkpoint "./phi4-mini-8da4w-mobile/pytorch_model_converted.bin" \
        --params "./phi4-mini-8da4w-mobile/config.json" \
        -kv \
        --use_sdpa_with_kv_cache \
        -X \
        --metadata '{"get_bos_id":199999, "get_eos_ids":[200020,199999]}' \
        --max_seq_length 512 \
        --max_context_length 512 \
        --output_name="phi4-mini-8da4w-mobile.pte"

Mobile Performance Characteristics
----------------------------------

The torchao-optimized 8da4w model provides:

- **Memory**: ~3.2GB on iPhone 15 Pro (vs ~12GB unquantized)
- **Speed**: ~17 tokens/sec on iPhone 15 Pro
- **Accuracy**: Maintained within 5-10% of original model on most benchmarks

**iOS Integration Example**:

.. code-block:: objective-c

    // Load the torchao-optimized PTE file
    NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"phi4-mini-8da4w-mobile" ofType:@"pte"];

    // ExecuTorch runtime automatically uses torchao's optimized kernels
    torch::executor::Result<torch::executor::Module> module_result =
        torch::executor::Module::load(modelPath.UTF8String);

Android integration follows similar patterns using the ExecuTorch Android API.

Evaluation and Benchmarking
############################

Model Quality Assessment
------------------------

Evaluate quantized models using lm-evaluation-harness:

.. code-block:: bash

    # Install evaluation framework
    pip install lm-eval[all]

    # Evaluate baseline model
    lm_eval --model hf \
            --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
            --tasks mmlu,arc_challenge,hellaswag,winogrande \
            --batch_size 8

    # Evaluate torchao-quantized model
    lm_eval --model hf \
            --model_args pretrained=nm-testing/Meta-Llama-3.1-8B-Instruct-W4A16-G128 \
            --tasks mmlu,arc_challenge,hellaswag,winogrande \
            --batch_size 8

Performance Benchmarking
------------------------

**Memory Usage Comparison**:

.. code-block:: python

    import torch
    from transformers import AutoModelForCausalLM
    import psutil
    import os

    def measure_memory_usage(model_id, quantization_config=None):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024 / 1024  # GB

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        mem_after = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        model_memory = mem_after - mem_before

        return model_memory

    # Compare memory usage
    baseline_memory = measure_memory_usage("meta-llama/Llama-3.1-8B-Instruct")

    from transformers import TorchAoConfig
    from torchao.quantization import Int4WeightOnlyConfig
    quant_config = TorchAoConfig(quant_type=Int4WeightOnlyConfig())
    quantized_memory = measure_memory_usage("meta-llama/Llama-3.1-8B-Instruct", quant_config)

    print(f"Baseline model: {baseline_memory:.2f} GB")
    print(f"Int4 quantized: {quantized_memory:.2f} GB")
    print(f"Memory reduction: {(1 - quantized_memory/baseline_memory)*100:.1f}%")

**Latency Benchmarking**:

.. code-block:: python

    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def benchmark_latency(model, tokenizer, prompt, num_runs=10):
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model.generate(inputs, max_new_tokens=100, do_sample=False)

        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(num_runs):
            with torch.no_grad():
                outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)

        torch.cuda.synchronize()
        end_time = time.time()

        avg_latency = (end_time - start_time) / num_runs
        tokens_generated = outputs.shape[1] - inputs.shape[1]
        throughput = tokens_generated / avg_latency

        return avg_latency, throughput

    # Benchmark both models
    prompt = "Explain the theory of relativity in simple terms."

    baseline_latency, baseline_throughput = benchmark_latency(baseline_model, tokenizer, prompt)
    quantized_latency, quantized_throughput = benchmark_latency(quantized_model, tokenizer, prompt)

    print(f"Baseline: {baseline_latency:.3f}s ({baseline_throughput:.1f} tok/s)")
    print(f"Quantized: {quantized_latency:.3f}s ({quantized_throughput:.1f} tok/s)")
    print(f"Speedup: {baseline_latency/quantized_latency:.2f}x")


Conclusion
##########

This tutorial demonstrated how torchao's quantization and sparsity techniques integrate seamlessly across the entire ML deployment stack:

- **HuggingFace Transformers** provides easy model loading with torchao quantization
- **vLLM** leverages torchao's optimized kernels for high-throughput serving
- **ExecuTorch** enables mobile deployment with torchao's mobile-optimized schemes

All these frameworks use torchao as the underlying optimization engine, ensuring consistent performance gains and ease of integration. The quantization techniques shown provide significant memory reduction (3-4x) and performance improvements (1.5-2x) while maintaining model quality within acceptable bounds for most applications.

For production deployments, always benchmark on your specific use case and hardware to validate the performance and accuracy trade-offs.
