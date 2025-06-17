##################################################
Inference Tutorial: From Quantization to Serving
##################################################

This tutorial demonstrates how to perform post-training quantization and deploy models for inference:

1. :ref:`Post-training Quantization with HuggingFace`: Using float8 dynamic quantization with HuggingFace integration
2. :ref:`High-throughput Serving with vLLM`: Deploying quantized models with vLLM
3. :ref:`Mobile Deployment with Executorch`: Lowering to ExecuTorch for on-device inference

All techniques shown here use torchao as the underlying optimization engine, seamlessly integrated through HuggingFace Transformers, vLLM, and ExecuTorch.

Post-training Quantization with HuggingFace
############################################

HuggingFace Transformers provides seamless integration with torchao quantization. The ``TorchAoConfig`` automatically applies torchao's optimized quantization algorithms during model loading.

Float8 Dynamic Quantization
------------------------------

Float8 dynamic quantization shows 36% reduction in model size with minimal accuracy loss:

.. code-block:: python

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

    model_id = "microsoft/Phi-4-mini-instruct"

    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow
    quant_config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    quantization_config = TorchAoConfig(quant_type=quant_config)
    quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Push to hub
    USER_ID = "YOUR_USER_ID"
    MODEL_NAME = model_id.split("/")[-1]
    save_to = f"{USER_ID}/{MODEL_NAME}-float8dq"
    quantized_model.push_to_hub(save_to, safe_serialization=False)
    tokenizer.push_to_hub(save_to)

    # Manual Testing
    prompt = "Hey, are you conscious? Can you talk to me?"
    messages = [
        {
            "role": "system",
            "content": "",
        },
        {"role": "user", "content": prompt},
    ]
    templated_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print("Prompt:", prompt)
    print("Templated prompt:", templated_prompt)
    inputs = tokenizer(
        templated_prompt,
        return_tensors="pt",
    ).to("cuda")
    generated_ids = quantized_model.generate(**inputs, max_new_tokens=128)
    output_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("Response:", output_text[0][len(prompt):])


[Optional] Float8 Dynamic Quantization + Semi-structured (2:4) sparsity
--------------------------------------------------------------------------

Torchao's sparsity support can be combined with quantization for additional performance gains. The Marlin sparse layout provides optimized kernels for 2:4 structured sparsity.

.. code-block:: python

    from torchao.quantization import Float8DynamicActivationFloat8SemiSparseWeightConfig

    # Combine sparsity with int4 quantization - both optimized by torchao
    quant_config = Float8DynamicActivationFloat8SemiSparseWeightConfig()
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

For more information on supported quantization and sparsity configurations, see https://huggingface.co/docs/transformers/main/en/quantization/torchao.

Inference with Transformers
---------------------------

Install the required packages:
.. code-block:: bash
    pip install git+https://github.com/huggingface/transformers@main
    pip install torchao
    pip install torch
    pip install accelerate

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

Model Quality Assessment
------------------------

Evaluate quantized models using lm-evaluation-harness:

.. code-block:: bash

    # Install evaluation framework
    # Need to install lm-eval from source: https://github.com/EleutherAI/lm-evaluation-harness#install

    # Evaluate baseline model
    lm_eval --model hf --model_args pretrained=microsoft/Phi-4-mini-instruct --tasks hellaswag --device cuda:0 --batch_size 8

    # Evaluate torchao-quantized model (float8dq)
    lm_eval --model hf --model_args pretrained=pytorch/Phi-4-mini-instruct-float8dq --tasks hellaswag --device cuda:0 --batch_size 8


Performance Benchmarking
------------------------------

**Latency Benchmarking**:

.. code-block:: bash

    # baseline
    python benchmarks/benchmark_latency.py --input-len 256 --output-len 256 --model microsoft/Phi-4-mini-instruct --batch-size 1

    # float8dq
    VLLM_DISABLE_COMPILE_CACHE=1 python benchmarks/benchmark_latency.py --input-len 256 --output-len 256 --model pytorch/Phi-4-mini-instruct-float8dq --batch-size 1

**Results (H100 machine)**

+----------------------------+----------------+------------------------------+
| Benchmark                  | Phi-4 mini-Ins | Phi-4-mini-instruct-float8dq |
+============================+================+==============================+
| latency (batch_size=1)     | 1.64s          | 1.41s (1.16x speedup)        |
+----------------------------+----------------+------------------------------+
| latency (batch_size=128)   | 3.1s           | 2.72s (1.14x speedup)        |
+----------------------------+----------------+------------------------------+


Memory Benchmarking
--------------------

**Memory Usage Comparison**:

.. code-block:: python

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

    # use "microsoft/Phi-4-mini-instruct" or "pytorch/Phi-4-mini-instruct-float8dq"
    model_id = "pytorch/Phi-4-mini-instruct-float8dq"
    quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    torch.cuda.reset_peak_memory_stats()

    prompt = "Hey, are you conscious? Can you talk to me?"
    messages = [
        {
            "role": "system",
            "content": "",
        },
        {"role": "user", "content": prompt},
    ]
    templated_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print("Prompt:", prompt)
    print("Templated prompt:", templated_prompt)
    inputs = tokenizer(
        templated_prompt,
        return_tensors="pt",
    ).to("cuda")
    generated_ids = quantized_model.generate(**inputs, max_new_tokens=128)
    output_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("Response:", output_text[0][len(prompt):])

    mem = torch.cuda.max_memory_reserved() / 1e9
    print(f"Peak Memory Usage: {mem:.02f} GB")


+-------------------+----------------+------------------------------+
| Benchmark         | Phi-4 mini-Ins | Phi-4-mini-instruct-float8dq |
+===================+================+==============================+
| Peak Memory (GB)  | 8.91           | 5.70 (36% reduction)         |
+-------------------+----------------+------------------------------+

Performance Breakdown
------------------------------
When using vLLM with torchao:

- **Float8 dynamic quantization**: Provides 36% VRAM reduction, 1.15x-1.2x speedup and little to no accuracy impact on H100
- **Sparsity Support**: Semi-structured (2:4) sparsity for faster inference (see [Accelerating Neural Network Training with Semi-Structured (2:4) Sparsity](https://pytorch.org/blog/accelerating-neural-network-training/) blog post)
- **KV Cache Quantization**: Enables long context inference with lower memory (see [KV Cache Quantization](https://github.com/pytorch/ao/blob/main/torchao/_models/llama/README.md))

High-throughput Serving with vLLM
##################################

vLLM automatically leverages torchao's optimized kernels when serving quantized models, providing significant throughput improvements.

Setting up vLLM with Quantized Models
--------------------------------------

First, install vLLM with torchao support:

.. code-block:: bash

    pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
    pip install torchao

Serving Quantized Models
-----------------------------

.. code-block:: bash

    vllm serve pytorch/Phi-4-mini-instruct-float8dq --tokenizer microsoft/Phi-4-mini-instruct -O3


Serving Performance Benchmarking
--------------------------------

**Serving Benchmarking**:

We benchmarked the throughput in a serving environment.

.. code-block:: bash

    # Setup: Get vllm source code
    git clone git@github.com:vllm-project/vllm.git

    # Install vllm
    VLLM_USE_PRECOMPILED=1 pip install --editable .

    # Run the benchmarks under vllm root folder:

    # Download sharegpt dataset:
    wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

    # Other datasets can be found in: https://github.com/vllm-project/vllm/tree/main/benchmarks
    # Note: you can change the number of prompts to be benchmarked with --num-prompts argument for benchmark_serving script.

    # For baseline
    # Server:
    vllm serve microsoft/Phi-4-mini-instruct --tokenizer microsoft/Phi-4-mini-instruct -O3
    # Client:
    python benchmarks/benchmark_serving.py --backend vllm --dataset-name sharegpt --tokenizer microsoft/Phi-4-mini-instruct --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --model microsoft/Phi-4-mini-instruct --num-prompts 1

    # For float8dq
    # Server:
    VLLM_DISABLE_COMPILE_CACHE=1 vllm serve pytorch/Phi-4-mini-instruct-float8dq --tokenizer microsoft/Phi-4-mini-instruct -O3
    # Client:
    python benchmarks/benchmark_serving.py --backend vllm --dataset-name sharegpt --tokenizer microsoft/Phi-4-mini-instruct --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --model pytorch/Phi-4-mini-instruct-float8dq --num-prompts 1

**Results (H100 machine)**

+----------------------------+----------------+------------------------------+
| Benchmark                  | Phi-4 mini-Ins | Phi-4-mini-instruct-float8dq |
+============================+================+==============================+
| serving (num_prompts=1)    | 1.35 req/s     | 1.57 req/s (1.16x speedup)   |
+----------------------------+----------------+------------------------------+
| serving (num_prompts=1000) | 66.68 req/s    | 80.53 req/s (1.21x speedup)  |
+----------------------------+----------------+------------------------------+

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

Mobile Deployment with ExecuTorch
##################################

ExecuTorch enables on-device inference using torchao's mobile-optimized quantization schemes. The 8da4w (8-bit dynamic activation, 4-bit weight) configuration is specifically designed for mobile deployment.

**Step 1: Untie Embedding Weights**

We want to quantize the embedding and lm_head differently. Since those layers are tied, we first need to untie the model:

.. code-block:: python

    from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    )
    import torch

    model_id = "microsoft/Phi-4-mini-instruct"
    untied_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(untied_model)
    from transformers.modeling_utils import find_tied_parameters
    print("tied weights:", find_tied_parameters(untied_model))
    if getattr(untied_model.config.get_text_config(decoder=True), "tie_word_embeddings"):
        setattr(untied_model.config.get_text_config(decoder=True), "tie_word_embeddings", False)

    untied_model._tied_weights_keys = []
    untied_model.lm_head.weight = torch.nn.Parameter(untied_model.lm_head.weight.clone())

    print("tied weights:", find_tied_parameters(untied_model))

    USER_ID = "YOUR_USER_ID"
    MODEL_NAME = model_id.split("/")[-1]
    save_to = f"{USER_ID}/{MODEL_NAME}-untied-weights"

    untied_model.push_to_hub(save_to)
    tokenizer.push_to_hub(save_to)

    # or save locally
    save_to_local_path = f"{MODEL_NAME}-untied-weights"
    untied_model.save_pretrained(save_to_local_path)
    tokenizer.save_pretrained(save_to)

**Step 2: Create Mobile-Optimized Quantization**

Quantizing the model for mobile deployment using torchao's Int8DynamicActivationIntxWeightConfig configuration:

.. code-block:: python

    from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    TorchAoConfig,
    )
    from torchao.quantization.quant_api import (
        IntxWeightOnlyConfig,
        Int8DynamicActivationIntxWeightConfig,
        ModuleFqnToConfig,
        quantize_,
    )
    from torchao.quantization.granularity import PerGroup, PerAxis
    import torch

    # we start from the model with untied weights
    model_id = "microsoft/Phi-4-mini-instruct"
    USER_ID = "YOUR_USER_ID"
    MODEL_NAME = model_id.split("/")[-1]
    untied_model_id = f"{USER_ID}/{MODEL_NAME}-untied-weights"
    untied_model_local_path = f"{MODEL_NAME}-untied-weights"

    embedding_config = IntxWeightOnlyConfig(
        weight_dtype=torch.int8,
        granularity=PerAxis(0),
    )
    linear_config = Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int4,
        weight_granularity=PerGroup(32),
        weight_scale_dtype=torch.bfloat16,
    )
    quant_config = ModuleFqnToConfig({"_default": linear_config, "model.embed_tokens": embedding_config})
    quantization_config = TorchAoConfig(quant_type=quant_config, include_embedding=True, untie_embedding_weights=True, modules_to_not_convert=[])

    # either use `untied_model_id` or `untied_model_local_path`
    quantized_model = AutoModelForCausalLM.from_pretrained(untied_model_id, torch_dtype=torch.float32, device_map="auto", quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Push to hub
    MODEL_NAME = model_id.split("/")[-1]
    save_to = f"{USER_ID}/{MODEL_NAME}-8da4w"
    quantized_model.push_to_hub(save_to, safe_serialization=False)
    tokenizer.push_to_hub(save_to)

    # Manual testing
    prompt = "Hey, are you conscious? Can you talk to me?"
    messages = [
        {
            "role": "system",
            "content": "",
        },
        {"role": "user", "content": prompt},
    ]
    templated_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print("Prompt:", prompt)
    print("Templated prompt:", templated_prompt)
    inputs = tokenizer(
        templated_prompt,
        return_tensors="pt",
    ).to("cuda")
    generated_ids = quantized_model.generate(**inputs, max_new_tokens=128)
    output_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("Response:", output_text[0][len(prompt):])


**Step 3: Export to ExecuTorch**

.. code-block:: bash

    # Install ExecuTorch
    git clone https://github.com/pytorch/executorch.git
    cd executorch
    ./install_requirements.sh

    # Convert checkpoint format for ExecuTorch
    python -m executorch.examples.models.phi_4_mini.convert_weights pytorch_model.bin pytorch_model_converted.bin

    # Export to PTE format with torchao optimizations preserved
    PARAMS="executorch/examples/models/phi_4_mini/config.json"
    python -m executorch.examples.models.llama.export_llama \
        --model "phi_4_mini" \
        --checkpoint "pytorch_model_converted.bin" \
        --params "$PARAMS" \
        -kv \
        --use_sdpa_with_kv_cache \
        -X \
        --metadata '{"get_bos_id":199999, "get_eos_ids":[200020,199999]}' \
        --max_seq_length 128 \
        --max_context_length 128 \
        --output_name="phi4-mini-8da4w.pte"


Mobile Performance Characteristics
----------------------------------

The torchao-optimized 8da4w model provides:

- **Memory**: ~3.2GB on iPhone 15 Pro
- **Speed**: ~17 tokens/sec on iPhone 15 Pro
- **Accuracy**: Maintained within 5-10% of original model on most benchmarks

For detailed instructions on testing the executorch model and reproducing benchmarks please refer to the [HF Phi-4-mini-instruct-8da4w model](https://huggingface.co/pytorch/Phi-4-mini-instruct-8da4w).

Conclusion
##########

This tutorial demonstrated how torchao's quantization and sparsity techniques integrate seamlessly across the entire ML deployment stack:

- **HuggingFace Transformers** provides easy model loading with torchao quantization
- **vLLM** leverages torchao's optimized kernels for high-throughput serving
- **ExecuTorch** enables mobile deployment with torchao's mobile-optimized schemes
- **lm-evaluation-harness** provides model quality assessment

All these frameworks use torchao as the underlying optimization engine, ensuring consistent performance gains and ease of integration. The quantization techniques shown provide significant memory reduction (3-4x) and performance improvements (1.5-2x) while maintaining model quality within acceptable bounds for most applications.

For production deployments, always benchmark on your specific use case and hardware to validate the performance and accuracy trade-offs.
