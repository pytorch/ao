# AWQ Quantization
Ported from https://github.com/mit-han-lab/llm-awq

## Benchmarks
Benchmarks are run on a machine with a single A100 GPU using the script in _models/llama which generates text in a latency optimized way (batchsize=1), evaluation was done
Using the lm_eval. The models used were meta-llama/Llama-2-7b-chat-hf

| Model       | Technique          | wikitext-perplexity | Tokens/Second | Memory Bandwidth (GB/s) | Peak Memory (GB) | Model Size (GB) |
| ----------- | ------------------ | ------------------- | ------------- | ----------------------- | ---------------- | --------------- |
| Llama-2-7B  | Base (bfloat16)    | 12.212              |  105.14       | 1389.35                 | 13.88            | 13.21           |
|             | int4wo-64          | 12.843              |  199.86       |  746.66                 |  4.50            |  3.74           |
|             | int4wo-64-GPTQ     | 12.489              |  199.86       |  746.66                 |  4.50            |  3.74           |
|             | awq                | 12.204              |  159.22       | 1069.87                 |  8.91            |  6.72           |