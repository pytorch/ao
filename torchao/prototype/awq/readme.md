# AWQ Quantization
Adapted from https://github.com/mit-han-lab/llm-awq

## Benchmarks
Evaluation perplexity numbers were calculated using the script in awq/example.py which calculates perplexity by concatenating wikitex2 test examples with newlines and dividing by context length. Group size of 64 was used for all quantization methods. For Llama-2-7b-chat-hf, performance benchmarks were calculated using the torchao/_models/llama/generate.py script and run on a 1xA100 80GB SXM4 instance. The awq-uint4 quantization method does not use an efficient fused kernel which is why performance is not great. awq-hqq uses tinygemm int4->bf16 kernel + hqq to provide better performance.

| Model              | Quantization | Perplexity | Tokens/sec | Throughput (GB/sec) | Peak Mem (GB) | Model Size (GB) |
|--------------------|--------------|------------|------------|---------------------|---------------|-----------------|
| Llama-2-7b-chat-hf | bfloat16     | 5.0309     | 107.38     | 1418.93             | 13.88         | 13.21           |
|                    | awq-uint4    | 5.2388     | 43.59      | 194.93              | 7.31          | 4.47            |
|                    | int4         | 5.28       | 201.14     | 751.42              | 4.87          | 3.74            |
|                    | awq-hqq      | 5.204      | 196.6      | 761.2               | 5.05          | 3.87            |
|                    | hqq          | 5.3419     | 209.19     | 804.32              | 4.89          | 3.84            |
| Llama-3-8b         | bfloat16     | 4.6269     |
|                    | awq-uint4    | 4.968      |
|                    | int4         | 5.04325    |
|                    | awq-hqq      | 4.8525     |
|                    | hqq          | 5.1277     |
| Llama-3.1-8b       | bfloat16     | 4.69732    |
|                    | awq-uint4    | 4.98163    |
|                    | int4         | 5.04091    |
|                    | awq-hqq      | 4.90632    |
|                    | hqq          | 5.14375    |



