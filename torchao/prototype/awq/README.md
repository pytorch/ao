# AWQ Quantization
Adapted from https://github.com/mit-han-lab/llm-awq

## Benchmarks
Evaluation perplexity numbers were calculated using the script in awq/example.py Group size of 64 was used for all quantization methods. For Llama-2-7b-chat-hf, performance benchmarks were calculated using the torchao/_models/llama/generate.py script and run on a 1xA100 80GB SXM4 instance. The awq-uint4 quantization method does not use an efficient fused kernel which is why performance is not great. awq-hqq uses tinygemm int4->bf16 kernel + hqq to provide better performance.

| Model              | Quantization | Tokens/sec | Throughput (GB/sec) | Peak Mem (GB) | Model Size (GB) |
|--------------------|--------------|------------|---------------------|---------------|-----------------|
| Llama-2-7b-chat-hf | bfloat16     | 107.38     | 1418.93             | 13.88         | 13.21           |
|                    | awq-hqq-int4 | 196.6      | 761.2               | 5.05          | 3.87            |
|                    | awq-uint4    | 43.59      | 194.93              | 7.31          | 4.47            |
|                    | int4wo-hqq   | 209.19     | 804.32              | 4.89          | 3.84            |
|                    | int4wo-64    | 201.14     | 751.42              | 4.87          | 3.74            |



The following tests were performed using LM eval and groupsize = 128

| Model              | Quantization | Perplexity | Truthful QA MC2 | WinoGrande | ARC challenge |
|--------------------|--------------|------------|-----------------|------------|---------------|
| Llama-3-8B-Instruct| bfloat16     | 10.936     | 0.540           | 0.783      | 0.567         |
|                    | awq-hqq-int4 | 11.383     | 0.522           | 0.772      | 0.543         |
|                    | awq-uint4    | 11.409     | 0.519           | 0.756      | 0.577         |
|                    | int4wo-hqq   | 11.905     | 0.528           | 0.757      | 0.563         |
|                    | int4wo-128   | 12.380     | 0.502           | 0.753      | 0.548         |






