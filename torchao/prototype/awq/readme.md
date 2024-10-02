# AWQ Quantization
Adapted from https://github.com/mit-han-lab/llm-awq

## Benchmarks
Benchmarks are run on a machine with a single RTX 3090 GPU using the script in awq/example.py which calculates perplexity by concatenating wikitex2 test examples with newlines and dividing by context length. Group size of 64 was used for both int4wo and awq-uint4. For awq quantization, c refers to number of calibration sequences and v refers to number of validation sequences. Calibration data is used to find the average magnitude of activations while validation data is used to find optimal equilization scales. Note c is always larger than v. Calibration data came from Pile dataset validation split.

| Model              | Quantization | Perplexity |
|--------------------|--------------|------------|
| Llama-2-7b-chat-hf | bfloat16     | 5.0309     |
|                    | awq-uint4    | 5.2388     |
|                    | int4         | 5.28       |
|                    | awq-hqq      | 5.204      |
|                    | hqq          | 5.3419     |
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



