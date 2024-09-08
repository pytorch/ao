# AWQ Quantization
Adapted from https://github.com/mit-han-lab/llm-awq

## Benchmarks
Benchmarks are run on a machine with a single RTX 3090 GPU using the script in awq/example.py which calculates perplexity by concatenating wikitex2 test examples with newlines and dividing by context length. Group size of 64 was used for both int4wo and awq-uint4. For awq quantization, c refers to number of calibration sequences and v refers to number of validation sequences. Calibration data is used to find the average magnitude of activations while validation data is used to find optimal equilization scales. Note c is always larger than v. Calibration data came from Pile dataset validation split.

| Model              | Quantization                | wikitext2-perplexity |
| ------------------ | ------------------------    |  ------------------- | 
| GPT-2              | Base (bfloat16)             | 30.1904              |
|                    | int4wo (tinygemm kernel)    | 519.73108            |
|                    | awq-uint4                   | 485.54907            |
| Llama-2-7b-hf      | Base (bfloat16)             | 5.47367              |
|                    | int4wo (tinygemm kernel)    | 5.73546              |
|                    | awq-uint4-c1-v1             | 5.72359              |
|                    | awq-uint4-c10-v1            | 5.72350              |

