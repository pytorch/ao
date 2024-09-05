# AWQ Quantization
Adapted from https://github.com/mit-han-lab/llm-awq

## Benchmarks
Benchmarks are run on a machine with a single RTX 3090 GPU using the script in awq/example.py which calculates perplexity by concatenating wikitex2 test examples with newlines and dividing by context length. The model used was openai-community/gpt2 with a context length of 1024. Group size of 64 was used for both int4wo and awq-uint4.

| Quantization                | wikitext2-perplexity |
| ------------------------    |  ------------------- | 
| Base (bfloat16)             | 30.1904              |
| int4wo (tinygemm kernel)    | 519.73108            |
| awq-uint4                   | 485.54907            |