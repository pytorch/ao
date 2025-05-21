# LLAMA

## Eval on Llama 3.1 8B and Llama 3.2 3B

We use lm-eval tasks for evaluating TorchAO Quantization APIs on HuggingFace models. The results are in the table below:

| Model Name | Quantization Technique    | Acc   |Acc Norm| Word perplexity| Model Size (GB)   |
|------------|---------------------------|-------|--------|----------------|-------------------|
| Llama 3.1 8B  | None                   | 60.01 | 78.84  |       7.33     | 15.01             |
| Llama 3.1 8B  | int4wo-128             | 58.10 | 77.06  |       8.25     | 4.76              |
| Llama 3.1 8B  | int8wo                 | 59.92 | 78.95  |       7.34     | 8.04              |
| Llama 3.1 8B  | int8dq                 | 60.01 | 78.82  |       7.45     | 8.03              |
| Llama 3.1 8B  | float8wo               | 59.83 | 78.61  |       7.37     | 8.03              |
| Llama 3.1 8B  | float8dq (PerRow)      | 59.86 | 78.57  |       7.41     | 8.04              |
| Llama 3.1 8B  | float8dq (PerTensor)   | 59.95 | 78.66  |       7.42     | 8.03              |
| Llama 3.1 8B  | gemlite (gp=128)       | 58.48 | 77.34  |       8.07     | 4.76              |

| Model Name | Quantization Technique    | Acc   |Acc Norm| Word perplexity| Model Size (GB)   |
|------------|---------------------------|-------|--------|----------------|-------------------|
| Llama 3.2 3B  | None                   | 55.27 | 73.70  |       9.26     | 6.43              |
| Llama 3.2 3B  | int4wo-128             | 53.13 | 71.31  |       10.36    | 2.29              |
| Llama 3.2 3B  | int8wo                 | 55.15 | 73.44  |       9.28     | 3.61              |
| Llama 3.2 3B  | int8dq                 | 55.00 | 73.29  |       9.43     | 3.61              |
| Llama 3.2 3B  | float8wo               | 55.18 | 73.58  |       9.31     | 3.61              |
| Llama 3.2 3B  | float8dq (PerRow)      | 55.18 | 73.37  |       9.33     | 3.61              |
| Llama 3.2 3B  | float8dq (PerTensor)   | 55.16 | 73.53  |       9.35     | 3.61              |
| Llama 3.2 3B  | gemlite (gp=128)       | 53.71 | 71.99  |      10.05     | 2.29              |

To generate the above results run:
```
sh benchmarks/_models/eval_hf_models.sh
```

To run lm-eval for a different hf-model with AO quantization technique, run:
```
python benchmarks/_models/eval_hf_models.py --model_id meta-llama/Llama-3.1-8B --quantization float8dq-row --tasks wikitext hellaswag
```
Replace model id, quantization and tasks with your desired values Please refer to ([HuggingFace <-> TorchAO](https://huggingface.co/docs/transformers/main/en//quantization/torchao)) integration docs for more details about the supported quantization techniques.

# SAM2
sam2 is a fork of https://github.com/facebookresearch/sam2 at commit c2ec8e14a185632b0a5d8b161928ceb50197eddc

It includes
- modifications to enable fullgraph=True compile
- `mask_to_rle_pytorch_2`
- small performance changes and fixes
- integration into torchao's packaging
