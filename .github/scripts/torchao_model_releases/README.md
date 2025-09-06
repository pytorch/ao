# Scripts for torchao model release and eval

Note: all commands below are run in directory: `.github/scripts/torchao_model_releases/`

## Release
### default options
By default, we release FP8, INT4, INT8-INT4 checkpoints, with model card pre-filled with template content, that can be modified later after we have eval results.

Examples:
```
# Note: first login with `huggingface-cli login`, the quantized model will be uploaded to
# the logged in user

# release with default quant options (FP8, INT4, INT8-INT4)
./release.sh --model_id Qwen/Qwen3-8B

# release a custom set of quant options
./release.sh --model_id Qwen/Qwen3-8B --quants INT4 FP8
```

### AWQ-INT4
[AWQ](https://arxiv.org/abs/2306.00978) is a technique to improve accuracy for weight only quantization. It improves accuracy by preserving "salient" weight channels that has high impact on the accuracy of output, through multiplying the weight channel by a scale, and do the reverse for the correspnoding activation, since activation is not quantized, there is no additional loss from activation, while the quantization loss from weight can be reduced.

After eval for INT4 checkpoint is done, we might find some task have a large accuracy drop compared to high precision baseline, in that case we can do a calibration for that task, with a few samples, tasks are selected from [lm-eval](https://github.com/EleutherAI/lm-eval\uation-harness/blob/main/lm_eval/tasks/README.md). You can follow [new task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md) to add new tasks to lm-eval.

Examples:
```
# release AWQ-INT4 model, calibrated with a specific task
# with some calibration_limit (number of samples)
python quantize_and_upload.py --model_id Qwen/Qwen3-8B --quant AWQ-INT4 --push_to_hub --task bbh --calibration_limit 2
```

## Eval
After we run the release script for a model, we can find new models in the huggingface hub page for the user, e.g. https://huggingface.co/torchao-testing, the models will have a model card that's filled in with template content, such as information about the model and eval instructions, there are a few things we need to fill in, including 1. peak memory usage, 2. latency when running model with vllm and 3. quality measurement using lm-eval.

### Single Script
The simplest is just to run all three evals. Please check out `Run Single Evals` section to make sure the environment is setup correctly. This includes:
1. install [vllm](https://github.com/vllm-project/vllm) from source and set `VLLM_DIR` to the soruce directory of vllm
2. install [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness)

```
sh eval.sh --eval_type all --model_ids Qwen/Qwen3-8B pytorch/Qwen3-8B-INT4
```

If `eval_type` is all, we'll also run summarize results for the list of `model_ids`, summarized results will be found in files: `summary_results_Qwen_Qwen3-8B.log` and `summary_results_pytorch_Qwen3-8B-INT4.log`.

Then we can fill in the blanks in the model cards of uploaded checkpoints.

### Separate Scripts
#### Memory Eval
```
sh eval.sh --eval_type memory --model_ids Qwen/Qwen3-8B
```

#### Latency Eval
For latency eval, make sure vllm is cloned and installed from source,
and `VLLM_DIR` should be set to the source directory of the cloned vllm repo.
```
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
export VLLM_DIR=path_to_vllm
```
see https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#set-up-using-python-only-build-without-compilation for more details.

After environment is setup, we can run eval:
```
sh eval.sh --eval_type latency --model_ids Qwen/Qwen3-8B --batch_sizes 1,256
```

#### Model Quality Eval
For model quality eval, we need to install lm-eval
```
pip install lm-eval
```
After environment is setup, we can run eval:
```
sh eval.sh --eval_type quality --model_ids Qwen/Qwen3-8B --tasks hellaswag,mmlu
```

# ### Summarize results
After we have finished all evals for each model, we can summarize the results with:
```
sh summarize_results.sh --model_ids Qwen/Qwen3-8B pytorch/Qwen3-8B-INT4
```
Summarized results files for above command: `summary_results_Qwen_Qwen3-8B.log` and `summary_results_pytorch_Qwen3-8B-INT4.log`

It will look through the current directory to find all the result files from memory, latency and quality evals and combine all the result information into a single file.
