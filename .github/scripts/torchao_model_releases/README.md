# Scripts for torchao Model Release and Eval

Note: all commands below should be run in directory: `.github/scripts/torchao_model_releases/`

## Frequently Used Commands
### Release and Eval Scripts for New Model Releases
```
MODEL=Qwen/Qwen3-8B
# Releasing all models: INT4, INT8, INT8-INT4
sh release.sh --model_id $MODEL --push_to_hub --populate_model_card_template

# INT8-INT4 requires additional steps to export and run so it's skipped from
# general eval here
# Need to set QMODEL_PREFIX properly before running eval
# QMODEL_PREFIX=pytorch/Qwen3-8B
sh eval.sh --model_ids $MODEL "$QMODEL_PREFIX-FP8" "$QMODEL_PREFIX-INT4"

# Some follow up evals
sh eval.sh --eval_type latency --batch_size 256 "$QMODEL_PREFIX-FP8"
sh eval.sh --eval_type quality --batch_size 256 "$QMODEL_PREFIX-INT8-INT4"

# Summarize all results
sh summarize_results.sh --model_ids $MODEL "$QMODEL_PREFIX-FP8" "$QMODEL_PREFIX-INT4" "$QMODEL_PREFIX-INT8-INT4" "$QMODEL_PREFIX-AWQ-INT4"
```

### AWQ Release and Eval
```
MODEL=Qwen/Qwen3-8B
TASK=mmlu_abstract_algebra
python quantize_and_upload.py --model_id $MODEL --quant AWQ-INT4 --push_to_hub --task $TASK --calibration_limit 10 --populate_model_card_template
sh eval.sh --model_ids $MODEL "$QMODEL_PREFIX-AWQ-INT4"
```

### Update Released Checkpoints in PyTorch
Sometimes we may have to update the checkpoints under a different user name (organization) without changing the model card, e.g. for INT4
```
MODEL=Qwen/Qwen3-8B
sh release.sh --model $MODEL --quants INT4 --push_to_hub --push_to_user_id pytorch
```

Or AWQ checkpoint:
```
MODEL=Qwen/Qwen3-8B
TASK=mmlu_abstract_algebra
python quantize_and_upload.py --model_id $MODEL --quant AWQ-INT4--task $TASK --calibration_limit 10 --push_to_hub --push_to_user_id pytorch
```

## Release Scripts
### default options
By default, we release FP8, INT4, INT8-INT4 checkpoints, with model card pre-filled with template content, that can be modified later after we have eval results.

Examples:
```
# Note: first login with `hf auth login`, the quantized model will be uploaded to the logged in user

# release with default quant options (FP8, INT4, INT8-INT4)
./release.sh --model_id Qwen/Qwen3-8B --push_to_hub

# release a custom set of quant options
./release.sh --model_id Qwen/Qwen3-8B --quants INT4 FP8 --push_to_hub
```

Note: for initial release, please include `--populate_model_card_template` to populate model card template.

### SmoothQuant-INT8-INT8
[SmoothQuant](https://arxiv.org/abs/2211.10438) smooths activation outliers by migrating quantization difficulty from activations to weights through a mathematically equivalent per-channel scaling transformation. That means SmoothQuant observes activation distribution before applying quantization.

Examples:
```
# release SmoothQuant-INT8-INT8 model, calibrated with a specific task
python quantize_and_upload.py --model_id Qwen/Qwen3-8B --quant SmoothQuant-INT8-INT8 --push_to_hub --task bbh --populate_model_card_template
```

### AWQ-INT4
Similar to SmoothQuant, [AWQ](https://arxiv.org/abs/2306.00978) improves accuracy by preserving "salient" weight channels that has high impact on the accuracy of output. The notable point is that AWQ uses activation distribution to find salient weights, not weight distribution, multiplying the weight channel by a scale, and doing the reverse for the corresponding activation. Since activation is not quantized, there is no additional loss from activation, while the quantization loss from weight can be reduced.

After eval for INT4 checkpoint is done, we might find some task have a large accuracy drop compared to high precision baseline, in that case we can do a calibration for that task, with a few samples, tasks are selected from [lm-eval](https://github.com/EleutherAI/lm-eval\uation-harness/blob/main/lm_eval/tasks/README.md). You can follow [new task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md) to add new tasks to lm-eval.

Examples:
```
# release AWQ-INT4 model, calibrated with a specific task
# with some calibration_limit (number of samples)
python quantize_and_upload.py --model_id Qwen/Qwen3-8B --quant AWQ-INT4 --push_to_hub --task bbh --calibration_limit 2
```

### Update checkpoints for a different user_id (e.g. pytorch)
Sometimes we may want to update the checkpoints for a different user id, without changing model card. For this we can use `--push_to_user_id`, e.g.

```
sh release.sh --model_id microsoft/Phi-4-mini-instruct --quants FP8 --push_to_hub --push_to_user_id pytorch
```

This will update `pytorch/Phi-4-mini-instruct-FP8` without changing the model card.

## Eval Scripts
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
For latency eval, make sure vllm is installed.
```
uv pip install vllm
```

Or install vllm nightly:
```
uv pip install vllm --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu126
```

After environment is setup, we can run eval:
```
sh eval.sh --eval_type latency --model_ids Qwen/Qwen3-8B --batch_sizes 1 256
```

#### Model Quality Eval
For model quality eval, we need to install lm-eval
```
uv pip install lm-eval
```
After environment is setup, we can run eval:
```
sh eval.sh --eval_type quality --model_ids Qwen/Qwen3-8B --tasks hellaswag mmlu
```

See https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks for all supported tasks.

Note: you can pass in `--use_cache` if the eval task failed during the middle of the run
and you don't want to re-run all evals and there is no change to the model checkpoint.
```
sh eval.sh --eval_type quality --model_ids Qwen/Qwen3-8B --tasks hellaswag mmlu --use_cache
```

#### Multi-modal Model Quality Eval
For multi-modal model quality eval, we need to install lmms-eval
```
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```
After environment is setup, we can run eval:
```
sh eval.sh --eval_type mm_quality --model_ids google/gemma-3-12b-it --mm_tasks chartqa --model_type gemma3 --mm_eval_batch_size 32
```

See https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models/simple for supported model types.
See https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks for supported multi-modal tasks.

Note: larger mm_eval_batch_size could speedup eval but may cause OOM, when that happens, please reduce the number

Note: you can pass in `--use_cache` if the eval task failed during the middle of the run
and you don't want to re-run all evals and there is no change to model checkpoint.
```
sh eval.sh --eval_type mm_quality --model_ids google/gemma-3-12b-it --mm_tasks chartqa --model_type gemma3 --mm_eval_batch_size 32 --use_cache
```

Alternatively, please feel free to use the example scripts directly from llms-eval repo: https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/examples/models to run the evaluation.

#### Summarize results
After we have finished all evals for each model, we can summarize the results with:
```
sh summarize_results.sh --model_ids Qwen/Qwen3-8B pytorch/Qwen3-8B-INT4
```
Summarized results files for above command: `summary_results_Qwen_Qwen3-8B.log` and `summary_results_pytorch_Qwen3-8B-INT4.log`

It will look through the current directory to find all the result files from memory, latency and quality evals and combine all the result information into a single file.
