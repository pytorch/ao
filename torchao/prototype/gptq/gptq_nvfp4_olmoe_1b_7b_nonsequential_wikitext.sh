#!/bin/bash

#
# A quick smoke test for non-sequential GPTQ on `allenai/OLMoE-1B-7B-0924`
#

COMMON_ARGS="--output-dir-prefix /home/dev/tmp/20260421 --model-id allenai/OLMoE-1B-7B-0924 --lm-eval-tasks wikitext --num-fewshot 0 --lm-eval-batch-size 16"

# baseline (bf16)
echo -e "\n\nbaseline (bf16)\n\n"
# python -u torchao/prototype/gptq/gptq_example.py $COMMON_ARGS --quantization none 
echo -e "done"

# nvfp4-rtn
echo -e "\n\nnvfp4-rtn\n\n"
# python -u torchao/prototype/gptq/gptq_example.py $COMMON_ARGS --quantization nvfp4-rtn
echo -e "done"

# nvfp4-gptq-nonsequential
echo -e "\n\nnvfp4-gptq-nonsequential\n\n"
# TODO(future PR): fix https://gist.github.com/vkuzo/51b2bfcee77fc193253faf007d99d694
# and enable this
# python -u torchao/prototype/gptq/gptq_example.py $COMMON_ARGS --quantization nvfp4-gptq-nonsequential --dataset-id c4 --dataset-split train
echo -e "done"

