#!/bin/bash

# terminate script on first error
set -e

launch() {
    echo "launching compile_fsdp $COMPILE, use_weight_dynamic_scaling $USE_WEIGHT_DYNAMIC_SCALING"

    # the NCCL_DEBUG setting is to avoid log spew
    # the CUDA_VISIBLE_DEVICES setting is for easy debugging
    NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1 python test/float8/test_fsdp.py \
        --compile_fsdp $COMPILE --use_weight_dynamic_scaling $USE_WEIGHT_DYNAMIC_SCALING

    echo "✅ All Tests Passed ✅"
}

if python -c 'import torch;print(torch.cuda.is_available())' | grep -q "False"; then
    echo "Skipping test_fsdp.sh because no CUDA devices are available."
    exit
fi

# COMPILE, USE_WEIGHT_DYNAMIC_SCALING
for i in False,False False,True True,False True,True
do
    IFS=","; set -- $i;
    COMPILE=$1; USE_WEIGHT_DYNAMIC_SCALING=$2
    launch
done
