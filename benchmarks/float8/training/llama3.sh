# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#!/bin/bash
# This script can be used to launch a torchtitan float8 training run
# with the given parameters,

# script arguments
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-1}
STEPS=${STEPS:-100}

# temporary log file which is deleted after performance data is parsed out and metrics are calculated.
LOG_FILE="/tmp/float8_training_log.txt"

# validate user has specified torchtitan root directory
if [ -z "${TORCHTITAN_ROOT}" ]; then
  echo "Error: TORCHTITAN environment variable is not set. Please set it before running this script."
  echo "Usage: TORCHTITAN_ROOT=<directory> ./llama3.sh"
  echo "Optional parameters configurable via environment variables:"
  echo " * FLOAT8_RECIPE_WITH_BEST_SETTINGS: "rowwise" or "tensorwise". if set, use float8 training in torchtitan with the specified recipe, including the additional settings which are optimal for that recipe. otherwise, use bf16 mixed precision training."
  echo " * MX_RECIPE: any valid MX recipe name. Note: only one of FLOAT8_RECIPE_WITH_BEST_SETTINGS and MX_RECIPE can be set."
  echo " * LOCAL_BATCH_SIZE: defaults to 1."
  echo " * STEPS: defaults to 100."
  echo " * EXTRA_ARGS: additional arguments to pass to the torchtitan training script."
  exit 1
fi

# validate recipe name
if [ -n "${FLOAT8_RECIPE_WITH_BEST_SETTINGS}" ] && [ -n "${MX_RECIPE}" ]; then
    echo "Error: both FLOAT8_RECIPE_WITH_BEST_SETTINGS and MX_RECIPE are set, please only set one of them." >&2
    exit 1
elif [ -n "${FLOAT8_RECIPE_WITH_BEST_SETTINGS}" ]; then
  if [ "${FLOAT8_RECIPE_WITH_BEST_SETTINGS}" == "tensorwise" ]; then
    FLOAT8_ARGS="--model.converters="float8" --float8.enable_fsdp_float8_all_gather --float8.precompute_float8_dynamic_scale_for_fsdp"
  else
    FLOAT8_ARGS="--model.converters="float8" --float8.recipe_name=${FLOAT8_RECIPE_WITH_BEST_SETTINGS}"
  fi
elif [ -n "${MX_RECIPE}" ]; then
    FLOAT8_ARGS="--model.converters="mx" --mx.recipe_name=${MX_RECIPE}"
else
    FLOAT8_ARGS=""
fi


# remember current directory to return to it later
original_dir=$(pwd)

# navigate to torchtitan root dir
cd ${TORCHTITAN_ROOT}

echo "float8 args: ${FLOAT8_ARGS}"

# run the command with the specified arguments
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ${TORCHTITAN_ROOT}/run_train.sh --training.steps=${STEPS} --training.local-batch-size=${LOCAL_BATCH_SIZE} --compile.enable ${FLOAT8_ARGS} ${EXTRA_ARGS} 2>&1 | tee ${LOG_FILE}

# return to original working directory
cd $original_dir

# parse logs to calculate top line metrics
python benchmarks/float8/training/parse_torchtitan_logs.py --log-file ${LOG_FILE}

# clean up logs
rm ${LOG_FILE}
