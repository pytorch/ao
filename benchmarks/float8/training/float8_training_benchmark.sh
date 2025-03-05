#!/bin/bash
# This script can be used to launch a torchtitan float8 training run
# with the given parameters,

# script arguments
BATCH_SIZE=${BATCH_SIZE:-1}
STEPS=${STEPS:-100}

# temporary log file which is deleted after performance data is parsed out and metrics are calculated.
LOG_FILE="/tmp/float8_training_log.txt"

# validate user has specified torchtitan root directory
if [ -z "${TORCHTITAN_ROOT}" ]; then
  echo "Error: TORCHTITAN environment variable is not set. Please set it before running this script."
  echo "Usage: TORCHTITAN_ROOT=<directory> ./float8_training_benchmark.sh"
  echo "Optional parameters configurable via environment variables:"
  echo " * FLOAT8_RECIPE_WITH_BEST_SETTINGS: "rowwise" or "tensorwise". if set, use float8 training in torchtitan with the specified recipe, including the additional settings which are optimal for that recipe. otherwise, use bf16 mixed precision training."
  echo " * BATCH_SIZE: defaults to 1."
  echo " * STEPS: defaults to 100."
  exit 1
fi

# validate recipe name
if [ -n "${FLOAT8_RECIPE_WITH_BEST_SETTINGS}" ]; then
  if [ "${FLOAT8_RECIPE_WITH_BEST_SETTINGS}" == "tensorwise" ]; then
    FLOAT8_ARGS="--model.converters="float8" --float8.enable_fsdp_float8_all_gather --float8.precompute_float8_dynamic_scale_for_fsdp --float8.force_recompute_fp8_weight_in_bwd"
  else
    FLOAT8_ARGS="--model.converters="float8" --float8.recipe_name=${FLOAT8_RECIPE_WITH_BEST_SETTINGS}"
  fi
fi


# remember current directory to return to it later
original_dir=$(pwd)

# navigate to torchtitan root dir
cd ${TORCHTITAN_ROOT}

echo "float8 args: ${FLOAT8_ARGS}"

# run the command with the specified arguments
CONFIG_FILE="./torchtitan/models/llama/train_configs/llama3_8b.toml" ${TORCHTITAN_ROOT}/run_train.sh --training.steps=${STEPS} --training.batch_size=${BATCH_SIZE} --training.compile ${FLOAT8_ARGS} 2>&1 | tee ${LOG_FILE}

# return to original working directory
cd $original_dir

# parse logs to calculate top line metrics
python parse_torchtitan_logs.py --log-file ${LOG_FILE}

# clean up logs
rm ${LOG_FILE}
