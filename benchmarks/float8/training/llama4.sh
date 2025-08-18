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
  echo "Usage: TORCHTITAN_ROOT=<directory> ./torchtitan_llama4.sh"
  echo " * EXTRA_ARGS: additional arguments to pass to the torchtitan training script."
  exit 1
fi

# remember current directory to return to it later
original_dir=$(pwd)

# navigate to torchtitan root dir
cd ${TORCHTITAN_ROOT}

# run the command with the specified arguments
CONFIG_FILE="./torchtitan/experiments/llama4/train_configs/debug_model.toml" ${TORCHTITAN_ROOT}/run_train.sh  ${EXTRA_ARGS} 2>&1 | tee ${LOG_FILE}

# return to original working directory
cd $original_dir

# parse logs to calculate top line metrics
python parse_torchtitan_logs.py --log-file ${LOG_FILE}

# clean up logs
rm ${LOG_FILE}
