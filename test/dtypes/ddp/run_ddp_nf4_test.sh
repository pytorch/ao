#!/bin/bash

set -euo pipefail
WORLD_SIZE=${1:-2}


# Test params
GLOBAL_BS=8
DIM=128
NUM_LINEARS=1
NUM_STEPS=3

PARAMS="--global_bs $GLOBAL_BS --dim $DIM --num_linears $NUM_LINEARS --num_steps $NUM_STEPS"
SAVE_DIR="checkpoints"
REF_DIR="${SAVE_DIR}/ref"
TEST_DIR="${SAVE_DIR}/test"
DDP_PROGRAM="ddp_nf4.py"
CHECK_PROGRAM="check_ddp_nf4.py"
REF_CMD="torchrun --nproc_per_node 1 $DDP_PROGRAM $PARAMS --save_dir $REF_DIR"
TEST_CMD="torchrun --nproc_per_node $WORLD_SIZE $DDP_PROGRAM $PARAMS --save_dir $TEST_DIR"
CHECK_CMD="python $CHECK_PROGRAM --ref_checkpoint_dir $REF_DIR --test_checkpoints_dir $TEST_DIR"
CLEANUP_CMD="rm -rf $SAVE_DIR"

echo "Step 1: Generating reference checkpoint..."
echo $REF_CMD
$REF_CMD
echo -e "\n --- \n"
sleep 2

echo "Step 2: Generating test checkpoints..."
echo $TEST_CMD
$TEST_CMD
echo -e "\n --- \n"
sleep 2

# Check params
echo "Step 3: Checking params..."
echo $CHECK_CMD
$CHECK_CMD
echo -e "\n --- \n"
sleep 2

# Cleanup
echo "Step 4: Cleaning up..."
echo $CLEANUP_CMD
$CLEANUP_CMD
echo -e "\n --- \n"
echo "Done!"
