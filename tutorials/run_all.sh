# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#!/bin/bash
FAILED=0
for dir in $(find . -type d); do
  if [ -f "$dir/run.sh" ]; then
    echo "Running: $dir/run.sh"
    CURRENT_DIR=$(pwd)
    cd "$dir"
    bash run.sh
    cd "$CURRENT_DIR"
  else
    for file in $(find "$dir" -maxdepth 1 -name "*.py"); do
      filename=$(basename "$file")
      if echo "$filename" | grep -q "tensor_parallel"; then
        echo "Running: torchrun --standalone --nnodes=1 --nproc-per-node=1 $file"
        torchrun --standalone --nnodes=1 --nproc-per-node=4 "$file"
        STATUS=$?
      else
        echo "Running: python $file"
        python "$file"
        STATUS=$?
      fi

      if [ $STATUS -ne 0 ]; then
        FAILED=1
        echo "Test failed: $file"
      fi
    done
  fi
done

if [ "$FAILED" -eq 1 ]; then
  echo "One or more tests failed"
  exit 1
else
  echo "All tests passed"
  exit 0
fi
