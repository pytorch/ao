#!/bin/bash
FAILED=0
find . -type d | while read dir; do
  if [ -f "$dir/run.sh" ]; then
    echo "Running: $dir/run.sh"
    CURRENT_DIR=$(pwd)
    cd "$dir"
    bash run.sh
    cd "$CURRENT_DIR"
  else
    find "$dir" -maxdepth 1 -name "*.py" | while read file; do
      filename=$(basename "$file")
      echo "filename: $filename"
      if [ "$filename" = *"tensor_parallel"* ]; then
        echo "Running: torchrun --standalone --nnodes=1 --nproc-per-node=1 $file"
        torchrun --standalone --nnodes=1 --nproc-per-node=4 "$file"
        STATUS=$?
        echo "Status: $STATUS"
      else
        echo "Running: python $file"
        python "$file"
        STATUS=$?
        echo "Status: $STATUS"
      fi

      if [ $STATUS -ne 0 ]; then
        FAILED=1
        echo "Test failed: $file"
      fi
    done
  fi
done

echo "Failed: $FAILED"
if (( FAILED == 1 )); then
  echo "One or more tests failed"
  exit 1
else
  echo "All tests passed"
  exit 0
fi
