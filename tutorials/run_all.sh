#!/bin/bash
find . -type d | while read dir; do
  if [ -f "$dir/run.sh" ]; then
    echo "Running: $dir/run.sh"
    pushd "$dir"
    bash run.sh
    popd
  else
    find "$dir" -maxdepth 1 -name "*.py" | while read file; do
      if [[ "$file" == *"tensor_parallel"* ]]; then
        echo "Running: torchrun --standalone --nnodes=1 --nproc-per-node=1 $file"
        torchrun --standalone --nnodes=1 --nproc-per-node=4 "$file"
      else
        echo "Running: python $file"
        python "$file"
      fi
    done
  fi
done
