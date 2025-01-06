#!/bin/bash

outputdir="/Users/cpuhrsch/blogs/tmp/sam2_amg_example_run_1"
while IFS= read -r filepath; do
  filename=$(basename "$filepath")
  dirname=$(basename "$(dirname "$filepath")")
  mkdir -p "${outputdir}"/"${dirname}"
  echo curl -w "\"%{time_total}s\\\\n\"" -s -X POST https://cpuhrsch--torchao-sam-2-cli-model-upload-rle.modal.run -F "image=@${filepath}" -o "${outputdir}"/"${dirname}"/"${filename}.json"
  echo "${filepath}" >> cmds_input_paths
  echo "${outputdir}"/"${dirname}"/"${filename}.json" >> cmds_output_paths
done < ~/data/sav_val_image_paths_shuf_1000
