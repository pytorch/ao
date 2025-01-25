#!/bin/bash

set -ex

# outputdir="/Users/cpuhrsch/blogs/tmp/sam2_amg_example_run_1"
# while IFS= read -r filepath; do
#   filename=$(basename "$filepath")
#   dirname=$(basename "$(dirname "$filepath")")
#   mkdir -p "${outputdir}"/"${dirname}"
#   echo curl -w "\"%{time_total}s\\\\n\"" -s -X POST https://cpuhrsch--torchao-sam-2-cli-model-upload-rle.modal.run -F "image=@${filepath}" -o "${outputdir}"/"${dirname}"/"${filename}.json"
#   echo "${filepath}" >> cmds_input_paths
#   echo "${outputdir}"/"${dirname}"/"${filename}.json" >> cmds_output_paths
# done < ~/data/sav_val_image_paths_shuf_1000

# time python cli_on_modal.py --task-type amg --input-paths ~/blogs/cmds_input_paths --output_directory /Users/cpuhrsch/blogs/tmp/sam2_amg_example_run_1_amg --output-rle False --meta-paths ~/blogs/cmds_meta_paths
# time python cli_on_modal.py --task-type sps --input-paths ~/blogs/cmds_input_paths --output_directory /Users/cpuhrsch/blogs/tmp/sam2_amg_example_run_1_sps --output-rle False --meta-paths ~/blogs/cmds_meta_paths
# time python cli_on_modal.py --task-type mps --input-paths ~/blogs/cmds_input_paths --output_directory /Users/cpuhrsch/blogs/tmp/sam2_amg_example_run_1_mps --output-rle False --meta-paths ~/blogs/cmds_meta_paths

# # amg
# modal deploy cli_on_modal.py
# time python cli_on_modal.py --task-type amg --input-paths ~/blogs/cmds_input_paths --output_directory ~/blogs/tmp/sam2_amg_example_run_1_amg --output-rle True --meta-paths ~/blogs/cmds_meta_paths | tee ~/blogs/amg_latencies

# # sps
# modal deploy cli_on_modal.py
# time python cli_on_modal.py --task-type sps --input-paths ~/blogs/cmds_input_paths --output_directory ~/blogs/tmp/sam2_amg_example_run_1_sps --output-rle True --meta-paths ~/blogs/cmds_meta_paths | tee ~/blogs/sps_latencies

# mps
modal deploy cli_on_modal.py
time python cli_on_modal.py --task-type mps --input-paths ~/blogs/cmds_input_paths --output_directory ~/blogs/tmp/sam2_amg_example_run_1_mps --output-rle True --meta-paths ~/blogs/cmds_meta_paths | tee ~/blogs/mps_latencies
