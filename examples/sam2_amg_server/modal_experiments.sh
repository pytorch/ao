# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#!/bin/bash

set -ex

# amg baseline
modal deploy cli_on_modal.py --name torchao-sam-2-cli-amg-baseline
mkdir -p ~/blogs/outputs/amg_baseline
time python cli_on_modal.py --task-type amg --input-paths ~/blogs/cmds_input_paths --output_directory ~/blogs/outputs/amg_baseline --output-rle True --meta-paths ~/blogs/cmds_meta_paths --name torchao-sam-2-cli-amg-baseline --baseline
modal app stop torchao-sam-2-cli-amg-baseline

# sps baseline
modal deploy cli_on_modal.py --name torchao-sam-2-cli-sps-baseline
mkdir -p ~/blogs/outputs/sps_baseline
time python cli_on_modal.py --task-type sps --input-paths ~/blogs/cmds_input_paths --output_directory ~/blogs/outputs/sps_baseline --output-rle True --meta-paths ~/blogs/cmds_meta_paths --name torchao-sam-2-cli-sps-baseline --baseline
modal app stop torchao-sam-2-cli-sps-baseline

# mps baseline
modal deploy cli_on_modal.py --name torchao-sam-2-cli-mps-baseline
mkdir -p ~/blogs/outputs/mps_baseline
time python cli_on_modal.py --task-type mps --input-paths ~/blogs/cmds_input_paths --output_directory ~/blogs/outputs/mps_baseline --output-rle True --meta-paths ~/blogs/cmds_meta_paths --name torchao-sam-2-cli-mps-baseline --baseline
modal app stop torchao-sam-2-cli-mps-baseline

# amg
modal deploy cli_on_modal.py --name torchao-sam-2-cli-amg
mkdir -p ~/blogs/outputs/amg
time python cli_on_modal.py --task-type amg --input-paths ~/blogs/cmds_input_paths --output_directory ~/blogs/outputs/amg --output-rle True --meta-paths ~/blogs/cmds_meta_paths --name torchao-sam-2-cli-amg
modal app stop torchao-sam-2-cli-amg

# sps
modal deploy cli_on_modal.py --name torchao-sam-2-cli-sps
mkdir -p ~/blogs/outputs/sps
time python cli_on_modal.py --task-type sps --input-paths ~/blogs/cmds_input_paths --output_directory ~/blogs/outputs/sps --output-rle True --meta-paths ~/blogs/cmds_meta_paths --name torchao-sam-2-cli-sps
modal app stop torchao-sam-2-cli-sps

# mps
modal deploy cli_on_modal.py --name torchao-sam-2-cli-mps
mkdir -p ~/blogs/outputs/mps
time python cli_on_modal.py --task-type mps --input-paths ~/blogs/cmds_input_paths --output_directory ~/blogs/outputs/mps --output-rle True --meta-paths ~/blogs/cmds_meta_paths --name torchao-sam-2-cli-mps
modal app stop torchao-sam-2-cli-mps

echo "amg vs baseline"
python compare_rle_lists.py ~/blogs/outputs/amg  ~/blogs/outputs/amg_baseline --compare-folders --strict
echo "sps vs baseline"
python compare_rle_lists.py ~/blogs/outputs/sps  ~/blogs/outputs/sps_baseline --compare-folders --strict
echo "mps vs baseline"
python compare_rle_lists.py ~/blogs/outputs/mps  ~/blogs/outputs/mps_baseline --compare-folders --strict
