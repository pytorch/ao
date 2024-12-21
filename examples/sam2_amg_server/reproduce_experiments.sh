#!/bin/bash

# Generate AMG data for various settings
python generate_data.py ~/checkpoints/sam2 large ~/blogs/sam2_amg_example/sav_val_image_paths_shuf_1000 ~/blogs/sam2_amg_example/output_data                                 --baseline --points-per-batch   64
python generate_data.py ~/checkpoints/sam2 large ~/blogs/sam2_amg_example/sav_val_image_paths_shuf_1000 ~/blogs/sam2_amg_example/output_data_ao                                         --points-per-batch   64
python generate_data.py ~/checkpoints/sam2 large ~/blogs/sam2_amg_example/sav_val_image_paths_shuf_1000 ~/blogs/sam2_amg_example/output_data_ao_ppb_1024                                --points-per-batch 1024
python generate_data.py ~/checkpoints/sam2 large ~/blogs/sam2_amg_example/sav_val_image_paths_shuf_1000 ~/blogs/sam2_amg_example/output_data_ao_ppb_1024_fast                           --points-per-batch 1024 --fast
python generate_data.py ~/checkpoints/sam2 large ~/blogs/sam2_amg_example/sav_val_image_paths_shuf_1000 ~/blogs/sam2_amg_example/output_data_ao_ppb_1024_fast_export                    --points-per-batch 1024 --fast --load_fast ~/exported_models/sam2_fast
python generate_data.py ~/checkpoints/sam2 large ~/blogs/sam2_amg_example/sav_val_image_paths_shuf_1000 ~/blogs/sam2_amg_example/output_data_ao_ppb_1024_fast_export_furious            --points-per-batch 1024 --fast --load_fast ~/exported_models/sam2_fast_furious --furious

# Calculating mIoU w.r.t. baseline AMG results


# Postprocessing baseline AMG data for SPS and MPS tasks


# Generate SPS data for various settings

# Calculating mIoU w.r.t. baseline SPS results

# Generate MPS data for various settings

# Calculating mIoU w.r.t. baseline MPS results
