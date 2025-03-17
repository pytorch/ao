# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# baseline
python eval_combo.py --coco_root_dir datasets/coco2017 --coco_slice_name val2017 --sam_checkpoint_base_path checkpoints --sam_model_type vit_h --point_sampling_cache_dir tmp/sam_coco_mask_center_cache --mask_debug_out_dir tmp/sam_eval_masks_out --batch_size 32 --num_workers 32 --use_compile max-autotune --use_half bfloat16 --device cuda --print_header True
# int8 dynamic quant (all)
python eval_combo.py --coco_root_dir datasets/coco2017 --coco_slice_name val2017 --sam_checkpoint_base_path checkpoints --sam_model_type vit_h --point_sampling_cache_dir tmp/sam_coco_mask_center_cache --mask_debug_out_dir tmp/sam_eval_masks_out --batch_size 32 --num_workers 32 --use_compile max-autotune --use_half bfloat16 --device cuda --compress int8_dynamic_quant
# 2:4 sparsity (all)
python eval_combo.py --coco_root_dir datasets/coco2017 --coco_slice_name val2017 --sam_checkpoint_base_path checkpoints --sam_model_type vit_h --point_sampling_cache_dir tmp/sam_coco_mask_center_cache --mask_debug_out_dir tmp/sam_eval_masks_out --batch_size 32 --num_workers 32 --use_compile max-autotune --use_half bfloat16 --device cuda --compress sparse_mlp_only
# 2:4 sparsity (mlp only)
python eval_combo.py --coco_root_dir datasets/coco2017 --coco_slice_name val2017 --sam_checkpoint_base_path checkpoints --sam_model_type vit_h --point_sampling_cache_dir tmp/sam_coco_mask_center_cache --mask_debug_out_dir tmp/sam_eval_masks_out --batch_size 32 --num_workers 32 --use_compile max-autotune --use_half bfloat16 --device cuda --compress sparse
# int8 dynamic quant + 2:4 sparsity (attn: int8, mlp lin1: int8+2:4 fuse mul, mlp lin2: 2:4 sparse)
python eval_combo.py --coco_root_dir datasets/coco2017 --coco_slice_name val2017 --sam_checkpoint_base_path checkpoints --sam_model_type vit_h --point_sampling_cache_dir tmp/sam_coco_mask_center_cache --mask_debug_out_dir tmp/sam_eval_masks_out --batch_size 32 --num_workers 32 --use_compile max-autotune --use_half bfloat16 --device cuda --compress int8_dynamic_quant_sparse
# int8 dynamic quant attn + int4 wo + sparse marlin lin 1 + 2:4 sparse lin2
python eval_combo.py --coco_root_dir datasets/coco2017 --coco_slice_name val2017 --sam_checkpoint_base_path checkpoints --sam_model_type vit_h --point_sampling_cache_dir tmp/sam_coco_mask_center_cache --mask_debug_out_dir tmp/sam_eval_masks_out --batch_size 32 --num_workers 32 --use_compile max-autotune --use_half float16 --device cuda --compress int4_weight_only_sparse
