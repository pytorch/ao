# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

SETUP_HOME=$(pwd)


mkdir -p checkpoints
mkdir -p datasets

mkdir -p tmp
mkdir -p tmp/sam_coco_mask_center_cache
mkdir -p tmp/sam_eval_masks_out

wget -nc -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -nc -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget -nc -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

mkdir -p datasets/coco2017
wget -nc -P datasets/coco2017 http://images.cocodataset.org/zips/val2017.zip
wget -nc -P datasets/coco2017 http://images.cocodataset.org/annotations/annotations_trainval2017.zip

cd datasets/coco2017 && unzip -n val2017.zip && cd $SETUP_HOME
cd datasets/coco2017 && unzip -n annotations_trainval2017.zip && cd $SETUP_HOME
