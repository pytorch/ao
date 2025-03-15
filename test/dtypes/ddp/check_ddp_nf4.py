# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from pathlib import Path

import torch

from torchao.dtypes.nf4tensor import NF4Tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_checkpoint_dir", type=str, required=True)
    parser.add_argument("--test_checkpoints_dir", type=str, required=True)

    args = parser.parse_args()

    ref_checkpoints = list(Path(args.ref_checkpoint_dir).glob("*.pt"))
    assert len(ref_checkpoints) == 1, "Expected exactly one reference checkpoint"
    ref_checkpoint = ref_checkpoints[0]
    ref_state_dict = torch.load(ref_checkpoint, weights_only=True, map_location="cpu")
    print(f"Ref checkpoint: {ref_checkpoint}")

    for path in Path(args.test_checkpoints_dir).glob("*.pt"):
        print(f"Checking {path}")
        state_dict = torch.load(path, weights_only=True, map_location="cpu")
        assert ref_state_dict.keys() == state_dict.keys()
        for name in ref_state_dict.keys():
            ref_param = ref_state_dict[name]
            test_param = state_dict[name]
            print(f"Checking {name} {type(ref_param)} {type(test_param)}")

            if isinstance(ref_param, NF4Tensor):
                ref_param = ref_param.get_original_weight()
                assert isinstance(test_param, NF4Tensor)
                test_param = test_param.get_original_weight()

            if not torch.allclose(ref_param, test_param, atol=1e-4, rtol=1e-4):
                diff = (ref_param - test_param).abs().max()
                print(f" \u2718 Param {name} differs by {diff}")
            else:
                print(f" \u2713 Param {name} is consistent")
        print("Passed!")
