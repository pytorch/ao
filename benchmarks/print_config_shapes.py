# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from torchao.kernel import autotuner

configs = autotuner._load_best_configs()

print("m,k,n")
for k, v in configs.items():
    a_shape = k[1]
    b_shape = k[4]
    M, K0 = a_shape
    K1, N = b_shape

    assert K0 == K1

    print(f"{M},{K0},{N}")
