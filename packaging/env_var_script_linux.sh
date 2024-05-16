# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is sourced into the environment before building a pip wheel. It
# should typically only contain shell variable assignments. Be sure to export
# any variables so that subprocesses will see them.

# Enable pybindings so that users can execute ExecuTorch programs from python.
if [[ ${CHANNEL:-nightly} == "nightly" ]]; then
  export TORCHAO_NIGHTLY=1
fi
