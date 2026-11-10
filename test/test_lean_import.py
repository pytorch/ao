# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys
import unittest

import torch


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestLeanImport(unittest.TestCase):
    def test_torchao_import_does_not_initialize_cuda(self):
        # patch torch.cuda.current_device to ensure it isn't called during
        # torchao import
        def _patched_current_device():
            raise AssertionError("do not call me")

        old_current_device = torch.cuda.current_device
        torch.cuda.current_device = _patched_current_device

        # the import below should not hit the assertion
        import torchao  # noqa: F401

        torch.cuda.current_device = old_current_device

    def test_torchao_import_is_fork_safe(self):
        # without PYTORCH_NVML_BASED_CUDA_CHECK=1, torch.cuda.is_available()
        # initializes the cuda driver, and forked children can't use cuda after
        code = (
            "import os, torch, torchao\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    torch.zeros(1, device='cuda')\n"
            "    os._exit(0)\n"
            "_, status = os.waitpid(pid, 0)\n"
            "assert os.waitstatus_to_exitcode(status) == 0\n"
        )
        env = dict(os.environ)
        env.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
        subprocess.run([sys.executable, "-c", code], check=True, env=env)


if __name__ == "__main__":
    unittest.main()
