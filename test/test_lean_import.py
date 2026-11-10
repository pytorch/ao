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

    @unittest.skipIf(
        torch.version.hip is not None,
        "rocm: import torchao still queries the gpu (float8/config.py)",
    )
    def test_torchao_import_is_fork_safe(self):
        # without PYTORCH_NVML_BASED_CUDA_CHECK=1, torch.cuda.is_available()
        # poisons fork: torch marks forked children as bad, and their first
        # cuda call raises
        code = (
            "import os, torch, torchao\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    try:\n"
            "        torch.zeros(1, device='cuda')\n"
            "    except BaseException:\n"
            "        import traceback\n"
            "        traceback.print_exc()\n"
            "        os._exit(1)\n"
            "    os._exit(0)\n"
            "_, status = os.waitpid(pid, 0)\n"
            "assert os.waitstatus_to_exitcode(status) == 0\n"
        )
        env = dict(os.environ)
        env.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipIf(
    torch.version.hip is not None,
    "rocm: import torchao still queries the gpu (float8/config.py)",
)
class TestLeanImportNoDeviceQuery(unittest.TestCase):
    def test_torchao_import_does_not_query_cuda(self):
        # runs without a gpu too: any device query during import raises
        code = (
            "import torch\n"
            "def _fail(*args, **kwargs):\n"
            "    raise AssertionError('import torchao queried the gpu')\n"
            "for name in ('is_available', 'device_count', 'get_device_capability',\n"
            "             'get_device_properties', '_lazy_init'):\n"
            "    setattr(torch.cuda, name, _fail)\n"
            "import torchao\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
