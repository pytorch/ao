# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys
import unittest

import torch

# Pretend torch was built with USE_DISTRIBUTED=0: the top-level
# torch.distributed package still exists, but it reports itself as unavailable
# and none of its submodules can be imported.
_IMPORT_WITHOUT_DISTRIBUTED = """
import sys

import torch

torch.distributed.is_available = lambda: False

for name in [n for n in sys.modules if n.startswith("torch.distributed.")]:
    del sys.modules[name]


class BlockDistributedSubmodules:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("torch.distributed."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


sys.meta_path.insert(0, BlockDistributedSubmodules())

import torchao  # noqa: F401
"""


class TestImportWithoutDistributed(unittest.TestCase):
    def test_torchao_import_without_torch_distributed(self):
        # https://github.com/pytorch/ao/issues/3452 - torchao must import on a
        # torch built with USE_DISTRIBUTED=0. Run this out of process so the
        # parent interpreter's module cache is left alone.
        result = subprocess.run(
            [sys.executable, "-c", _IMPORT_WITHOUT_DISTRIBUTED],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"importing torchao without torch.distributed failed:\n{result.stderr}",
        )


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


if __name__ == "__main__":
    unittest.main()
