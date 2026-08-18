# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

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


class TestWalkPackages(unittest.TestCase):
    def test_walk_packages_does_not_raise(self):
        # Optional backends that are not built must signal their absence with
        # ImportError, which pkgutil.walk_packages and other importlib-based
        # discovery tolerate. Anything else escapes and breaks the walk for
        # every caller. See https://github.com/pytorch/ao/issues/4577
        # Run in a subprocess so the parent module cache is untouched.
        code = (
            "import pkgutil, torchao\n"
            "[None for _ in pkgutil.walk_packages(torchao.__path__, prefix='torchao.')]\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual(
            result.returncode,
            0,
            f"walking torchao raised a non-ImportError:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
