# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Regression test for https://github.com/pytorch/ao/issues/4577.

`torchao.experimental.ops.mps` eagerly calls `_load_torchao_mps_lib()` at
import time. On any non-macOS platform where the `torch.ops.torchao`
MPS operators aren't already registered (e.g. an XPU wheel on Windows),
this used to raise a bare `RuntimeError`. `pkgutil.walk_packages` /
`pkgutil.iter_modules` - used by libraries like diffusers and transformers
for optional-backend discovery - only swallow `ImportError` while probing
submodule importability; any other exception aborts the whole package walk.
These tests run on any platform (they don't require macOS or MPS hardware)
and lock in that:
  1. failing to load the MPS lib raises ImportError, not RuntimeError.
  2. pkgutil.walk_packages over torchao's tree no longer aborts because of
     this module.
"""

import pkgutil
import sys

from torch.testing._internal.common_utils import TestCase, run_tests


class TestMpsLibImportError(TestCase):
    def test_load_torchao_mps_lib_raises_import_error(self):
        if sys.platform == "darwin":
            self.skipTest(
                "this test targets the non-macOS failure path; on macOS "
                "the lib may load successfully"
            )
        # `torchao.experimental.ops.mps.utils` calls `_load_torchao_mps_lib()`
        # eagerly in the parent package's `__init__.py`, so the failure
        # surfaces as soon as anything under `mps` is imported for the
        # first time in this process.
        with self.assertRaises(ImportError):
            import torchao.experimental.ops.mps  # noqa: F401

    def test_pkgutil_walk_packages_survives_missing_mps_lib(self):
        import torchao

        # This must not raise: pkgutil only catches ImportError internally,
        # so if `torchao.experimental.ops.mps` ever raises anything else on
        # import, this walk aborts for every downstream caller (diffusers,
        # transformers, etc.), not just torchao users who touch MPS.
        modules = list(pkgutil.walk_packages(torchao.__path__, prefix="torchao."))
        names = {m.name for m in modules}
        self.assertIn("torchao.experimental.ops.mps", names)


if __name__ == "__main__":
    run_tests()
