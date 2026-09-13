# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Regression test for https://github.com/pytorch/ao/issues/4318.

`torchao/prototype/attention/fp8_fa3/setup.py` imported a module,
`torchao.prototype.attention.config`, that has never existed in the repo,
and separately called `setup_fp8_backend` with a signature
(`config`, `sdpa_fn=...`) that doesn't match the real function in
`shared_utils/setup.py`. The module was also unreachable dead code -
nothing in torchao imported it. It has been removed; this test locks in
that the `fp8_fa3` package and the public attention API still import
cleanly (this doesn't require CUDA/Hopper hardware, only import-time
correctness).
"""

import importlib

from torch.testing._internal.common_utils import TestCase, run_tests


class TestFp8Fa3PackageImport(TestCase):
    def test_fp8_fa3_package_imports(self):
        module = importlib.import_module("torchao.prototype.attention.fp8_fa3")
        for name in module.__all__:
            self.assertTrue(hasattr(module, name))

    def test_fp8_fa3_setup_module_removed(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torchao.prototype.attention.fp8_fa3.setup")

    def test_attention_api_imports(self):
        from torchao.prototype.attention.api import apply_low_precision_attention

        self.assertTrue(callable(apply_low_precision_attention))


if __name__ == "__main__":
    run_tests()
