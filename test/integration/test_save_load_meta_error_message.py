# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Regression test for https://github.com/pytorch/ao/issues/4339 (bug 2).

`TestSaveLoadMeta._test_handle_save_load_meta_impl` in test_integration.py
has two near-identical assertions comparing a float reference (`ref_f`)
against a compiled-quantized output. The second one checked
`SQNR(ref_f, test) > min_sqnr` (correct) but its error message printed
`SQNR(ref_f, ref_q)` (`ref_q` is the *other* assertion's compiled reference,
computed earlier in the method) instead of `SQNR(ref_f, test)` -- so on
failure it reported the wrong number, from the wrong tensor pair.

The method itself needs `torch.compile(mode="max-autotune")` plus a real
quantization API and isn't meaningfully runnable/verifiable without a GPU.
What *is* CPU-verifiable is the fixed source line itself, so this reads the
source file directly (`_test_handle_save_load_meta_impl` is wrapped by
`@torch.no_grad()`/`@run_supported_device_dtype`, which defeats
`inspect.getsource` on the function object) and checks that every assertion
message expression matches its own assertion condition's variable -- i.e.
that this exact copy-paste mistake (message referencing a stale variable
from an earlier, unrelated assertion) cannot silently reappear.
"""

import re
from pathlib import Path

from torch.testing._internal.common_utils import TestCase, run_tests


class TestSaveLoadMetaErrorMessage(TestCase):
    def test_second_sqnr_assertion_message_matches_its_condition(self):
        source = (Path(__file__).parent / "test_integration.py").read_text()
        start = source.index("def _test_handle_save_load_meta_impl(")
        end = source.index("\n    def test_save_load_dqtensors(", start)
        method_source = source[start:end]

        # Find every `assert SQNR(ref_f, <var>) > min_sqnr, (f"... {SQNR(ref_f, <var2>)} ...")`
        # block and check <var> == <var2> for each one found.
        pattern = re.compile(
            r"assert SQNR\(ref_f,\s*(\w+)\)\s*>\s*min_sqnr,\s*\(\s*"
            r'f"got sqnr:\s*\{SQNR\(ref_f,\s*(\w+)\)\}',
        )
        matches = pattern.findall(method_source)
        self.assertEqual(
            len(matches), 2, f"expected 2 SQNR assertions, found {len(matches)}"
        )
        for condition_var, message_var in matches:
            self.assertEqual(
                condition_var,
                message_var,
                f"assertion checks SQNR(ref_f, {condition_var}) but its "
                f"failure message would print SQNR(ref_f, {message_var}) "
                f"instead - message must reference the same tensor the "
                f"condition actually checked",
            )


if __name__ == "__main__":
    run_tests()
