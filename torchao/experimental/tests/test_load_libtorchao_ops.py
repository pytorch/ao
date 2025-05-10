# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestLibTorchAoOpsLoader(unittest.TestCase):
    def test_find_and_load_success(self):
        mock_paths = [Path("/test/path1")]
        mock_lib = MagicMock()
        mock_lib.__str__.return_value = "/test/path1/libtorchao_ops_aten.so"

        with patch("pathlib.Path.glob", return_value=[mock_lib]):
            with patch("torch.ops.load_library") as mock_load:
                from ..op_lib import find_and_load_libtorchao_ops

                find_and_load_libtorchao_ops(mock_paths)

                mock_load.assert_called_once_with("/test/path1/libtorchao_ops_aten.so")

    def test_no_library_found(self):
        mock_paths = [Path("/test/path1"), Path("/test/path2")]

        with patch("pathlib.Path.glob", return_value=[]):
            from ..op_lib import find_and_load_libtorchao_ops

            with self.assertRaises(FileNotFoundError):
                find_and_load_libtorchao_ops(mock_paths)

    def test_multiple_libraries_error(self):
        mock_paths = [Path("/test/path1")]
        mock_lib1 = MagicMock()
        mock_lib2 = MagicMock()
        mock_libs = [mock_lib1, mock_lib2]

        with patch("pathlib.Path.glob", return_value=mock_libs):
            from ..op_lib import find_and_load_libtorchao_ops

            try:
                find_and_load_libtorchao_ops(mock_paths)
                self.fail("Expected AssertionError was not raised")
            except AssertionError as e:
                expected_error_msg = f"Expected to find one libtorchao_ops_aten.* library at {mock_paths[0]}, but found 2"
                self.assertIn(expected_error_msg, str(e))


if __name__ == "__main__":
    unittest.main()
