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


if __name__ == "__main__":
    unittest.main()
