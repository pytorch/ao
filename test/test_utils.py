import unittest
from unittest.mock import patch
from torchao.utils import torch_version_at_least

class TestTorchVersionAtLeast(unittest.TestCase):
    def test_torch_version_at_least(self):
        test_cases = [
            ("2.5.0a0+git9f17037", "2.5.0", True),
            ("2.5.0a0+git9f17037", "2.4.0", True),
            ("2.5.0.dev20240708+cu121", "2.5.0", True),
            ("2.5.0.dev20240708+cu121", "2.4.0", True),
            ("2.5.0", "2.4.0", True),
            ("2.5.0", "2.5.0", True),
            ("2.4.0", "2.4.0", True),
            ("2.4.0", "2.5.0", False),
        ]

        for torch_version, compare_version, expected_result in test_cases:
            with patch('torch.__version__', torch_version):
                result = torch_version_at_least(compare_version)

                self.assertEqual(result, expected_result, f"Failed for torch.__version__={torch_version}, comparing with {compare_version}")
                print(f"{torch_version}: {result}")

if __name__ == '__main__':
    unittest.main()
