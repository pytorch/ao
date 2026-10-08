import unittest

import torch
from torch import nn

from torchao.prototype.quantization.module_swap.data_getters import (
    get_module_input_data,
)


class TestGetModuleInputData(unittest.TestCase):
    def test_success_removes_only_capture_hook(self):
        model = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 4))
        data = torch.randn(4, 16)
        target = model[1]
        seen = []
        existing_hook = target.register_forward_hook(
            lambda module, inputs, output: seen.append(inputs[0].detach())
        )
        self.addCleanup(existing_hook.remove)
        original_hooks = tuple(target._forward_hooks)

        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                seen.clear()
                captured = get_module_input_data(model, data, target, batch_size)
                torch.testing.assert_close(captured, model[0](data))
                self.assertFalse(captured.requires_grad)
                self.assertEqual(len(seen), len(data) // batch_size)
                self.assertEqual(tuple(target._forward_hooks), original_hooks)
                self.assertEqual(model(data).shape, (4, 4))

    def test_forward_error_removes_capture_hook_and_allows_retry(self):
        # Exercise both an upstream failure and a failure inside the target.
        for target_index in (0, 1):
            with self.subTest(target_index=target_index):
                model = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 4))
                target = model[target_index]
                data = torch.randn(2, 16)
                with self.assertRaises(RuntimeError):
                    get_module_input_data(model, data[:, :-1], target, batch_size=1)

                self.assertEqual(model(data).shape, (2, 4))
                self.assertEqual(tuple(target._forward_hooks), ())
                captured = get_module_input_data(model, data, target, batch_size=1)
                expected = data if target_index == 0 else model[0](data)
                torch.testing.assert_close(captured, expected)
                self.assertEqual(tuple(target._forward_hooks), ())

    def test_existing_hook_error_is_preserved(self):
        model = nn.Sequential(nn.Linear(16, 8))
        data = torch.randn(2, 16)
        error = ValueError("existing hook failed")

        def failing_hook(module, inputs, output):
            raise error

        existing_hook = model[0].register_forward_hook(failing_hook)
        self.addCleanup(existing_hook.remove)
        original_hooks = tuple(model[0]._forward_hooks)
        with self.assertRaises(ValueError) as raised:
            get_module_input_data(model, data, model[0], batch_size=1)

        self.assertIs(raised.exception, error)
        self.assertEqual(tuple(model[0]._forward_hooks), original_hooks)


if __name__ == "__main__":
    unittest.main()
