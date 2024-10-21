import unittest
import torch
from torchao.dtypes.uintx.uintx import UintxTensor, to_uintx

class TestUintxSlicing(unittest.TestCase):
    def setUp(self):
        self.original_data = torch.tensor([10, 25, 40, 55, 5, 20, 35, 50], dtype=torch.uint8)
        self.uintx_tensor = to_uintx(self.original_data, torch.uint6)
        return self.uintx_tensor


    def test_basic_slicing(self):
        sliced_uintx = self.uintx_tensor[2:6]
        sliced_data = sliced_uintx.get_plain()
        expected_slice = torch.tensor([40, 55, 5, 20], dtype=torch.uint8)
        self.assertTrue(torch.all(sliced_data == expected_slice),
                        f"Expected: {expected_slice}, Got: {sliced_data}")

    def test_step_slicing(self):
        step_sliced_uintx = self.uintx_tensor[1::2]
        step_sliced_data = step_sliced_uintx.get_plain()
        expected_step_slice = torch.tensor([25, 55, 20, 50], dtype=torch.uint8)
        self.assertTrue(torch.all(step_sliced_data == expected_step_slice),
                        f"Expected: {expected_step_slice}, Got: {step_sliced_data}")

    def test_negative_indexing(self):
        negative_sliced_uintx = self.uintx_tensor[-3:]
        negative_sliced_data = negative_sliced_uintx.get_plain()
        expected_negative_slice = torch.tensor([20, 35, 50], dtype=torch.uint8)
        self.assertTrue(torch.all(negative_sliced_data == expected_negative_slice),
                        f"Expected: {expected_negative_slice}, Got: {negative_sliced_data}")

    def test_single_element_indexing(self):
        single_element = self.uintx_tensor[4].get_plain()
        expected_element = torch.tensor([5], dtype=torch.uint8)
        self.assertEqual(single_element.item(), expected_element.item(),
                         f"Expected: {expected_element.item()}, Got: {single_element.item()}")

    #TODO: Implement these tests
    def test_slice_assignment(self):
        new_data = torch.tensor([60, 61], dtype=torch.uint8)
        self.uintx_tensor[3:5] = to_uintx(new_data, torch.uint6)
        modified_data = self.uintx_tensor.get_plain()
        expected_modified = torch.tensor([10, 25, 40, 60, 61, 20, 35, 50], dtype=torch.uint8)
        self.assertTrue(torch.all(modified_data == expected_modified),
                        f"Expected: {expected_modified}, Got: {modified_data}")

    def test_out_of_bounds_slicing(self):
        out_of_bounds_uintx = self.uintx_tensor[5:10]
        out_of_bounds_data = out_of_bounds_uintx.get_plain()
        expected_out_of_bounds = torch.tensor([20, 35, 50], dtype=torch.uint8)
        self.assertTrue(torch.all(out_of_bounds_data == expected_out_of_bounds),
                        f"Expected: {expected_out_of_bounds}, Got: {out_of_bounds_data}")

if __name__ == "__main__":
    unittest.main()

   