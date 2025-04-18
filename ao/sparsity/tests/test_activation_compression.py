import torch
import unittest
from ..activation_compression import ActivationCompressor, CompressedActivation

class TestActivationCompression(unittest.TestCase):
    def setUp(self):
        self.compressor = ActivationCompressor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_compression_decompression(self):
        # Create a sparse tensor
        tensor = torch.zeros(10, 10, device=self.device)
        tensor[0, 0] = 1.0
        tensor[5, 5] = 2.0
        tensor[9, 9] = 3.0
        
        # Compress
        compressed = self.compressor.compress_tensor(tensor)
        
        # Verify compression
        self.assertEqual(compressed['values'].shape[0], 3)  # Should have 3 non-zero values
        self.assertEqual(compressed['indices'].shape[0], 3)  # Should have 3 indices
        
        # Decompress
        decompressed = self.compressor.decompress_tensor(compressed)
        
        # Verify decompression
        self.assertTrue(torch.allclose(tensor, decompressed))
        
    def test_compression_ratio(self):
        # Create a sparse tensor with 10% non-zero values
        tensor = torch.zeros(100, 100, device=self.device)
        num_non_zero = 1000  # 10% of 10000
        indices = torch.randint(0, 100, (num_non_zero, 2), device=self.device)
        values = torch.rand(num_non_zero, device=self.device)
        tensor[indices[:, 0], indices[:, 1]] = values
        
        # Compress
        compressed = self.compressor.compress_tensor(tensor)
        
        # Calculate compression ratio
        original_size = tensor.element_size() * tensor.numel()
        compressed_size = (
            compressed['values'].element_size() * compressed['values'].numel() +
            compressed['indices'].element_size() * compressed['indices'].numel()
        )
        
        # Verify compression ratio is better than 1:1
        self.assertLess(compressed_size, original_size)
        
    def test_compressed_activation_module(self):
        # Create a simple model with compressed activation
        model = CompressedActivation()
        model.train()  # Enable training mode
        
        # Create input tensor
        x = torch.randn(10, 10, device=self.device)
        
        # Forward pass
        y = model(x)
        
        # Verify compression happened
        self.assertIsNotNone(model.compressed_data)
        
        # Backward pass
        grad_output = torch.ones_like(y)
        grad_input = model.backward(grad_output)
        
        # Verify decompression happened
        self.assertIsNone(model.compressed_data)
        
if __name__ == '__main__':
    unittest.main() 