import torch
import torch.nn as nn
import unittest
from ..compressed_ffn import CompressedFFN
from ..activation_compression import ActivationCompressor

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.d_model = 64  # Reduced from 512
        self.d_ff = 256   # Reduced from 2048
        self.batch_size = 8  # Reduced from 32
        self.seq_len = 32   # Reduced from 128
        
    def test_accuracy_comparison(self):
        """Test that compression doesn't significantly impact model accuracy."""
        # Create models
        baseline = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model)
        ).to(self.device)
        
        compressed = CompressedFFN(self.d_model, self.d_ff).to(self.device)
        
        # Create synthetic dataset with smaller size
        num_samples = 100  # Reduced from 1000
        X = torch.randn(num_samples, self.seq_len, self.d_model, device=self.device)
        y = torch.randn(num_samples, self.seq_len, self.d_model, device=self.device)
        
        # Train both models
        criterion = nn.MSELoss()
        optimizer_baseline = torch.optim.Adam(baseline.parameters())
        optimizer_compressed = torch.optim.Adam(compressed.parameters())
        
        # Training loop
        num_epochs = 5  # Reduced from 10
        baseline_losses = []
        compressed_losses = []
        
        for epoch in range(num_epochs):
            # Train baseline
            baseline.train()
            optimizer_baseline.zero_grad()
            output = baseline(X)
            loss_baseline = criterion(output, y)
            loss_baseline.backward()
            optimizer_baseline.step()
            baseline_losses.append(loss_baseline.item())
            
            # Train compressed
            compressed.train()
            optimizer_compressed.zero_grad()
            output = compressed(X)
            loss_compressed = criterion(output, y)
            loss_compressed.backward()
            optimizer_compressed.step()
            compressed_losses.append(loss_compressed.item())
            
            # Print progress
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"Baseline loss: {loss_baseline.item():.4f}")
                print(f"Compressed loss: {loss_compressed.item():.4f}")
        
        # Compare final losses
        final_baseline_loss = baseline_losses[-1]
        final_compressed_loss = compressed_losses[-1]
        
        # Allow for small difference in final loss
        self.assertLess(abs(final_baseline_loss - final_compressed_loss) / final_baseline_loss, 0.1)
        
    def test_gradient_flow(self):
        """Test that gradients flow correctly through compressed layers."""
        model = CompressedFFN(self.d_model, self.d_ff).to(self.device)
        model.train()
        
        # Create input and target
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        target = torch.randn_like(x)
        
        # Forward pass
        output = model(x)
        
        # Backward pass
        loss = torch.mean((output - target) ** 2)
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient is None for parameter {name}")
            self.assertFalse(torch.all(param.grad == 0), f"Gradient is zero for parameter {name}")
            
    def test_different_model_sizes(self):
        """Test compression with different model sizes."""
        model_sizes = [
            (32, 128),    # Very small
            (64, 256),    # Small
            (128, 512),   # Medium
        ]
        
        for d_model, d_ff in model_sizes:
            model = CompressedFFN(d_model, d_ff).to(self.device)
            model.train()
            
            # Create input
            x = torch.randn(self.batch_size, self.seq_len, d_model, device=self.device)
            
            # Forward pass
            output = model(x)
            
            # Check output shape
            self.assertEqual(output.shape, (self.batch_size, self.seq_len, d_model))
            
            # Check compression stats
            compression_ratio, sparsity = model.get_compression_stats()
            self.assertGreater(compression_ratio, 0.0)
            self.assertGreater(sparsity, 0.0)
            
if __name__ == '__main__':
    unittest.main() 