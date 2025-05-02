import torch
import torch.nn as nn
import time
import argparse
from typing import Dict, List, Tuple
from ..compressed_ffn import CompressedFFN
from ..activation_compression import ActivationCompressor

class BaselineFFN(nn.Module):
    """Baseline FFN without compression for comparison."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.activation(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x

def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, int, int],
    num_iterations: int = 100,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark a model's performance.
    
    Args:
        model: Model to benchmark
        input_shape: Shape of input tensor (batch_size, seq_len, d_model)
        num_iterations: Number of iterations to run
        device: Device to run on
        
    Returns:
        Dictionary with benchmark results
    """
    model = model.to(device)
    model.train()
    
    # Create input tensor
    x = torch.randn(*input_shape, device=device)
    
    # Warmup
    for _ in range(10):
        _ = model(x)
        torch.cuda.synchronize()
    
    # Measure forward pass time
    forward_times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        _ = model(x)
        torch.cuda.synchronize()
        forward_times.append(time.time() - start_time)
    
    # Measure memory usage
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    _ = model(x)
    final_memory = torch.cuda.memory_allocated()
    memory_usage = final_memory - initial_memory
    
    return {
        'forward_time_mean': sum(forward_times) / len(forward_times),
        'forward_time_std': torch.tensor(forward_times).std().item(),
        'memory_usage': memory_usage
    }

def run_benchmarks(
    batch_sizes: List[int] = [8, 16, 32, 64],
    seq_lens: List[int] = [32, 64, 128, 256],
    d_models: List[int] = [512, 1024, 2048],
    d_ffs: List[int] = [2048, 4096, 8192],
    device: str = 'cuda'
):
    """
    Run comprehensive benchmarks comparing compressed and baseline models.
    
    Args:
        batch_sizes: List of batch sizes to test
        seq_lens: List of sequence lengths to test
        d_models: List of model dimensions to test
        d_ffs: List of feed-forward dimensions to test
        device: Device to run on
    """
    results = []
    
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            for d_model in d_models:
                for d_ff in d_ffs:
                    if d_ff < d_model * 2:
                        continue  # Skip invalid configurations
                        
                    input_shape = (batch_size, seq_len, d_model)
                    
                    # Create models
                    baseline = BaselineFFN(d_model, d_ff)
                    compressed = CompressedFFN(d_model, d_ff)
                    
                    # Run benchmarks
                    baseline_results = benchmark_model(baseline, input_shape, device=device)
                    compressed_results = benchmark_model(compressed, input_shape, device=device)
                    
                    # Calculate speedup and memory savings
                    speedup = baseline_results['forward_time_mean'] / compressed_results['forward_time_mean']
                    memory_savings = baseline_results['memory_usage'] / compressed_results['memory_usage']
                    
                    results.append({
                        'batch_size': batch_size,
                        'seq_len': seq_len,
                        'd_model': d_model,
                        'd_ff': d_ff,
                        'baseline_time': baseline_results['forward_time_mean'],
                        'compressed_time': compressed_results['forward_time_mean'],
                        'speedup': speedup,
                        'baseline_memory': baseline_results['memory_usage'],
                        'compressed_memory': compressed_results['memory_usage'],
                        'memory_savings': memory_savings
                    })
                    
                    # Print results
                    print(f"\nConfiguration: batch_size={batch_size}, seq_len={seq_len}, "
                          f"d_model={d_model}, d_ff={d_ff}")
                    print(f"Speedup: {speedup:.2f}x")
                    print(f"Memory savings: {memory_savings:.2f}x")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark compression performance')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run benchmarks on')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for benchmark results')
    args = parser.parse_args()
    
    # Run benchmarks
    results = run_benchmarks(device=args.device)
    
    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main() 