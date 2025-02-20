import yaml
import torch
from typing import Dict, List, Any, Tuple
from pathlib import Path
from itertools import product

class BenchmarkConfig:
    def __init__(self, quantization: str, params: Dict[str, Any], matrix_shape: List[int]):
        self.quantization = quantization
        self.m, self.k, self.n = matrix_shape
        self.precision = self._parse_precision(params['precision'])
        self.compile = params.get('compile', False)
        self.device = params.get('device', 'cuda')
        self.name = f'benchmark_{self.quantization}_m{self.m}_k{self.k}_n{self.n}'

    @staticmethod
    def _parse_precision(precision_str: str) -> torch.dtype:
        """Convert string precision to torch dtype"""
        return getattr(torch, precision_str.split('.')[-1])

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for main function"""
        return {
            'quantization': self.quantization,
            'm': self.m,
            'k': self.k,
            'n': self.n,
            'precision': self.precision,
            'compile': self.compile,
            'device': self.device
        }

def load_benchmark_configs(config_path: str) -> List[BenchmarkConfig]:
    """Load benchmark configurations from YAML file"""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    quantizations = config_data['quantizations']
    params = config_data['model_params']
    matrix_shapes = params['matrix_shapes']
    
    configs = []
    # Generate all combinations of quantizations and matrix shapes
    for quant, shape in product(quantizations, matrix_shapes):
        configs.append(BenchmarkConfig(quant, params, shape))
    
    return configs

def run_benchmarks_from_config(config_path: str) -> None:
    """Run benchmarks using configurations from YAML file"""
    from bench_inference_quant import run
    
    configs = load_benchmark_configs(config_path)
    for config in configs:
        print(f"\nRunning benchmark: {config.name}")
        run(**config.to_dict())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run benchmarks from config file")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark_config.yml",
        help="Path to benchmark configuration file"
    )
    args = parser.parse_args()
    run_benchmarks_from_config(args.config) 