from itertools import product
from typing import Any, Dict, List, Tuple

import torch
import yaml


class BenchmarkConfig:
    def __init__(
        self,
        quantization: str,
        params: Dict[str, Any],
        shape_name: str,
        shape: List[int],
    ):
        self.quantization = quantization
        self.m, self.k, self.n = shape
        self.shape_name = shape_name
        self.precision = self._parse_precision(params["precision"])
        self.compile = params.get("compile", False)
        self.device = params.get("device", "cuda")
        self.model_type = params.get("model_type", "linear")
        self.name = f"benchmark_{self.quantization}_{self.shape_name}_m{self.m}_k{self.k}_n{self.n}"

    @staticmethod
    def _parse_precision(precision_str: str) -> torch.dtype:
        """Convert string precision to torch dtype"""
        return getattr(torch, precision_str.split(".")[-1])

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for main function"""
        return {
            "quantization": self.quantization,
            "m": self.m,
            "k": self.k,
            "n": self.n,
            "precision": self.precision,
            "compile": self.compile,
            "device": self.device,
            "model_type": self.model_type,
        }


def get_shapes_for_config(shape_config: Dict[str, Any]) -> List[Tuple[str, List[int]]]:
    """Get shapes for a given configuration"""
    name = shape_config["name"]
    if name == "custom":
        return [(name, shape) for shape in shape_config["shapes"]]
    # else:
    #     return [(name, shape) for shape in get_name_to_shapes_iter(name, None, None, None)]


def load_benchmark_configs(config_path: str) -> List[BenchmarkConfig]:
    """Load benchmark configurations from YAML file"""
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    quantizations = config_data["quantizations"]
    params = config_data["model_params"]

    configs = []
    # Process each shape configuration
    for shape_config in params["matrix_shapes"]:
        shapes = get_shapes_for_config(shape_config)
        # Generate combinations for each shape
        for quant, (shape_name, shape) in product(quantizations, shapes):
            configs.append(BenchmarkConfig(quant, params, shape_name, shape))

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
        help="Path to benchmark configuration file",
    )
    args = parser.parse_args()
    run_benchmarks_from_config(args.config)
