import json
import torch
import platform
import os

def get_arch_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    else:
        # This returns x86_64 or arm64 (for aarch64)
        return platform.machine()


def write_json_result(output_json_path, headers, row):
    """
    Write the result into JSON format, so that it can be uploaded to the benchmark database
    to be displayed on OSS dashboard. The JSON format is defined at
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    mapping_headers = {headers[i]: v for i, v in enumerate(row)}
    record = {
        "benchmark": {
            "name": "TorchAO benchmark",
            "mode": "inference",
            "dtype": mapping_headers["dtype"],
            "extra_info": {
                "device": mapping_headers["device"],
                "arch": mapping_headers["arch"],
            },
        },
        "model": {
            "name": mapping_headers["name"],
            "type": "model",
            "origins": ["pytorch"],
        },
        "metric": {
            "name": mapping_headers["metric"],
            "benchmark_values": [mapping_headers["actual"]],
            "target_value": mapping_headers["target"],
        },
    }

    with open(f"{os.path.splitext(output_json_path)[0]}.json", "a") as f:
        print(json.dumps(record), file=f)
