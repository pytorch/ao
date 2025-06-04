# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import hashlib
import json
import os
import platform
import time

import torch


def get_arch_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    else:
        # This returns x86_64 or arm64 (for aarch64)
        return platform.machine()


def write_json_result_ossci(output_json_path, headers, row):
    """
    Write the result into JSON format, so that it can be uploaded to the benchmark database
    to be displayed on OSS dashboard. The JSON format is defined at
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database

    OSS CI version, that will leave many fields to be filled in by CI
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
                "min_sqnr": mapping_headers["min_sqnr"],
                # True means compile is enabled, False means eager mode
                "compile": mapping_headers["compile"],
            },
        },
        "model": {
            "name": mapping_headers["name"],
            "type": "model",
            "origins": ["torchao"],
        },
        "metric": {
            "name": mapping_headers["metric"],
            "benchmark_values": [mapping_headers["actual"]],
            "target_value": mapping_headers["target"],
        },
    }

    with open(f"{os.path.splitext(output_json_path)[0]}.json", "a") as f:
        print(json.dumps(record), file=f)


def write_json_result_local(output_json_path, headers, row):
    """
    Write the result into JSON format, so that it can be uploaded to the benchmark database
    to be displayed on OSS dashboard. The JSON format is defined at
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database

    Local version (filling in dummy values for fields that should be populated by CI)
    """
    mapping_headers = {headers[i]: v for i, v in enumerate(row)}
    today = datetime.date.today()
    sha_hash = hashlib.sha256(str(today).encode("utf-8")).hexdigest()
    first_second = datetime.datetime.combine(today, datetime.time.min)
    workflow_id = int(first_second.timestamp())
    job_id = workflow_id + 1
    record = {
        "timestamp": int(time.time()),
        "schema_version": "v3",
        "name": "devvm local benchmark",
        "repo": "pytorch/ao",
        "head_branch": "main",
        "head_sha": sha_hash,
        "workflow_id": workflow_id,
        "run_attempt": 1,
        "job_id": job_id,
        "benchmark": {
            "name": "TorchAO benchmark",
            "mode": "inference",
            "dtype": mapping_headers["dtype"],
            "extra_info": {
                "device": mapping_headers["device"],
                "arch": mapping_headers["arch"],
                "min_sqnr": mapping_headers["min_sqnr"],
                # True means compile is enabled, False means eager mode
                "compile": mapping_headers["compile"],
            },
        },
        "model": {
            "name": mapping_headers["name"],
            "type": "model",
            "origins": ["torchao"],
        },
        "metric": {
            "name": mapping_headers["metric"],
            "benchmark_values": [mapping_headers["actual"]],
            "target_value": mapping_headers["target"],
        },
    }

    with open(f"{os.path.splitext(output_json_path)[0]}.json", "a") as f:
        print(json.dumps(record), file=f)
