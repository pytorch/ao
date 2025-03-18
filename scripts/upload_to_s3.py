# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import io
import json
import os
from functools import lru_cache
from typing import Any

import boto3


@lru_cache
def get_s3_resource() -> Any:
    return boto3.resource("s3")


def upload_to_s3(
    bucket_name: str,
    key: str,
    json_path: str,
) -> None:
    print(f"Writing {json_path} documents to S3")
    data = []
    with open(f"{os.path.splitext(json_path)[0]}.json", "r") as f:
        for l in f.readlines():
            data.append(json.loads(l))

    body = io.StringIO()
    for benchmark_entry in data:
        json.dump(benchmark_entry, body)
        body.write("\n")

    try:
        get_s3_resource().Object(
            f"{bucket_name}",
            f"{key}",
        ).put(
            Body=body.getvalue(),
            ContentType="application/json",
        )
    except Exception as e:
        print("fail to upload to s3:", e)
        return
    print("Done!")


if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser(
        description="Upload benchmark result json file to clickhouse"
    )
    parser.add_argument(
        "--json-path",
        type=str,
        help="json file path to upload to click house",
        required=True,
    )
    args = parser.parse_args()
    today = datetime.date.today()
    today = datetime.datetime.combine(today, datetime.time.min)
    today_timestamp = str(int(today.timestamp()))
    print("Today timestamp:", today_timestamp)
    import subprocess

    # Execute the command and capture the output
    output = subprocess.check_output(["hostname", "-s"])
    # Decode the output from bytes to string
    hostname = output.decode("utf-8").strip()
    upload_to_s3(
        "ossci-benchmarks",
        f"v3/pytorch/ao/{hostname}/torchao-models-" + today_timestamp + ".json",
        args.json_path,
    )
