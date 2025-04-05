# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    local_dir: str = field(
        default=None, metadata={"help": "Local Path of storing inputs and outputs "}
    )
    input_model_filename: Optional[str] = field(
        default="test-input", metadata={"help": "Input model relative manifold path"}
    )
    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative manifold path"}
    )
    output_model_local_path: str = field(
        default=None, metadata={"help": "Output model local path, do not set manually"}
    )
    w_bits: Optional[int] = field(
        default=32,
        metadata={
            "help": "#bits to use for quantization; use 16 for evaluating base model. choices=[4, 8, 32]"
        },
    )
    contain_weight_clip_val: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Set contain_weight_clip_val=True when load a trained quantized model."
        },
    )

@dataclass
class DataArguments:
    max_train_samples: Optional[int] = field(
        default=-1,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=-1,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    train_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Train data local path"}
    )
    eval_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Eval data local path"}
    )



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="/tmp/output/")
    model_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated). 512 or 1024"
        },
    )
    qat: Optional[bool] = field(default=False)


def process_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(model_args.local_dir, exist_ok=True)

    assert model_args.output_model_local_path is None

    model_args.output_model_local_path = os.path.join(
        model_args.local_dir, "models", str(model_args.output_model_filename)
    )

    return model_args, data_args, training_args
