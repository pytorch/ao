# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from utils import DTYPE2STR, benchmark_main_helper2, product_dict

from torchao.sparsity.prototype.fast_sparse_training import swap_linear_with_semi_sparse_linear_

from transformers import AutoModelForQuestionAnswering, logging
from benchmark_sam import get_sam_model, checkpoint_path

logging.set_verbosity_error()

min_run_time = 0.5
device = torch.device("cuda")


configs = [
    "dense",
    "all",
    # (),
    # # ("attention.self", ),
    # # ("attention.output", ),
    # # ("intermediate", ),
    # # ("output", ),
    # ("attention"),
    # ("intermediate", "output"),
    # ("attention.output", "intermediate", "output"),
    # ("attention.self", "attention.output", "intermediate", "output"),
]

CASES = list(
    product_dict(
        model_str=["vit-b/"],
        config = configs,
        batch_size=[1],
        dtype=[torch.bfloat16],
    )
)

class BertTest(nn.Module):

    def __init__(self, model_str, config, batch_size, dtype, bw) -> None:
        super().__init__()
        self.label = model_str
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_str, torch_dtype=dtype).to(device)
        self.bert_layer = self.model.bert.encoder.layer[0]
        hidden_size = self.bert_layer.attention.self.query.weight.shape[0]
        self.input = torch.randn([batch_size, 384, hidden_size]).to(dtype).to(device)
        self.grad = torch.randn([batch_size, 384, hidden_size]).to(dtype).to(device)
        swap_linear_with_semi_sparse_linear_(self.bert_layer, config)
        config_str = " ".join(config)
        self.sub_label = f"{DTYPE2STR[dtype]} ({self.label} | {batch_size} | {config_str}"
        self.to(device).to(dtype)

    def fw(self):
        out = self.bert_layer(self.input)[0]
        self.out = out

    def bw(self):
        self.out.backward(self.grad, retain_graph=True)


class SAMTest(nn.Module):

    def __init__(self, model_str, config, batch_size, dtype, bw) -> None:
        super().__init__()
        self.label = "sam"
        self.model, self.input = get_sam_model(batchsize=batch_size)
        self.model = self.model.to(dtype)
        self.input = self.input.to(device)
        self.grad = torch.clone(self.input)
        sparse_config = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "mlp" in name:
                if config != "dense":
                    sparse_config.append(name)

        if config != "dense":
            swap_linear_with_semi_sparse_linear_(self.model, sparse_config)

        self.sub_label = f"{DTYPE2STR[dtype]} ({self.label} | {batch_size} | {config}"

    def fw(self):
        out = self.model(self.input)
        self.out = out

    def bw(self):
        self.out.backward(self.grad, retain_graph=True)


functions = {
    "runtime": SAMTest
}

benchmark_main_helper2(
    "sam_fw_bw",
    fw=True,
    # bw=True,
    cases=CASES,
    functions=functions,
    cuda_graph=False,
    min_run_time=min_run_time,
)
