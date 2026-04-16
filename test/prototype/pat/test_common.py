# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared test utilities for PAT optimizer tests."""

import os
import socket

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import init_device_mesh


class TwoLayerMLP(torch.nn.Module):
    def __init__(self, input_size, output_size, fc_multiplier: int = 4):
        super().__init__()
        middle_size = fc_multiplier * input_size
        self.fc1 = torch.nn.Linear(input_size, middle_size)
        self.fc2 = torch.nn.Linear(middle_size, output_size)

    @staticmethod
    def _linear_prune_config():
        default_config = {"group_type": "ElemGrouper", "prox_type": "ProxLasso"}
        return {(torch.nn.Linear, "weight"): default_config}

    @staticmethod
    def _group_lasso_prune_config():
        default_config = {
            "group_type": "Dim0Grouper",
            "prox_type": "ProxGroupLasso",
        }
        return {(torch.nn.Linear, "weight"): default_config}

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def make_prox_kwargs(gamma, **overrides):
    kwargs = {
        "gamma": gamma,
        "gamma_index_slope": 0.0,
        "disable_vmap": False,
        "is_svd_grouper": False,
    }
    kwargs.update(overrides)
    return kwargs


def optim_step(model, optimizer, dummy_input, label, step):
    output = model(dummy_input[step : step + 1])
    loss = F.cross_entropy(output, label[step : step + 1])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


class DistributedTestMixin:
    """Mixin providing setUpClass/tearDownClass for single-rank distributed tests."""

    mesh = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.environ.setdefault("MASTER_ADDR", "localhost")
        if "MASTER_PORT" not in os.environ:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", 0))
                port = s.getsockname()[1]
            os.environ["MASTER_PORT"] = str(port)
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
        cls.mesh = init_device_mesh("cpu", (1, 1))

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if dist.is_initialized():
            dist.destroy_process_group()
