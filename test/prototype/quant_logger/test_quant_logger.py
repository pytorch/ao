# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import csv
import importlib
import pathlib
import sys
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import torchao.prototype.quant_logger
from torchao.prototype.quant_logger import (
    add_activation_loggers,
    enable_log_stats_to_file,
    enable_log_tensor_save_tensors_to_disk,
    log_parameter_info,
    reset_counter,
)
from torchao.utils import torch_version_at_least

torch.manual_seed(0)


def get_toy_model(dim1, dim2):
    return nn.Sequential(
        nn.Linear(dim1, dim2, bias=False),
        nn.ReLU(),
        nn.Linear(dim2, dim1, bias=False),
    )


class ModelWithLoop(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x0):
        x1 = self.fc(x0)
        x2 = F.relu(x1)
        x3 = self.fc(x2)
        return x3


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not torch_version_at_least("2.10.0"), "Need pytorch 2.10+")
class TestQuantLogger(unittest.TestCase):
    def setUp(self):
        # Reload module to restore default log_tensor op (tests may redefine it)
        importlib.reload(torchao.prototype.quant_logger.quant_logger)
        importlib.reload(torchao.prototype.quant_logger)

    def test_log_activations_simple(self):
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            M, K, N = 2, 32, 64
            x = torch.randn(M, K)
            m = get_toy_model(K, N)
            add_activation_loggers(m)
            m(x)

            output = mock_stdout.getvalue()
            print(output, file=sys.__stdout__)
            lines = output.strip().split("\n")

            self.assertIn("t=act, c=0, fqn='0.weight', op='linear'", lines[0])
            self.assertIn("t=act, c=1, fqn='2.weight', op='linear'", lines[1])

            # Parse and verify MKN shape from extra argument
            import re

            mkn_pattern = r"extra='MKN=(\d+)\|(\d+)\|(\d+)'"

            # First linear: input (M, K) @ weight (N, K).T -> (M, N)
            match0 = re.search(mkn_pattern, lines[0])
            self.assertIsNotNone(match0)
            self.assertEqual(int(match0.group(1)), M)
            self.assertEqual(int(match0.group(2)), K)
            self.assertEqual(int(match0.group(3)), N)

            # Second linear: input (M, N) @ weight (K, N).T -> (M, K)
            match1 = re.search(mkn_pattern, lines[1])
            self.assertIsNotNone(match1)
            self.assertEqual(int(match1.group(1)), M)
            self.assertEqual(int(match1.group(2)), N)
            self.assertEqual(int(match1.group(3)), K)

    def test_log_parameter_info_simple(self):
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            K, N = 4, 32
            m = get_toy_model(K, N)
            log_parameter_info(m)

            output = mock_stdout.getvalue()
            print(output, file=sys.__stdout__)
            lines = output.strip().split("\n")
            self.assertIn("t=param, c=0, fqn='0.weight', op='',", lines[0])
            self.assertIn("t=param, c=1, fqn='2.weight', op='',", lines[1])

    def test_loop_simple(self):
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            dim = 32
            x = torch.randn(dim, dim)
            m = ModelWithLoop(dim)

            log_parameter_info(m)
            reset_counter()

            add_activation_loggers(m)
            m(x)

            output = mock_stdout.getvalue()
            print(output, file=sys.__stdout__)
            lines = output.strip().split("\n")
            self.assertIn("t=param, c=0, fqn='fc.weight', op='',", lines[0])
            self.assertIn("t=param, c=1, fqn='fc.bias', op='',", lines[1])
            self.assertIn("t=act, c=0, fqn='fc.weight', op='linear',", lines[2])
            self.assertIn("t=act, c=1, fqn='fc.weight', op='linear',", lines[3])

    def test_custom_logging_fn(self):
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:

            @torch.library.custom_op("quant_logger::log_tensor", mutates_args=("x",))
            def log_tensor(
                x: torch.Tensor, fqn: str, op: str, tag: str, extra: str | None = None
            ) -> None:
                min_val = torch.min(x)
                print(f"custom {tag=}, {fqn=}, {min_val=}")

            M, K, N = 2, 4, 6
            x = torch.randn(M, K)
            m = get_toy_model(K, N)
            add_activation_loggers(m)
            m(x)

            output = mock_stdout.getvalue()
            print(output, file=sys.__stdout__)
            lines = output.strip().split("\n")
            self.assertIn("custom tag='act', fqn='0.weight',", lines[0])
            self.assertIn("custom tag='act', fqn='2.weight',", lines[1])

    def test_custom_logging_fn_save_tensors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            save_dir = tmp_path / "activations"
            enable_log_tensor_save_tensors_to_disk(str(save_dir))

            M, K, N = 2, 4, 6
            x = torch.randn(M, K)
            m = get_toy_model(K, N)
            log_parameter_info(m)
            add_activation_loggers(m)
            m(x)

            # Check saved tensors can be loaded from disk
            param_0 = torch.load(save_dir / "0.weight__param.pt")
            param_2 = torch.load(save_dir / "2.weight__param.pt")
            act_0 = torch.load(save_dir / "0.weight_linear_act.pt")
            act_2 = torch.load(save_dir / "2.weight_linear_act.pt")

            self.assertEqual(param_0.shape, (N, K))
            self.assertEqual(param_2.shape, (K, N))
            self.assertEqual(act_0.shape, (M, K))
            self.assertEqual(act_2.shape, (M, N))

    def test_custom_logging_fn_save_stats(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            filename = tmp_path / "stats.csv"
            enable_log_stats_to_file(str(filename))

            M, K, N = 2, 32, 64
            x = torch.randn(M, K)
            m = get_toy_model(K, N)
            log_parameter_info(m)
            add_activation_loggers(m)
            m(x)

            with open(filename, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["tag"], "param")
            self.assertEqual(rows[0]["fqn"], "0.weight")
            self.assertEqual(rows[1]["tag"], "param")
            self.assertEqual(rows[1]["fqn"], "2.weight")
            self.assertEqual(rows[2]["tag"], "act")
            self.assertEqual(rows[2]["fqn"], "0.weight")
            self.assertEqual(rows[3]["tag"], "act")
            self.assertEqual(rows[3]["fqn"], "2.weight")

    def test_opt_125m(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            model_name = "facebook/opt-125m"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            filename = tmp_path / "stats.csv"
            enable_log_stats_to_file(str(filename))

            log_parameter_info(model)
            add_activation_loggers(model)

            prompt = "Hello, world!"
            inputs = tokenizer(prompt, return_tensors="pt")
            _outputs = model.generate(**inputs, max_new_tokens=1)

            with open(filename, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Check at least one parameter was logged
            param_rows = [r for r in rows if r["tag"] == "param"]
            self.assertGreater(len(param_rows), 0)
            self.assertEqual(param_rows[0]["fqn"], "model.decoder.embed_tokens.weight")

            # Check at least one activation was logged
            act_rows = [r for r in rows if r["tag"] == "act"]
            self.assertGreater(len(act_rows), 0)
            self.assertEqual(
                act_rows[0]["fqn"], "model.decoder.layers.0.self_attn.q_proj.weight"
            )


if __name__ == "__main__":
    unittest.main()
