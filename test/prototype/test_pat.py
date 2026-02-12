# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import random
import unittest

import torch
import torch.nn.functional as F
from torch import nn
from torch.testing._internal import common_utils

from torchao.prototype.pat.group import (
    AttentionHeadGrouperDim0,
    AttentionHeadGrouperDim1,
    PackedSVDGrouper,
    QKGrouper,
    QKSVDGrouper,
    SVDGrouper,
)
from torchao.prototype.pat.layers.masked_layernorm import MaskedLayerNorm
from torchao.prototype.pat.optim import (
    ProxGroupLasso,
    ProxLasso,
    ProxNuclearNorm,
    PruneOptimizer,
)
from torchao.prototype.pat.utils import get_param_groups


class TestMaskedLayerNorm(common_utils.TestCase):
    @common_utils.parametrize("batch", [1, 4])
    @common_utils.parametrize("seq_len", [2, 8])
    @common_utils.parametrize("embed_dim", [16, 64])
    def test_masked_layernorm(self, batch=1, seq_len=2, embed_dim=16):
        dim2_nz = embed_dim // 2
        embed = torch.randn(batch, seq_len, embed_dim)
        embed[..., dim2_nz:] = 0

        masked_layer_norm = MaskedLayerNorm(embed_dim)
        layer_norm = nn.LayerNorm(dim2_nz)
        with torch.no_grad():
            layer_norm.weight.copy_(masked_layer_norm.weight[:dim2_nz])
            layer_norm.bias.copy_(masked_layer_norm.bias[:dim2_nz])

        out = masked_layer_norm(embed)
        expected_out = layer_norm(embed[..., :dim2_nz])
        torch.testing.assert_close(out[..., :dim2_nz], expected_out)


class MHADummyModel(nn.Module):
    def __init__(self, embed_dim, num_heads, n_cls):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=False)
        self.classifier = nn.Linear(embed_dim, n_cls)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        out = self.classifier(attn_output)
        return out


class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, output_size, fc_multiplier: int = 4):
        super(TwoLayerMLP, self).__init__()
        middle_size = fc_multiplier * input_size
        self.fc1 = torch.nn.Linear(input_size, middle_size)
        self.fc2 = torch.nn.Linear(middle_size, output_size)

    @staticmethod
    def _linear_prune_config():
        default_config = {"group_type": "ElemGrouper", "prox_type": "ProxLasso"}
        return {(torch.nn.Linear, "weight"): default_config}

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class TestQKGrouper(common_utils.TestCase):
    def __init__(self, methodName):
        super(TestQKGrouper, self).__init__(methodName)
        self.reg_lambda = 1.0
        self.prox_map = ProxGroupLasso(self.reg_lambda)

    @staticmethod
    def _get_qk(p, embed_dim, qk_reg_index):
        qk = p[:embed_dim] if qk_reg_index == 0 else p[embed_dim : (embed_dim * 2)]
        return qk

    def get_gamma(self, p):
        """Heuristic that uses the mean of the group to set gamma."""
        p_col = p[:, 0]
        gamma = (1 - p_col.mean()) * torch.linalg.vector_norm(p_col)
        gamma.div_(self.prox_map.tau(p_col))
        return gamma

    def _test_post_prune(self, p, qk_orig, embed_dim, qk_reg_index, gamma):
        qk = self._get_qk(p, embed_dim, qk_reg_index)
        nz_mask = qk.sum(dim=0).ne(0)
        self.assertTrue(nz_mask.eq(0).any(), "No columns of Q/K were pruned")

        # original columns that are <= gamma are pruned
        expect_nz_mask = qk_orig.gt(gamma).all(dim=0)
        torch.testing.assert_close(nz_mask > 1, expect_nz_mask, atol=0, rtol=0)

    def _test_mha_inner(self, p, embed_dim, qk_reg_index):
        qk_orig = self._get_qk(p, embed_dim, qk_reg_index).clone()
        qk_no_prune = self._get_qk(p, embed_dim, int(not qk_reg_index)).clone()
        v_orig = p[(embed_dim * 2) :].clone()
        qk_pack_dim = 0
        with QKGrouper(p, qk_pack_dim, qk_reg_index) as grouper:
            self.assertTrue(grouper.p.equal(qk_orig))

            gamma = self.get_gamma(grouper.p)
            _ = torch.vmap(
                self.prox_map.apply_, in_dims=(grouper.in_dims, None), out_dims=0
            )(grouper.p, gamma)

        self._test_post_prune(p, qk_orig, embed_dim, qk_reg_index, gamma)

        # unregularized query or key was not modified
        no_prune = self._get_qk(p, embed_dim, int(not qk_reg_index))
        torch.testing.assert_close(no_prune, qk_no_prune, atol=0, rtol=0)

        # value was not modified
        v = p[(embed_dim * 2) :]
        torch.testing.assert_close(v, v_orig, atol=0, rtol=0)

    @common_utils.parametrize("embed_dim", [16, 64])
    @common_utils.parametrize("num_heads", [2, 4])
    @common_utils.parametrize("qk_reg_index", [0, 1])
    def test_pytorch_mha(self, embed_dim=16, num_heads=4, qk_reg_index=0):
        assert embed_dim % num_heads == 0, (
            f"{embed_dim=} must be divisible by {num_heads=}"
        )

        # single in_proj_weight of shape (embed_dim * 3, embed_dim)
        model = nn.MultiheadAttention(embed_dim, num_heads, bias=False)
        p = model.in_proj_weight.detach()
        self._test_mha_inner(p, embed_dim, qk_reg_index)

    @common_utils.parametrize("qk_reg_index", [0, 1])
    def test_e2e_optimizer(self, embed_dim=64, qk_reg_index=0):
        n_cls = 3
        model = MHADummyModel(embed_dim, num_heads=4, n_cls=n_cls)
        prune_config = {
            "mha.in_proj_weight": {
                "group_type": "QKGrouper",
                "prox_type": "ProxGroupLasso",
                "qk_pack_dim": 0,
                "qk_reg_index": qk_reg_index,
            }
        }
        param_groups = get_param_groups(model, prune_config, verbose=False)
        self.assertEqual(len(param_groups), 3)

        p = model.mha.in_proj_weight.detach()
        qk_orig = self._get_qk(p, embed_dim, qk_reg_index).clone()

        # set lr to gamma since we run a single step
        gamma = self.get_gamma(qk_orig)
        optimizer = PruneOptimizer(
            torch.optim.SGD(param_groups, lr=gamma), reg_lambda=self.reg_lambda
        )

        data = torch.randn(1, 8, embed_dim)
        label = torch.arange(0, n_cls) * data.mean(axis=-1, keepdim=True)
        output = model(data)
        loss = nn.functional.mse_loss(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self._test_post_prune(p, qk_orig, embed_dim, qk_reg_index, gamma)


class TestAttentionHeadGrouper(common_utils.TestCase):
    def __init__(self, methodName):
        super(TestAttentionHeadGrouper, self).__init__(methodName)
        self.reg_lambda = 1.0
        self.prox_map = ProxGroupLasso(self.reg_lambda)

    @staticmethod
    def _get_view_shape_reduce_dim(dim, num_heads, head_pack_dim):
        if head_pack_dim == 0:
            view_shape = (num_heads, -1, dim)
            reduce_dim = (1, 2)
        else:
            view_shape = (dim, num_heads, -1)
            reduce_dim = (0, 2)
        return view_shape, reduce_dim

    def _test_post_prune(self, p, p_orig, head_pack_dim, view_shape, reduce_dim, gamma):
        nz_mask = p.view(*view_shape).sum(dim=reduce_dim).ne(0)
        self.assertTrue(nz_mask.eq(0).any(), "No groups of p were pruned")

        # original groups that are <= gamma are pruned
        expect_nz_mask = p_orig.view(*view_shape).gt(gamma).all(dim=reduce_dim)
        torch.testing.assert_close(nz_mask > 1, expect_nz_mask, atol=0, rtol=0)

    def get_gamma(self, p, head_pack_dim, view_shape):
        """Heuristic that uses the mean of the group to set gamma."""
        p = p.view(*view_shape)
        p_group = p[0] if head_pack_dim == 0 else p[:, 0]
        gamma = (1 - p_group.mean()) * torch.linalg.vector_norm(p_group)
        gamma.div_(self.prox_map.tau(p_group))
        return gamma

    @common_utils.parametrize("dim", [64, 128])
    @common_utils.parametrize("head_pack_dim", [0, 1])
    def test_head_grouper(self, dim=16, head_pack_dim=0, head_dim_ratio=8):
        assert dim % head_dim_ratio == 0, (
            f"{dim=} must be divisible by {head_dim_ratio=}"
        )
        num_heads = dim // 8
        packed_dim = dim * num_heads
        shape = (dim, packed_dim) if head_pack_dim == 0 else (packed_dim, dim)
        model = nn.Linear(*shape, bias=False)
        p = model.weight.detach()
        p_orig = p.clone()
        view_shape, reduce_dim = self._get_view_shape_reduce_dim(
            dim, num_heads, head_pack_dim
        )
        grouper_cls = (
            AttentionHeadGrouperDim0 if head_pack_dim == 0 else AttentionHeadGrouperDim1
        )
        with grouper_cls(p, num_heads) as grouper:
            gamma = self.get_gamma(grouper.p, head_pack_dim, view_shape)
            _ = torch.vmap(
                self.prox_map.apply_, in_dims=(grouper.in_dims, None), out_dims=0
            )(grouper.p, gamma)
            self.assertEqual(grouper.p.size(head_pack_dim), num_heads)
        self._test_post_prune(p, p_orig, head_pack_dim, view_shape, reduce_dim, gamma)


class TestSVDGrouper(common_utils.TestCase):
    def __init__(self, methodName):
        super(TestSVDGrouper, self).__init__(methodName)
        self.reg_lambda = 1.0
        self.prox_map = ProxNuclearNorm(self.reg_lambda)

    @common_utils.parametrize("embed_dim", (16, 64))
    def test_grouper(self, embed_dim=16):
        model = torch.nn.Linear(embed_dim, embed_dim)
        p = model.weight
        with SVDGrouper(p) as grouper:
            gamma = grouper.p.mean()
            p_orig = grouper.p.clone()
            torch.vmap(
                self.prox_map.apply_, in_dims=(grouper.in_dims, None), out_dims=0
            )(grouper.p, gamma)
            expect_nz_mask = p_orig.gt(gamma)
            torch.testing.assert_close(grouper.p.ne(0), expect_nz_mask, atol=0, rtol=0)

    @common_utils.parametrize("embed_dim", (16, 64))
    @common_utils.parametrize("pack_dim", (0, 1))
    def test_qk_grouper(self, embed_dim=16, pack_dim=0):
        shape = [embed_dim, embed_dim]
        shape[int(not pack_dim)] *= 3
        model = torch.nn.Linear(*shape)
        p = model.weight
        with QKSVDGrouper(p, pack_dim=pack_dim) as grouper:
            gamma = grouper.p.mean()
            p_orig = grouper.p.clone()
            torch.vmap(
                self.prox_map.apply_, in_dims=(grouper.in_dims, None), out_dims=0
            )(grouper.p, gamma)
            expect_nz_mask = p_orig.gt(gamma)
            torch.testing.assert_close(grouper.p.ne(0), expect_nz_mask, atol=0, rtol=0)

    @common_utils.parametrize("embed_dim", (16, 64))
    @common_utils.parametrize("pack_dim", (0, 1))
    def test_packed_grouper(self, embed_dim=16, npack=3, pack_dim=0):
        shape = [embed_dim, embed_dim]
        shape[int(not pack_dim)] *= npack
        model = torch.nn.Linear(*shape)
        p = model.weight
        with PackedSVDGrouper(p, npack, pack_dim=pack_dim) as grouper:
            gamma = grouper.p.mean(0).mean()
            p_orig = grouper.p.clone()
            torch.vmap(
                self.prox_map.apply_, in_dims=(grouper.in_dims, None), out_dims=0
            )(grouper.p.flatten(), gamma)
            torch.testing.assert_close(
                grouper.p.ne(0), p_orig.gt(gamma), atol=0, rtol=0
            )
            self.assertEqual(p.data_ptr(), grouper._p.data_ptr())


class TestPruneOptimizer(common_utils.TestCase):
    def __init__(self, methodName):
        super(TestPruneOptimizer, self).__init__(methodName)
        self.reg_lambda = 1.0
        self.prox_map = ProxLasso(self.reg_lambda)

    @staticmethod
    def _optim_step(model, optimizer, dummy_input, label, step):
        output = model(dummy_input[step : step + 1])
        loss = F.cross_entropy(output, label[step : step + 1])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    @common_utils.parametrize("warmup_steps", [0, 5])
    def test_warmup_steps(self, warmup_steps=0, total_steps=10):
        model = TwoLayerMLP(input_size=10, output_size=2)
        prune_config = model._linear_prune_config()
        param_groups = get_param_groups(model, prune_config, verbose=False)
        optimizer = PruneOptimizer(
            torch.optim.SGD(param_groups, lr=0.1),
            reg_lambda=self.reg_lambda,
            warmup_steps=warmup_steps,
        )

        dummy_input = torch.randn(total_steps, 10)
        label = torch.randint(0, 2, (total_steps,))
        for step in range(total_steps):
            self._optim_step(model, optimizer, dummy_input, label, step)
            if step < warmup_steps:
                for group in optimizer.regularized_param_groups():
                    for p in group["params"]:
                        assert "latent" in optimizer.state[p]
            else:
                for group in optimizer.regularized_param_groups():
                    for p in group["params"]:
                        # check that param magnitude is smaller than latent
                        latent = optimizer.state[p]["latent"]
                        self.assertTrue(
                            torch.linalg.norm(p) < torch.linalg.norm(latent)
                        )

    @common_utils.parametrize("healing_start_step", [3, 5])
    def test_healing_start_step(self, healing_start_step=5, total_steps=10):
        model = TwoLayerMLP(input_size=10, output_size=2)
        prune_config = model._linear_prune_config()
        param_groups = get_param_groups(model, prune_config, verbose=False)
        optimizer = PruneOptimizer(
            torch.optim.AdamW(param_groups, lr=0.01),
            reg_lambda=self.reg_lambda,
            warmup_steps=0,
            healing_start_step=healing_start_step,
        )

        dummy_input = torch.randn(total_steps, 10)
        label = torch.randint(0, 2, (total_steps,))
        prev_pruned_p_dct = {}
        for step in range(total_steps):
            self._optim_step(model, optimizer, dummy_input, label, step)
            if step == healing_start_step:
                for group in optimizer.regularized_param_groups():
                    for p in group["params"]:
                        if p.count_nonzero().item():
                            prev_pruned_p_dct[p.data_ptr()] = p.clone().detach()
            elif step > healing_start_step:
                for group in optimizer.regularized_param_groups():
                    for p in group["params"]:
                        prev_pruned_p = prev_pruned_p_dct.get(p.data_ptr())
                        if prev_pruned_p is not None:
                            zero_mask = prev_pruned_p.eq(0)
                            self.assertTrue(p.grad[zero_mask].eq(0).all())
        self.assertFalse(prev_pruned_p is None)


common_utils.instantiate_parametrized_tests(TestMaskedLayerNorm)
common_utils.instantiate_parametrized_tests(TestQKGrouper)
common_utils.instantiate_parametrized_tests(TestAttentionHeadGrouper)
common_utils.instantiate_parametrized_tests(TestSVDGrouper)
common_utils.instantiate_parametrized_tests(TestPruneOptimizer)

if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    unittest.main()
