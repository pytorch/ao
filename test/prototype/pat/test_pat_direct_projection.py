# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Regressions for latent and direct projected PAT optimizer updates."""

import copy
import math
from unittest import mock

import torch
from torch.testing._internal import common_utils

from test.prototype.pat.test_common import TwoLayerMLP, optim_step
from torchao.prototype.pat.optim import (
    MinSparsityConstraint,
    PruneOptimizer,
    build_prune_optimizer,
)
from torchao.prototype.pat.utils import get_param_groups


def _param_groups(model, min_sparsity=0.5, prox_freq=1, scheduled=False):
    config = {
        (torch.nn.Linear, "weight"): {
            "group_type": "Dim0Grouper",
            "prox_type": "MinSparsityConstraint",
            "min_sparsity": min_sparsity,
            "min_sparsity_schedule": scheduled,
            "prox_freq": prox_freq,
        }
    }
    return get_param_groups(model, config, verbose=False)


def _base_optimizer(groups, name, lr=0.1):
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, foreach=False)
    return torch.optim.AdamW(
        groups, lr=lr, betas=(0.8, 0.9), weight_decay=0.01, foreach=False
    )


def _optimizer(
    model,
    name="sgd",
    latent_weights=None,
    warmup_steps=2,
    healing_start_step=8,
    prox_freq=1,
    scheduled=False,
    min_sparsity=0.5,
    lr=0.1,
    reg_lambda=0.0,
    builder=False,
):
    base = _base_optimizer(
        _param_groups(model, min_sparsity, prox_freq, scheduled), name, lr
    )
    kwargs = {} if latent_weights is None else {"latent_weights": latent_weights}
    if builder:
        return build_prune_optimizer(
            base,
            prune_reg_lambda=reg_lambda,
            prune_warmup_steps=warmup_steps,
            prune_healing_start_step=healing_start_step,
            **kwargs,
        )
    return PruneOptimizer(
        base,
        warmup_steps=warmup_steps,
        healing_start_step=healing_start_step,
        reg_lambda=reg_lambda,
        **kwargs,
    )


def _linear_model():
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(3, 6, dtype=torch.float32)
    with torch.no_grad():
        model.weight.copy_(torch.arange(1, 19, dtype=torch.float32).reshape(6, 3) / 4)
        model.bias.copy_(torch.arange(1, 7, dtype=torch.float32) / 10)
    return model


def _mlp_and_data(steps):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = TwoLayerMLP(input_size=4, output_size=2).float()
    generator = torch.Generator().manual_seed(1234)
    inputs = torch.randn(steps, 4, dtype=torch.float32, generator=generator)
    labels = torch.randint(0, 2, (steps,), generator=generator)
    return model, inputs, labels


def _set_dense_gradients(model, step):
    for p in model.parameters():
        gradient = torch.linspace(0.2, 1.1, p.numel(), dtype=p.dtype, device=p.device)
        p.grad = gradient.reshape_as(p) * (1 + step / 10)


def _dense_step(model, optimizer, step):
    optimizer.zero_grad()
    _set_dense_gradients(model, step)
    optimizer.step()


def _moment_keys(name):
    return ("momentum_buffer",) if name == "sgd" else ("step", "exp_avg", "exp_avg_sq")


class TestDirectProjection(common_utils.TestCase):
    def _assert_no_latent(self, optimizer):
        self.assertNotIn("_state", optimizer.__dict__)
        for state in optimizer.state.values():
            self.assertNotIn("latent", state)
        for state in optimizer.state_dict()["state"].values():
            self.assertNotIn("latent", state)

    def _assert_pair_equal(self, model_a, optimizer_a, model_b, optimizer_b):
        self.assertEqual(model_a.state_dict(), model_b.state_dict(), atol=0, rtol=0)
        self.assertEqual(
            optimizer_a.state_dict(), optimizer_b.state_dict(), atol=0, rtol=0
        )
        self.assertEqual(optimizer_a.num_steps, optimizer_b.num_steps)

    def test_omitted_default_matches_explicit_latent_trajectory_and_state(self):
        for name in ("sgd", "adamw"):
            for warmup in (0, 2):
                for prox_freq in (1, 3):
                    with self.subTest(name=name, warmup=warmup, prox_freq=prox_freq):
                        model_a, inputs, labels = _mlp_and_data(12)
                        model_b = copy.deepcopy(model_a)
                        kwargs = {
                            "name": name,
                            "warmup_steps": warmup,
                            "prox_freq": prox_freq,
                            "scheduled": True,
                        }
                        default = _optimizer(model_a, **kwargs)
                        explicit = _optimizer(model_b, latent_weights=True, **kwargs)
                        self.assertTrue(default.latent_weights)
                        self.assertTrue(explicit.latent_weights)
                        for step in range(12):
                            optim_step(model_a, default, inputs, labels, step)
                            optim_step(model_b, explicit, inputs, labels, step)
                            self._assert_pair_equal(model_a, default, model_b, explicit)
                            for group in default.regularized_param_groups():
                                for p in group["params"]:
                                    self.assertIn("latent", default.state[p])
                        zeroed = model_a.fc1.weight.detach().eq(0).all(dim=1)
                        self.assertTrue(zeroed.any())
                        self.assertGreater(
                            default.state[model_a.fc1.weight]["latent"][zeroed]
                            .abs()
                            .sum()
                            .item(),
                            0.0,
                        )

    def test_direct_mode_never_uses_latent_and_keeps_base_moments(self):
        for name in ("sgd", "adamw"):
            for warmup in (0, 2):
                for prox_freq in (1, 3):
                    with self.subTest(name=name, warmup=warmup, prox_freq=prox_freq):
                        model = _linear_model()
                        optimizer = _optimizer(
                            model,
                            name=name,
                            latent_weights=False,
                            warmup_steps=warmup,
                            prox_freq=prox_freq,
                        )
                        self.assertEqual(len(optimizer.state), 0)
                        self._assert_no_latent(optimizer)
                        with (
                            mock.patch.object(
                                optimizer,
                                "_init_latent_state",
                                side_effect=AssertionError,
                            ),
                            mock.patch.object(
                                optimizer,
                                "save_latent_params",
                                side_effect=AssertionError,
                            ),
                            mock.patch.object(
                                optimizer,
                                "restore_latent_params",
                                side_effect=AssertionError,
                            ),
                        ):
                            for step in range(12):
                                _dense_step(model, optimizer, step)
                                self._assert_no_latent(optimizer)
                                for p in model.parameters():
                                    state = optimizer.state[p]
                                    for key in _moment_keys(name):
                                        self.assertIn(key, state)
                                        self.assertTrue(
                                            torch.isfinite(state[key]).all()
                                        )
                                    moment = state[_moment_keys(name)[-1]]
                                    self.assertEqual(moment.shape, p.shape)
                                    self.assertGreater(moment.abs().sum().item(), 0.0)

    def test_direct_updates_match_manual_base_step_and_projection(self):
        for name in ("sgd", "adamw"):
            for warmup in (0, 2):
                for prox_freq in (1, 3):
                    with self.subTest(name=name, warmup=warmup, prox_freq=prox_freq):
                        model = _linear_model()
                        reference = copy.deepcopy(model)
                        optimizer = _optimizer(
                            model,
                            name=name,
                            latent_weights=False,
                            warmup_steps=warmup,
                            healing_start_step=100,
                            prox_freq=prox_freq,
                        )
                        base = _base_optimizer(_param_groups(reference), name)
                        prox = MinSparsityConstraint(reg_lambda=0.0, min_sparsity=0.5)
                        gamma = 0.0
                        for step in range(10):
                            previous_zeros = reference.weight.detach().eq(0).all(dim=1)
                            _dense_step(model, optimizer, step)
                            _dense_step(reference, base, step)
                            # The oracle never calls PruneOptimizer or its helpers.
                            if step >= warmup:
                                gamma += 0.1
                                if (step - warmup) % prox_freq == 0:
                                    with torch.no_grad():
                                        prox.apply_(reference.weight, gamma=gamma)
                                elif previous_zeros.any():
                                    self.assertTrue(
                                        model.weight[previous_zeros].ne(0).any()
                                    )
                            self.assertEqual(
                                model.state_dict(),
                                reference.state_dict(),
                                atol=0,
                                rtol=0,
                            )
                            for p, q in zip(
                                model.parameters(), reference.parameters(), strict=True
                            ):
                                for key in _moment_keys(name):
                                    self.assertEqual(
                                        optimizer.state[p][key],
                                        base.state[q][key],
                                        atol=0,
                                        rtol=0,
                                    )
                            self.assertEqual(optimizer.param_groups[0]["gamma"], gamma)
                            self.assertEqual(optimizer.num_steps, step + 1)
                            self._assert_no_latent(optimizer)

    def test_zero_gradient_preserves_projected_point_on_skipped_steps(self):
        for warmup in (0, 2):
            with self.subTest(warmup=warmup):
                model = _linear_model()
                optimizer = PruneOptimizer(
                    torch.optim.SGD(_param_groups(model, prox_freq=3), lr=0.1),
                    warmup_steps=warmup,
                    healing_start_step=100,
                    latent_weights=False,
                )
                for _ in range(warmup + 1):
                    for p in model.parameters():
                        p.grad = torch.zeros_like(p)
                    optimizer.step()
                projected = copy.deepcopy(model.state_dict())
                zeroed = model.weight.detach().eq(0).all(dim=1)
                self.assertEqual(zeroed.sum().item(), 3)
                gamma = optimizer.param_groups[0]["gamma"]
                # Includes two skipped steps, a projection, then two more skips.
                for _ in range(5):
                    for p in model.parameters():
                        p.grad = torch.zeros_like(p)
                    optimizer.step()
                    gamma += 0.1
                    self.assertEqual(model.state_dict(), projected, atol=0, rtol=0)
                    self.assertTrue(model.weight[zeroed].eq(0).all())
                    self.assertEqual(optimizer.param_groups[0]["gamma"], gamma)
                    self._assert_no_latent(optimizer)

    def test_healing_freezes_actual_zeros_with_dense_gradients_and_momentum(self):
        for name in ("sgd", "adamw"):
            for latent_weights in (False, True):
                for warmup in (0, 2):
                    with self.subTest(
                        name=name, latent_weights=latent_weights, warmup=warmup
                    ):
                        model = _linear_model()
                        optimizer = _optimizer(
                            model,
                            name=name,
                            latent_weights=latent_weights,
                            warmup_steps=warmup,
                            healing_start_step=5,
                            prox_freq=3,
                            scheduled=True,
                        )
                        for step in range(5):
                            _dense_step(model, optimizer, step)
                        frozen = model.weight.detach().eq(0)
                        self.assertEqual(frozen.all(dim=1).sum().item(), 3)
                        self.assertTrue((~frozen).any())
                        key = "momentum_buffer" if name == "sgd" else "exp_avg"
                        self.assertTrue(
                            optimizer.state[model.weight][key][frozen].ne(0).all()
                        )
                        gamma = optimizer.param_groups[0]["gamma"]
                        for step in range(5, 9):
                            before = model.weight.detach().clone()
                            _set_dense_gradients(model, step)
                            self.assertTrue(model.weight.grad[frozen].ne(0).all())
                            optimizer.step()
                            self.assertTrue(model.weight.grad[frozen].eq(0).all())
                            self.assertTrue(model.weight.grad[~frozen].ne(0).all())
                            self.assertTrue(model.weight[frozen].eq(0).all())
                            self.assertFalse(
                                torch.equal(model.weight[~frozen], before[~frozen])
                            )
                            self.assertTrue(
                                optimizer.state[model.weight][key][frozen].ne(0).all()
                            )
                            self.assertEqual(optimizer.param_groups[0]["gamma"], gamma)
                            if not latent_weights:
                                self._assert_no_latent(optimizer)

    def test_builder_forwards_mode_and_schedule_arguments(self):
        for name in ("sgd", "adamw"):
            for latent_weights in (None, True, False):
                with self.subTest(name=name, latent_weights=latent_weights):
                    model_a = _linear_model()
                    model_b = copy.deepcopy(model_a)
                    kwargs = {
                        "name": name,
                        "latent_weights": latent_weights,
                        "warmup_steps": 2,
                        "healing_start_step": 6,
                        "prox_freq": 2,
                        "reg_lambda": 0.7,
                    }
                    built = _optimizer(model_a, builder=True, **kwargs)
                    direct = _optimizer(model_b, **kwargs)
                    self.assertIsInstance(built, PruneOptimizer)
                    self.assertEqual(built.latent_weights, latent_weights is not False)
                    self.assertEqual(built.warmup_steps, 2)
                    self.assertEqual(built.healing_start_step, 6)
                    self.assertEqual(built.param_groups[0]["reg_lambda"], 0.7)
                    for step in range(9):
                        _dense_step(model_a, built, step)
                        _dense_step(model_b, direct, step)
                        self._assert_pair_equal(model_a, built, model_b, direct)
                        if latent_weights is False:
                            self._assert_no_latent(built)

    def test_both_modes_reach_target_sparsity_before_healing(self):
        for latent_weights in (False, True):
            for scheduled in (False, True):
                for target in (0.0, 0.5, 1.0):
                    with self.subTest(
                        latent_weights=latent_weights,
                        scheduled=scheduled,
                        target=target,
                    ):
                        model = _linear_model()
                        optimizer = _optimizer(
                            model,
                            latent_weights=latent_weights,
                            warmup_steps=2,
                            healing_start_step=7,
                            prox_freq=3,
                            scheduled=scheduled,
                            min_sparsity=target,
                            lr=0.01,
                        )
                        n_rows = model.weight.shape[0]
                        for step in range(10):
                            _dense_step(model, optimizer, step)
                            if step in (2, 5) or step >= 6:
                                effective_target = target
                                if scheduled and step < 6:
                                    progress = (step - 2) / (7 - 2)
                                    effective_target *= 1 - (1 - progress) ** 3
                                expected_rows = math.ceil(effective_target * n_rows)
                                # Step 6 is off cadence but must set the final mask.
                                rows = (
                                    model.weight.detach().eq(0).all(dim=1).sum().item()
                                )
                                self.assertEqual(rows, expected_rows)
                                self.assertEqual(
                                    optimizer.relative_sparsity, expected_rows / n_rows
                                )
                                self.assertEqual(
                                    optimizer.state[model.weight][
                                        "sparsity_frac"
                                    ].item(),
                                    expected_rows / n_rows,
                                )
                        if not latent_weights:
                            self._assert_no_latent(optimizer)

    def test_state_dict_is_delegated_and_does_not_serialize_mode(self):
        for name in ("sgd", "adamw"):
            for latent_weights in (False, True):
                with self.subTest(name=name, latent_weights=latent_weights):
                    model = _linear_model()
                    optimizer = _optimizer(
                        model, name=name, latent_weights=latent_weights
                    )
                    for step in range(5):
                        _dense_step(model, optimizer, step)
                    base = optimizer.base_optimizer
                    with mock.patch.object(
                        base, "state_dict", wraps=base.state_dict
                    ) as delegated:
                        checkpoint = optimizer.state_dict()
                    delegated.assert_called_once_with()
                    self.assertEqual(checkpoint, base.state_dict(), atol=0, rtol=0)
                    self.assertEqual(set(checkpoint), {"state", "param_groups"})
                    self.assertNotIn("latent_weights", checkpoint)
                    for group in checkpoint["param_groups"]:
                        self.assertNotIn("latent_weights", group)
                    for state in checkpoint["state"].values():
                        self.assertNotIn("latent_weights", state)
                    resumed = _optimizer(
                        copy.deepcopy(model), name=name, latent_weights=latent_weights
                    )
                    resumed.load_state_dict(copy.deepcopy(checkpoint))
                    self.assertEqual(resumed.latent_weights, latent_weights)
                    self.assertEqual(resumed.state_dict(), checkpoint, atol=0, rtol=0)

    def test_reconstruct_load_and_continue_in_each_training_phase(self):
        for name in ("sgd", "adamw"):
            for latent_weights in (False, True):
                for checkpoint_step in (1, 6, 10):
                    with self.subTest(
                        name=name,
                        latent_weights=latent_weights,
                        checkpoint_step=checkpoint_step,
                    ):
                        model_a, inputs, labels = _mlp_and_data(12)
                        kwargs = {
                            "name": name,
                            "latent_weights": latent_weights,
                            "warmup_steps": 2,
                            "healing_start_step": 8,
                            "prox_freq": 3,
                            "scheduled": True,
                        }
                        optimizer_a = _optimizer(model_a, **kwargs)
                        for step in range(checkpoint_step):
                            optim_step(model_a, optimizer_a, inputs, labels, step)
                        checkpoint = copy.deepcopy(optimizer_a.state_dict())
                        model_b = copy.deepcopy(model_a)
                        optimizer_b = _optimizer(model_b, **kwargs)
                        optimizer_b.load_state_dict(copy.deepcopy(checkpoint))
                        self.assertEqual(optimizer_b.latent_weights, latent_weights)
                        self._assert_pair_equal(
                            model_a, optimizer_a, model_b, optimizer_b
                        )
                        for p, q in zip(
                            model_a.parameters(), model_b.parameters(), strict=True
                        ):
                            self.assertNotEqual(p.data_ptr(), q.data_ptr())
                            for key, value in optimizer_a.state[p].items():
                                if isinstance(value, torch.Tensor):
                                    self.assertNotEqual(
                                        value.data_ptr(),
                                        optimizer_b.state[q][key].data_ptr(),
                                    )
                        for step in range(checkpoint_step, 12):
                            optim_step(model_a, optimizer_a, inputs, labels, step)
                            optim_step(model_b, optimizer_b, inputs, labels, step)
                            self._assert_pair_equal(
                                model_a, optimizer_a, model_b, optimizer_b
                            )
                            if not latent_weights:
                                self._assert_no_latent(optimizer_a)
                                self._assert_no_latent(optimizer_b)

    def test_legacy_latent_checkpoint_loads_into_default_mode(self):
        for name in ("sgd", "adamw"):
            for prox_freq in (1, 3):
                with self.subTest(name=name, prox_freq=prox_freq):
                    legacy_model = _linear_model()
                    base = _base_optimizer(
                        _param_groups(legacy_model, prox_freq=prox_freq), name
                    )
                    group = base.param_groups[0]
                    group.update(gamma=0.0, reg_lambda=0.0, num_steps=0)
                    prox = MinSparsityConstraint(reg_lambda=0.0, min_sparsity=0.5)
                    warmup, checkpoint_step = 2, 5
                    # Build legacy state independently, including skipped prox steps.
                    for step in range(9):
                        if step == checkpoint_step:
                            checkpoint = copy.deepcopy(base.state_dict())
                            resumed_model = copy.deepcopy(legacy_model)
                            resumed = _optimizer(
                                resumed_model,
                                name=name,
                                warmup_steps=warmup,
                                healing_start_step=100,
                                prox_freq=prox_freq,
                            )
                            self.assertTrue(resumed.latent_weights)
                            resumed.load_state_dict(copy.deepcopy(checkpoint))
                            self.assertEqual(
                                resumed.state_dict(), checkpoint, atol=0, rtol=0
                            )
                            self.assertIn("latent", resumed.state[resumed_model.weight])
                        _set_dense_gradients(legacy_model, step)
                        with torch.no_grad():
                            if step == warmup:
                                base.state[legacy_model.weight]["latent"].copy_(
                                    legacy_model.weight
                                )
                            elif step > warmup:
                                legacy_model.weight.copy_(
                                    base.state[legacy_model.weight]["latent"]
                                )
                            base.step()
                            state = base.state[legacy_model.weight]
                            if step < warmup:
                                if "latent" not in state:
                                    state["latent"] = (
                                        legacy_model.weight.detach().clone()
                                    )
                            else:
                                state["latent"].copy_(legacy_model.weight)
                                group["gamma"] += 0.1
                                if (step - warmup) % prox_freq == 0:
                                    zeros, _ = prox.apply_(
                                        legacy_model.weight, gamma=group["gamma"]
                                    )
                                    state["sparsity_frac"] = (
                                        zeros / legacy_model.weight.numel()
                                    )
                        group["num_steps"] = step + 1
                        if step >= checkpoint_step:
                            _dense_step(resumed_model, resumed, step)
                            self.assertEqual(
                                legacy_model.state_dict(),
                                resumed_model.state_dict(),
                                atol=0,
                                rtol=0,
                            )
                            self.assertEqual(
                                base.state_dict(), resumed.state_dict(), atol=0, rtol=0
                            )
                            self.assertEqual(resumed.num_steps, step + 1)


if __name__ == "__main__":
    common_utils.run_tests()
