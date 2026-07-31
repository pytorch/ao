# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Iterable

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

from packaging import version
from torchao.quantization.utils import compute_error
from torchao.utils import is_sm_at_least_90


def get_blockwise_fp8_distributed_skip_reason(
    *,
    triton_module,
    min_cuda_devices: int,
    sm90_reason: str,
) -> str | None:
    if not torch.cuda.is_available():
        return "CUDA not available"
    if torch.cuda.device_count() < min_cuda_devices:
        return f"Need at least {min_cuda_devices} CUDA devices"
    if not is_sm_at_least_90():
        return sm90_reason
    if version.parse(triton_module.__version__) < version.parse("3.3.0"):
        return "Triton version < 3.3.0"
    return None


def full_tensor(
    tensor: torch.Tensor,
    *,
    unwrap_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    tensor = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
    return unwrap_fn(tensor) if unwrap_fn is not None else tensor


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 2e-2,
    rtol: float = 2e-2,
    min_sqnr: float | None = None,
    unwrap_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    actual = full_tensor(actual, unwrap_fn=unwrap_fn).float()
    expected = full_tensor(expected, unwrap_fn=unwrap_fn).float()
    if min_sqnr is not None:
        sqnr = compute_error(expected, actual).item()
        assert sqnr >= min_sqnr, f"SQNR {sqnr} must be >= {min_sqnr}"
        return
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def broadcast_module(
    module: torch.nn.Module,
    *,
    unwrap_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    for param in module.parameters():
        dist.broadcast(full_tensor(param, unwrap_fn=unwrap_fn), src=0)


def get_replicated_local_batch(
    *,
    replica_count: int,
    replica_index: int,
    iter_idx: int,
    sample_shape: tuple[int, ...],
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(100 + iter_idx)
    global_input = torch.randn(
        (replica_count, *sample_shape),
        device=device,
        dtype=dtype,
    )
    global_target = torch.randn_like(global_input)
    dist.broadcast(global_input, src=0)
    dist.broadcast(global_target, src=0)
    return (
        global_input[replica_index].contiguous(),
        global_target[replica_index].contiguous(),
    )


def assert_parameters_are_dtensors(parameters: Iterable[torch.Tensor]) -> None:
    for param in parameters:
        assert isinstance(param, DTensor)


def allreduce_reference_grads(
    model: torch.nn.Module,
    *,
    world_size: int,
    group=None,
    unwrap_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    for param in model.parameters():
        assert param.grad is not None
        grad = full_tensor(param.grad, unwrap_fn=unwrap_fn)
        dist.all_reduce(grad, group=group)
        grad.div_(world_size)


def assert_dtensor_parameter_grads_match(
    ref_parameters: Iterable[torch.nn.Parameter],
    dist_parameters: Iterable[torch.nn.Parameter],
    *,
    min_sqnr: float | None = None,
    unwrap_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    for ref_param, dist_param in zip(ref_parameters, dist_parameters, strict=True):
        assert ref_param.grad is not None
        assert dist_param.grad is not None
        assert isinstance(dist_param, DTensor)
        assert isinstance(dist_param.grad, DTensor)
        assert_close(
            dist_param.grad,
            ref_param.grad,
            min_sqnr=min_sqnr,
            unwrap_fn=unwrap_fn,
        )


def assert_dtensor_parameter_values_match(
    ref_parameters: Iterable[torch.nn.Parameter],
    dist_parameters: Iterable[torch.nn.Parameter],
    *,
    min_sqnr: float | None = None,
    unwrap_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    for ref_param, dist_param in zip(ref_parameters, dist_parameters, strict=True):
        assert isinstance(dist_param, DTensor)
        assert_close(
            dist_param,
            ref_param,
            min_sqnr=min_sqnr,
            unwrap_fn=unwrap_fn,
        )


class BlockwiseFP8DTensorTestMixin:
    def _build_cuda_mesh(self):
        torch.cuda.set_device(self.rank % torch.cuda.device_count())
        mesh = self.build_device_mesh()
        mesh._device_type = "cuda"
        return mesh

    def _local_shard(self, tensor: torch.Tensor, placement, *, contiguous: bool):
        if isinstance(placement, Replicate):
            return tensor.contiguous() if contiguous else tensor
        if not isinstance(placement, Shard):
            raise AssertionError(f"Unsupported placement: {placement}")

        self.assertEqual(tensor.size(placement.dim) % self.world_size, 0)
        chunk = tensor.size(placement.dim) // self.world_size
        local = tensor.narrow(placement.dim, self.rank * chunk, chunk)
        return local.contiguous() if contiguous else local

    def _dtensor_from_global(
        self,
        mesh,
        tensor: torch.Tensor,
        placement,
        *,
        contiguous: bool,
    ):
        local_tensor = self._local_shard(tensor, placement, contiguous=contiguous)
        return local_tensor, DTensor.from_local(
            local_tensor,
            mesh,
            [placement],
            run_check=False,
        )

    def _assert_dtensor_matches(
        self,
        actual: DTensor,
        expected_local: torch.Tensor,
        expected_global: torch.Tensor,
        expected_placement,
        expected_global_shape=None,
        *,
        atol: float = 0.0,
        rtol: float = 0.0,
        min_global_sqnr: float | None = None,
    ) -> None:
        self.assertIsInstance(actual, DTensor)
        self.assertEqual(actual.placements, (expected_placement,))
        if expected_global_shape is not None:
            self.assertEqual(tuple(actual.shape), tuple(expected_global_shape))
        torch.testing.assert_close(actual.to_local(), expected_local, atol=0, rtol=0)
        actual_global = actual.redistribute(placements=[Replicate()]).to_local()
        if min_global_sqnr is not None:
            self.assertGreaterEqual(
                compute_error(expected_global, actual_global).item(),
                min_global_sqnr,
            )
            return
        torch.testing.assert_close(
            actual_global,
            expected_global,
            atol=atol,
            rtol=rtol,
        )

    def _assert_distributed_output(
        self,
        dist_output: DTensor,
        expected_local_output: torch.Tensor,
        expected_global_output: torch.Tensor,
        expected_placement,
        expected_global_shape,
        *,
        global_atol: float = 0.0,
        global_rtol: float = 0.0,
    ) -> None:
        self._assert_dtensor_matches(
            dist_output,
            expected_local_output,
            expected_global_output,
            expected_placement,
            expected_global_shape,
            atol=global_atol,
            rtol=global_rtol,
        )
