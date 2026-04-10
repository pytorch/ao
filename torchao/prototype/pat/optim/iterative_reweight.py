import sys

from torch import Tensor


class IterativeReweight:
    def __init__(
        self,
        reweight_freq: int,
        reweight_end_step: int = sys.maxsize,
        eps: float = 1e-3,
    ):
        assert reweight_freq > 0, (
            f"Expected reweight_freq to be positive but got {reweight_freq}"
        )
        assert eps > 0, f"Expected eps to be positive but got {eps}"
        self.reweight_freq = reweight_freq
        self.reweight_end_step = reweight_end_step
        self.eps = eps

    def should_update(self, step: int) -> bool:
        return step % self.reweight_freq == 0 and step <= self.reweight_end_step

    def __call__(self, group_norm: Tensor, sigma: Tensor) -> Tensor:
        """Assume that group_norm is already divided by tau"""
        return 1.0 / (group_norm / (sigma + self.eps) + self.eps)
