import torch
from torch import Tensor

def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    See https://pytorch.org/vision/main/generated/torchvision.ops.nms.html
    """
    return torch.ops.torchao.nms.default(boxes, scores, iou_threshold)


# Defines the meta kernel / fake kernel / abstract impl
@torch.library.impl_abstract("torchao::nms")
def _(dets, scores, iou_threshold):
    torch._check(dets.dim() == 2, lambda: f"boxes should be a 2d tensor, got {dets.dim()}D")
    torch._check(dets.size(1) == 4, lambda: f"boxes should have 4 elements in dimension 1, got {dets.size(1)}")
    torch._check(scores.dim() == 1, lambda: f"scores should be a 1d tensor, got {scores.dim()}")
    torch._check(
        dets.size(0) == scores.size(0),
        lambda: f"boxes and scores should have same number of elements in dimension 0, got {dets.size(0)} and {scores.size(0)}",
    )
    ctx = torch._custom_ops.get_ctx()
    num_to_keep = ctx.create_unbacked_symint()
    return dets.new_empty(num_to_keep, dtype=torch.long)


def prepack_fp6_weight(fp6_weight: Tensor) -> Tensor:
    return torch.ops.torchao.prepack_fp6_weight.default(fp6_weight)


@torch.library.impl_abstract("torchao::prepack_fp6_weight")
def _(fp6_weight):
    torch._check(fp6_weight.dim() == 2, lambda: f"weight should be a 2d tensor, got {fp6_weight.dim()}D")
    return torch.empty_like(fp6_weight)


def fake_fp6_to_fp6(fake_fp6_tensor: Tensor) -> Tensor:
    return torch.ops.torchao.fake_fp6_to_fp6.default(fake_fp6_tensor)


@torch.library.impl_abstract("torchao::fake_fp6_to_fp6")
def _(fake_fp6_tensor):
    torch._check(fake_fp6_tensor.dim() == 2, lambda: f"weight should be a 2d tensor, got {fake_fp6_tensor.dim()}D")
    torch._check(fake_fp6_tensor.dtype is torch.float16, lambda: f"weight must be FP16, got {fake_fp6_tensor.dtype}")
    M, K = fake_fp6_tensor.shape
    torch._check(K % 4  == 0, lambda: f"second dimension must be a multiple of 4, got {K}")
    return torch.empty((M, K * 6 // 8), dtype=torch.uint8, device=fake_fp6_tensor.device)


def fp16act_fp6weight_linear(_in_feats: Tensor, _weights: Tensor, _scales: Tensor, splitK: int = 1) -> Tensor:
    return torch.ops.torchao.fp16act_fp6weight_linear.default(_in_feats, _weights, _scales, splitK)


def fp6_weight_dequant(fp6_tensor: Tensor, fp16_scale: Tensor) -> Tensor:
    return torch.ops.torchao.fp6_weight_dequant.default(fp6_tensor, fp16_scale)
