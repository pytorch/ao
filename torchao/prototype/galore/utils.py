# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import torch


def get_orthogonal_matrix(weights, rank, type):
    module_params = weights

    if module_params.data.dtype != torch.float:
        float_data = False
        original_type = module_params.data.dtype
        original_device = module_params.data.device
        matrix = module_params.data.float()
    else:
        float_data = True
        matrix = module_params.data

    U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)

    # make the smaller matrix always to be orthogonal matrix
    if type == "right":
        # A = U[:, :rank] @ torch.diag(s[:rank])
        B = Vh[:rank, :]

        if not float_data:
            B = B.to(original_device).type(original_type)
        return B
    elif type == "left":
        A = U[:, :rank]
        # B = torch.diag(s[:rank]) @ Vh[:rank, :]
        if not float_data:
            A = A.to(original_device).type(original_type)
        return A
    elif type == "full":
        A = U[:, :rank]
        B = Vh[:rank, :]
        if not float_data:
            A = A.to(original_device).type(original_type)
            B = B.to(original_device).type(original_type)
        return [A, B]
    else:
        raise ValueError("type should be left, right or full")


class TestGaLoreProjector:
    def __init__(
        self,
        rank=128,
        scale=1.0,
        proj_type="std",
    ):
        self.rank = rank
        self.scale = scale

        if proj_type != "std":
            raise ("Only std projection is supported")

        self.proj_type = proj_type

        self.ortho_matrix = None

    def update_orthogonal_matrix(self, full_rank_grad):
        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            self.ortho_matrix = get_orthogonal_matrix(
                full_rank_grad, self.rank, type="right"
            )
        else:
            self.ortho_matrix = get_orthogonal_matrix(
                full_rank_grad, self.rank, type="left"
            )

    def project(self, full_rank_grad):
        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        else:
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)

        return low_rank_grad

    def project_back(self, low_rank_grad):
        if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        else:
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)

        return full_rank_grad * self.scale


def make_copy(*args):
    return [t.detach().clone() for t in args]


# def adam_step(
#     exp_avg,
#     exp_avg2,
#     grad,
#     galore_proj,
#     params,
#     step_size=1e-4,
#     beta1=BETA1,
#     beta2=BETA2,
#     eps=EPS,
# ):
#     grad = galore_proj.project(grad)
#     exp_avg = beta1 * exp_avg + (1 - beta1) * grad
#     exp_avg2 = beta2 * exp_avg2 + (1 - beta2) * torch.square(grad)
#     denom = exp_avg2.sqrt() + eps
#     norm_grad = exp_avg / denom
#     norm_grad = galore_proj.project_back(norm_grad)
#     # params = params - step_size * norm_grad
#     return exp_avg, exp_avg2, denom, norm_grad
