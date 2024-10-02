"""Based on https://github.com/facebookresearch/SpinQuant"""

import torch
from torch import nn
import tqdm  # TODO: remove this dependency

from torchao.quantization.hadamard_utils import apply_exact_had_to_linear, random_hadamard_matrix, get_hadK, matmul_hadU_cuda
from torchao._models.llama.model import FeedForward, Transformer, Attention



@torch.inference_mode()
def rotate_model_r2(model):
    torch.manual_seed(0)
    R2 = random_hadamard_matrix(model.config.head_dim, "cuda")
    
    # Apply R2 rotation to all multi-head self-attention modules
    for m in model.modules():
        if isinstance(m, Attention):
            head_dim = model.config.head_dim

            # Rotate W_o
            apply_exact_had_to_linear(m.wo, had_dim=head_dim, output=False, R2=R2)

            # Extract W_v
            kv_size = model.config.n_local_heads * head_dim
            wq, wk, wv = m.wqkv.weight.data.split([model.config.dim, kv_size, kv_size], dim=0)
            out_features, in_features = wv.shape
            wv_mod = nn.Linear(in_features, out_features, bias=False, device=wq.device, dtype=wq.dtype)
            wv_mod.weight.data = wv

            # Rotate W_v
            apply_exact_had_to_linear(wv_mod, had_dim=head_dim, output=True, R2=R2)

            m.wqkv.weight.data = torch.cat([wq, wk, wv_mod.weight.data], dim=0)


@torch.inference_mode()
def rotate_model_r4(model):
    layers = [layer for layer in model.layers]
    for idx, layer in enumerate(
        tqdm.tqdm(layers, unit="layer", desc="Applying R4 rotation to W_down")
    ):
        # Rotate the MLP output weights
        W = layers[idx].feed_forward.w2
        print(f"Min/max before R4 rotation: {W.weight.data.min().item():.2f}/{W.weight.data.max().item():.2f}")
        apply_exact_had_to_linear(
            W, had_dim=-1, output=False
        )  # apply exact (inverse) hadamard on the weights of mlp output
        print(f"Min/max *after* R4 rotation: {W.weight.data.min().item():.2f}/{W.weight.data.max().item():.2f}")


class ActQuantWrapper(nn.Module):  # TODO: rename to FeedForwardR4Wrapper
    # Based on SpinQuant utils/quant_utils:ActQuantWrapper

    def __init__(self, module: "FeedForward", had_K, K, fp32_had=False):
        super().__init__()
        self.module = module
        self.register_buffer("had_K", had_K)
        self.K = K
        self.fp32_had = fp32_had

    def forward(self, x):
        x_dtype = x.dtype

        # x -> x @ R4
        if self.fp32_had:  # Full Hadamard in FP32
            x = matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
        else:  # Full Hadamard in FP16
            x = matmul_hadU_cuda(x, self.had_K, self.K)

        # Quantization should happen here when using dynamic quantization. 
        # TODO: check this

        # FFN forward
        x = self.module(x)

        return x

def replace_feedforward_layers(module, had_K, K, fp32_had):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name == "w2":  # FeedForward last layer
            setattr(module, name, ActQuantWrapper(child, had_K, K, fp32_had))
        else:
            replace_feedforward_layers(child, had_K, K, fp32_had)


def add_activation_wrapper_r3_r4(model):
    # Modify the forward pass to rotate the activations at specific points
    
    # R3
    # TODO

    # R4
    # eval_utils/main.py:36
    fp32_had = False   # default used in SpinQuant
    had_K, K = get_hadK(model.config.intermediate_size)
    replace_feedforward_layers(model, had_K, K, fp32_had)  # TODO: check the result

    # for attr in dir(model):
    #     tmp = getattr(model, attr)
    #     if type(tmp) == FeedForward:
    #         setattr(model, attr, ActQuantWrapper(tmp, had_K, K, fp32_had))



def apply_spinquant_to_llama(model: Transformer):
    """
    Apply SpinQuant to Llama: https://arxiv.org/abs/2405.16406
    
    Currently, this only applies the R2 + R3 + R4 rotation matrices to the model (not R1).
    """
    rotate_model_r2(model)
    rotate_model_r4(model)
    add_activation_wrapper_r3_r4(model)
