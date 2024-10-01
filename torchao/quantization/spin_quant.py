"""Based on https://github.com/facebookresearch/SpinQuant"""

import torch
from torch import nn

from torchao.quantization.hadamard_utils import apply_exact_had_to_linear


def R4_rotate_down_proj_weights(layer: "TransformerBlock"):
    # Rotate the MLP output weights and bias.
    W = layer.feed_forward.w2
    apply_exact_had_to_linear(
        W, had_dim=-1, output=False
    )  # apply exact (inverse) hadamard on the weights of mlp output


def rotate_model_r2(model):
    from torchao._models.llama.model import Attention

    torch.manual_seed(0)
    R2 = _random_hadamard_matrix(model.config.head_dim, "cuda")
    
    # Apply R2 rotation to all multi-head self-attention modules
    for m in model.modules():
        if isinstance(m, Attention):
            assert isinstance(m.wqkv, nn.Linear)
            assert isinstance(m.wo, nn.Linear)

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
def rotate_model_r3_r4(model):
    import tqdm  # TODO: remove this dependency
    head_dim = model.config.dim // model.config.n_head

    layers = [layer for layer in model.layers]
    for idx, layer in enumerate(
        tqdm.tqdm(layers, unit="layer", desc="Applying R4 rotation to W_down")
    ):
        R4_rotate_down_proj_weights(layers[idx])


def apply_spinquant_to_llama(model: "Transformer"):
    """
    Apply SpinQuant to Llama: https://arxiv.org/abs/2405.16406
    
    Currently, this only applies the R2 + R3 + R4 rotation matrices to the model.
    """
    rotate_model_r2(model)
    rotate_model_r3_r4(model)
