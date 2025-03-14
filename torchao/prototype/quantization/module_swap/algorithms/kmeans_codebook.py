import torch
import torch.nn as nn

from torchao.prototype.quantization.module_swap.quantized_modules import QuantizedLinear
from torchao.prototype.quantization.module_swap.quantizers import CodeBookQuantizer


def kmeans_codebook(
    model: nn.Module,
    niter: int = 30,
    nredo: int = 1,
    dtype: torch.dtype = torch.float32,
) -> None:
    import faiss

    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, QuantizedLinear):
                if isinstance(layer.weight_quantizer, CodeBookQuantizer):
                    weight = layer.weight
                    codebook_dim = layer.weight_quantizer.codebook_dim
                    weight = weight.reshape(
                        weight.shape[0] * (weight.shape[1] // codebook_dim),
                        codebook_dim,
                    )
                    num_centroids = layer.weight_quantizer.codebook.shape[0]
                    kmeans = faiss.Kmeans(
                        weight.shape[1],
                        num_centroids,
                        niter=niter,
                        nredo=nredo,
                        verbose=True,
                        gpu=True if torch.cuda.is_available() else False,
                    )
                    kmeans.train(weight.to(device="cpu", dtype=dtype))
                    C = kmeans.centroids

                    layer.weight_quantizer.codebook.data = torch.FloatTensor(C).to(
                        weight.dtype
                    )
