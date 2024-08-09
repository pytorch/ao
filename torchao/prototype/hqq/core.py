import torch
import math
from torch import Tensor, float16, float32
from typing import Union


# Shrinking operator (proximal operator for the lp norm)
def shrink_lp_op(x: Tensor, beta: float, lp_norm: float) -> Tensor:
    if lp_norm == 1:
        return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
    else:
        return torch.sign(x) * torch.nn.functional.relu(
            torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), lp_norm - 1)
        )


# Proximal solver || W - dequantize(quantize(W))||_p^p
@torch.inference_mode()
def optimize_weights_proximal_legacy(
    tensor: Tensor,
    scale: Tensor,
    zero: Tensor,
    min_max: list,
    axis: int = 0,
    dtype: Union[torch.dtype, None] = None,
    device: Union[str, None] = None,
    verbose: bool = False,
    opt_params: dict = {
        "lp_norm": 0.7,
        "beta": 1e1,
        "kappa": 1.01,
        "iters": 20,
        "early_stop": True,
    },
) -> tuple:
    lp_norm, beta, kappa, iters, early_stop = (
        opt_params["lp_norm"],
        opt_params["beta"],
        opt_params["kappa"],
        opt_params["iters"],
        opt_params["early_stop"],
    )

    device = tensor.device if (device is None) else torch.device(device)

    if dtype is None:
        dtype = float16 if (device.type == "cuda") else float32

    W_f = tensor.to(dtype=dtype, device=device)
    scale = scale.to(dtype=dtype, device=device)
    zero = zero.to(dtype=dtype, device=device)

    best_error = 1e4
    for i in range(iters):
        W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])
        W_r = (W_q - zero) / scale
        W_e = shrink_lp_op(W_f - W_r, beta, lp_norm)
        zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
        beta *= kappa

        current_error = float(torch.abs(W_f - W_r).mean())
        if verbose:
            print("Iter " + str(i + 1), " | Error: " + str(current_error))
        if early_stop:
            if current_error < best_error:
                best_error = current_error
            else:
                break

    scale = scale.to(tensor.device)
    zero = zero.to(tensor.device)
    del W_f, W_q, W_r, W_e
    torch.cuda.empty_cache()

    W_q = torch.round(tensor * scale + zero).clamp(min_max[0], min_max[1])
    return W_q, scale, zero


# Default: fast with early stopping
optimize_weights_proximal = optimize_weights_proximal_legacy


# Mainly used to check if the group-size is divisible by numel()
def is_divisible(val1: int, val2: int) -> bool:
    return int(val2 * math.ceil(val1 / val2)) == val1


# Converts hqq format W_dequant = (W_q - zero)*scale into affinequantized format: (W_q - mid_point)*scale_ao + zero_ao
def convert_to_affinequantized_format(W_q, scale, zero, nbits, shape):
    quant_min = 0
    quant_max = 2**nbits - 1
    mid_point = (quant_max + quant_min + 1) / 2
    zero_ao = ((mid_point - zero.float()) * scale.float()).to(zero.dtype)
    scale_ao = scale
    W_q_ao = W_q.view(shape)
    return W_q_ao, scale_ao, zero_ao


# Main HQQ Quantizer - simplified, no bitpacking.
class HQQQuantizer:
    optimize_weights = optimize_weights_proximal

    @classmethod
    def quantize(
        cls,
        tensor: Tensor,
        nbits: float = 4,
        group_size: int = 64,
        optimize: bool = True,
        axis: int = 1,
        compute_dtype: torch.dtype = float16,
        device: str = "cuda",
        verbose: bool = False,  # to check the optimizer error
        raw_output: bool = False,  # If True, it will return the quant params in hqq lib format
    ) -> tuple:
        assert axis in [0, 1], "axis should be either 0 or 1"
        if group_size is not None:
            assert is_divisible(tensor.numel(), group_size), (
                "group_size should be divisble by the total tensor dimensions. shape: "
                + str(tensor.shape)
                + ", group_size: "
                + str(group_size)
            )

        W = tensor.to(device=device, dtype=torch.float32)
        shape = W.shape

        # Reshape for grouping
        if group_size is not None:
            W = (
                W.reshape([-1, group_size])
                if (axis == 1)
                else W.reshape([group_size, -1])
            )

        # Get min/max values
        _min = W.min(axis=axis, keepdim=True)[0]
        _max = W.max(axis=axis, keepdim=True)[0]

        max_v = round(2**nbits - 1)
        min_v = 0
        min_max = [min_v, max_v]

        # Clamp to avoid fp16 issues
        scale = (max_v / (_max - _min)).clamp(max=2e4)
        zero = -_min * scale

        # Round zero as in: https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/quantizer.py#L42C9-L42C14
        if nbits in [4]:
            zero = torch.round(zero)

        # Fine-tune weights
        if optimize:
            W_q, scale, zero = HQQQuantizer.optimize_weights(
                tensor=W,
                scale=scale,
                zero=zero,
                min_max=min_max,
                axis=axis,
                device=device,
                verbose=verbose,
            )
        else:
            W_q = torch.round(W * scale + zero).clamp(min_max[0], min_max[1])

        # Store meta-data (we invert the scale for dequantization)
        scale = 1.0 / scale

        # Convert to affienquantized format
        if raw_output is False:
            W_q, scale, zero = convert_to_affinequantized_format(
                W_q, scale, zero, nbits, shape
            )

        # Make sure all the weights are in the right compute_dtype/device
        W_q = W_q.to(dtype=torch.uint8, device=device)
        scale = scale.to(dtype=compute_dtype, device=device)
        zero = zero.to(dtype=compute_dtype, device=device)

        # cleanup
        del W, _min, _max
        torch.cuda.empty_cache()

        return W_q, scale, zero, shape
