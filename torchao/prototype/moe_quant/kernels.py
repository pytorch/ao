import torch
import torch.nn.functional as F
import warnings
from torchao.quantization.utils import _torchtitan_available, _fbgemm_available

grouped_gemm_fp8_rowwise = None
if _fbgemm_available:
    try:
        from fbgemm_gpu.experimental.gemm.triton_gemm.grouped_gemm import grouped_gemm_fp8_rowwise
    except:
        pass    

__all__ = ["fp8_dq_moe_op",
           "manual_pad",
           "torchtitan_pad",
        ]


def fp8_dq_moe_op(input, w1, w2, w3, expert_indices, scores, fast_accum=True, use_fbgemm_kernel=True):
    # parameters
    orig_in_shape = input.shape
    input.reshape(-1, orig_in_shape[-1])
    num_tokens, dim = input.shape
    num_experts, expert_dim, _ = w1.shape
    scores = scores.view(-1, scores.shape[-1])
    top_k = scores.shape[-1]
    total_activations = num_tokens*top_k
    
    # preprocess indices
    expert_indices = expert_indices.view(-1)
    activation_shuffle = expert_indices.argsort(stable=True)
    token_shuffle = activation_shuffle.div(top_k).floor().to(torch.int64)
    num_tokens_per_expert = torch.histc(expert_indices, bins=num_experts, min=0, max=num_experts).to(torch.int32)
    
    # get data for weights
    w1_fp8 = w1.original_weight_tensor.tensor_impl.float8_data
    w1_scale = w1.original_weight_tensor.tensor_impl.scale.squeeze()
    w1_qfunc = w1.input_quant_func
    w1_quant_kwargs = w1.quant_kwargs

    w3_fp8 = w3.original_weight_tensor.tensor_impl.float8_data
    w3_scale = w3.original_weight_tensor.tensor_impl.scale.squeeze()

    w2_fp8 = w2.original_weight_tensor.tensor_impl.float8_data
    w2_scale = w2.original_weight_tensor.tensor_impl.scale.squeeze()
    w2_qfunc = w2.input_quant_func
    w2_quant_kwargs = w2.quant_kwargs
    
    # quantize input
    q_input = w1_qfunc(input, **w1_quant_kwargs)
    q_input_data = q_input.tensor_impl.float8_data
    q_input_scale = q_input.tensor_impl.scale.squeeze()


    if use_fbgemm_kernel:
        # quant without padding
        input_fp8 = q_input_data[token_shuffle]
        input_scale = q_input_scale[token_shuffle] if q_input_scale.numel()>1 else q_input_scale
        
        @torch._dynamo.disable()
        def do_group_gemms(input_fp8, input_scale, w1_fp8, w1_scale, w2_fp8, w2_scale, w3_fp8, w3_scale, num_tokens_per_expert, w2_qfunc, w2_quant_kwargs):
            assert grouped_gemm_fp8_rowwise is not None, "fbgemm kernel requires fbgemm-gpu-genai to be installed: https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/experimental/gen_ai/README.md"
            y1 = grouped_gemm_fp8_rowwise(input_fp8, w1_fp8.reshape(-1, w1_fp8.shape[-1]), num_tokens_per_expert, input_scale, w1_scale.reshape(-1), use_fast_accum=True, _use_warp_specialization=False)
            y3 = grouped_gemm_fp8_rowwise(input_fp8, w3_fp8.reshape(-1, w3_fp8.shape[-1]), num_tokens_per_expert, input_scale, w3_scale.reshape(-1), use_fast_accum=True, _use_warp_specialization=False)
            y = F.silu(y1)*y3
            y_q = w2_qfunc(y, **w2_quant_kwargs)
            y_fp8 = y_q.tensor_impl.float8_data
            y_scale = y_q.tensor_impl.scale.squeeze()
            
            # TODO use _scatter_add_indices to combine the last group gemm with the final_out calculation
            out = grouped_gemm_fp8_rowwise(y_fp8, w2_fp8.view(-1, w2_fp8.shape[-1]), num_tokens_per_expert, y_scale, w2_scale.view(-1), use_fast_accum=fast_accum, _use_warp_specialization=False)
            return out
        
        out = do_group_gemms(input_fp8, input_scale, w1_fp8, w1_scale, w2_fp8, w2_scale, w3_fp8, w3_scale, num_tokens_per_expert, w2_qfunc, w2_quant_kwargs)

        # unpad and combine output with weights
        sorted_scores = scores.reshape(-1,1)[activation_shuffle]
        out = out*sorted_scores

        # sum weighted outputs
        final_out = torch.zeros_like(input)
        final_out = final_out.scatter_add(
            dim=0,
            index=token_shuffle.unsqueeze(-1).expand(total_activations, dim).to(torch.int64),
            src=out
        )
        final_out = final_out.reshape(orig_in_shape)
        return final_out

    else:
        # padding
        alignment = 16
        if _torchtitan_available:
            num_ranks = 1
            padded_indices, m_sizes, m_offsets = torchtitan_pad(num_tokens_per_expert, alignment, num_ranks)
        else:
            padded_indices, m_sizes, m_offsets = manual_pad(num_tokens_per_expert, alignment)

        pad_len = padded_indices.shape[0]
        valid_values = padded_indices >= 0

        # shuffle/pad input
        input_fp8 = torch.zeros((pad_len, q_input_data.shape[-1]), dtype=q_input_data.dtype, device=q_input_data.device)
        input_scale = torch.zeros(pad_len, dtype=q_input_scale.dtype, device=q_input_scale.device)
        input_fp8[valid_values] = q_input_data[token_shuffle]
        input_scale[valid_values] = q_input_scale[token_shuffle] if q_input_scale.numel()>1 else q_input_scale


        y1 = torch._scaled_grouped_mm(input_fp8, w1_fp8.transpose(-2, -1), input_scale, w1_scale, offs=m_offsets, out_dtype=torch.bfloat16, use_fast_accum=fast_accum)
        y3 = torch._scaled_grouped_mm(input_fp8, w3_fp8.transpose(-2, -1), input_scale, w3_scale, offs=m_offsets, out_dtype=torch.bfloat16, use_fast_accum=fast_accum)
        y = F.silu(y1)*y3
        y_q = w2_qfunc(y, **w2_quant_kwargs)

        y_fp8 = y_q.tensor_impl.float8_data
        y_scale = y_q.tensor_impl.scale.squeeze()
        out = torch._scaled_grouped_mm(y_fp8, w2_fp8.transpose(-2, -1), y_scale, w2_scale, offs=m_offsets, out_dtype=torch.bfloat16, use_fast_accum=fast_accum)
        
        # unpad and combine output with weights
        out = out[valid_values]
        sorted_scores = scores.reshape(-1,1)[activation_shuffle]
        out = out*sorted_scores

        # sum weighted outputs
        final_out = torch.zeros_like(input)
        final_out = final_out.scatter_add(
            dim=0,
            index=token_shuffle.unsqueeze(-1).expand(total_activations, dim).to(torch.int64),
            src=out
        )
        final_out = final_out.reshape(orig_in_shape)
        return final_out

def torchtitan_pad(num_tokens_per_expert, alignment, num_ranks):
    from torchtitan.experiments.kernels.moe.indices import generate_permute_indices 
    num_experts = num_tokens_per_expert.shape[0]

    # pad to nearest multiple of alignment that's greater than 0
    padded_sizes = (((num_tokens_per_expert + (num_tokens_per_expert==0))/alignment).ceil() * alignment)
    pad_len = int(padded_sizes.sum().item())

    padded_indices, m_sizes, m_offsets = generate_permute_indices(
        num_tokens_per_expert,
        num_experts,
        num_ranks,
        pad_len,
        alignment,
        use_cpu=False
    )
    return padded_indices, m_sizes, m_offsets

def manual_pad(num_tokens_per_expert, alignment):
    num_experts = num_tokens_per_expert.shape[0]

    m_sizes = ((((num_tokens_per_expert + (num_tokens_per_expert==0))/alignment).ceil() * alignment)).to(torch.int32)
    pad_len = int(m_sizes.sum().item())

    padded_indices = torch.zeros(pad_len, dtype=torch.int32, device=num_tokens_per_expert.device)-1
    start_tok_index = 0
    start_pad_index = 0
    for i in range(num_experts):
        end_tok_index = int(start_tok_index+num_tokens_per_expert[i].item())
        end_pad_index = int(start_pad_index+num_tokens_per_expert[i].item())
        padded_indices[start_pad_index:end_pad_index] = torch.arange(start_tok_index, end_tok_index, dtype=torch.int32, device=num_tokens_per_expert.device)
        start_tok_index = end_tok_index
        start_pad_index = start_pad_index + int(m_sizes[i].item())
    m_offsets = m_sizes.cumsum(0).to(torch.int32)
    return padded_indices, m_sizes, m_offsets
