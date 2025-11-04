import torch
from torch._higher_order_ops.out_dtype import out_dtype


def cpu_x_token_assym_fp8_w_group_sym_int4_gemm(
    x_i8,
    x_scale,
    x_zero_point,
    weight_int4,
    weight_scale,
    weight_zero_point,
    bias_fp32,
    group_size,
):
    # For groupwise quantization, we need to handle the computation differently
    # weight_i4 shape: [out_features, in_features]
    # weight_scale shape: [out_features, in_features // group_size]
    # weight_zero_point shape: [out_features, in_features // group_size]
    out_features, in_features = weight_int4.shape
    num_groups = in_features // group_size

    # scales in xnnpack are stored as bf16 and converted to fp32 for computation
    weight_scale = weight_scale.to(torch.bfloat16).to(torch.float32)

    assert x_i8.dim() == 2, "x_i8 must be 2D tensor"
    # Reshape for group-wise processing
    # x: [batch_size, in_features] -> [batch_size, num_groups, group_size]
    batch_size = x_i8.shape[0]
    x_i8_grouped = x_i8.view(batch_size, num_groups, group_size)

    # weight: [out_features, in_features] -> [out_features, num_groups, group_size]
    weight_i4_grouped = weight_int4.view(out_features, num_groups, group_size)

    # Convert to int16 for computation
    x_i32_grouped = x_i8_grouped.to(torch.int32)
    weight_i32_grouped = weight_i4_grouped.to(torch.int32)

    # Perform groupwise integer linear operation
    acc_fp32 = torch.zeros(
        batch_size, out_features, dtype=torch.float32, device=x_i8.device
    )

    for group_idx in range(num_groups):
        # Extract current group
        x_group = x_i32_grouped[:, group_idx, :]  # [batch_size, group_size]
        weight_group = weight_i32_grouped[:, group_idx, :]  # [out_features, group_size]
        weight_group_col_sum = weight_group.sum(dim=-1)  # [out_features]

        # Get scale for this group
        weight_scale_group = weight_scale[:, group_idx]  # [out_features]

        # Integer matmul: [batch_size, group_size] @ [group_size, out_features] -> [batch_size, out_features]
        group_acc = out_dtype(
            torch.ops.aten.linear.default,
            torch.int32,
            x_group,
            weight_group,
            None,
        )

        # Output has to be scaled by x_scale * weight_scale_group
        # However we will first scale by weight_scale_group, that is accounting
        # only for scale of weight, and then scale by x_scale at the end because
        # x_scale applies to all groups
        acc_fp32 = acc_fp32 + group_acc.to(torch.float32) * weight_scale_group.view(
            1, -1
        )

        # we must also subtract x_zero_point * weight_group_sum
        # since (X - x_zero_point) * W = X * W - x_zero_point * W
        weights_col_sum_adjusted = (
            weight_group_col_sum.to(torch.float32).view(1, -1)
            * x_zero_point.view(-1, 1)
            * weight_scale_group.view(1, -1)
        )
        acc_fp32 = acc_fp32 - weights_col_sum_adjusted
    x_scale_multiplier = x_scale.view(-1, 1)
    out_fp32 = acc_fp32 * x_scale_multiplier
    if bias_fp32 is not None:
        out_fp32 = out_fp32 + bias_fp32

    return out_fp32


def naive_x_token_assym_fp8_w_group_sym_int4_gemm(
    act_q,
    act_scale,
    act_zp,
    w_q,
    w_scale,
    w_group_size,
) -> torch.Tensor:
    #
    # now we have the scales/zero_points/quant values for both gemm operands
    # below is a manual slow gemm with integer operands and float rescaling,
    # implemented using eager PyTorch ops. This should be slow but closely
    # (but not exactly) matching a real int8,int8->int32 gemm with
    # rescaling, with the only difference being that the sum inside of the
    # dot product is done in float32 right now.
    #
    q_output = torch.zeros(
        act_q.shape[0],
        w_q.shape[0],
        dtype=torch.float32,
        device=act_q.device,
    )
    for m_idx in range(act_q.shape[0]):
        for n_idx in range(w_q.shape[0]):
            for g_idx in range(w_q.shape[1] // w_group_size):
                k_start = g_idx * w_group_size
                k_end = k_start + w_group_size
                act_chunk = act_q[m_idx][k_start:k_end]
                w_chunk = w_q[n_idx][k_start:k_end]

                # (act_q - act_zp) * w_q
                # result still in int32
                elem_int32 = (act_chunk - act_zp[m_idx]) * w_chunk

                # sum((act_q - act_zp) * w_q)
                # this is in float32, so likely a small deviation from the real
                # kernel, where the entire dot product would be in int32
                sum_float32 = torch.sum(elem_int32)

                # scale
                act_scale_tmp = act_scale[m_idx].squeeze(-1)
                w_scale_tmp = w_scale[n_idx][g_idx].squeeze(-1).bfloat16().float()
                sum_scaled = sum_float32 * act_scale_tmp * w_scale_tmp

                # accumulate
                q_output[m_idx][n_idx] += sum_scaled

    return q_output
