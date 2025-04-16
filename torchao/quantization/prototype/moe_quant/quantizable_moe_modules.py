class MOEFeedForwardAOQuantizable(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
        self.cond_ffn = ConditionalFeedForwardAOQuantizable(config)
        self.dim = config.dim
        self.num_activated_experts = config.num_activated_experts
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        x = x.view(-1, self.dim) # x: [T, D]
        scores = self.gate(x) # [T, E]
        expert_weights = F.softmax(scores, dim=-1)
        expert_weights, expert_indices = torch.topk(expert_weights, self.num_activated_experts, dim=-1) # [T, A], [T, A]
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True).to(x.dtype) # [T, A]
        out = self.cond_ffn(x, expert_indices, expert_weights, self.num_activated_experts)
        return out.reshape(batch_size, -1, self.dim)


class ConditionalFeedForwardAOQuantizable(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w1 = nn.Parameter(torch.empty(config.num_experts, config.intermediate_size, config.dim)) # E, I, D
        self.w2 = nn.Parameter(torch.empty(config.num_experts, config.dim, config.intermediate_size)) # E, D, I
        self.w3 = nn.Parameter(torch.empty(config.num_experts, config.intermediate_size, config.dim)) # E, I, D
        self.num_experts = config.num_experts
    def forward(
        self, x: Tensor,        # T, D
        expert_indices: Tensor, # T, A
        expert_weights: Tensor,  # T, A
        num_activated_experts: int,
        ) -> Tensor:
        num_tokens, dim = x.shape
        num_token_activations = num_tokens * num_activated_experts

        if x.shape[0]==1: #only 1 token (can be done without graph breaks when compiled)
            outs = []
            expert_indices=expert_indices.squeeze()
            # collect used experts
            w1 = self.w1[expert_indices]
            w2 = self.w2[expert_indices]
            w3 = self.w3[expert_indices]

            # run token through each expert
            for index in range(num_activated_experts):
                cur_out = F.linear( F.silu(F.linear(x, w1[index])) * F.linear(x, w3[index]), w2[index])
                outs.append(cur_out)

            # combine outputs
            final_out = (torch.cat(outs, dim=0) * expert_weights.view(-1,1)).sum(dim=0).unsqueeze(-1)
            return final_out
        else:
            expert_list = [x for x in range(self.num_experts)]
            
            # shuffle tokens into groups for each expert
            ordered_token_activations = expert_indices.view(-1).argsort(stable=True) # [A]
            ordered_token_indices = ordered_token_activations.div(num_activated_experts).floor().to(torch.int64) #  [T]

            num_tokens_per_expert = torch.histc(expert_indices, bins=self.num_experts+1, min=-1, max=self.num_experts) #  [E+1] (added leading 0 so can be used for indexing)
            cum_tokens_per_expert = num_tokens_per_expert.cumsum(0).to(torch.int64)  #  [E+1]
            
            # without quant this is compilable, with quant it throws an error. 
            # Even without quant there's a graph break here so not a huge loss
            @torch._dynamo.disable()
            def group_tokens_by_expert(ordered_token_indices, cum_tokens_per_expert, expert_list):
                token_indices_per_expert = [ordered_token_indices[cum_tokens_per_expert[expert]:cum_tokens_per_expert[expert+1]] for expert in expert_list] # [T'(e1)], [T'(e2)] ...
                return token_indices_per_expert
            token_indices_per_expert = group_tokens_by_expert(ordered_token_indices, cum_tokens_per_expert, expert_list)
            tokens_grouped_by_expert = [x[indices] for indices in token_indices_per_expert]

            # calculate outputs for each expert
            outs = []
            for cur_x, expert in zip(tokens_grouped_by_expert,expert_list):

                w1=self.w1[expert] # I, D
                w2=self.w2[expert] # D, I
                w3=self.w3[expert] # I, D

                cur_out = F.linear( F.silu(F.linear(cur_x, w1)) * F.linear(cur_x, w3), w2) # [T'(e), D]
                outs.append(cur_out)

            # weigh outputs
            ordered_outs = torch.cat(outs, dim=0) # [T*A, D]
            ordered_token_activation_weights = expert_weights.view(-1,1)[ordered_token_activations].view(-1,1) # [T*A, 1]
            weighted_ordered_outs = ordered_outs*ordered_token_activation_weights # [T*A, D]
            
            # sum weighted token-activation outputs together for each token
            final_out = torch.zeros_like(x) #  [T, D]
            final_out = final_out.scatter_add(dim=0, index=ordered_token_indices.unsqueeze(-1).expand(num_token_activations,dim).to(torch.int64), src=weighted_ordered_outs)
        return final_out
