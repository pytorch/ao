import torch
import unittest
import random
from typing import List, Optional, Tuple
from itertools import product
import torchao
from torchao.kv_cache import PagedAttentionCache, PagedTensor

class NiaveCache:
    def __init__(self):
        self.past_key = None
        self.past_value = None

    def expand_cache(self, beam_size):
        self.past_key = self.past_key.repeat_interleave(beam_size, dim=0)
        self.past_value = self.past_value.repeat_interleave(beam_size, dim=0)

    def update(self, key, value, layer_idx=0):
        if self.past_key is None:
            self.past_key = key
            self.past_value = value
        else:
            self.past_key = torch.cat((self.past_key, key), dim=2)
            self.past_value = torch.cat((self.past_value, value), dim=2)
        return self.past_key, self.past_value

    def reorder_cache(self, beam_idx):
        self.past_key = self.past_key.index_select(0, beam_idx)
        self.past_value = self.past_value.index_select(0, beam_idx)


class MHAModule(torch.nn.Module):
    def __init__(self, head_dim, num_heads, num_kv_heads):
        super(MHAModule, self).__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.scale = head_dim**-0.5
        self.q = torch.nn.Linear(
            self.num_heads * self.head_dim, self.num_heads * self.head_dim
        )
        self.k = torch.nn.Linear(
            self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        )
        self.v = torch.nn.Linear(
            self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        )

    def forward(self, inputs, kv_cache):
        query = self.q(inputs)
        key = self.k(inputs)
        value = self.v(inputs)
        batch_size = inputs.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        key = key.view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        updated_key, updated_value = kv_cache.update(key, value, 0)
        if isinstance(updated_key, torch.Tensor):
            updated_key = updated_key.repeat_interleave(
                self.num_heads // self.num_kv_heads, dim=1
            )
            updated_value = updated_value.repeat_interleave(
                self.num_heads // self.num_kv_heads, dim=1
            )
        output = torch.nn.functional.scaled_dot_product_attention(
            query, updated_key, updated_value, scale=self.scale
        )
        return output


class PagedAttentionCachePagedTensorTest(unittest.TestCase):
    def _test_paged_attention_cache(
        self,
        num_blocks,
        block_size,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        device,
        dtype,
        batch_size,
        beam_size,
    ):
        num_layers = 1
        prompt_len = 32
        mha_model = MHAModule(head_dim, num_query_heads, num_key_value_heads).to(
            device=device, dtype=dtype
        )
        naive_cache = NiaveCache()
        pagedcache = PagedAttentionCache(
            num_blocks,
            block_size,
            num_key_value_heads,
            head_dim,
            num_layers,
            device,
            dtype,
        )
        # enable prompt sharing for the first token, fork
        pagedcache.set_batch2seq_for_prompt_sharing(batch_size, beam_size)
        pagedcache.allocate(batch_size, prompt_len)
        prompt_inputs = torch.randn(
            batch_size,
            prompt_len,
            num_query_heads * head_dim,
            device=device,
            dtype=dtype,
        )
        paged_output = mha_model(prompt_inputs, pagedcache)
        naive_output = mha_model(prompt_inputs, naive_cache)
        torch.allclose(paged_output, naive_output)

        beam_idx = torch.arange(
            0, batch_size * beam_size, beam_size, device=device, dtype=torch.int64
        ).repeat_interleave(beam_size)
        naive_cache.expand_cache(beam_size)
        naive_cache.reorder_cache(beam_idx)
        pagedcache.reorder_cache(beam_idx)

        # Next token
        pagedcache.allocate(batch_size * beam_size, 1)
        next_inputs = torch.randn(
            batch_size * beam_size,
            1,
            num_query_heads * head_dim,
            device=device,
            dtype=dtype,
        )

        paged_output = mha_model(next_inputs, pagedcache)   
        naive_output = mha_model(next_inputs, naive_cache)
        torch.allclose(paged_output, naive_output, atol=1e-3, rtol=1e-3)  

        for i in range(batch_size):
            beam_idx[i * beam_size : (i + 1) * beam_size] = torch.randint(
                i * beam_size,
                (i + 1) * beam_size,
                (1, beam_size),
                device=device,
                dtype=torch.int64,
            )
        naive_cache.reorder_cache(beam_idx)
        pagedcache.reorder_cache(beam_idx)

        # Next token
        pagedcache.allocate(batch_size * beam_size, 1)
        prompt_inputs = torch.randn(
            batch_size * beam_size,
            1,
            num_query_heads * head_dim,
            device=device,
            dtype=dtype,
            )
        paged_output = mha_model(prompt_inputs, pagedcache)
        naive_output = mha_model(prompt_inputs, naive_cache)
        torch.allclose(paged_output, naive_output, atol=1e-3, rtol=1e-3)

    def test_paged_attention_kv_cache(self):
        # num_blocks, block_size, num_query_heads, num_key_value_heads, head_dim, device, dtype, batch_size, beam_size
        num_blocks = 128
        block_sizes = [16, 32]
        num_query_heads = [40]
        num_key_value_heads = [40, 10, 1]
        head_dim = [64, 128]
        device = ["cpu"]
        dtypes = [torch.float, torch.float16, torch.bfloat16]
        batch_size = [1, 8]  
        beam_size = [1, 4]  
        for (
            block_size,
            num_query_head,
            num_key_value_head,
            head_dim,
            device,
            dtype,
            batch_size,
            beam_size,
        ) in product(
            block_sizes,
            num_query_heads,
            num_key_value_heads,
            head_dim,
            device,
            dtypes,
            batch_size,
            beam_size,
        ):
            self._test_paged_attention_cache(
                num_blocks,
                block_size,
                num_query_head,
                num_key_value_head,
                head_dim,
                device,
                dtype,
                batch_size,
                beam_size,
            )


# class PagedAttentionTest(unittest.TestCase):
#     def create_kv_caches(
#         self,
#         num_blocks: int,
#         block_size: int,
#         num_layer: int,
#         num_head: int,
#         head_size: int,
#         dtype: torch.dtype,
#         seed: int,
#     ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
#         torch.random.manual_seed(seed)
#         torch.manual_seed(seed)

#         scale = head_size**-0.5
#         key_cache_shape = (num_blocks, num_head, block_size,  head_size)
#         key_caches = []
#         for _ in range(num_layer):
#             key_cache = torch.empty(size=key_cache_shape, dtype=dtype)
#             key_cache.uniform_(-scale, scale)
#             key_caches.append(key_cache)

#         value_cache_shape = (num_blocks, num_head, block_size,  head_size)
#         value_caches = []
#         for _ in range(num_layer):
#             value_cache = torch.empty(size=value_cache_shape, dtype=dtype)
#             value_cache.uniform_(-scale, scale)
#             value_caches.append(value_cache)
#         return key_caches, value_caches

#     def ref_masked_attention(
#         self,
#         query: torch.Tensor,
#         key: torch.Tensor,
#         value: torch.Tensor,
#         scale: float,
#         attn_mask: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         attn_weights = torch.einsum("qhd,khd->hqk", query, key).float()
#         attn_weights = attn_weights * scale
#         if attn_mask is not None:
#             attn_weights = attn_weights + attn_mask.float()
#         attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
#         out = torch.einsum("hqk,khd->qhd", attn_weights, value)
#         return out

#     def ref_single_query_cached_kv_attention(
#         self,
#         output: torch.Tensor,
#         query: torch.Tensor,
#         num_queries_per_kv: int,
#         key_cache: torch.Tensor,
#         value_cache: torch.Tensor,
#         block_tables: torch.Tensor,
#         context_lens: torch.Tensor,
#         scale: float,
#         attn_mask: Optional[torch.Tensor],
#     ) -> None:
#         num_query_heads = query.shape[1]
#         num_kv_head = value_cache.shape[1]
#         head_size = value_cache.shape[3]
#         block_size = value_cache.shape[2]
#         num_seqs = query.shape[0]

#         block_tables = block_tables.cpu().tolist()
#         context_lens = context_lens.cpu().tolist()
#         for i in range(num_seqs):
#             q = query[i].unsqueeze(0)
#             block_table = block_tables[i]
#             context_len = int(context_lens[i])

#             keys = []
#             values = []
#             for j in range(context_len):
#                 key = torch.empty(
#                     num_kv_head, head_size, dtype=query.dtype, device="cpu"
#                 )
#                 value = torch.empty(
#                     num_kv_head, head_size, dtype=query.dtype, device="cpu"
#                 )
#                 for k in range(num_kv_head):
#                     block_number = int(block_table[j // block_size])
#                     block_offset = j % block_size
#                     key[k, :] = key_cache[block_number, k, block_offset, :]
#                     value[k, :] = value_cache[block_number, k, block_offset, :]
#                 keys.append(key)
#                 values.append(value)
#             keys = torch.stack(keys, dim=0)
#             values = torch.stack(values, dim=0)
#             if num_queries_per_kv > 1:
#                 # Handle MQA and GQA
#                 keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
#                 values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)
#             # out = self.ref_masked_attention(q, keys, values, scale, attn_mask[i])
#             out = self.ref_masked_attention(q, keys, values, scale, None)
#             out = out.view(num_query_heads, head_size)
#             output[i].copy_(out, non_blocking=True)

#     def _test_paged_attention_func(
#         self,
#         num_seqs: int,
#         num_head: Tuple[int, int],
#         head_size: int,
#         num_blocks: int,
#         block_size: int,
#         dtype: torch.dtype,
#         seed: int,
#     ) -> None:
#         random.seed(seed)
#         torch.random.manual_seed(seed)
#         torch.manual_seed(seed)
#         max_seq_len = 512
#         scale = float(1.0 / (head_size**0.5))
#         num_query_heads, num_kv_head = num_head
#         query = torch.empty(
#             num_seqs, num_query_heads, head_size, dtype=dtype, device="cpu"
#         )
#         query.uniform_(-scale, scale)
#         assert num_query_heads % num_kv_head == 0
#         num_queries_per_kv = num_query_heads // num_kv_head
#         head_mapping = torch.repeat_interleave(
#             torch.arange(num_kv_head, dtype=torch.int32, device="cpu"),
#             num_queries_per_kv,
#         )
#         attn_mask = None
#         context_lens = [random.randint(1, max_seq_len) for _ in range(num_seqs)]
#         context_lens[-1] = max_seq_len
#         max_context_len = max_seq_len  # max(context_lens)
#         attn_mask = torch.zeros(num_seqs, 1, 1, max_context_len, dtype=dtype)
#         for i in range(num_seqs):
#             attn_mask[i, :, :, context_lens[i] :].fill_(-10000.0)
#         paded_context_lens = torch.tensor(
#             [max_context_len for _ in range(num_seqs)]
#         ).to(torch.int32)
#         context_lens = torch.tensor(context_lens, dtype=torch.int, device="cpu")

#         # Create the block tables.NUM_PREFILL_SEQS
#         max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
#         block_tables = []
#         for _ in range(num_seqs):
#             block_table = [
#                 random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
#             ]
#             block_tables.append(block_table)
#         block_tables = torch.tensor(block_tables, dtype=torch.int, device="cpu")

#         # Create the KV caches.
#         key_caches, value_caches = self.create_kv_caches(
#             num_blocks, block_size, 1, num_kv_head, head_size, dtype, seed
#         )
#         key_cache, value_cache = key_caches[0], value_caches[0]
#         output = torch.empty_like(query.unsqueeze(2))
#         torch.ops.torchao.paged_attention(
#             output,
#             query.unsqueeze(2),
#             key_cache,
#             value_cache,
#             head_mapping,
#             scale,
#             block_tables,
#             paded_context_lens,
#             block_size,
#             attn_mask,
#         )
#         output = output.squeeze(2)
#         #Run the reference implementation.
#         ref_output = torch.empty_like(query)
#         self.ref_single_query_cached_kv_attention(
#             ref_output,
#             query,
#             num_queries_per_kv,
#             key_cache,
#             value_cache,
#             block_tables,
#             context_lens,
#             scale,
#             attn_mask,
#         )
#         assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-3)

#     def test_paged_attention(self):
#         num_blocks = 128
#         dtypes = [torch.bfloat16, torch.float, torch.float16]
#         num_gen_seqs = [2]  # Arbitrary values for testing
#         num_heads = [(40, 40), (64, 16)]  # Arbitrary values for testing
#         head_sizes = [64, 256]
#         block_sizes = [16, 32]
#         seeds = [0]
#         for (
#             num_seqs,
#             num_head,
#             head_size,
#             block_size,
#             dtype,
#             seed,
#         ) in product(
#             num_gen_seqs,
#             num_heads,
#             head_sizes,
#             block_sizes,
#             dtypes,
#             seeds,
#         ):
#             pass
#             self._test_paged_attention_func(
#                 num_seqs,
#                 num_head,
#                 head_size,
#                 num_blocks,
#                 block_size,
#                 dtype,
#                 seed,
#             )


if __name__ == "__main__":
    test = unittest.main()