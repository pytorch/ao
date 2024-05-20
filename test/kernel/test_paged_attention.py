import torch
import unittest
import random
from typing import List, Optional, Tuple
from itertools import product
import torchao


class PagedAttentionTest(unittest.TestCase):
    def create_kv_caches(
        self,
        num_blocks: int,
        block_size: int,
        num_layer: int,
        num_head: int,
        head_size: int,
        dtype: torch.dtype,
        seed: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        torch.random.manual_seed(seed)
        torch.manual_seed(seed)

        scale = head_size**-0.5
        key_cache_shape = (num_blocks, block_size, num_head, head_size)
        key_caches = []
        for _ in range(num_layer):
            key_cache = torch.empty(size=key_cache_shape, dtype=dtype)
            key_cache.uniform_(-scale, scale)
            key_caches.append(key_cache)

        value_cache_shape = (num_blocks, block_size, num_head, head_size)
        value_caches = []
        for _ in range(num_layer):
            value_cache = torch.empty(size=value_cache_shape, dtype=dtype)
            value_cache.uniform_(-scale, scale)
            value_caches.append(value_cache)
        return key_caches, value_caches

    def ref_masked_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_weights = torch.einsum("qhd,khd->hqk", query, key).float()
        attn_weights = attn_weights * scale
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask.float()
        attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
        out = torch.einsum("hqk,khd->qhd", attn_weights, value)
        return out

    def ref_single_query_cached_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        num_queries_per_kv: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        scale: float,
        attn_mask: Optional[torch.Tensor],
    ) -> None:
        num_query_heads = query.shape[1]
        num_kv_head = value_cache.shape[2]
        head_size = value_cache.shape[3]
        block_size = value_cache.shape[1]
        num_seqs = query.shape[0]

        block_tables = block_tables.cpu().tolist()
        context_lens = context_lens.cpu().tolist()
        for i in range(num_seqs):
            q = query[i].unsqueeze(0)
            block_table = block_tables[i]
            context_len = int(context_lens[i])

            keys = []
            values = []
            for j in range(context_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size

                k = key_cache[block_number, block_offset, :, :]
                k = k.reshape(num_kv_head, head_size)
                keys.append(k)

                v = value_cache[block_number, block_offset, :, :]
                values.append(v)
            keys = torch.stack(keys, dim=0)
            values = torch.stack(values, dim=0)
            if num_queries_per_kv > 1:
                # Handle MQA and GQA
                keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
                values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)
            # out = self.ref_masked_attention(q, keys, values, scale, attn_mask[i])
            out = self.ref_masked_attention(q, keys, values, scale, None)
            out = out.view(num_query_heads, head_size)
            output[i].copy_(out, non_blocking=True)

    def _test_paged_attention_func(
        self,
        num_seqs: int,
        num_head: Tuple[int, int],
        head_size: int,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        seed: int,
    ) -> None:
        random.seed(seed)
        torch.random.manual_seed(seed)
        torch.manual_seed(seed)
        max_seq_len = 512
        scale = float(1.0 / (head_size**0.5))
        num_query_heads, num_kv_head = num_head
        query = torch.empty(
            num_seqs, num_query_heads, head_size, dtype=dtype, device="cpu"
        )
        query.uniform_(-scale, scale)
        assert num_query_heads % num_kv_head == 0
        num_queries_per_kv = num_query_heads // num_kv_head
        head_mapping = torch.repeat_interleave(
            torch.arange(num_kv_head, dtype=torch.int32, device="cpu"),
            num_queries_per_kv,
        )
        attn_mask = None
        context_lens = [random.randint(1, max_seq_len) for _ in range(num_seqs)]
        context_lens[-1] = max_seq_len
        max_context_len = max_seq_len  # max(context_lens)
        attn_mask = torch.zeros(num_seqs, 1, 1, max_context_len, dtype=dtype)
        for i in range(num_seqs):
            attn_mask[i, :, :, context_lens[i] :].fill_(-10000.0)
        paded_context_lens = torch.tensor(
            [max_context_len for _ in range(num_seqs)]
        ).to(torch.int32)
        context_lens = torch.tensor(context_lens, dtype=torch.int, device="cpu")

        # Create the block tables.NUM_PREFILL_SEQS
        max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
        block_tables = []
        for _ in range(num_seqs):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
        block_tables = torch.tensor(block_tables, dtype=torch.int, device="cpu")

        # Create the KV caches.
        key_caches, value_caches = self.create_kv_caches(
            num_blocks, block_size, 1, num_kv_head, head_size, dtype, seed
        )
        key_cache, value_cache = key_caches[0], value_caches[0]

        output = torch.empty_like(query)
        torch.ops.torchao.paged_attention(
            output,
            query.unsqueeze(1),
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            paded_context_lens,
            block_size,
            attn_mask,
        )

        # Run the reference implementation.
        ref_output = torch.empty_like(query)
        self.ref_single_query_cached_kv_attention(
            ref_output,
            query,
            num_queries_per_kv,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            scale,
            attn_mask,
        )
        assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-3)

    def test_paged_attention(self):
        num_blocks = 128
        dtypes = [torch.bfloat16, torch.float, torch.float16]
        num_gen_seqs = [2]  # Arbitrary values for testing
        num_heads = [(40, 40), (64, 16)]  # Arbitrary values for testing
        head_sizes = [64, 256]
        block_sizes = [16, 32]
        seeds = [0]
        for (
            num_seqs,
            num_head,
            head_size,
            block_size,
            dtype,
            seed,
        ) in product(
            num_gen_seqs,
            num_heads,
            head_sizes,
            block_sizes,
            dtypes,
            seeds,
        ):
            self._test_paged_attention_func(
                num_seqs,
                num_head,
                head_size,
                num_blocks,
                block_size,
                dtype,
                seed,
            )


if __name__ == "__main__":
    test = unittest.main()