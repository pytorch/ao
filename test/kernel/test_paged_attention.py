import torch
import unittest
import random
from itertools import product
import torchao
from torchao.kv_cache import PagedAttentionCache, PagedTensor

class NaiveCache:
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

@unittest.skipIf(torch.cuda.is_available(), "CUDA is not enabled yet")
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
        device = ['cpu']
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

if __name__ == "__main__":
    test = unittest.main()
