import torch
from torch.testing._internal.common_utils import TestCase, IS_FBCODE
from torch.testing._internal.optests import opcheck
from typing import List, Optional, Tuple
import torchao
from torchao.quantization.utils import TORCH_VERSION_AFTER_2_4
import unittest
import random
from parameterized import parameterized
from itertools import product



# torch.testing._internal.optests.generate_tests.OpCheckError: opcheck(op, ...):
# test_faketensor failed with module 'torch' has no attribute '_custom_ops' (scroll up for stack trace)
@unittest.skipIf(IS_FBCODE, "Skipping the test in fbcode since we don't have TARGET file for kernels")
class TestOps(TestCase):
    def _create_tensors_with_iou(self, N, iou_thresh):
        # force last box to have a pre-defined iou with the first box
        # let b0 be [x0, y0, x1, y1], and b1 be [x0, y0, x1 + d, y1],
        # then, in order to satisfy ops.iou(b0, b1) == iou_thresh,
        # we need to have d = (x1 - x0) * (1 - iou_thresh) / iou_thresh
        # Adjust the threshold upward a bit with the intent of creating
        # at least one box that exceeds (barely) the threshold and so
        # should be suppressed.
        boxes = torch.rand(N, 4) * 100
        boxes[:, 2:] += boxes[:, :2]
        boxes[-1, :] = boxes[0, :]
        x0, y0, x1, y1 = boxes[-1].tolist()
        iou_thresh += 1e-5
        boxes[-1, 2] += (x1 - x0) * (1 - iou_thresh) / iou_thresh
        scores = torch.rand(N)
        return boxes, scores

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "skipping when torch verion is 2.3 or lower")
    def test_nms(self):
        iou = 0.2
        boxes, scores = self._create_tensors_with_iou(1000, iou)
        boxes = boxes.cuda()
        scores = scores.cuda()

        # smoke test
        _ = torchao.ops.nms(boxes, scores, iou)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.nms, (boxes, scores, iou), test_utils=test_utils)

    def _create_fp6_inputs(self, BS: int, OC: int, IC: int):
        # Randomly initialize each bytes. The highest value for randint() is set the the max value of uint32_t.
        fp6_weight = torch.randint(4294967295, (OC, IC // 16 * 3)).to(torch.int)
        fp16_scale = torch.rand(OC).half() + 0.5
        fp16_activation = torch.rand(BS, IC).half() + 0.5
        return fp6_weight, fp16_scale, fp16_activation

    def test_prepack_fp6_weight(self):
        OC = 256
        IC = 256
        fp6_weight, _, _ = self._create_fp6_inputs(0, OC, IC)

        # smoke test
        torchao.ops.prepack_fp6_weight(fp6_weight)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.prepack_fp6_weight, (fp6_weight,), test_utils=test_utils)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fp16_to_fp6(self):
        OC = 256
        IC = 256

        # in this fp6, we use 3 bits for exponent and 2 bits for mantissa
        # also, we don't have nan/inf
        fp6_absmax = 28.0  # 2 ** (0b111 - 0b011) * (1 + 0.5 + 0.25), where E=111, M=11
        fp6_absmin = 0.0625  # 2 ** (-0b010) * 0.25, where E=000, M=01 (subnormal number)
        fp16_weight = torch.randn((OC, IC), dtype=torch.float16)
        fp16_weight.clip_(-fp6_absmax, fp6_absmax)
        fp16_weight[fp16_weight.abs() < fp6_absmin] = 0

        # smoke test
        torchao.ops.fp16_to_fp6(fp16_weight)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.fp16_to_fp6, (fp16_weight,), test_utils=test_utils)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fp16act_fp6weight_linear(self):
        BS = 2
        OC = 256
        IC = 256
        splitK = 1
        fp6_weight, fp16_scale, fp16_activation = self._create_fp6_inputs(BS, OC, IC)

        fp6_weight_packed = torchao.ops.prepack_fp6_weight(fp6_weight)
        act_cuda = fp16_activation.cuda()
        weight_cuda = fp6_weight_packed.cuda()
        scale_cuda = fp16_scale.cuda()

        # smoke test
        torchao.ops.fp16act_fp6weight_linear(act_cuda, weight_cuda, scale_cuda, splitK)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.fp16act_fp6weight_linear, (act_cuda, weight_cuda, scale_cuda, splitK), test_utils=test_utils)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fp6_weight_dequant(self):
        OC = 256
        IC = 256
        fp6_weight, fp16_scale, _ = self._create_fp6_inputs(0, OC, IC)

        # smoke test
        torchao.ops.fp6_weight_dequant(fp6_weight, fp16_scale)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.fp6_weight_dequant, (fp6_weight, fp16_scale), test_utils=test_utils)

    # adapted from https://github.com/usyd-fsalab/fp6_llm/blob/main/tests/python/kernel_test.py
    @parameterized.expand([(1, 2048, 4096, 5), (2, 8192, 8192, 6)])
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fp6_matmul_correctness(self, BS, OC, IC, splitK):
        fp6_weight, fp16_scale, fp16_activation = self._create_fp6_inputs(BS, OC, IC)

        fp6_weight_packed = torchao.ops.prepack_fp6_weight(fp6_weight)
        act_cuda = fp16_activation.cuda()
        weight_cuda = fp6_weight_packed.cuda()
        scale_cuda = fp16_scale.cuda()

        results_fp6 = torchao.ops.fp16act_fp6weight_linear(act_cuda, weight_cuda, scale_cuda, splitK)

        fp16_weight = torchao.ops.fp6_weight_dequant(fp6_weight, fp16_scale).cuda()
        results_fp16 = act_cuda @ fp16_weight.T

        error = (results_fp6 - results_fp16).abs()
        relative_error = error / results_fp16.abs()
        assert relative_error.mean() < 1e-2
    
    def _create_kv_caches(
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

    def _ref_masked_attention(
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

    def _ref_paged_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        num_queries_per_kv: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        scale: float,
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
            # out = self._ref_masked_attention(q, keys, values, scale, attn_mask[i])
            out = self._ref_masked_attention(q, keys, values, scale, None)
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
        key_caches, value_caches = self._create_kv_caches(
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
        self._ref_paged_attention(
            ref_output,
            query,
            num_queries_per_kv,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            scale,
        )
        assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-3)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available") 
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
    unittest.main()
