import torch
import torch.nn as nn
import functools
from typing import List, Tuple, Optional, Dict, Any
import copy

HANDLED_FUNCTIONS = {}


class PagedTensor(object):
    def __init__(
        self,
        cache: torch.Tensor, #The cache tensor from the PagedAttentionCache object, which is shared accross iterations.
        block_tables: torch.Tensor,#The block tables for each sequence in the batch which is used to mapping logical block to physical blocks.
        context_lens: torch.Tensor,#The context lens for each sequence in the batch.
    ):
        self.block_tables = block_tables
        self.cache = cache
        self.context_lens = context_lens

    def __repr__(self):
        return f"PagedTensor({self.cache.shape})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, PagedTensor)) for t in types
        ):
            return NotImplementedError(
                "{} is not supported by PagedTensor".format(func)
            )
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


def implements(torch_function):
    """Register a torch function override for PagedTensor"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements(torch.nn.functional.scaled_dot_product_attention)
def scaled_dot_product_attention(
    input, key_tensor, value_tensor, attn_mask=None, scale=None
):
    query = input
    key_cache = key_tensor.cache
    value_cache = value_tensor.cache
    block_tables = key_tensor.block_tables
    context_lens = key_tensor.context_lens
    output = torch.empty_like(query)
    torch.ops.torchao.paged_attention(
        output,
        query,
        key_cache,
        value_cache,
        scale,
        block_tables,
        context_lens,
        attn_mask,
    )
    return output


class PagedAttentionCache(object):
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_key_value_heads: int,
        head_dim: int,
        num_layers: int,
        device="cpu",
        dtype=None,
    ) -> None:
        super().__init__()

        # model info
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_layers = num_layers

        # Cache tensor info
        self.dtype = dtype if dtype is not None else torch.float32
        self.device = device
        self.num_blocks = num_blocks
        self.block_size = block_size

        cache_shape = (
            self.num_blocks,
            self.num_key_value_heads,
            self.block_size,
            self.head_dim,
        )

        # KV caches for each layer
        self.key_caches = [
            torch.zeros(cache_shape, dtype=self.dtype, device=device)
            for _ in range(num_layers)
        ]
        self.value_caches = [
            torch.zeros(cache_shape, dtype=self.dtype, device=device)
            for _ in range(num_layers)
        ]

        self.seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )

        # paged cache runtime information
        self.free_blocks = list(range(num_blocks))  # free blocks
        self.block_ref_count = [
            0
        ] * self.num_blocks  # init the reference count for each physical block
        self.block_tables = (
            dict()
        )  # mapping logical block to physical blocks for each sequence

        # The follow two states are shared accross layer but only for the current decode step. Need to update for every decode step.
        self.batch2seq = None  # mapping batch index to {seq_id0, seq_id1, ...} to enable prompt sharing.
        self.slots_mapping = None  # mapping logical slots to physical slots.

    def _copy_on_write(self, src_block_idx: int, dst_block_idx: int):
        """
        Copy the content of src_block_idx to dst_block_idx.

        Args:
            src_block_idx (int): The index of the source block.
            dst_block_idx (int): The index of the destination block.
        """
        for layer_idx in range(self.num_layers):
            self.key_caches[layer_idx][dst_block_idx] = self.key_caches[layer_idx][
                src_block_idx
            ].clone()
            self.value_caches[layer_idx][dst_block_idx] = self.value_caches[layer_idx][
                src_block_idx
            ].clone()

    def allocate(self, batch_size: int, key_len: int) -> None:
        """
        Allocate physical slots for a every sequence with key_len tokens in this batcch.

        Args:
        - batch_size (int): The batch size of the sequence.
        - key_len (int): The length of the key.

        Returns:
        - None
        """
        self.slots_mapping = []
        past_context_len = self.seen_tokens
        if self.batch2seq is None:
            self.set_batch2seq_for_prompt_sharing(batch_size, 1)
        for i in range(batch_size):
            seq_idx = self.batch2seq[i][0]
            # Scenario 1: New seqence: allocate blocks for this sequence
            if seq_idx not in self.block_tables:
                needed_blocks = (key_len + self.block_size - 1) // self.block_size
                if needed_blocks > len(self.free_blocks):
                    raise AssertionError(
                        f"No space in KV cache to store new token state. needed_blocks: {needed_blocks}, free_blocks: {self.free_blocks}"
                    )
                blocks = self.free_blocks[:needed_blocks]
                self.free_blocks = self.free_blocks[needed_blocks:]
                self.block_tables[seq_idx] = blocks
                for block_idx in blocks:
                    self.block_ref_count[block_idx] += 1
            # Senario 2: Partial processed sequence: find free slots in the allocated blocks or allocate new blocks
            else:
                seq_len = key_len + past_context_len
                target_blocks = (seq_len + self.block_size - 1) // self.block_size
                new_blocks = target_blocks - len(self.block_tables[seq_idx])

                if new_blocks > len(self.free_blocks):
                    raise AssertionError(
                        f"PagedAttentionCache: No enough free blocks to allocate for sequence {seq_idx}."
                    )

                if new_blocks > 0:  # allocate new blocks
                    candidate_blocks = self.free_blocks[:new_blocks]
                    self.block_tables[seq_idx].extend(self.free_blocks[:new_blocks])
                    self.free_blocks = self.free_blocks[new_blocks:]
                    for block_idx in candidate_blocks:
                        self.block_ref_count[block_idx] += 1
                else:
                    last_block = self.block_tables[seq_idx][-1]
                    # sharing the last block with other sequences, need to allocate a new block and copy the last block
                    if self.block_ref_count[last_block] > 1:
                        if len(self.free_blocks) == 0:
                            raise AssertionError(
                                f"PagedAttentionCache: No enough free blocks to allocate for sequence {seq_idx}."
                            )
                        new_block = self.free_blocks.pop()
                        self.block_tables[seq_idx][-1] = new_block
                        self.block_ref_count[new_block] += 1
                        self.block_ref_count[last_block] -= 1
                        self._copy_on_write(last_block, new_block)

            slots = []
            # the slots for this sequence
            for j in range(key_len):
                token_id = j + past_context_len
                block_idx = token_id // self.block_size
                block_offset = token_id % self.block_size
                slots.append(
                    self.block_tables[seq_idx][block_idx] * self.block_size
                    + block_offset
                )
            self.slots_mapping.append(slots)
        self.slots_mapping = torch.tensor(
            self.slots_mapping, dtype=torch.long, device=self.device
        )
        # step 2): fork new sequences to enable prompt sharing
        for batch_idx in range(batch_size):
            seq_ids = self.batch2seq[batch_idx]
            # fork the blocks allocated for the first sequence to other sequences in the batch
            for seq_id in seq_ids[1:]:
                self._fork(seq_ids[0], seq_id)

    def _free(self, seq_idx: int):
        """
        Frees the blocks allocated for the given sequence index.

        Args:
            seq_idx (int): The index of the sequence whose blocks are to be freed.

        Raises:
            AssertionError: If the given sequence index is not present in the block tables.
        """

        if seq_idx not in self.block_tables:
            raise AssertionError(
                f"PagedAttentionCache: Sequence index {seq_idx} is not present in the block tables."
            )

        for block_idx in self.block_tables[seq_idx]:
            self.block_ref_count[block_idx] -= 1
            if self.block_ref_count[block_idx] == 0:
                self.free_blocks.append(block_idx)

    def _fork(self, seq_idx: int, new_seq_idx: int):
        """
        Forks the blocks allocated for seq_idx to new_seq_idx.

        Args:
            seq_idx (int): The index of the sequence to be forked.
            new_seq_idx (int): The index of the new sequence.

        Raises:
            AssertionError: If seq_idx is not in block_tables or if new_seq_idx is already in block_tables.
        """
        if seq_idx not in self.block_tables:
            raise AssertionError(
                f"PagedAttentionCache: Sequence index {seq_idx} is not present in the block tables."
            )

        self.block_tables[new_seq_idx] = copy.deepcopy(self.block_tables[seq_idx])
        for block_idx in self.block_tables[seq_idx]:
            self.block_ref_count[block_idx] += 1

    def set_batch2seq_for_prompt_sharing(self, batch_size: int, beam_size: int):
        """
        Set the batch2seq mapping for prompt sharing.

        Args:
            batch_size (int): The batch size.
            beam_size (int): The beam size.
        """
        self.batch2seq = {}
        for i in range(batch_size):
            self.batch2seq[i] = [i * beam_size + j for j in range(beam_size)]

    def _reshape_and_cache(
        self,
        slot_mapping: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        """
        Reshapes and caches the key and value states based on the given slot mapping.

        Args:
            slot_mapping (List[List[int]]): A list of lists representing the slot mapping.
            key_states (torch.Tensor): The key states tensor.
            value_states (torch.Tensor): The value states tensor.
            layer_idx (int): The index of the layer.

        Returns:
            None
        """
        slot_mapping = slot_mapping.to(torch.int)
        block_indicies = torch.div(slot_mapping, self.block_size, rounding_mode="floor")
        block_indicies = block_indicies.cpu().tolist()
        block_offsets = slot_mapping % self.block_size
        block_offsets = block_offsets.cpu().tolist()
        batch_size = key_states.size(0)
        seq_len = key_states.size(2)
        for i in range(batch_size):
            for seq_idx in range(seq_len):
                block_idx = block_indicies[i][seq_idx]
                block_offset = block_offsets[i][seq_idx]
                for head_idx in range(self.num_key_value_heads):
                    self.key_caches[layer_idx][block_idx, head_idx, block_offset, :] = (
                        key_states[i, head_idx, seq_idx, :]
                    )
                    self.value_caches[layer_idx][
                        block_idx, head_idx, block_offset, :
                    ] = value_states[i, head_idx, seq_idx, :]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. PagedAttentionCache does not have a maximum length."""
        RuntimeError("PagedAttentionCache does not have a maximum sequence length.")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with key and value states for a specific layer.

        Args:
            key_states (torch.Tensor): The new key states tensor of shape [batch, head, seq, dim].
            value_states (torch.Tensor): The new value states tensor of shape [batch, head, seq, dim].
            layer_idx (int): The index of the layer.
            cache_kwargs (Dict[str, Any]): Additional arguments for the cache subclass.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the updated key states and value states tensors(entire context token states).

        Raises:
            AssertionError: If the batch size is inconsistent with the existing cache.
        """
        batch_size = key_states.shape[0]  # [batch, head, seq, dim]
        cur_len = key_states.shape[-2]

        # # slots info for key/value are same for every layer and allocate should be called before model.forward() to reduce the allocation overhead
        # AssertionError(
        #     self.slots_mapping is not None,
        #     "PagedAttentionCache: Please first call allocate() of this object to get target positions in paged cache before the model.forward().",
        # )
        # cache key_states & value_states
        self._reshape_and_cache(self.slots_mapping, key_states, value_states, layer_idx)

        if layer_idx == self.num_layers - 1:
            self.seen_tokens += cur_len
            self.slot_mapping = None

        if (
            self.seen_tokens == 0
            or self.seen_tokens == cur_len
            and layer_idx == self.num_layers - 1
        ):  # first token
            return key_states, value_states
        else:  # Next token
            if layer_idx == self.num_layers - 1:
                # last layer already updated self.seen_tokens
                context_lens = torch.tensor(
                    [self.seen_tokens for _ in range(batch_size)],
                    dtype=torch.int32,
                )
            else:
                context_lens = torch.tensor(
                    [self.seen_tokens + cur_len for _ in range(batch_size)],
                    dtype=torch.int32,
                )
            block_tables_t = []
            for seq_idx in range(batch_size):
                block_tables_t.append(self.block_tables[seq_idx])
            block_tables_t = torch.tensor(
                block_tables_t, dtype=torch.int32, device=self.device
            )
            return PagedTensor(
                self.key_caches[layer_idx], block_tables_t, context_lens
            ), PagedTensor(self.value_caches[layer_idx], block_tables_t, context_lens)

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        """
        Reorder the cache according to the beam index. The beam index is a tensor of shape (batch_size,)
        and the sequence id can be get from the self.batch2seq.
        """
        freed_seqs = []
        new_block_tables = self.block_tables.copy()
        num_beams = beam_idx.numel() // len(self.batch2seq)
        for batch_idx, target_batch_idx in enumerate(beam_idx.tolist()):
            target_seq_id = self.batch2seq[target_batch_idx // num_beams][0]
            seq_id = self.batch2seq[batch_idx // num_beams][0]
            freed_seqs.append(seq_id)
            new_block_tables[seq_id] = []
            for block in self.block_tables[target_seq_id]:
                self.block_ref_count[block] += 1
                new_block_tables[seq_id].append(block)
        for seq_idx in freed_seqs:
            self._free(seq_idx)
        self.block_tables = new_block_tables
        self.batch2seq = None
