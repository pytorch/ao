Inference
---------
In continuation to the previous tutorials about pretraining and finetuning, in this tutorial we'll show recipes for post-training quantization and serving the quantized model

The tutorial focuses on 3 receipes for post-training quantization and serving the quantized model:
1. :ref:`Post-training Quantization and Serving a model on HuggingFace`


Post-training Quantization and Serving
######################################

Part 3 (inference): Move/duplicate Jerry’s Phi-4 model card instructions to doc page
Part 3: Move code snippets from HF transformers torchao guide to this tutorial

Post-training Quantization using HuggingFace
------------------------------------------------


Evaluating the model
--------------------

Serving it on vLLM
--------------------

Sparsify using HuggingFace
##########################

Part 3: Add sparsity torchao huggingface integration


Lower to Executorch
###################

From the executorch root directory run the following command to lower the model to executorch format:

.. code:: console
    python -m examples.models.llama.export_llama --checkpoint "${LLAMA_QUANTIZED_CHECKPOINT:?}" -p "${LLAMA_PARAMS:?}" -kv --use_sdpa_with_kv_cache -qmode 8da4w --group_size 256 -d fp32 --metadata '{"get_bos_id":128000, "get_eos_id":128001}' --embedding-quantize 4,32 --output_name="llama3_8da4w.pte"

This will generate a file called ``llama3_8da4w.pte`` in the current directory. This file is the quantized and lowered model that can be used for inference.

# Evaluate model
# python -m examples.models.llama.eval_llama \
# 	-c "${LLAMA_QUANTIZED_CHECKPOINT:?}" \
# 	-p "${LLAMA_PARAMS:?}" \
# 	-t "${LLAMA_TOKENIZER:?}" \
# 	-kv \
# 	-d fp32 \
# 	--tasks mmlu \
# 	--num_fewshot 5 \
# 	--max_seq_len 8192 \
# 	--max_context_len 8192
