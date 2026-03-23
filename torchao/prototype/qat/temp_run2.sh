CUDA_VISIBLE_DEVICES=6 python torchao/prototype/qat/temp_eval.py --checkpoint ./checkpoints/qwen3-30b-a3b-sft > logs/eval_sft_nvfp4.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python torchao/prototype/qat/temp_eval.py --checkpoint ./checkpoints/qwen3-30b-a3b-sft-qat > logs/eval_qat_nvfp4.log 2>&1 &
wait
