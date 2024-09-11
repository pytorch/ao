export CHECKPOINT_PATH=../../../checkpoints # path to checkpoints folder

export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
# auto-round w/o quant_lm_head
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization autoround
# auto-round w/ quant_lm_head
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization autoround-cuda-1

export MODEL_REPO=meta-llama/Meta-Llama-3-8B
# auto-round w/o quant_lm_head
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantizatio autoround
# auto-round w/ quant_lm_head
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization autoround-cpu-1
