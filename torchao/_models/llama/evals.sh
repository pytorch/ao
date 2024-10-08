export CHECKPOINT_PATH=../../../checkpoints # path to checkpoints folder

export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization autoround        # auto-round w/o quant_lm_head
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization autoround-cuda-1 # auto-round w/ quant_lm_head

export MODEL_REPO=meta-llama/Meta-Llama-3-8B
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization autoround-cpu    # auto-round w/o quant_lm_head
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization autoround-cuda-1 # auto-round w/ quant_lm_head

export MODEL_REPO=meta-llama/Meta-Llama-3.1-8B
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization autoround-cpu    # auto-round w/o quant_lm_head
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization autoround-cuda-1 # auto-round w/ quant_lm_head
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int8wo
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int4wo-64
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization float8wo
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization float8dq-tensor
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization float8dq-row

# Testing on additional tasks
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int8wo --tasks 'winogrande' 'arc_challenge'
python eval.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --quantization int8wo --tasks 'mmlu' 'truthfulqa_mc2'
