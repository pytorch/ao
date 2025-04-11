export MODEL_REPO=mistralai/Mixtral-8x7B-Instruct-v0.1
export CHECKPOINT_PATH=~/checkpoints/

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --compile
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --compile

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant int8wo --compile
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int8wo --compile

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant int8wo-base --compile
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int8wo-base --compile

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant int4wo --compile
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int4wo --compile

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant int4wo-base --compile
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int4wo-base --compile

# EXPERT CHOICE
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant int8dq --compile
# # # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int8dq --compile
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant int8dq-base --compile
# # # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int8dq-base --compile

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant fp8wo --compile
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant fp8wo --compile

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant fp8wo-base --compile
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant fp8wo-base --compile

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant fp8dq --compile
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant fp8dq --compile

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant fp8dq-base --compile
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant fp8dq-base --compile

# ARM
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant intxdq --device cpu
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant intxdq --compile --device cpu
