export MODEL_REPO=mistralai/Mixtral-8x7B-Instruct-v0.1
export CHECKPOINT_PATH=checkpoints/

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int4wo-base

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int4wo-base --compile

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int4wo-base --compile --compile_mode "max-autotune"

######### MULTI TOKEN #######

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant noquant # GOOD
# Average tokens/sec: 18.33
# Average tokens/sec including batches 146.65
# Memory used: 95.35 GB
# model size: 93.62

# grouped_mm_decomposed
# Average tokens/sec: 13.43
# Average tokens/sec including batches 107.42
# Memory used: 95.35 GB
# model size: 93.62

# multi token path
# Average tokens/sec: 14.31
# Average tokens/sec including batches 114.47
# Memory used: 95.35 GB
# model size: 93.62

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant noquant --compile # GOOD
# Average tokens/sec: 24.14
# Average tokens/sec including batches 193.11
# Memory used: 95.25 GB
# model size: 93.62

# grouped_mm_decomposed
# Average tokens/sec: 6.20
# Average tokens/sec including batches 49.56
# Memory used: 96.39 GB
# model size: 93.62

# multi token path
# Average tokens/sec: 9.19
# Average tokens/sec including batches 73.50
# Memory used: 95.25 GB
# model size: 93.62

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant noquant --compile --compile_mode "max-autotune"
# Average tokens/sec: 23.95
# Average tokens/sec including batches 191.63
# Memory used: 95.25 GB
# model size: 93.62

# grouped_mm_decomposed
# Average tokens/sec: 6.20
# Average tokens/sec including batches 49.56
# Memory used: 96.39 GB
# model size: 93.62

# multi token path
# Average tokens/sec: 8.61
# Average tokens/sec including batches 68.90
# Memory used: 96.28 GB
# model size: 93.62

######### SINGLE TOKEN #######

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant noquant
# Average tokens/sec: 33.69
# Memory used: 95.28 GB
# model size: 93.43

# single token
# Average tokens/sec: 23.24
# Memory used: 95.28 GB
# model size: 93.43

# grouped_mm_decomposed
# Average tokens/sec: 18.84
# Memory used: 95.28 GB
# model size: 93.43

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1
# Average tokens/sec: 23.29
# Memory used: 97.76 GB
# model size: 93.43
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant noquant --compile
# Average tokens/sec: 68.24
# Memory used: 95.28 GB
# model size: 93.43

# single token
# Average tokens/sec: 77.22
# Memory used: 95.28 GB
# model size: 93.43

# grouped_mm_decomposed
# Average tokens/sec: 7.11
# Memory used: 96.29 GB
# model size: 93.43

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --compile
# Average tokens/sec: 74.80
# Memory used: 97.80 GB
# model size: 93.43

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant noquant --compile --compile_mode "max-autotune"
# Average tokens/sec: 74.36
# Memory used: 97.80 GB
# model size: 93.43

# single token
# Average tokens/sec: 78.48
# Memory used: 95.28 GB
# model size: 93.43

# grouped_mm_decomposed
# Average tokens/sec: 6.98
# Memory used: 96.29 GB
# model size: 93.43

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --compile --compile_mode "max-autotune"
# Average tokens/sec: 74.82
# Memory used: 97.80 GB
# model size: 93.43

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --compile
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --compile

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant int8wo --compile
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int8wo --compile

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant int8wo-base --compile
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int8wo-base --compile

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant int4wo --compile
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int4wo --compile

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant int4wo-base --compile
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int4wo-base --compile

# # # EXPERT CHOICE
# # # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant int8dq --compile
# # # # # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int8dq --compile
# # # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant int8dq-base --compile
# # # # # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant int8dq-base --compile

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant fp8wo --compile
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant fp8wo --compile

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant fp8wo-base --compile
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant fp8wo-base --compile

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant fp8dq --compile
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant fp8dq --compile

# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 1 --moe_quant fp8dq-base --compile
# python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant fp8dq-base --compile

# # ARM
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant intxdq --device cpu
# # python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --batch_size 8 --moe_quant intxdq --compile --device cpu
