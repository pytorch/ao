NGPU=${NGPU:-4}
torchrun --nproc_per_node=$NGPU --local-ranks-filter=0 -m pytest test/prototype/moe_qat/test_distributed.py -s -v
