torchrun --nproc_per_node=4 --local-ranks-filter=0 -m pytest test/prototype/moe_training/test_fsdp_tp.py -s
