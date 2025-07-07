torchrun --nproc_per_node=2 --local-ranks-filter=0 -m pytest test/prototype/moe_training/test_tp.py -s
