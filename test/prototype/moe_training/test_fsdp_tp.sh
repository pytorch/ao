torchrun --nproc_per_node=2 --local-ranks-filter=0 -m pytest test_fsdp_tp.py -s
