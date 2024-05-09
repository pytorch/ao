#! /bin/bash

# [Batch sizes to test]
# If you want to test the performance of FP6-LLM for larger inference batch sizes, 
# which typically happens during prompt processing, 
# please revise this file by simply "commenting" and "uncommenting".

N=(1 2 4 8 16 32 64)
SplitK=(5 6 7 6)

# BS <=64
#N=(1 2 4 8 16 32 64)
#SplitK=(5 6 7 6)

# BS = 128
#N=(128)
#SplitK=(5 3 3 3)

# BS = 256
#N=(256)
#SplitK=(4 3 2 3)

# BS = 512
#N=(512)
#SplitK=(2 5 2 4)

# BS = 1024
#N=(1024)
#SplitK=(1 2 1 2)

# BS >= 2048
# N = (2048, 4096, 8192, 16384)
#SplitK=(1 1 1 1)

# Benchmarking the specific Matrix Shape from llama2-70b
M=(10240 8192  57344 8192)
K=(8192  8192  8192  28672)

#mkdir -p Profiling
for ((i=0;i<${#M[@]};i++)) 
do
    for BS in ${N[@]} 
    do
        #ncu -f -o Profiling/M${M[i]}K${K[i]}N${BS} --set full \
        python fp6_test.py --OC=${M[i]} --IC=${K[i]} --BS=${BS} --splitK=${SplitK[i]}  
    done
done
