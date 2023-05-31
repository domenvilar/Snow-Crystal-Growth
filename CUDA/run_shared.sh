#!/bin/sh

n=60
m=60

# alpha=1.
# beta=0.99
# gamma=0.001

# alpha=1.
# beta=0.9285
# gamma=0.001

alpha=1.
beta=0.9
gamma=0.001


write_to_file=0
block_size_x=8
block_size_y=8
is_shared_mem=1 # 0: without shared memory, 1: with shared memory

file_name="analysis_shared.csv"


# # Load the CUDA module
# module load CUDA/10.1.243-GCC-8.3.0

# # Compile the CUDA code
# nvcc -DBLOCK_SIZE_X=$block_size_x -DBLOCK_SIZE_Y=$block_size_y shared_implementation.cu -O2 -o shared_implementation

# # Execute the command
# srun --reservation=fri -G1 -n1 shared_implementation $n $n $alpha $beta $gamma $write_to_file $block_size_x $block_size_y $is_shared_mem $is_block $is_column $is_row
# shape of block
is_block=1
is_column=0
is_row=0
for block_size in 8 16 32
do
    for ((n=1000; n<=10000; n+=1000))
    do
        # Compile the CUDA code
        nvcc -DBLOCK_SIZE_X=$block_size -DBLOCK_SIZE_Y=$block_size shared_implementation.cu -O2 -o shared_implementation

        # Execute the command
        srun --reservation=fri -G1 -n1 shared_implementation $n $n $alpha $beta $gamma $write_to_file $block_size $block_size $is_shared_mem $is_block $is_column $is_row $file_name
    done
done

