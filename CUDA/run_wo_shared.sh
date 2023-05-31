#!/bin/sh

n=10000
m=10000

# alpha=1.
# beta=0.99
# gamma=0.001

alpha=1.
beta=0.9285
gamma=0.001

# alpha=1.
# beta=0.9
# gamma=0.001

write_to_file=0
block_size_x=64
block_size_y=64
is_shared_mem=0 # 0: without shared memory, 1: with shared memory
# shape of block
is_block=1
is_column=0
is_row=0
file_name="analysis_wo_shared_shape.csv"
file_name="test.csv"
k=1


# # Load the CUDA module
# module load CUDA/10.1.243-GCC-8.3.0

# # Compile the CUDA code
# nvcc wo_shared_implementation.cu -O2 -o wo_shared_implementation

# # Execute the command
# srun --reservation=fri -G1 -n1 wo_shared_implementation $n $n $alpha $beta $gamma $write_to_file $n $block_size_y $is_shared_mem $is_block $is_column $is_row $file_name $k


# for ((n=1000; n<=10000; n+=1000))
# do
#     # Compile the CUDA code
#     nvcc wo_shared_implementation.cu -O2 -o wo_shared_implementation

#     # Execute the command
#     srun --reservation=fri -G1 -n1 wo_shared_implementation $n $n $alpha $beta $gamma $write_to_file $block_size_x $block_size_y $is_shared_mem $is_block $is_column $is_row $file_name $k
# done

# Loop through different block sizes
# is_block=1
# is_column=0
# is_row=0
# for ((n=1000; n<=10000; n+=1000))
# do
#     for block_size in 8 16 32 64 128 256 512
#     do  
#         # Compile the CUDA code
#         nvcc wo_shared_implementation.cu -O2 -o wo_shared_implementation

#         # Execute the command
#         srun --reservation=fri -G1 -n1 wo_shared_implementation $n $n $alpha $beta $gamma $write_to_file $block_size $block_size $is_shared_mem $is_block $is_column $is_row $file_name  $k

#     done
# done

# # Loop through different column sizes
# is_block=0
# is_column=1
# is_row=0
# for ((n=1000; n<=10000; n+=1000))
# do
#     for col_size in 8 16 32 64 128 256 512
#     do  
#         # Compile the CUDA code
#         nvcc wo_shared_implementation.cu -O2 -o wo_shared_implementation

#         # Execute the command
#         srun --reservation=fri -G1 -n1 wo_shared_implementation $n $n $alpha $beta $gamma $write_to_file $col_size $n $is_shared_mem $is_block $is_column $is_row $file_name $k

#     done
# done


# Loop through different row sizes
# is_block=0
# is_column=0
# is_row=1
# for ((n=1000; n<=10000; n+=1000))
# do
#     for row_size in 8 16 32 64 128 256 512
#     do  
#         # Compile the CUDA code
#         nvcc wo_shared_implementation.cu -O2 -o wo_shared_implementation

#         # Execute the command
#         srun --reservation=fri -G1 -n1 wo_shared_implementation $n $n $alpha $beta $gamma $write_to_file $n $row_size $is_shared_mem $is_block $is_column $is_row $file_name $k

#     done
# done

# Look at the effect of writing to file
write_to_file=1
for k in 1 2 4 8 16 32
do
    # Compile the CUDA code
    nvcc wo_shared_implementation.cu -O2 -o wo_shared_implementation

    # Execute the command
    srun --reservation=fri -G1 -n1 wo_shared_implementation 1000 1000 $alpha $beta $gamma $write_to_file $block_size_x $block_size_y $is_shared_mem $is_block $is_column $is_row $file_name $k
done