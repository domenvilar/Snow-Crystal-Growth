#include "board.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// nvcc test.cu -O2 -o test
// srun --reservation=fri -G1 -n1 test

// Grid dimensions

__global__ void updateGrid(unsigned int* d_board, unsigned int* d_board_new, float* d_levels, float* d_levels_new, int n_rows, int n_cols, float alpha, float gamma)
{

    // precompute coefficients
    float alpha12 = alpha / 12.;
    float alpha2 = alpha / 2.;
    char neighs_even_col[6][2] = { { -1, 0 }, { -1, 1 }, { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, -1 } };
    char neighs_odd_col[6][2] = { { -1, 0 }, { 0, 1 }, { 1, 1 }, { 1, 0 }, { 1, -1 }, { 0, -1 } };
    char(*neighs)[2]; // the pointer to the neighbor offset arrays

    int col = blockIdx.x * blockDim.x + threadIdx.x; // column
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row

    int idx = row * n_cols + col;

    // Check if the coordinates are within the grid bounds
    // and current cell is not frozen
    // 1,2,3 cells receive water
    if (col < n_cols && row < n_rows && d_board[idx] != 0) {
        neighs = (col % 2 == 0) ? neighs_even_col : neighs_odd_col;

        // find the accumulation of water from the neighbors
        for (int k = 0; k < 6; k++) {
            int neigh_row = row + neighs[k][0]; // y
            int neigh_col = col + neighs[k][1]; // x

            // neighboring edge and unreceptive cells contributes water
            if (d_board[neigh_row * n_cols + neigh_col] < 2) {
                d_levels_new[idx] += alpha12 * d_levels[neigh_row * n_cols + neigh_col];
            }
        }

        // if the cell is boundary
        if (d_board[idx] == 2) {
            d_levels_new[idx] += gamma; // add the water vapor

            // if the cell freezes, set the flag and update the neighbor cells
            if (d_levels_new[idx] >= 1) {
                d_board_new[idx] = 3;

                for (int k = 0; k < 6; k++) {
                    int neigh_row = row + neighs[k][0];
                    int neigh_col = col + neighs[k][1];

                    // we dont update edge cells
                    if (d_board[neigh_row * n_cols + neigh_col] != 0) {
                        atomicMax(&d_board_new[neigh_row * n_cols + neigh_col], 2);
                    }
                }
            }

        } else {
            // otherwise the water also difuses out of the cell
            d_levels_new[idx] -= alpha2 * d_levels[idx];
        }
    }
}

void saveBoardToFile(unsigned int* board, unsigned int n, unsigned int m, const char* filename)
{
    FILE* fp = fopen(filename, "w+");
    if (fp != NULL) {
        fwrite(board, sizeof(unsigned int), n * m, fp);
        fclose(fp);
        // printf("Board saved to file: %s\n", filename);
    } else {
        printf("Failed to open file for writing: %s\n", filename);
    }
}

void saveLevelsToFile(float* levels, unsigned int n, unsigned int m, const char* filename)
{
    FILE* fp = fopen(filename, "w+");
    if (fp != NULL) {
        fwrite(levels, sizeof(float), n * m, fp);
        fclose(fp);
        // printf("Levels saved to file: %s\n", filename);
    } else {
        printf("Failed to open file for writing: %s\n", filename);
    }
}

int main(int argc, char* argv[])
{

    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    float alpha = atof(argv[3]);
    float beta = atof(argv[4]);
    float gamma = atof(argv[5]);
    int write_to_file = atoi(argv[6]);
    int block_size_x = atoi(argv[7]);
    int block_size_y = atoi(argv[8]);
    int is_shared_mem = atoi(argv[9]); // 0: without shared memory, 1: with shared memory

    // shape of block
    int is_block = atoi(argv[10]);
    int is_column = atoi(argv[11]);
    int is_row = atoi(argv[12]);
    char* filename = argv[13];
    int k = atoi(argv[14]);

    FILE* fp = fopen(filename, "a");
    if (fp == NULL) {
        printf("Failed to open the file for writing.\n");
        return 1;
    }
    // fprintf(fp, "n,m,alpha,beta,gamma,block_size_x,block_size_y,copy_time,kernel_time,write_time,is_shared_mem,is_block,is_column,is_row\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // n rows and m columns
    unsigned int** board = board_initialize(n, m);
    unsigned int** board_new = board_initialize(n, m);
    float** levels = levels_initialize(n, m, beta);
    float** levels_new = levels_initialize(n, m, beta);

    // Step 1: Allocate device memory for the arrays
    // 0 -> edge cell
    // 1 -> unreceptive
    // 2 -> boundary
    // 3 -> frozen
    unsigned int* d_board;
    unsigned int* d_board_new;
    float* d_levels;
    float* d_levels_new;

    cudaEventRecord(start);

    cudaMalloc((void**)&d_board, n * m * sizeof(unsigned int));
    cudaMalloc((void**)&d_board_new, n * m * sizeof(unsigned int));
    cudaMalloc((void**)&d_levels, n * m * sizeof(float));
    cudaMalloc((void**)&d_levels_new, n * m * sizeof(float));

    cudaMemcpy(d_board, board[0], n * m * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_board_new, board_new[0], n * m * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_levels, levels[0], n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_levels_new, levels_new[0], n * m * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float copyTime;
    cudaEventElapsedTime(&copyTime, start, stop);

    int numBlocksX = (m + block_size_x - 1) / block_size_x;
    int numBlocksY = (n + block_size_y - 1) / block_size_y;

    printf("numBlocksX: %d\n", numBlocksX);
    printf("numBlocksY: %d\n", numBlocksY);

    printf("# n=%d\n# m=%d\n", n, m);
    printf("# alpha=%.1f\n# beta=%.4f\n# gamma=%.3f\n", alpha, beta, gamma);
    printf("# block_size_x=%d\n# block_size_y=%d\n", block_size_x, block_size_y);

    // Set grid and block dimensions
    dim3 gridSize(numBlocksX, numBlocksY);
    dim3 blockSize(block_size_x, block_size_y);

    float allKernelTime = 0;
    float allWriteTime = 0;
    for (int iter = 0; iter < n / 3; iter++) {

        cudaEventRecord(start);
        // Launch the kernel to update the grid
        updateGrid<<<gridSize, blockSize>>>(d_board, d_board_new, d_levels, d_levels_new, n, m, alpha, gamma);

        // Copy data from d_board_new to d_board on the GPU
        cudaMemcpy(d_board, d_board_new, n * m * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_levels, d_levels_new, n * m * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float kernelTime;
        cudaEventElapsedTime(&kernelTime, start, stop);
        allKernelTime += kernelTime;

        if (write_to_file && iter % k == 0) {
            cudaEventRecord(start);
            unsigned int* board_result = new unsigned int[n * m];
            cudaMemcpy(board_result, d_board, n * m * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            float* levels_result = new float[n * m];
            cudaMemcpy(levels_result, d_levels, n * m * sizeof(float), cudaMemcpyDeviceToHost);

            // Save the board to a file
            const char* board_filename = "Data/array.bin";
            saveBoardToFile(board_result, n, m, board_filename);

            // Save the levels to a file
            const char* levels_filename = "Data/levels.bin";
            saveLevelsToFile(levels_result, n, m, levels_filename);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float writeTime;
            cudaEventElapsedTime(&writeTime, start, stop);
            allWriteTime += writeTime;
        }
    }

    cudaFree(d_board);
    cudaFree(d_board_new);
    cudaFree(d_levels);
    cudaFree(d_levels_new);

    float secPerIter = allKernelTime / (n / 3);

    printf("Copy time: %f ms\n", copyTime);
    printf("t[s]/iter: %f ms\n", secPerIter);
    printf("Write time: %f ms\n\n", allWriteTime);
    // Append the data to the file
    fprintf(fp, "%d,%d,%f,%f,%f,%d,%d,%f,%f,%f,%d,%d,%d,%d,%d\n", n, m, alpha, beta, gamma, block_size_x, block_size_y, copyTime, secPerIter, allWriteTime, is_shared_mem, is_block, is_column, is_row, k);

    return 0;
}