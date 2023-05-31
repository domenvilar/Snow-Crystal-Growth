#include "board.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// nvcc test.cu -O2 -o test
// srun --reservation=fri -G1 -n1 test

// Grid dimensions
// #define BLOCK_SIZE_X 8 // Number of threads per block
// #define BLOCK_SIZE_Y 8 // Number of threads per block

__global__ void updateGrid(unsigned int* d_board, unsigned int* d_board_new, float* d_levels, float* d_levels_new, int n_rows, int n_cols, float alpha, float gamma)
{
    // printf("Block (%d, %d) started\n", blockIdx.x, blockIdx.y);
    __shared__ unsigned int s_board[BLOCK_SIZE_X][BLOCK_SIZE_Y];
    __shared__ float s_levels[BLOCK_SIZE_X][BLOCK_SIZE_Y];

    __shared__ unsigned int s_board_new[BLOCK_SIZE_X][BLOCK_SIZE_Y];
    __shared__ float s_levels_new[BLOCK_SIZE_X][BLOCK_SIZE_Y];

    // precompute coefficients
    float alpha12 = alpha / 12.;
    float alpha2 = alpha / 2.;
    char neighs_even_col[6][2] = { { -1, 0 }, { -1, 1 }, { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, -1 } };
    char neighs_odd_col[6][2] = { { -1, 0 }, { 0, 1 }, { 1, 1 }, { 1, 0 }, { 1, -1 }, { 0, -1 } };
    char(*neighs)[2]; // the pointer to the neighbor offset arrays

    int col_stacked = blockIdx.x * blockDim.x + threadIdx.x; // column
    int row_stacked = blockIdx.y * blockDim.y + threadIdx.y; // row
    int col = threadIdx.x;
    int row = threadIdx.y;

    // offset
    int col_offset = (blockIdx.x) * 4 + 2;
    int row_offset = (blockIdx.y) * 4 + 2;

    // idx v globalnem pomnilniku
    int col_global = col_stacked - col_offset;
    int row_global = row_stacked - row_offset;
    int idx = row_global * n_cols + col_global;

    // Load current cell and buffered cells into shared memory
    if (row_global < n_rows && col_global < n_cols && row_global >= 0 && col_global >= 0) {
        s_board[row][col] = d_board[idx];
        s_levels[row][col] = d_levels[idx];

        s_board_new[row][col] = d_board_new[idx];
        s_levels_new[row][col] = d_levels_new[idx];
    }

    __syncthreads();

    // Check if the coordinates are within the grid bounds
    // and current cell is not frozen or edge
    // 0 -> edge cell
    // 1 -> unreceptive
    // 2 -> boundary
    // 3 -> frozen
    // 1,2,3 cells receive water, calculalte water levels for 1x buffered edge
    // each thread calculates one cell, so we need to check if the cell is within the grid bounds
    // we calculate water levels for 1x buffered edge -> we can update cell state for 2x buffered edge
    if (row_global < n_rows && col_global < n_cols && row_global >= 0 && col_global >= 0 && s_board[row][col] != 0 && col > 0 && col < BLOCK_SIZE_X - 1 && row > 0 && row < BLOCK_SIZE_Y - 1) {
        neighs = (col_global % 2 == 0) ? neighs_even_col : neighs_odd_col; // to me se skrbi ce je prov
        // find the accumulation of water from the neighbors
        for (int k = 0; k < 6; k++) {
            int neigh_row = row + neighs[k][0]; // y
            int neigh_col = col + neighs[k][1]; // x

            // edge and unreceptive cells contributes water
            if (s_board[neigh_row][neigh_col] < 2) {
                s_levels_new[row][col] += alpha12 * s_levels[neigh_row][neigh_col];
                // printf("levels %f cell type %d\n", s_levels_new[row][col], s_board[row][col]);
            }
        }

        // if the cell is boundary
        // add the water vapor and check if it freezes
        if (s_board[row][col] == 2) {
            s_levels_new[row][col] += gamma; // add the water vapor

            // if the cell freezes neighbors become boundary
            if (s_levels_new[row][col] >= 1) {
                s_board_new[row][col] = 3;

                // printf("cell (%d, %d) froze\n", row_global, col_global);
                for (int k = 0; k < 6; k++) {
                    int neigh_row = row + neighs[k][0];
                    int neigh_col = col + neighs[k][1];

                    // we dont update edge cells
                    if (s_board[neigh_row][neigh_col] != 0) {
                        atomicMax(&s_board_new[neigh_row][neigh_col], 2);
                    }
                }
            }

        } else if (s_board[row][col] == 1) {
            // otherwise the water also difuses out of the unreceptive cell
            s_levels_new[row][col] -= alpha2 * s_levels[row][col];
        }
    }
}

void saveBoardToFile(unsigned int* board, unsigned int n, unsigned int m, const char* filename)
{
    FILE* fp = fopen(filename, "wb+");
    if (fp != NULL) {
        fwrite(board, sizeof(unsigned int), n * m, fp);
        fclose(fp);
        // printf("Board saved to file: %s\n", filename);
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

    printf("BLOCK SIZE X: %d\n", BLOCK_SIZE_X);

    FILE* fp = fopen(filename, "a");
    if (fp == NULL) {
        printf("Failed to open the file for writing.\n");
        return 1;
    }

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

    printf("# n=%d\n# m=%d\n", n, m);
    printf("# alpha=%.1f\n# beta=%.4f\n# gamma=%.3f\n", alpha, beta, gamma);
    printf("# block_size_x=%d\n# block_size_y=%d\n", block_size_x, block_size_y);

    int numBlocksX = (m + BLOCK_SIZE_X - 5) / (BLOCK_SIZE_X - 4);
    int numBlocksY = (n + BLOCK_SIZE_Y - 5) / (BLOCK_SIZE_Y - 4);

    // printf("numBlocksX: %d\n", numBlocksX);
    // printf("numBlocksY: %d\n", numBlocksY);

    // Set grid and block dimensions
    dim3 gridSize(numBlocksX, numBlocksY);
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    float allKernelTime = 0;
    float allWriteTime = 0;
    for (int iter = 0; iter < n / 3; iter++) {
        cudaEventRecord(start);
        // Launch the kernel to update the grid
        updateGrid<<<gridSize, blockSize>>>(d_board, d_board_new, d_levels, d_levels_new, n, m, alpha, gamma);

        // Copy data from d_board_new to d_board on the GPU
        cudaMemcpy(d_board, d_board_new, n * m * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_levels, d_levels_new, n * m * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float kernelTime;
        cudaEventElapsedTime(&kernelTime, start, stop);
        allKernelTime += kernelTime;

        if (write_to_file) {
            cudaEventRecord(start);
            unsigned int* board_result = new unsigned int[n * m];
            cudaMemcpy(board_result, d_board, n * m * sizeof(unsigned int), cudaMemcpyDeviceToHost);

            // Save the board to a file
            char filename[50];
            snprintf(filename, sizeof(filename), "Data/array_%d.bin", iter);
            saveBoardToFile(board_result, n, m, filename);

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
    fprintf(fp, "%d,%d,%f,%f,%f,%d,%d,%f,%f,%f,%d,%d,%d,%d\n", n, m, alpha, beta, gamma, block_size_x, block_size_y, copyTime, secPerIter, allWriteTime, is_shared_mem, is_block, is_column, is_row);

    return 0;
}