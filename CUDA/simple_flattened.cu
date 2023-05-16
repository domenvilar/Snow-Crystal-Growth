#include "board.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// nvcc test.cu -O2 -o test
// srun --reservation=fri -G1 -n1 test

// Grid dimensions
#define BLOCK_SIZE 16 // Number of threads per block 16x16

__global__ void updateGrid(unsigned int* d_board, unsigned int* d_board_new, float* d_levels, float* d_levels_new, int n_rows, int n_cols, float alpha, float gamma)
{

    __shared__ unsigned int shared_board[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    __shared__ float shared_levels[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    // precompute coefficients
    float alpha12 = alpha / 12.;
    float alpha2 = alpha / 2.;
    char neighs_even_col[6][2] = { { -1, 0 }, { -1, 1 }, { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, -1 } };
    char neighs_odd_col[6][2] = { { -1, 0 }, { 0, 1 }, { 1, 1 }, { 1, 0 }, { 1, -1 }, { 0, -1 } };
    char(*neighs)[2]; // the pointer to the neighbor offset arrays

    int col = blockIdx.x * blockDim.x + threadIdx.x; // column
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row

    int idx = row * n_cols + col;

    // buffered
    int shared_row = threadIdx.y + 1;
    int shared_col = threadIdx.x + 1;
    int shared_idx = shared_row * (BLOCK_SIZE + 2) + shared_col;

    // Load current cell and buffered cells into shared memory
    if (row < n_rows && col < n_cols) {
        shared_board[shared_row][shared_col] = d_board[idx];
        shared_levels[shared_row][shared_col] = d_levels[idx];

        // Load halo cells
        // left edge
        if (threadIdx.x == 0) {
            shared_board[shared_row][shared_col - 1] = d_board[row * n_cols + max(col - 1, 0)];
            shared_levels[shared_row][shared_col - 1] = d_levels[row * n_cols + max(col - 1, 0)];

            // top left corner
            if (threadIdx.y == 0) {
                shared_board[shared_row - 1][shared_col - 1] = d_board[max(row - 1, 0) * n_cols + max(col - 1, 0)];
                shared_levels[shared_row - 1][shared_col - 1] = d_levels[max(row - 1, 0) * n_cols + max(col - 1, 0)];
            }
            // bottom left corner
            if (threadIdx.y == blockDim.y - 1) {
                shared_board[shared_row + 1][shared_col - 1] = d_board[min(row + 1, n_rows - 1) * n_cols + max(col - 1, 0)];
                shared_levels[shared_row + 1][shared_col - 1] = d_levels[min(row + 1, n_rows - 1) * n_cols + max(col - 1, 0)];
            }
        }
        // right edge
        if (threadIdx.x == blockDim.x - 1) {
            shared_board[shared_row][shared_col + 1] = d_board[row * n_cols + min(col + 1, n_cols - 1)];
            shared_levels[shared_row][shared_col + 1] = d_levels[row * n_cols + min(col + 1, n_cols - 1)];

            // top right corner
            if (threadIdx.y == 0) {
                shared_board[shared_row - 1][shared_col + 1] = d_board[max(row - 1, 0) * n_cols + min(col + 1, n_cols - 1)];
                shared_levels[shared_row - 1][shared_col + 1] = d_levels[max(row - 1, 0) * n_cols + min(col + 1, n_cols - 1)];
            }
            // bottom right corner
            if (threadIdx.y == blockDim.y - 1) {
                shared_board[shared_row + 1][shared_col + 1] = d_board[min(row + 1, n_rows - 1) * n_cols + min(col + 1, n_cols - 1)];
                shared_levels[shared_row + 1][shared_col + 1] = d_levels[min(row + 1, n_rows - 1) * n_cols + min(col + 1, n_cols - 1)];
            }
        }
        // top edge
        if (threadIdx.y == 0) {
            shared_board[shared_row - 1][shared_col] = d_board[max(row - 1, 0) * n_cols + col];
            shared_levels[shared_row - 1][shared_col] = d_levels[max(row - 1, 0) * n_cols + col];
        }
        // bottom edge
        if (threadIdx.y == blockDim.y - 1) {
            shared_board[shared_row + 1][shared_col] = d_board[min(row + 1, n_rows - 1) * n_cols + col];
            shared_levels[shared_row + 1][shared_col] = d_levels[min(row + 1, n_rows - 1) * n_cols + col];
        }
    }

    // Synchronize to ensure all data is loaded into shared memory
    __syncthreads();

    // Check if the coordinates are within the grid bounds
    // and current cell is not frozen or edge
    if (col < n_cols && row < n_rows && d_board[idx] != 3 && d_board[idx] != 0) {
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
    FILE* fp = fopen(filename, "wb+");
    if (fp != NULL) {
        fwrite(board, sizeof(unsigned int), n * m, fp);
        fclose(fp);
        printf("Board saved to file: %s\n", filename);
    } else {
        printf("Failed to open file for writing: %s\n", filename);
    }
}

int main()
{
    int n = 50, m = 60;
    float alpha = 1., beta = 0.9, gamma = 0.001;

    int N_steps = 14;
    printf("Initializing board..\n");

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

    cudaMalloc((void**)&d_board, n * m * sizeof(unsigned int));
    cudaMalloc((void**)&d_board_new, n * m * sizeof(unsigned int));
    cudaMalloc((void**)&d_levels, n * m * sizeof(float));
    cudaMalloc((void**)&d_levels_new, n * m * sizeof(float));

    cudaMemcpy(d_board, board[0], n * m * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_board_new, board_new[0], n * m * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_levels, levels[0], n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_levels_new, levels_new[0], n * m * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocksX = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int numBlocksY = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("numBlocksX: %d\n", numBlocksX);
    printf("numBlocksY: %d\n", numBlocksY);

    // Set grid and block dimensions
    dim3 gridSize(numBlocksX, numBlocksY);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    for (int iter = 0; iter < 1000; iter++) {

        // Launch the kernel to update the grid
        updateGrid<<<gridSize, blockSize>>>(d_board, d_board_new, d_levels, d_levels_new, n, m, alpha, gamma);

        /* d_board = d_board_new;
        d_levels = d_levels_new; */

        // Copy data from d_board_new to d_board on the GPU
        cudaMemcpy(d_board, d_board_new, n * m * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_levels, d_levels_new, n * m * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

        unsigned int* board_result = new unsigned int[n * m];
        cudaMemcpy(board_result, d_board, n * m * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        // Save the board to a file
        char filename[50];
        snprintf(filename, sizeof(filename), "Data/array_%d.bin", iter);
        saveBoardToFile(board_result, n, m, filename);
    }

    // Copy the final grid from the device to the host
    // unsigned int* board_result = new unsigned int[n * m];
    /* cudaMemcpy(board_result, d_board, n * m * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Save the board to a file
    FILE* fp = fopen("array.bin", "wb+");

    if (fp != NULL) {
        fwrite(board_result, sizeof(unsigned int), n * m, fp);
        fclose(fp);
        printf("Done writing successfully\n");
    } */

    // Print the board
    /* for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%u ", board_result[i * m + j]);
        }
        printf("\n");
    }

    // Print the board
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%u ", board[i][j]);
        }
        printf("\n");
    } */

    cudaFree(d_board);
    cudaFree(d_board_new);
    cudaFree(d_levels);
    cudaFree(d_levels_new);

    return 0;
}
