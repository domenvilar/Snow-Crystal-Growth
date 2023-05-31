#include "board.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// nvcc test.cu -O2 -o test
// srun --reservation=fri -G1 -n1 test

// Grid dimensions
#define BLOCK_SIZE 8 // Number of threads per block

__device__ int lock = 0;

__global__ void updateGrid(unsigned int* d_board, unsigned int* d_board_new, float* d_levels, float* d_levels_new, int n_rows, int n_cols, float alpha, float gamma)
{
    // printf("Block (%d, %d) started\n", blockIdx.x, blockIdx.y);
    __shared__ unsigned int s_board[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_levels[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ unsigned int s_board_new[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_levels_new[BLOCK_SIZE][BLOCK_SIZE];

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
    // 1,2,3 cells can receive water, calculalte water levels for 1x buffered edge
    if (row_global < n_rows && col_global < n_cols && row_global >= 0 && col_global >= 0 && s_board[row][col] != 0 && col > 0 && col < BLOCK_SIZE - 1 && row > 0 && row < BLOCK_SIZE - 1) {
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

    // Synchronize threads to ensure all shared memory updates are complete
    __syncthreads();

    // Copy values from shared memory back to global memory
    if (row + 2 < blockDim.x && col + 2 < blockDim.y && row > 1 && col > 1 && row_global < n_rows && col_global < n_cols && row_global >= 0 && col_global >= 0) {
        d_board_new[idx] = s_board_new[row][col];
        d_levels_new[idx] = s_levels_new[row][col];
    }

    // Only one thread per block prints the shared memory

    /*
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // Acquire the lock
        while (atomicCAS(&lock, 0, 1) != 0)
            ;

        // print the shared levels
        printf("After calculation Block (row %d, col %d)\n", blockIdx.y, blockIdx.x);
        printf("levels after\n");
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                printf("%f ", s_levels_new[i][j]);
            }
            printf("\n");
        }
        printf("\n");

        // print the shared board
        printf("board after\n");
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                printf("%d ", s_board_new[i][j]);
            }
            printf("\n");
        }

        // Release the lock
        // Release the lock
        atomicExch(&lock, 0);
    }
    __syncthreads();

    */
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
    int n = 60, m = 60;
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

    int numBlocksX = (m + BLOCK_SIZE - 5) / (BLOCK_SIZE - 4);
    int numBlocksY = (n + BLOCK_SIZE - 5) / (BLOCK_SIZE - 4);

    printf("numBlocksX: %d\n", numBlocksX);
    printf("numBlocksY: %d\n", numBlocksY);

    // Set grid and block dimensions
    dim3 gridSize(numBlocksX, numBlocksY);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    // print board
    printf("Initial board:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%d ", board[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    // print levels
    printf("Initial levels:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%.2f ", levels[i][j]);
        }
        printf("\n");
    }

    for (int iter = 0; iter < 30; iter++) {

        // Launch the kernel to update the grid
        updateGrid<<<gridSize, blockSize>>>(d_board, d_board_new, d_levels, d_levels_new, n, m, alpha, gamma);

        // Copy data from d_board_new to d_board on the GPU
        cudaMemcpy(d_board, d_board_new, n * m * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_levels, d_levels_new, n * m * sizeof(float), cudaMemcpyDeviceToDevice);

        unsigned int* board_result = new unsigned int[n * m];
        cudaMemcpy(board_result, d_board, n * m * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                printf("%u ", board_result[i * m + j]);
            }
            printf("\n");
        }
        printf("\n");

        // Save the board to a file
        char filename[50];
        snprintf(filename, sizeof(filename), "Data/array_%d.bin", iter);
        saveBoardToFile(board_result, n, m, filename);
    }

    cudaFree(d_board);
    cudaFree(d_board_new);
    cudaFree(d_levels);
    cudaFree(d_levels_new);

    return 0;
}