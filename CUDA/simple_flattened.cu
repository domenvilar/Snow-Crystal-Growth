#include "board.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// nvcc test.cu -O2 -o test
// srun --reservation=fri -G1 -n1 test

// Grid dimensions
#define BLOCK_SIZE 4 // Number of threads per block

__device__ int lock = 0;

__global__ void updateGrid(unsigned int* d_board, unsigned int* d_board_new, float* d_levels, float* d_levels_new, int n_rows, int n_cols, float alpha, float gamma)
{

    __shared__ unsigned int s_board[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    __shared__ float s_levels[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    __shared__ unsigned int s_board_new[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    __shared__ float s_levels_new[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    // precompute coefficients
    float alpha12 = alpha / 12.;
    float alpha2 = alpha / 2.;
    char neighs_even_col[6][2] = { { -1, 0 }, { -1, 1 }, { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, -1 } };
    char neighs_odd_col[6][2] = { { -1, 0 }, { 0, 1 }, { 1, 1 }, { 1, 0 }, { 1, -1 }, { 0, -1 } };
    char(*neighs)[2]; // the pointer to the neighbor offset arrays

    int col = blockIdx.x * blockDim.x + threadIdx.x; // column
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row

    // idx v globalnem pomnilniku
    int idx = row * n_cols + col;

    // idx znotraj shared memory
    int shared_col = threadIdx.x + 1;
    int shared_row = threadIdx.y + 1;

    // Load current cell and buffered cells into shared memory
    if (row < n_rows && col < n_cols) {
        s_board[shared_row][shared_col] = d_board[idx];
        s_levels[shared_row][shared_col] = d_levels[idx];

        s_board_new[shared_row][shared_col] = d_board_new[idx];
        s_levels_new[shared_row][shared_col] = d_levels_new[idx];

        // upper edge
        if (threadIdx.y == 0) {
            s_board[0][shared_col] = d_board[(row - 1) * n_cols + col];
            s_levels[0][shared_col] = d_levels[(row - 1) * n_cols + col];
            s_board_new[0][shared_col] = d_board_new[(row - 1) * n_cols + col];
            s_levels_new[0][shared_col] = d_levels_new[(row - 1) * n_cols + col];
        }

        // lower edge
        if (threadIdx.y == blockDim.y - 1) {
            s_board[shared_row + 1][shared_col] = d_board[(row + 1) * n_cols + col];
            s_levels[shared_row + 1][shared_col] = d_levels[(row + 1) * n_cols + col];
            s_board_new[shared_row + 1][shared_col] = d_board_new[(row + 1) * n_cols + col];
            s_levels_new[shared_row + 1][shared_col] = d_levels_new[(row + 1) * n_cols + col];
        }

        // right edge
        if (threadIdx.x == blockDim.x - 1) {
            s_board[shared_row][shared_col + 1] = d_board[row * n_cols + col + 1];
            s_levels[shared_row][shared_col + 1] = d_levels[row * n_cols + col + 1];
            s_board_new[shared_row][shared_col + 1] = d_board_new[row * n_cols + col + 1];
            s_levels_new[shared_row][shared_col + 1] = d_levels_new[row * n_cols + col + 1];
        }

        // left edge
        if (threadIdx.x == 0) {
            s_board[shared_row][0] = d_board[row * n_cols + col - 1];
            s_levels[shared_row][0] = d_levels[row * n_cols + col - 1];
            s_board_new[shared_row][0] = d_board_new[row * n_cols + col - 1];
            s_levels_new[shared_row][0] = d_levels_new[row * n_cols + col - 1];
        }

        // top left corner
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            s_board[0][0] = d_board[(row - 1) * n_cols + col - 1];
            s_levels[0][0] = d_levels[(row - 1) * n_cols + col - 1];
            s_board_new[0][0] = d_board_new[(row - 1) * n_cols + col - 1];
            s_levels_new[0][0] = d_levels_new[(row - 1) * n_cols + col - 1];
        }

        // top right corner
        if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0) {
            s_board[0][shared_col + 1] = d_board[(row - 1) * n_cols + col + 1];
            s_levels[0][shared_col + 1] = d_levels[(row - 1) * n_cols + col + 1];
            s_board_new[0][shared_col + 1] = d_board_new[(row - 1) * n_cols + col + 1];
            s_levels_new[0][shared_col + 1] = d_levels_new[(row - 1) * n_cols + col + 1];
        }

        // bottom left corner
        if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1) {
            s_board[shared_row + 1][0] = d_board[(row + 1) * n_cols + col - 1];
            s_levels[shared_row + 1][0] = d_levels[(row + 1) * n_cols + col - 1];
            s_board_new[shared_row + 1][0] = d_board_new[(row + 1) * n_cols + col - 1];
            s_levels_new[shared_row + 1][0] = d_levels_new[(row + 1) * n_cols + col - 1];
        }

        // bottom right corner
        if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
            s_board[shared_row + 1][shared_col + 1] = d_board[(row + 1) * n_cols + col + 1];
            s_levels[shared_row + 1][shared_col + 1] = d_levels[(row + 1) * n_cols + col + 1];
            s_board_new[shared_row + 1][shared_col + 1] = d_board_new[(row + 1) * n_cols + col + 1];
            s_levels_new[shared_row + 1][shared_col + 1] = d_levels_new[(row + 1) * n_cols + col + 1];
        }
    }

    __syncthreads();

    // Only one thread per block prints the shared memory

    /* if (threadIdx.x == 0 && threadIdx.y == 0) {
        // Acquire the lock
        while (atomicCAS(&lock, 0, 1) != 0)
            ;
        printf("Before calculation Block (%d, %d)\n", blockIdx.x, blockIdx.y);
        // Print the shared memory
        for (int i = 0; i < BLOCK_SIZE + 2; i++) {
            for (int j = 0; j < BLOCK_SIZE + 2; j++) {
                printf("%d ", s_board_new[i][j]);
            }
            printf("\n");
        }
        printf("\n");

        // print the shared levels
        printf("levels\n");
        for (int i = 0; i < BLOCK_SIZE + 2; i++) {
            for (int j = 0; j < BLOCK_SIZE + 2; j++) {
                printf("%f ", s_levels_new[i][j]);
            }
            printf("\n");
        }
        printf("\n");

        // Release the lock
        lock = 0;
    } */
    // Check if the coordinates are within the grid bounds
    // and current cell is not frozen or edge
    // 0 -> edge cell
    // 1 -> unreceptive
    // 2 -> boundary
    // 3 -> frozen
    // only unreceptive and boundary cells can receive water
    if (col < n_cols && row < n_rows && s_board[shared_row][shared_col] != 0) {
        neighs = (col % 2 == 0) ? neighs_even_col : neighs_odd_col;

        // find the accumulation of water from the neighbors
        for (int k = 0; k < 6; k++) {
            int neigh_row = shared_row + neighs[k][0]; // y
            int neigh_col = shared_col + neighs[k][1]; // x

            // edge and unreceptive cells contributes water
            if (s_board[neigh_row][neigh_col] < 2) {
                s_levels_new[shared_row][shared_col] += alpha12 * s_levels[neigh_row][neigh_col];
            }
        }

        // if the cell is boundary
        // add the water vapor and check if it freezes
        if (s_board[shared_row][shared_col] == 2) {
            s_levels_new[shared_row][shared_col] += gamma; // add the water vapor

            // if the cell freezes neighbors become boundary
            if (s_levels_new[shared_row][shared_col] >= 1) {
                s_board_new[shared_row][shared_col] = 3;

                for (int k = 0; k < 6; k++) {
                    int neigh_row = shared_row + neighs[k][0];
                    int neigh_col = shared_col + neighs[k][1];

                    // we dont update edge cells
                    if (s_board[neigh_row][neigh_col] != 0) {
                        atomicMax(&s_board_new[neigh_row][neigh_col], 2);
                    }
                }
            }

        } else if (s_board[shared_row][shared_col] == 1) {
            // otherwise the water also difuses out of the unreceptive cell
            s_levels_new[shared_row][shared_col] -= alpha2 * s_levels[shared_row][shared_col];
        }
    }

    // Synchronize threads to ensure all shared memory updates are complete
    __syncthreads();

    // Only one thread per block prints the shared memory

    /* if (threadIdx.x == 0 && threadIdx.y == 0) {
        // Acquire the lock
        while (atomicCAS(&lock, 0, 1) != 0)
            ;

        printf("After calculation Block (%d, %d)\n", blockIdx.x, blockIdx.y);
        // Print the shared memory
        for (int i = 0; i < BLOCK_SIZE + 2; i++) {
            for (int j = 0; j < BLOCK_SIZE + 2; j++) {
                printf("%d ", s_board_new[i][j]);
            }
            printf("\n");
        }
        printf("\n");

        // print the shared levels
        printf("levels\n");
        for (int i = 0; i < BLOCK_SIZE + 2; i++) {
            for (int j = 0; j < BLOCK_SIZE + 2; j++) {
                printf("%f ", s_levels_new[i][j]);
            }
            printf("\n");
        }
        printf("\n");

        // Release the lock
        lock = 0;
    } */

    // Copy values from shared memory back to global memory
    if (row + 1 < n_rows && col - 1 < n_cols) {
        d_board_new[idx] = s_board_new[shared_row][shared_col];
        d_levels_new[idx] = s_levels_new[shared_row][shared_col];
    }
    __syncthreads();

    // Write buffered edge cells to global memory using atomic operations
    if (threadIdx.x == 0 && col > 0) {
        // Left edge
        if (s_board_new[shared_row][shared_col - 1] == 2) // col is global index in d_board
            atomicMax(&d_board_new[row * n_cols + col - 1], 2);

        // Top left corner
        if (threadIdx.y == 0 && s_board_new[shared_row - 1][shared_col - 1] == 2) {
            atomicMax(&d_board_new[(row - 1) * n_cols + (col - 1)], 2);
        }
        // Bottom left corner
        if (threadIdx.y == blockDim.y - 1 && s_board_new[shared_row + 1][shared_col - 1] == 2) {
            atomicMax(&d_board_new[(row + 1) * n_cols + (col - 1)], 2);
        }
    }

    if (threadIdx.x == blockDim.x - 1 && col < n_cols - 1) {
        // Right edge
        if (s_board_new[shared_row][shared_col + 1] == 2)
            atomicMax(&d_board_new[row * n_cols + col + 1], 2);

        // Top right corner
        if (threadIdx.y == 0 && s_board_new[shared_row - 1][shared_col + 1] == 2) {
            atomicMax(&d_board_new[(row - 1) * n_cols + (col + 1)], 2);
        }

        // Bottom right corner
        if (threadIdx.y == blockDim.y - 1 && s_board_new[shared_row + 1][shared_col + 1] == 2) {
            atomicMax(&d_board_new[(row + 1) * n_cols + (col + 1)], 2);
        }
    }

    // top edge
    if (threadIdx.y == 0 && row > 0 && s_board_new[shared_row - 1][shared_col] == 2) {
        atomicMax(&d_board_new[(row - 1) * n_cols + col], 2);
    }

    // bottom edge
    if (threadIdx.y == blockDim.y - 1 && row < n_rows - 1 && s_board_new[shared_row + 1][shared_col] == 2) {
        atomicMax(&d_board_new[(row + 1) * n_cols + col], 2);
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

    int numBlocksX = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int numBlocksY = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

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

    for (int iter = 0; iter < 30; iter++) {

        // Launch the kernel to update the grid
        updateGrid<<<gridSize, blockSize, (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) + (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2)>>>(d_board, d_board_new, d_levels, d_levels_new, n, m, alpha, gamma);

        /* d_board = d_board_new;
        d_levels = d_levels_new; */

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
