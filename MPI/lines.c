// module load OpenMPI/4.1.0-GCC-10.2.0 
// mpicc mpi_test.c -o mpi_test
// srun --reservation=fri --mpi=pmix -n2 -N2 ./mpi_test

#include <stdlib.h>
#include <stdio.h>
#include <string.h> 
#include <mpi.h>
#include "input_output.h"
#include "board_lines.h"



int main(int argc, char *argv[])
{
    int rank; // process rank 
    int root = 0;
	int num_p; // total number of processes 
	int source; // sender rank
	int destination; // receiver rank 
	int tag = 0; // message tag 
    unsigned char **board, **myboard, **myboard_new;
    unsigned char *boardptr = NULL;
    float **levels, **mylevels, **mylevels_new;
    float *levelsptr = NULL;

    int myrows, mystart; // the number of unique rows for each process and the starting row index in the global array

    int n = 51, m = 55; // m is the width, common to all processes

    float alpha = 1., beta = 0.9, gamma = 1e-3;

    int N; // the number of processes
    int N_jumps = 20;
    int b = 2; // the buffer zone size, at least 1

    int *counts = NULL; // the number of elements in each process, including buffer zones
    int *offsets = NULL; // offsets of each process in the global array, including buffer zones
    int *return_offsets = NULL; // like offsets, but without buffer zones
    int *return_counts = NULL; // like counts, but without buffer zones

    int top_offset,bottom_offset,top_add,bottom_add,top_buffer,bottom_buffer;
    int i,j;
    char filename[100];

    MPI_Status status; // message status

    MPI_Init(&argc, &argv); // initialize MPI 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get process rank 
	MPI_Comm_size(MPI_COMM_WORLD, &N); // get number of processes

    if (rank == root)
    { 
        board = board_initialize(n,m);
        levels = levels_initialize(n,m,beta);
        counts = (int *)malloc(sizeof(int) * N);
        offsets = (int *)malloc(sizeof(int) * N);
        return_offsets = (int *)malloc(sizeof(int) * N);
        return_counts = (int *)malloc(sizeof(int) * N);

        for (i = 0; i<N; i++){
            counts[i] = ((i + 1) * (n - 2)/N - i * (n - 2)/N + 2 * b) * m;
            offsets[i] = (i * (n-2)/N - b + 1) * m;
            return_offsets[i] = m * (i * (n - 2) / N) + m;
            return_counts[i] = ((i + 1) * (n - 2)/N - i * (n - 2)/N ) * m;
        }

        counts[0] = ((n-2)/N + b + 1)*m;
        counts[N-1] = (n - (N-1)*(n-2)/N - 1 + b)*m;
        offsets[0] = 0;

        boardptr = *board;
        levelsptr = *levels;

        // printf("\n");
        // for (int i=0; i<N;i++) printf(" (%d, %d)",return_offsets[i],return_counts[i]);
        // printf("\n");
        // printf("Root board:\n");
        // board_print(board,n,m);
    }

    // calculate the number of rows in each process, including the buffer zones
    if (rank != 0 && rank != N -1) {
        myrows = (rank + 1) * (n - 2)/N - rank * (n - 2)/N + 2 * b;
    }
    else if (rank == 0){
        myrows = (n-2)/N + b + 1;
    }
    else if (rank == N - 1){
        myrows = n - (N-1)*(n-2)/N - 1 + b;
    }

    myboard = board_initialize(myrows,m);
    myboard_new = board_initialize(myrows,m);
    mylevels = levels_initialize(myrows,m,beta);
    mylevels_new = levels_initialize(myrows,m,beta);

    // scatter the data to different processes
    MPI_Scatterv(boardptr,counts,offsets,MPI_UNSIGNED_CHAR,*myboard,myrows*m,MPI_UNSIGNED_CHAR,root,MPI_COMM_WORLD);
    MPI_Scatterv(levelsptr,counts,offsets,MPI_FLOAT,*mylevels,myrows*m,MPI_FLOAT,root,MPI_COMM_WORLD);

    copy_levels(mylevels,mylevels_new,myrows,m);
    copy_mat(myboard,myboard_new,myrows,m);

    top_offset = (rank == 0) ? 1 : b;
    bottom_offset = (rank == N-1) ? 1 : b;
    // the shift in the final line of calculation at the top and bottom for each process
    top_add = (rank == 0) ? 0 : 1;
    bottom_add = (rank == N-1) ? 0 : 1;
    
    top_buffer, bottom_buffer; // sets the number of lines at the top and bottom of each segment that aren't processed

    for (i=0; i<N_jumps; i++) {
        // run independenly as long as the buffers permit

        top_buffer = 1;
        bottom_buffer = 1;

        for (j = 0; j<max(1,b-1); j++){
 
            board_step_lines(mylevels,mylevels_new,myboard,myboard_new,myrows,m,alpha,gamma,top_buffer,bottom_buffer);
            // copy the results to the original board
            copy_levels(mylevels_new,mylevels,myrows,m);
            copy_mat(myboard_new,myboard,myrows,m);
            //board_update(&myboard,&myboard_new);
            //levels_update(&mylevels,&mylevels_new);
            top_buffer += top_add;
            bottom_buffer += bottom_add;
        }

        // exchange data
        if (rank > 0 && rank < N-1){
            // send the bottom rows and receive the top rows
            MPI_Sendrecv(myboard[myrows - 2*b],b*m,MPI_UNSIGNED_CHAR,rank + 1,0,
                        *myboard,b*m,MPI_UNSIGNED_CHAR,rank - 1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            // send the top rows and recieive the bottom rows
            MPI_Sendrecv(myboard[b],b*m,MPI_UNSIGNED_CHAR,rank - 1,0,
                         myboard[myrows - b],b*m,MPI_UNSIGNED_CHAR,rank + 1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            
            //the same, but for levels
            MPI_Sendrecv(mylevels[myrows - 2*b],b*m,MPI_FLOAT,rank + 1,0,
                        *mylevels,b*m,MPI_FLOAT,rank - 1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Sendrecv(mylevels[b],b*m,MPI_FLOAT,rank - 1,0,
                         mylevels[myrows - b],b*m,MPI_FLOAT,rank + 1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        else if (rank == 0){
            // proces 0 exchanges data with process 1
            MPI_Sendrecv(myboard[myrows - 2*b],b*m,MPI_UNSIGNED_CHAR,1,0,
                         myboard[myrows - b], b*m, MPI_UNSIGNED_CHAR,1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            // levels
            MPI_Sendrecv(mylevels[myrows - 2*b],b*m,MPI_FLOAT,1,0,
                         mylevels[myrows - b], b*m, MPI_FLOAT,1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        else if (rank == N-1) {
            // process N - 1 exchanges data with process N - 2
            MPI_Sendrecv(myboard[b],b*m,MPI_UNSIGNED_CHAR,N-2,0,
                        *myboard, b*m, MPI_UNSIGNED_CHAR,N-2,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            // levels
            MPI_Sendrecv(mylevels[b],b*m,MPI_FLOAT,N-2,0,
                        *mylevels, b*m, MPI_FLOAT,N-2,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }


        process_after_exchange(myboard,myrows,m,top_offset,bottom_offset);

        // copy the values to the _new structures
        copy_levels(mylevels,mylevels_new,myrows,m);
        copy_mat(myboard,myboard_new,myrows,m);

    }

    MPI_Gatherv(myboard[top_offset],(myrows - top_offset - bottom_offset)*m,MPI_UNSIGNED_CHAR,
                boardptr,return_counts,return_offsets,MPI_UNSIGNED_CHAR,root,MPI_COMM_WORLD);

    MPI_Gatherv(mylevels[top_offset],(myrows - top_offset - bottom_offset)*m,MPI_FLOAT,
                levelsptr,return_counts,return_offsets,MPI_FLOAT,root,MPI_COMM_WORLD);

    if (rank == root){
        printf("Root board\n");
        board_print(board,n,m);
        save_array(boardptr,"lines_snowflake.bin",n,m);
        save_levels(levelsptr,"lines_snowflake_levels.bin",n,m);
    }


    // free memory
    if (rank == root) {
        board_free(board);
        levels_free(levels);
        
    }

    board_free(myboard);
    levels_free(mylevels);
    free(counts);
    free(offsets);
    free(return_offsets);
    free(return_counts);

    MPI_Finalize();


    return 0;
}