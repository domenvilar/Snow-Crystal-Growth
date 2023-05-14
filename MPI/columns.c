// module load OpenMPI/4.1.0-GCC-10.2.0 
// mpicc mpi_test.c -o mpi_test
// srun --reservation=fri --mpi=pmix -n2 -N2 ./mpi_test

#include <stdlib.h>
#include <stdio.h>
#include <string.h> 
#include <mpi.h>
#include "input_output.h"
#include "board_cols.h"

/*So when you send a vector you give a pointer where the vector starts and the stride,block and numbe parameters
determine how the data is picked from the selected buffer.
The sent data can be thought as being compacted into a block, removing any voids due to the block size being less than
stride.
On the receiving end then this data can be thought as the compressed version of the vector and if the received data
type is a vector, this contiguous received data is unpacked the same way it was unpacked, but this time for the 
vector on the receiving end.*/

void set_to_value(unsigned char **mat,int rows,int cols)
{
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            mat[i][j] = 10 + j;
        }
    }
}

void set_to_number(unsigned char **mat,int num,int rows,int cols)
{
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            mat[i][j] = num*10;
        }
    }
}

void shrink(unsigned char **mat, int n, int m, int top_buffer, int bottom_buffer)
{
    // precompute coefficients
    int i,j;

    int buffer = max(top_buffer,bottom_buffer);

    for (i = 1; i<n-1; i++){
        for (j = top_buffer; j<m-bottom_buffer; j++){
            mat[i][j] = (mat[i][j]/10) * 10 + buffer;
        }
    }
}

int main(int argc, char *argv[])
{
    int rank; // process rank 
    int root = 0;
	int num_p; // total number of processes 
	int source; // sender rank
	int destination; // receiver rank 
	int tag = 0; // message tag 
    unsigned char **board, **myboard, **myboard_new,**boardT,**myboardT;
    unsigned char *boardptr = NULL;
    float **levels, **mylevels, **mylevels_new;
    float *levelsptr = NULL;

    int mycols, mystart; // the number of rows for each process (including buffer) and the starting column index in the global array

    int n = 51, m = 55; // m is the width, common to all processes

    float alpha = 1., beta = 0.9, gamma = 1e-3;

    int N; // the number of processes
    int N_jumps = 7;
    int b = 4; // the buffer zone size, at least 1

    int *counts = NULL; // the number of elements in each process, including buffer zones
    int *offsets = NULL; // offsets of each process in the global array, including buffer zones
    int *return_offsets = NULL; // like offsets, but without buffer zones
    int *return_counts = NULL; // like counts, but without buffer zones

    int top_offset,bottom_offset,top_add,bottom_add,top_buffer,bottom_buffer;
    int i,j;
    char filename[100];

    MPI_Status status; // message status

    MPI_Datatype coltype_scatter_send,col_buff,return_buff,col_buff_lvl,return_buff_lvl;

    MPI_Init(&argc, &argv); // initialize MPI 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get process rank 
	MPI_Comm_size(MPI_COMM_WORLD, &N); // get number of processes


    if (rank == root)
    { 
        board = board_initialize(n,m);
        //boardT = transpose_board(board,n,m); // transpose the board
        //set_to_value(board,n,m);
        levels = levels_initialize(n,m,beta);
        counts = (int *)malloc(sizeof(int) * N);
        offsets = (int *)malloc(sizeof(int) * N);
        return_offsets = (int *)malloc(sizeof(int) * N);
        return_counts = (int *)malloc(sizeof(int) * N);

        for (i = 0; i<N; i++){
            counts[i] = ((i + 1) * (m - 2)/N - i * (m - 2)/N + 2 * b);
            offsets[i] = (i * (m-2)/N - b + 1);
            return_offsets[i] = (i * (m - 2) / N) + 1;
            return_counts[i] = ((i + 1) * (m - 2)/N - i * (m - 2)/N );
        }

        counts[0] = (m-2)/N + b + 1;
        counts[N-1] = m - (N-1)*(m-2)/N - 1 + b;
        offsets[0] = 0;

        boardptr = *board;
        levelsptr = *levels;

        printf("\n");
        for (int i=0; i<N;i++) printf(" (%d, %d)",offsets[i],counts[i]);
        printf("\n");
        // board_print(board,n,m);
    }

    // calculate the number of rows in each process, including the buffer zones
    if (rank != 0 && rank != N -1) {
        mycols = (rank + 1) * (m - 2)/N - rank * (m - 2)/N + 2 * b;
        mystart = rank * (m-2)/N - b + 1;
    }
    else if (rank == 0){
        mycols = (m-2)/N + b + 1;
        mystart = 0;
    }
    else if (rank == N - 1){
        mycols = m - (N-1)*(m-2)/N - 1 + b;
        mystart = rank * (m-2)/N - b + 1;
    }

    myboard = board_initialize(n,mycols);
    myboard_new = board_initialize(n,mycols);
    mylevels = levels_initialize(n,mycols,beta);
    mylevels_new = levels_initialize(n,mycols,beta);

    top_offset = (rank == 0) ? 1 : b;
    bottom_offset = (rank == N-1) ? 1 : b;

    top_add = (rank == 0) ? 0 : 1;
    bottom_add = (rank == N-1) ? 0 : 1;

    // define the buffer area to be exchanged between the cells
    MPI_Type_vector(n,b,mycols,MPI_UNSIGNED_CHAR,&col_buff);
    MPI_Type_vector(n-2,mycols - top_offset - bottom_offset,mycols,MPI_UNSIGNED_CHAR,&return_buff); // the return area
    // the same as above, but for levels
    MPI_Type_vector(n,b,mycols,MPI_FLOAT,&col_buff_lvl);
    MPI_Type_vector(n-2,mycols - top_offset - bottom_offset,mycols,MPI_FLOAT,&return_buff_lvl);
    // commit the types
    MPI_Type_commit(&col_buff);
    MPI_Type_commit(&return_buff);
    MPI_Type_commit(&col_buff_lvl);
    MPI_Type_commit(&return_buff_lvl);

    // scatter the columns
    if (rank == 0){
        // for all but the root process
        for (i = 1; i < N; i++)
        {
            MPI_Type_vector(n,counts[i],m,MPI_UNSIGNED_CHAR,&coltype_scatter_send);
            MPI_Type_commit(&coltype_scatter_send);
            MPI_Send(&board[0][offsets[i]],1,coltype_scatter_send,i,0,MPI_COMM_WORLD);

            MPI_Type_vector(n,counts[i],m,MPI_FLOAT,&coltype_scatter_send);
            MPI_Type_commit(&coltype_scatter_send);
            MPI_Send(&levels[0][offsets[i]],1,coltype_scatter_send,i,0,MPI_COMM_WORLD);
        }

        // store the root process board directly
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < mycols; j++)
            {
                myboard[i][j] = board[i][j];
                mylevels[i][j] = levels[i][j];
            }
        }
    }
    else {
        // receiving end of the non-root processes
        MPI_Recv(*myboard,mycols*n,MPI_UNSIGNED_CHAR,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(*mylevels,mycols*n,MPI_FLOAT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }

    copy_levels(mylevels,mylevels_new,n,mycols);
    copy_mat(myboard,myboard_new,n,mycols);

    // TESTING
    //set_to_number(myboard,rank+1,n,mycols); // for testing which process writes to which area


    for (i=0; i<N_jumps; i++){
        top_buffer = 1;
        bottom_buffer = 1;
        
        for (j = 0; j < max(1,b-1); j++)
        {
            board_step_cols(mylevels,mylevels_new,myboard,myboard_new,n,mycols,alpha,gamma,top_buffer,bottom_buffer,mystart);

            copy_levels(mylevels_new,mylevels,n,mycols);
            copy_mat(myboard_new,myboard,n,mycols);

            top_buffer += top_add;
            bottom_buffer += bottom_add;
        }

        // exchange data
        if (rank > 0 && rank < N-1){
            // rightward pass - receive from the left and send to the right
            MPI_Sendrecv(&myboard[0][mycols - 2*b],1,col_buff,rank + 1, 0,
                        *myboard,1,col_buff,rank - 1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            // leftward pass - receive from the right and send to the left
            MPI_Sendrecv(&myboard[0][b],1,col_buff,rank - 1, 0,
                         &myboard[0][mycols-b],1,col_buff,rank + 1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            
            // the same, but for levels
            MPI_Sendrecv(&mylevels[0][mycols - 2*b],1,col_buff_lvl,rank + 1, 0,
                        *mylevels,1,col_buff_lvl,rank - 1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Sendrecv(&mylevels[0][b],1,col_buff_lvl,rank - 1, 0,
                         &mylevels[0][mycols-b],1,col_buff_lvl,rank + 1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        else if (rank == 0){
            MPI_Sendrecv(&myboard[0][mycols - 2*b],1,col_buff,1,0,
                         &myboard[0][mycols - b], 1, col_buff,1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            // levels
            MPI_Sendrecv(&mylevels[0][mycols - 2*b],1,col_buff_lvl,1,0,
                         &mylevels[0][mycols - b], 1, col_buff_lvl,1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        else {
            // rank N - 1
            MPI_Sendrecv(&myboard[0][b],1,col_buff,N-2,0,
                        *myboard,1,col_buff,N-2,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            // levels
            MPI_Sendrecv(&mylevels[0][b],1,col_buff_lvl,N-2,0,
                         *mylevels,1,col_buff_lvl,N-2,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }

        process_after_exchange(myboard,n,mycols,top_offset,bottom_offset,mystart);

        // copy the values to the _new structures
        copy_levels(mylevels,mylevels_new,n,mycols);
        copy_mat(myboard,myboard_new,n,mycols);

    }

    // gather the columns
    if (rank == 0){
        // for all but the root process
        for (i = 1; i < N; i++)
        {
            MPI_Type_vector(n-2,return_counts[i],m,MPI_UNSIGNED_CHAR,&coltype_scatter_send);
            MPI_Type_commit(&coltype_scatter_send);
            MPI_Recv(&board[1][return_offsets[i]],1,coltype_scatter_send,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            // levels
            MPI_Type_vector(n-2,return_counts[i],m,MPI_FLOAT,&coltype_scatter_send);
            MPI_Type_commit(&coltype_scatter_send);
            MPI_Recv(&levels[1][return_offsets[i]],1,coltype_scatter_send,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }

        // store the root process board directly
        for (i = 1; i < n-1; i++)
        {
            for (j = 1; j < mycols-b; j++)
            {
                board[i][j] = myboard[i][j];
                levels[i][j] = mylevels[i][j];
            }
        }
        
    }
    else {
        // receiving end of the non-root processes
        MPI_Send(&myboard[1][top_offset],1,return_buff,0,0,MPI_COMM_WORLD);
        MPI_Send(&mylevels[1][top_offset],1,return_buff_lvl,0,0,MPI_COMM_WORLD);
    }


    if (rank == 2){
        printf("rank %d board\n",rank);
        board_print(myboard,n,mycols);
    }

    if (rank == root){
        printf("Root board\n");
        board_print(board,n,m);
        save_array(boardptr,"cols_snowflake.bin",n,m);
        save_levels(levelsptr,"cols_snowflake_levels.bin",n,m);
    }
    // free memory
    if (rank == root) {
        board_free(board);
        levels_free(levels);
        MPI_Type_free(&coltype_scatter_send);
    }

    board_free(myboard);
    levels_free(mylevels);
    free(counts);
    free(offsets);
    free(return_offsets);
    free(return_counts);

    MPI_Type_free(&col_buff);
    MPI_Type_free(&return_buff);
    MPI_Type_free(&col_buff_lvl);
    MPI_Type_free(&return_buff_lvl);
    MPI_Finalize();


    return 0;
}