#include "input_output.h"
#include "board.h"


int main(){

    int n = 51, m = 55;
    float alpha = 1.,beta = 0.9, gamma = 0.001;

    int N_steps = 14;
    char filename[20];

    printf("Running a grid %d x %d for %d steps\n",n,m,N_steps);

    unsigned char **board = board_initialize(n,m);
    unsigned char **board_new = board_initialize(n,m);
    float **levels = levels_initialize(n,m,beta);
    float **levels_new = levels_initialize(n,m,beta);


    for (int i = 0; i<N_steps; i++){
        board_step(levels,levels_new,board,board_new,n,m,alpha,gamma);

        copy_levels(levels_new,levels,n,m);
        copy_mat(board_new,board,n,m);

        //sprintf(filename, "snowflake_new%d.bin",i);
        //save_array(board_new[0],filename,n,m);

        // board_update(&board,&board_new);
        //sprintf(filename, "snowflake%d.bin",i);
        //save_array(board[0],filename,n,m);
        // levels_update(&levels,&levels_new);
    }
    

    save_array(board[0],"snowflake_sequential.bin",n,m);
    save_levels(levels[0],"snowflake_sequential_levels.bin",n,m);

    board_free(board);
    board_free(board_new);
    levels_free(levels);
    levels_free(levels_new);


    return 0;
}