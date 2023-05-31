#include "board.h"


void board_step_lines(float **levels,float **levels_new,
                unsigned char **mat, unsigned char **mat_new,
                int n, int m,
                float alpha, float gamma, int top_buffer, int bottom_buffer)
{
    // precompute coefficients
    float alpha12 = alpha/12.;
    float alpha2 = alpha/2.;
    const char (*neighs)[2]; // the pointer to the neighbor offset arrays

    int ni,nj; // the neighbor indices
    int i,j,k;

    // copy the matrix since I can't find a better way
    // for (i = top_buffer; i<n-bottom_buffer; i++){
    //     for (j = 1; j<m-1; j++){
    //         mat_new[i][j] = mat[i][j];
    //     }
    // }
    
    for (i = top_buffer; i<n-bottom_buffer; i++){
        for (j = 1; j<m-1; j++){

            if (mat[i][j] == 3) continue; // if the cell is frozen, skip this step

            neighs = (j%2 == 0) ? neighs_even_col_h : neighs_odd_col_h; // determine the neighbor pointer
            
            //levels_new[i][j] = levels[i][j]; // copy the old value
            
            // find the accumulation of water from the neighbors
            for (k = 0; k<6; k++){
                ni = i + neighs[k][0];
                nj = j + neighs[k][1];
                // no water comes from receptive (2,3) cells
                if (mat[ni][nj] < 2) levels_new[i][j] += alpha12 * levels[ni][nj]; 
                    
            }
            
            // if the cell is boundary
            if (mat[i][j] == 2){
                levels_new[i][j] += gamma; // add the water vapor

                // if the cell freezes, set the flag and update the neighbor cells
                if (levels_new[i][j] >= 1){
                    mat_new[i][j] = 3;

                    for (k = 0; k<6; k++){
                        ni = i + neighs[k][0];
                        nj = j + neighs[k][1];
                        mat_new[ni][nj] = max(2,mat_new[ni][nj]);
                    }
                }
            }
            else levels_new[i][j] -= alpha2 * levels[i][j]; // otherwise the water also difuses out of the cell
        }
    }
}

void process_after_exchange(unsigned char **mat,int n, int m, int top_offset, int bottom_offset)
{
    const char (*neighs)[2]; // the pointer to the neighbor offset arrays

    int nj,ni; // the neighbor indices
    int j,k;

    for (j=1; j<m-1; j++){
        neighs = (j%2 == 0) ? neighs_even_col_h : neighs_odd_col_h;

        for (k=0; k<6; k++){

            ni = neighs[k][0];
            nj = j + neighs[k][1];

            if (mat[n-bottom_offset][j] == 3 && ni<=0) // bottom row buffer
                mat[ni + n - bottom_offset][nj] = max(2,mat[ni + n - bottom_offset][nj]);
            if (mat[n-bottom_offset-1][j] == 3) // bottom row host
                mat[ni + n - bottom_offset-1][nj] = max(2,mat[ni + n - bottom_offset-1][nj]);

            if (mat[top_offset-1][j] == 3 && ni>=0) // top row buffer
                mat[ni + top_offset - 1][nj] = max(2,mat[ni + top_offset - 1][nj]);
            if (mat[top_offset][j] == 3) // top row host
                mat[ni + top_offset][nj] = max(2,mat[ni + top_offset][nj]);
        }
    }
}

void process_after_exchange_v2(unsigned char **mat,int n, int m, int top_offset, int bottom_offset)
{
    const char (*neighs)[2]; // the pointer to the neighbor offset arrays

    int nj,ni; // the neighbor indices
    int j,k;

    for (j=1; j<m-1; j++){
        neighs = (j%2 == 0) ? neighs_even_col_h : neighs_odd_col_h;

        for (k=0; k<6; k++){

            ni = neighs[k][0];
            nj = j + neighs[k][1];

            if (mat[n-bottom_offset][j] == 3) // bottom row buffer
                mat[ni + n - bottom_offset][nj] = max(2,mat[ni + n - bottom_offset][nj]);
            
            if (mat[n-bottom_offset-1][j] == 3) // bottom row host
                mat[ni + n - bottom_offset-1][nj] = max(2,mat[ni + n - bottom_offset-1][nj]);

            if (mat[top_offset-1][j] == 3) // top row buffer
                mat[ni + top_offset - 1][nj] = max(2,mat[ni + top_offset - 1][nj]);
            if (mat[top_offset][j] == 3) // top row host
                mat[ni + top_offset][nj] = max(2,mat[ni + top_offset][nj]);
        }
    }
}

void process_after_exchange_old(unsigned char **mat,int n, int m, int top_offset, int bottom_offset)
{
    const char (*neighs)[2]; // the pointer to the neighbor offset arrays

    int nj,ni; // the neighbor indices
    int j,k;

    for (j=1; j<m-1; j++){
        neighs = (j%2 == 0) ? neighs_even_col_h : neighs_odd_col_h;
        for (k=0; k<6; k++){

            ni = neighs[k][0];
            nj = j + neighs[k][1];

            if (mat[n-1][j] == 3 && ni<0) // bottom row
                mat[ni + n - 1][nj] = max(2,mat[ni + n - 1][nj]);
            
            if (mat[n-2][j] == 3) // bottom row
                mat[ni + n - 2][nj] = max(2,mat[ni + n - 2][nj]);

            if (mat[0][j] == 3 && ni>0) // top row
                mat[ni][nj] = max(2,mat[ni][nj]);
            if (mat[1][j] == 3) // top row
                mat[ni + 1][nj] = max(2,mat[ni + 1][nj]);
        }
    }
}
