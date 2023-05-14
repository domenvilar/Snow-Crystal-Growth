#include "board.h"

void board_step_cols(float **levels,float **levels_new,
                unsigned char **mat, unsigned char **mat_new,
                int n, int m,
                float alpha, float gamma, int left_buffer, int right_buffer, int base_offset)
{
    // precompute coefficients
    float alpha12 = alpha/12.;
    float alpha2 = alpha/2.;
    const char (*neighs)[2]; // the pointer to the neighbor offset arrays

    int ni,nj; // the neighbor indices
    int i,j,k;

    // copy the matrix since I can't find a better way
    // for (i = left_buffer; i<n-right_buffer; i++){
    //     for (j = 1; j<m-1; j++){
    //         mat_new[i][j] = mat[i][j];
    //     }
    // }
    
    for (i = 1; i<n-1; i++){
        for (j = left_buffer; j<m-right_buffer; j++){

            if (mat[i][j] == 3) continue; // if the cell is frozen, skip this step

            neighs = ((j+base_offset)%2 == 0) ? neighs_even_col : neighs_odd_col; // determine the neighbor pointer
            
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


// TODO TEST, maybe also thoroughly verify the exchanges (by lines also, not only cols)
void process_after_exchange(unsigned char **mat,int n, int m,
                            int left_offset, int right_offset,int base_offset)
{   // pointers to neighbor arrays for the four columns we are working with
    const char (*neighsLh)[2] = ((left_offset + base_offset)%2 == 0) ? neighs_even_col : neighs_odd_col; // the pointer to the neighbor offset arrays
    const char (*neighsRh)[2] = ((m-right_offset + base_offset)%2 == 0) ? neighs_even_col : neighs_odd_col;
    const char (*neighsLb)[2] = ((left_offset - 1 + base_offset)%2 == 0) ? neighs_even_col : neighs_odd_col; // the pointer to the neighbor offset arrays
    const char (*neighsRb)[2] = ((m-right_offset-1 + base_offset)%2 == 0) ? neighs_even_col : neighs_odd_col;

    int njLh,njRh,njLb,njRb,niLh,niRh,niLb,niRb; // the neighbor indices
    int i,k;

    for (i=1; i<n-1; i++){

        for (k=0; k<6; k++){

            niLh = i + neighsLh[k][0];
            niRb = i + neighsRh[k][0];
            niLb = i + neighsLb[k][0];
            niRh = i + neighsRb[k][0];

            njLh = neighsLh[k][1];
            njRh = neighsRh[k][1];
            njLb = neighsLb[k][1];
            njRb = neighsRb[k][1];

            if (mat[i][left_offset] == 3 && njLh<0) // left column host
                mat[niLh][left_offset + njLh] = max(2,mat[niLh][njLh + left_offset]);

            if (mat[i][left_offset-1] == 3 && njLb>0) // left column buffer
                mat[niLb][left_offset - 1 + njLb] = max(2,mat[niLb][left_offset - 1 + njLb]);

            if (mat[i][m - right_offset - 1] == 3 && njRh>0) // right column host
                mat[niRh][m - right_offset - 1 + njRh] = max(2,mat[niRh][njRh + m - right_offset - 1]);

            if (mat[i][m - right_offset] == 3 && njRb<0) // right column buffer
                mat[niRb][m - right_offset + njRb] = max(2,mat[niRb][m - right_offset + njRb]);

        }
    }
}