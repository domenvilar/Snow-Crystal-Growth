#include <stdio.h>
#include <stdlib.h>

const char neighs_even_col[6][2] = { { -1, 0 }, { -1, 1 }, { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, -1 } };
const char neighs_odd_col[6][2] = { { -1, 0 }, { 0, 1 }, { 1, 1 }, { 1, 0 }, { 1, -1 }, { 0, -1 } };

/*my own definition of max since stdlib of this version of C doesn't provide it*/
unsigned char max(unsigned char a, unsigned char b)
{
    return (a > b) ? a : b;
}

unsigned char** board_initialize_simple(unsigned int n, unsigned int m)
{
    int k, l;

    unsigned char* bd = (unsigned char*)malloc(sizeof(unsigned char) * n * m);
    unsigned char** b = (unsigned char**)malloc(sizeof(unsigned char*) * n);

    for (k = 0; k < n; k++)
        b[k] = &bd[k * m]; // assign pointers to values at the start of the row

    return b;
}

float** levels_initialize_simple(unsigned int n, unsigned int m)
{
    int k, l;

    float* bd = (float*)malloc(sizeof(float) * n * m);
    float** b = (float**)malloc(sizeof(float*) * n);

    for (k = 0; k < n; k++)
        b[k] = &bd[k * m]; // assign pointers to values at the start of the row

    return b;
}

/*
initializes the board with n rows and m columns
*/
unsigned char** board_initialize(unsigned int n, unsigned int m)
{
    int k, l;

    unsigned char* bd = (unsigned char*)malloc(sizeof(unsigned char) * n * m);
    unsigned char** b = (unsigned char**)malloc(sizeof(unsigned char*) * n);

    for (k = 0; k < n; k++)
        b[k] = &bd[k * m]; // assign pointers to values at the start of the row

    // set all edge columns to 0
    for (k = 0; k < n; k++) {
        b[k][0] = 0; // left column
        b[k][m - 1] = 0; // right column
    }

    // set the top and bottom row to 0
    for (k = 0; k < m; k++) {
        b[0][k] = 0; // left column
        b[n - 1][k] = 0; // right column
    }

    // set all but the border values to 1
    for (k = 1; k < n - 1; k++) {
        for (l = 1; l < m - 1; l++) {
            b[k][l] = 1;
        }
    }

    // set the center cell to be frozen and the neighboring boundary
    int center_i, center_j;
    center_i = n / 2;
    center_j = m / 2;
    const char(*neighs)[2] = (center_j % 2 == 0) ? neighs_even_col : neighs_odd_col;

    for (int i = 0; i < 6; i++) {
        b[center_i + neighs[i][0]][center_j + neighs[i][1]] = 2;
    }

    b[center_i][center_j] = 3;

    return b;
}

/*initializes the water levels array of all cells be equal to beta*/
float** levels_initialize(unsigned int n, unsigned int m, float beta)
{
    int k, l;
    beta = 0.9;

    float* bd = (float*)malloc(sizeof(float) * n * m);
    float** b = (float**)malloc(sizeof(float*) * n);

    for (k = 0; k < n; k++)
        b[k] = &bd[k * m]; // assign pointers to values at the start of the row

    // set all values to beta
    for (k = 0; k < n; k++) {
        for (l = 0; l < m; l++) {
            b[k][l] = beta;
        }
    }

    int center_i, center_j;
    center_i = n / 2;
    center_j = m / 2;
    b[center_i][center_j] = 1.;

    return b;
}

/*frees the memory*/
void board_free(unsigned char** b)
{
    free(*b);
    free(b);
}

/*frees the memory*/
void levels_free(float** b)
{
    free(*b);
    free(b);
}
/*switches the board the pointers point to*/
void board_update(unsigned char*** b, unsigned char*** bn)
{
    unsigned char** bt;

    bt = *b;
    *b = *bn;
    *bn = bt;
}

/*switches the levels the pointers point to*/
void levels_update(float*** b, float*** bn)
{
    float** bt;

    bt = *b;
    *b = *bn;
    *bn = bt;
}

void copy_mat(unsigned char** source, unsigned char** dest, int n, int m)
{
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            dest[i][j] = source[i][j];
        }
    }
}

void copy_levels(float** source, float** dest, int n, int m)
{
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            dest[i][j] = source[i][j];
        }
    }
}

void board_step(float** levels, float** levels_new,
    unsigned char** mat, unsigned char** mat_new,
    int n, int m,
    float alpha, float gamma)
{
    // precompute coefficients
    float alpha12 = alpha / 12.;
    float alpha2 = alpha / 2.;
    const char(*neighs)[2]; // the pointer to the neighbor offset arrays

    int ni, nj; // the neighbor indices
    int i, j, k;

    // copy the matrix since I can't find a better way
    // for (i = 1; i<n-1; i++){
    //     for (j = 1; j<m-1; j++){
    //         mat_new[i][j] = mat[i][j];
    //     }
    // }

    // i row, j column
    for (i = 1; i < n - 1; i++) {
        for (j = 1; j < m - 1; j++) {

            if (mat[i][j] == 3)
                continue; // if the cell is frozen, skip this step

            neighs = (j % 2 == 0) ? neighs_even_col : neighs_odd_col; // determine the neighbor pointer
            // levels_new[i][j] = levels[i][j];
            //  find the accumulation of water from the neighbors
            for (k = 0; k < 6; k++) {
                ni = i + neighs[k][0];
                nj = j + neighs[k][1];
                // no water comes from receptive (2,3) cells
                if (mat[ni][nj] < 2)
                    levels_new[i][j] += alpha12 * levels[ni][nj];
            }

            // if the cell is boundary
            if (mat[i][j] == 2) {
                levels_new[i][j] += gamma; // add the water vapor

                // if the cell freezes, set the flag and update the neighbor cells
                if (levels_new[i][j] >= 1) {
                    mat_new[i][j] = 3;

                    for (k = 0; k < 6; k++) {
                        ni = i + neighs[k][0];
                        nj = j + neighs[k][1];
                        mat_new[ni][nj] = max(2, mat_new[ni][nj]);
                    }
                }
            } else
                levels_new[i][j] -= alpha2 * levels[i][j]; // otherwise the water also difuses out of the cell
        }
    }
}

void board_print(unsigned char** b, int n, int m)
{
    int k, l;

    //	system("@cls||clear");
    for (k = 0; k < n; k++) {
        for (l = 0; l < m; l++)
            printf(" %d", b[k][l]);
        printf("\n");
    }
    printf("\n");
}

void levels_print(float** b, int n, int m)
{
    int k, l;

    //	system("@cls||clear");
    for (k = 0; k < n; k++) {
        for (l = 0; l < m; l++)
            printf(" %.1f", b[k][l]);
        printf("\n");
    }
    printf("\n");
}
