#include <stdio.h>
#include <stdlib.h>

const char neighs_even_col_h[6][2] = { { -1, 0 }, { -1, 1 }, { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, -1 } };
const char neighs_odd_col_h[6][2] = { { -1, 0 }, { 0, 1 }, { 1, 1 }, { 1, 0 }, { 1, -1 }, { 0, -1 } };
/*
initializes the board with n rows and m columns
*/
unsigned int** board_initialize(unsigned int n, unsigned int m)
{
    int k, l;

    unsigned int* bd = (unsigned int*)malloc(sizeof(unsigned int) * n * m);
    unsigned int** b = (unsigned int**)malloc(sizeof(unsigned int*) * n);

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
    const char(*neighs)[2] = (center_j % 2 == 0) ? neighs_even_col_h : neighs_odd_col_h;

    for (int i = 0; i < 6; i++) {
        b[center_i + neighs[i][0]][center_j + neighs[i][1]] = 2;
    }

    b[center_i][center_j] = 3;
    // printf("center_i: %d, center_j: %d\n", center_i, center_j);

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
