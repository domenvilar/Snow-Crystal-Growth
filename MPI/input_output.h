#include <stdio.h>

/*
buffer is the pointer to the start of the sequence
filename is the name of the file to be saved
size_x and size_y are x and y dimensions of the board
*/
int save_array(unsigned char *buffer,const char *filename,int size_x, int size_y)
{
    FILE *fp = fopen(filename,"wb+");
    size_t num_elements;
    if (fp != NULL){
        num_elements = fwrite(buffer,sizeof(unsigned char),size_x*size_y,fp);
        fclose(fp);
    }

    if (num_elements != size_x*size_y){
        printf("Unsuccessful data storage of file %s",filename);
        return 1;
    }
    else{
        return 0;
    }
}

int save_levels(float *buffer,const char *filename,int size_x, int size_y)
{
    FILE *fp = fopen(filename,"wb+");
    size_t num_elements;
    if (fp != NULL){
        num_elements = fwrite(buffer,sizeof(float),size_x*size_y,fp);
        fclose(fp);
    }

    if (num_elements != size_x*size_y){
        printf("Unsuccessful data storage of file %s",filename);
        return 1;
    }
    else{
        return 0;
    }
}