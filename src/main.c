#include "read_idx.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    uint8_t *data;
    size_t data_size;
    
    read_idx("data/train-labels-idx1-ubyte", &data, &data_size);
    printf("We have %ld labels, only print 5.\n", data_size);
    for (int i = 0; i < 5; i++) {
        printf("%d\n", *(data+i));
    }
    free(data);

    read_idx("data/train-images-idx3-ubyte", &data, &data_size);
    printf("We have %ld images, only print 1.\n", data_size/784); // Each image is 28*28 pixels (so 784 bytes)
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            printf("%d\t", *(data+2*784+28*i+j));
        }
        printf("\n");
    }
    free(data);

    return 0;
}