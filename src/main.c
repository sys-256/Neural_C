#include "read_idx.h"
#include "libmatrix.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    double *data_in, *data_out;
    size_t data_in_size, data_out_size;

    libmatrix_generate_weights(3, 5, &data_in, &data_in_size);
    libmatrix_relu(data_in, data_in_size, &data_out, &data_out_size);
    
    for (size_t i = 0; i < data_out_size; i++) {
        printf("%1.3f ", *(data_out+i));
    }
    printf("\n");
    free(data_in);
    free(data_out);

    double data_relu[] = {-0.33, 0.33, 0, -28394, 281022};
    libmatrix_relu_prime(data_relu, 5);
    for (size_t i = 0; i < 5; i++) {
        printf("%1.3f ", *(data_relu+i));
    }

    double data_softmax[] = {557.32599628, 656.53432109, 737.34115319, 741.98236534, 537.17001307,
                    706.15147083, 420.91040211, 622.37374273, 122.03657538, 706.17970965};
    libmatrix_softmax(data_softmax, 10);
    for (size_t i = 0; i < 10; i++) {
        printf("%1.30f\n", *(data_softmax+i));
    }

    double data_predict[] = {557.32599628, 656.53432109, 741.98236534, 737.34115319};
    size_t result_predict;
    libmatrix_predict(data_predict, 4, &result_predict);
    printf("%ld: %1.30f\n", result_predict, *(data_predict+result_predict));

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