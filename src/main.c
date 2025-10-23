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
    
    double in_buf[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    size_t in_rows = 3, in_cols = 2;
    double *out_buf;
    size_t out_rows, out_cols;

    libmatrix_transpose(in_buf, in_rows, in_cols, &out_buf, &out_rows, &out_cols);
    
    for (size_t i = 0; i < in_rows; i++) {
        for (size_t j = 0; j < in_cols; j++) {
            printf("%1.2f ", *(in_buf+in_cols*i+j));
        }
        printf("\n");
    }
    for (size_t i = 0; i < out_rows; i++) {
        for (size_t j = 0; j < out_cols; j++) {
            printf("%1.2f ", *(out_buf+out_cols*i+j));
        }
        printf("\n");
    }
    printf("\n");
    free(out_buf);
    
    double dot_in1_buf[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    size_t dot_in1_rows = 3, dot_in1_cols = 2;
    double dot_in2_buf[] = {1.0, 3.0, 5.0, 2.0, 4.0, 6.0};
    size_t dot_in2_rows = 2, dot_in2_cols = 3;
    double *dot_out_buf;
    size_t dot_out_rows, dot_out_cols;

    libmatrix_dot_product(dot_in1_buf, dot_in1_rows, dot_in1_cols, dot_in2_buf, dot_in2_rows, dot_in2_cols, &dot_out_buf, &dot_out_rows, &dot_out_cols);
    for (size_t i = 0; i < dot_out_rows; i++) {
        for (size_t j = 0; j < dot_out_cols; j++) {
            printf("%1.2f ", *(dot_out_buf+dot_out_cols*i+j));
        }
        printf("\n");
    }
    printf("\n");

    double *const_out_buf;
    size_t const_out_size;
    libmatrix_const_product(dot_out_buf, dot_out_cols * dot_out_rows, 0.5, &const_out_buf, &const_out_size);
    for (size_t i = 0; i < dot_out_rows; i++) {
        for (size_t j = 0; j < dot_out_cols; j++) {
            printf("%1.2f ", *(const_out_buf+dot_out_cols*i+j));
        }
        printf("\n");
    }
    printf("\n");
    free(const_out_buf);
    free(dot_out_buf);
    
    double add_in1_buf[] = {1.0, 2.6, 3.2, 4.0, 5.0, 6.0};
    size_t add_in1_rows = 3, add_in1_cols = 2;
    double add_in2_buf[] = {1.0, 3.4, 5.1, 2.1, 4.0, 6.0};
    size_t add_in2_rows = 3, add_in2_cols = 2;
    double *add_out_buf;
    size_t add_out_size;

    libmatrix_minus(add_in1_buf, add_in1_rows * add_in1_cols, add_in2_buf, add_in2_rows * add_in2_cols, &add_out_buf, &add_out_size);
    for (size_t i = 0; i < add_in1_rows; i++) {
        for (size_t j = 0; j < add_in1_cols; j++) {
            printf("%1.2f ", *(add_out_buf+add_in1_cols*i+j));
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");
    free(add_out_buf);

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