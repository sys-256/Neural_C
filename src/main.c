#include <stdio.h>
#include <stdlib.h>

#include "read_idx.h"
#include "libneural_helper.h"
#include "libneural_matrix.h"

int main() {
    // START: test libneural_helper_generate_weights, libneural_helper_relu
    double *relu_data_in, *relu_data_out;

    libneural_helper_generate_weights(3, 5, &relu_data_in);
    printf("libneural_helper_generate_weights: ");
    for (size_t i = 0; i < 3*5; i++) {
        printf("%1.3f ", *(relu_data_in+i));
    }
    printf("\n");

    libneural_helper_relu(relu_data_in, 3*5, &relu_data_out);
    printf("libneural_helper_relu: ");
    for (size_t i = 0; i < 3*5; i++) {
        printf("%1.3f ", *(relu_data_out+i));
    }
    printf("\n");
    printf("\n");
    free(relu_data_in);
    free(relu_data_out);
    // END: test libneural_helper_generate_weights, libneural_helper_relu

    // START: test libneural_helper_relu_prime
    double relu_prime_data[] = {-0.33, 0.33, 0, -28394, 281022};
    libneural_helper_relu_prime(relu_prime_data, 5); // 5 elements in array
    printf("libneural_helper_relu_prime: ");
    for (size_t i = 0; i < 5; i++) {
        printf("%1.3f ", *(relu_prime_data+i));
    }
    printf("\n");
    printf("\n");
    // END: test libneural_helper_relu_prime

    // START: test libneural_helper_softmax
    double softmax_data[] = {557.32599628, 656.53432109, 737.34115319, 741.98236534, 537.17001307,
                    706.15147083, 420.91040211, 622.37374273, 122.03657538, 706.17970965};
    libneural_helper_softmax(softmax_data, 10);
    printf("libneural_helper_softmax:\n");
    for (size_t i = 0; i < 10; i++) {
        printf("%1.3f\n", *(softmax_data+i));
    }
    printf("\n");
    // END: test libneural_helper_softmax

    // START: test libneural_helper_predict
    double predict_data[] = {557.32599628, 656.53432109, 741.98236534, 737.34115319};
    size_t predict_result;
    libneural_helper_predict(predict_data, 4, &predict_result); // 4 elements in result
    printf("libneural_helper_predict: result=%ld, element=%1.3f\n", predict_result, *(predict_data+predict_result));
    printf("\n");
    // END: test libneural_helper_predict
    
    // START: test libneural_matrix_transpose
    double transpose_in_buf[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    size_t transpose_in_rows = 3,  transpose_in_cols = 2;
    double *transpose_out_buf;
    libneural_matrix_transpose(transpose_in_buf, transpose_in_rows, transpose_in_cols, &transpose_out_buf);
    printf("libneural_matrix_transpose original:\n");
    for (size_t i = 0; i < transpose_in_rows; i++) {
        for (size_t j = 0; j < transpose_in_cols; j++) {
            printf("%1.2f ", *(transpose_in_buf+transpose_in_cols*i+j));
        }
        printf("\n");
    }
    printf("libneural_matrix_transpose transposed:\n");
    for (size_t i = 0; i < transpose_in_cols; i++) {
        for (size_t j = 0; j < transpose_in_rows; j++) {
            printf("%1.2f ", *(transpose_out_buf+transpose_in_rows*i+j));
        }
        printf("\n");
    }
    printf("\n");
    free(transpose_out_buf);
    // END: test libneural_matrix_transpose
    
    // START: test libneural_matrix_dot_product, libneural_matrix_const_product
    double dot_in1_buf[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    size_t dot_in1_rows = 3, dot_in1_cols = 2;
    double dot_in2_buf[] = {1.0, 3.0, 5.0, 2.0, 4.0, 6.0};
    size_t dot_in2_cols = 3;
    double *dot_out_buf;

    libneural_matrix_dot_product(dot_in1_buf, dot_in1_rows, dot_in1_cols, dot_in2_buf, dot_in2_cols, &dot_out_buf);
    printf("libneural_matrix_dot_product:\n");
    for (size_t i = 0; i < dot_in1_rows; i++) {
        for (size_t j = 0; j < dot_in2_cols; j++) {
            printf("%1.2f ", *(dot_out_buf+dot_in2_cols*i+j));
        }
        printf("\n");
    }
    printf("\n");
    
    double *const_out_buf;
    libneural_matrix_const_product(dot_out_buf, dot_in1_rows * dot_in2_cols, 0.5, &const_out_buf);
    printf("libneural_matrix_const_product:\n");
    for (size_t i = 0; i < dot_in2_cols; i++) {
        for (size_t j = 0; j < dot_in2_cols; j++) {
            printf("%1.2f ", *(const_out_buf+dot_in2_cols*i+j));
        }
        printf("\n");
    }
    printf("\n");
    free(const_out_buf);
    free(dot_out_buf);
    // END: test libneural_matrix_dot_product, libneural_matrix_const_product
    
    // START: test libneural_matrix_minus
    double minus_in1_buf[] = {1.0, 2.6, 3.2, 4.0, 5.0, 6.0};
    double minus_in2_buf[] = {1.0, 3.4, 5.1, 2.1, 4.0, 6.0};
    size_t minus_in_rows = 3, minus_in_cols = 2;
    double *minus_out_buf;

    libneural_matrix_minus(minus_in1_buf, minus_in2_buf, minus_in_rows * minus_in_cols, &minus_out_buf);
    printf("libneural_matrix_minus:\n");
    for (size_t i = 0; i < minus_in_rows; i++) {
        for (size_t j = 0; j < minus_in_cols; j++) {
            printf("%1.2f ", *(minus_out_buf+minus_in_cols*i+j));
        }
        printf("\n");
    }
    printf("\n");
    free(minus_out_buf);
    // END: test libneural_matrix_minus

    // START: test read_idx
    uint8_t *idx_data;
    size_t idx_data_size;
    
    read_idx("data/train-labels-idx1-ubyte", &idx_data, &idx_data_size);
    printf("We have %ld labels, only print 5.\n", idx_data_size);
    for (int i = 0; i < 5; i++) {
        printf("%d\n", *(idx_data+i));
    }
    free(idx_data);
    printf("\n");

    read_idx("data/train-images-idx3-ubyte", &idx_data, &idx_data_size);
    printf("We have %ld images, only print 1.\n", idx_data_size/784); // Each image is 28*28 pixels (so 784 bytes)
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            printf("%d\t", *(idx_data+2*784+28*i+j));
        }
        printf("\n");
    }
    free(idx_data);
    // END: test read_idx

    return 0;
}