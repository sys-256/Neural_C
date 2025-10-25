#include <stdio.h>
#include <stdlib.h>

#include "read_idx.h"
#include "libneural_helper.h"
#include "libneural_matrix.h"
#include "libneural_train.h"

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
    relu_data_in = NULL;
    free(relu_data_out);
    relu_data_out = NULL;
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
    transpose_out_buf = NULL;
    // END: test libneural_matrix_transpose
    
    // START: test libneural_matrix_dot_product, libneural_matrix_hadamard_product, libneural_matrix_const_product
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

    double hadamard_in1_buf[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double hadamard_in2_buf[] = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5};
    size_t hadamard_size = 6;
    libneural_matrix_hadamard_product(hadamard_in1_buf, hadamard_in2_buf, hadamard_size);
    printf("libneural_matrix_hadamard_product:\n");
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 2; j++) {
            printf("%1.2f ", *(hadamard_in1_buf+2*i+j));
        }
        printf("\n");
    }
    printf("\n");

    
    libneural_matrix_const_product(dot_out_buf, dot_in1_rows * dot_in2_cols, 0.5);
    printf("libneural_matrix_const_product:\n");
    for (size_t i = 0; i < dot_in2_cols; i++) {
        for (size_t j = 0; j < dot_in2_cols; j++) {
            printf("%1.2f ", *(dot_out_buf+dot_in2_cols*i+j));
        }
        printf("\n");
    }
    printf("\n");
    free(dot_out_buf);
    dot_out_buf = NULL;
    // END: test libneural_matrix_dot_product, libneural_matrix_const_product
    
    // START: test libneural_matrix_minus
    double minus_in1_buf[] = {1.0, 2.6, 3.2, 4.0, 5.0, 6.0};
    double minus_in2_buf[] = {1.0, 3.4, 5.1, 2.1, 4.0, 6.0};
    size_t minus_in_rows = 3, minus_in_cols = 2;

    libneural_matrix_minus(minus_in1_buf, minus_in2_buf, minus_in_rows * minus_in_cols);
    printf("libneural_matrix_minus:\n");
    for (size_t i = 0; i < minus_in_rows; i++) {
        for (size_t j = 0; j < minus_in_cols; j++) {
            printf("%1.2f ", *(minus_in1_buf+minus_in_cols*i+j));
        }
        printf("\n");
    }
    printf("\n");
    // END: test libneural_matrix_minus
    
    // START: test libneural_train_feed_forward, libneural_train_back_propgation
    // Test with 3-5-2 model and random input and weights
    double back_prop_input[] = {0.840, 0.394, 0.783};
    double back_prop_w1[] = {0.197, 0.996, 0.422, 0.208, 0.689, 0.566, 0.392, 0.981, 0.768, 0.748, 0.734, 0.445, 0.759, 0.965, 0.452};
    double back_prop_w2[] = {0.806, 0.800, 0.804, 0.131, 0.141, 0.912, 0.084, 0.624, 0.238, 0.373};
    double back_prop_output[] = {0, 1};
    double *back_prop_z1, *back_prop_a1, *back_prop_a2;
    
    libneural_train_feed_forward(back_prop_input, back_prop_w1, back_prop_w2, 3, 5, 2, &back_prop_z1, &back_prop_a1, &back_prop_a2);
    printf("libneural_train_feed_forward: %.15f %0.15f\n", *(back_prop_a2), *(back_prop_a2+1));
    
    libneural_train_back_propagation(back_prop_input, back_prop_w1, back_prop_w2, back_prop_output, 3, 5, 2, 0.5);
    printf("updated w1: ");
    for (size_t i = 0; i < 3*5; i++) {
        printf("%1.3f ", *(back_prop_w1+i));
    }
    printf("\n");
    printf("updated w2: ");
    for (size_t i = 0; i < 5*2; i++) {
        printf("%1.3f ", *(back_prop_w2+i));
    }
    printf("\n");
    printf("\n");
    
    free(back_prop_z1);
    free(back_prop_a1);
    free(back_prop_a2);
    back_prop_z1 = NULL;
    back_prop_a1 = NULL;
    back_prop_a2 = NULL;
    // END: test libneural_train_feed_forward, libneural_train_back_propgation

    // START: test read_idx
    uint8_t *idx_data;
    size_t idx_data_size;
    
    read_idx("data/train-labels-idx1-ubyte", &idx_data, &idx_data_size);
    printf("We have %ld labels, only print 5.\n", idx_data_size);
    for (int i = 0; i < 5; i++) {
        printf("%d\n", *(idx_data+i));
    }
    free(idx_data);
    idx_data = NULL;
    printf("\n");

    read_idx("data/train-images-idx3-ubyte", &idx_data, &idx_data_size);
    printf("We have %ld images, only print 1.\n", idx_data_size/784); // Each image is 28*28 pixels (so 784 bytes)
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            printf("%d\t", *(idx_data+2*784+28*i+j));
        }
        printf("\n");
    }
    printf("\n");
    free(idx_data);
    idx_data = NULL;
    // END: test read_idx
    
    // START: test libneural_train_train
    double *w1, *w2;
    size_t input_size = 784, hidden_size = 20, output_size = 10;
    double alpha = 0.01;
    size_t iterations = 15;
    libneural_helper_generate_weights(input_size, hidden_size, &w1);
    libneural_helper_generate_weights(hidden_size, output_size, &w2);

    uint8_t *train_images, *train_labels;
    size_t train_images_size, train_labels_size;

    read_idx("data/train-images-idx3-ubyte", &train_images, &train_images_size);
    read_idx("data/train-labels-idx1-ubyte", &train_labels, &train_labels_size);

    // Convert array of uint8_t to array of doubles
    double *train_images_converted = malloc(train_images_size * sizeof(double));
    for (size_t i = 0; i < train_images_size; i++) {
        *(train_images_converted+i) = (double) *(train_images+i);
    }
    libneural_matrix_const_product(train_images_converted, train_images_size, 1.0/255.0); // Divide by max value of pixel

    libneural_train_train(train_images_converted, w1, w2, train_labels, input_size, hidden_size, output_size, train_labels_size, alpha, iterations);
    free(w1);
    w1 = NULL;
    free(w2);
    w2 = NULL;
    free(train_images);
    train_images = NULL;
    free(train_labels);
    train_labels = NULL;
    free(train_images_converted);
    train_images_converted = NULL;
    // END: test libneural_train_train

    return 0;
}