#include <stdlib.h>

#include "libneural_matrix.h"

uint8_t libneural_matrix_transpose(double *input, size_t input_rows, size_t input_cols, double **output) {
    *output = malloc(input_cols * input_rows * sizeof(double)); // Allocate memory, output is of type double
    if (!*output) return LIBNEURAL_MATRIX_ERR_OOM; // Out of memory

    for (size_t i = 0; i < input_cols; i++) {
        for (size_t j = 0; j < input_rows; j++) {
            *(*output+input_rows*i+j) = *(input+input_cols*j+i); // Set [A.T]_ij to [A]_ji
        }
    }
    return LIBNEURAL_MATRIX_OK;
}

uint8_t libneural_matrix_dot_product(double *input1, size_t input1_rows, size_t input1_cols, double *input2, size_t input2_cols, double **output) {
    *output = malloc(input1_rows * input2_cols * sizeof(double)); // Allocate memory, output is of type double
    if (!*output) return LIBNEURAL_MATRIX_ERR_OOM; // Out of memory

    for (size_t i = 0; i < input1_rows; i++) {
        for (size_t j = 0; j < input2_cols; j++) {
            *(*output+input2_cols*i+j) = 0.0; // Set C_ij to 0
            for (size_t k = 0; k < input1_cols; k++) { // k goes from 0 to n=input1_cols=input2_rows
                *(*output+input2_cols*i+j) += *(input1+input1_cols*i+k) * *(input2+input2_cols*k+j); // C_ij += A_ik*B_kj
            }
        }
    }
    return LIBNEURAL_MATRIX_OK;
}

uint8_t libneural_matrix_hadamard_product(double *input1, double *input2, size_t input_size) {
    // Copy element1 * element2 into every element
    for (size_t i = 0; i < input_size; i++) {
        *(input1+i) *= *(input2+i);
    }
    return LIBNEURAL_MATRIX_OK;
}

uint8_t libneural_matrix_const_product(double *input, size_t input_size, double constant) {
    // Copy element * constant into every element
    for (size_t i = 0; i < input_size; i++) {
        *(input+i) *= constant;
    }
    return LIBNEURAL_MATRIX_OK;
}

uint8_t libneural_matrix_minus(double *input1, double *input2, size_t input_size, double **output) {
    *output = malloc(input_size * sizeof(double)); // Allocate memory, output is of type double
    if (!*output) return LIBNEURAL_MATRIX_ERR_OOM; // Out of memory

    // Copy input1 element + input2 element into every element of the output
    for (size_t i = 0; i < input_size; i++) {
        *(*output+i) = *(input1+i) - *(input2+i);
    }
    return LIBNEURAL_MATRIX_OK;
}
