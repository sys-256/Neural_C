#include "libmatrix.h"
#include <stdlib.h>
#include <math.h>

uint8_t libmatrix_generate_weights(uint64_t rows, uint64_t columns, double **output) {
    // The starting weights may be the same each time, so don't seed
    
    *output = malloc(rows * columns * sizeof(double)); // Allocate memory, output is of type double
    if (!*output) return LIBMATRIX_ERR_OOM; // Out of memory

    // Initialize every value in the matrix with a random number in range [0, 1]
    for (size_t i = 0; i < rows * columns; i++) {
        *(*output+i) = (double) rand() / RAND_MAX;
    }
    return LIBMATRIX_OK;
}

uint8_t libmatrix_relu(double *input, size_t input_size, double **output) {
    *output = malloc(input_size * sizeof(double)); // Allocate memory, output is of type double
    if (!*output) return LIBMATRIX_ERR_OOM; // Out of memory

    // Copy max(0, input element) into every element of the output
    for (size_t i = 0; i < input_size; i++) {
        double value = *(input+i);
        if (value > 0) {
            *(*output+i) = value;
        } else {
            *(*output+i) = 0;
        }
    }
    return LIBMATRIX_OK;
}

uint8_t libmatrix_relu_prime(double *input, size_t size) {
    // Copy element > 0 into every element of the matrix
    for (size_t i = 0; i < size; i++) {
        double value = *(input+i);
        if (value > 0) {
            *(input+i) = 1;
        } else {
            *(input+i) = 0;
        }
    }
    return LIBMATRIX_OK;
}

uint8_t libmatrix_softmax(double *input, size_t size) {
    if (size < 1) { // Flattened matrix needs at least one element
        return LIBMATRIX_ERR_INPUT;
    }

    // Find max
    double max = *(input); // Initialize with first element of matrix
    for (size_t i = 1; i < size; i++) { // We know the first element isn't larger than max, so start at i=1
        if (*(input+i) > max) {
            max = *(input+i);
        }
    }

    // Subtract max from every element, then apply exp on every element
    for (size_t i = 0; i < size; i++) {
        *(input+i) = exp(*(input+i) - max);
    }

    // Get sum of np.exp(z)
    double sum = *(input);
    for (size_t i = 1; i < size; i++) { // We initialized with first element, so start at i=1
        sum += *(input+i);
    }

    // Divide every element by sum
    for (size_t i = 0; i < size; i++) {
        *(input+i) /= sum;
    }
    return LIBMATRIX_OK;
}

uint8_t libmatrix_predict(double *input, size_t size, size_t *result) {
    if (size < 1) { // Flattened matrix needs at least one element
        return LIBMATRIX_ERR_INPUT;
    }

    *result = 0; // Initialize with first element of matrix
    for (size_t i = 1; i < size; i++) { // We know the first element isn't larger than max, so start at i=1
        if (*(input+i) > *(input+*result)) {
            *result = i;
        }
    }
    return LIBMATRIX_OK;
}

uint8_t libmatrix_transpose(double *input, size_t input_rows, size_t input_cols, double **output) {
    *output = malloc(input_cols * input_rows * sizeof(double)); // Allocate memory, output is of type double
    if (!*output) return LIBMATRIX_ERR_OOM; // Out of memory

    for (size_t i = 0; i < input_cols; i++) {
        for (size_t j = 0; j < input_rows; j++) {
            *(*output+input_rows*i+j) = *(input+input_cols*j+i); // Set [A.T]_ij to [A]_ji
        }
    }
    return LIBMATRIX_OK;
}

uint8_t libmatrix_dot_product(double *input1, size_t input1_rows, size_t input1_cols, double *input2, size_t input2_cols, double **output) {
    *output = malloc(input1_rows * input2_cols * sizeof(double)); // Allocate memory, output is of type double
    if (!*output) return LIBMATRIX_ERR_OOM; // Out of memory

    for (size_t i = 0; i < input1_rows; i++) {
        for (size_t j = 0; j < input2_cols; j++) {
            *(*output+input2_cols*i+j) = 0.0; // Set C_ij to 0
            for (size_t k = 0; k < input1_cols; k++) { // k goes from 0 to n=input1_cols=input2_rows
                *(*output+input2_cols*i+j) += *(input1+input1_cols*i+k) * *(input2+input2_cols*k+j); // C_ij += A_ik*B_kj
            }
        }
    }
    return LIBMATRIX_OK;
}

uint8_t libmatrix_const_product(double *input, size_t input_size, double constant, double **out_buf) {
    *out_buf = malloc(input_size * sizeof(double)); // Allocate memory, out_buf is of type double
    if (!*out_buf) return LIBMATRIX_ERR_OOM; // Out of memory

    // Copy input element * constant into every element of the output
    for (size_t i = 0; i < input_size; i++) {
        *(*out_buf+i) = *(input+i) * constant;
    }
    return LIBMATRIX_OK;
}

uint8_t libmatrix_minus(double *input1, double *input2, size_t input_size, double **output) {
    *output = malloc(input_size * sizeof(double)); // Allocate memory, out_buf is of type double
    if (!*output) return LIBMATRIX_ERR_OOM; // Out of memory

    // Copy input1 element + input2 element into every element of the output
    for (size_t i = 0; i < input_size; i++) {
        *(*output+i) = *(input1+i) - *(input2+i);
    }
    return LIBMATRIX_OK;
}
