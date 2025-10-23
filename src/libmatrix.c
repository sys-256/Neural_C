#include "libmatrix.h"
#include <stdlib.h>
#include <math.h>


int libmatrix_generate_weights(uint64_t rows, uint64_t columns, double **out_buf, size_t *out_size) {
    // The starting weights may be the same each time, so don't seed
    
    *out_size = rows * columns;
    *out_buf = malloc(*out_size * sizeof(double)); // Allocate memory, out_buf is of type double
    if (!*out_buf) return LIBMATRIX_ERR_OOM; // Out of memory

    // Initialize every value in the matrix with a random number in range [0, 1)
    for (size_t i = 0; i < *out_size; i++) {
        *(*out_buf+i) = (double) rand() / RAND_MAX;
    }
    return LIBMATRIX_OK;
}

int libmatrix_relu(double *in_buf, size_t in_size, double **out_buf, size_t *out_size) {
    *out_size = in_size;
    *out_buf = malloc(*out_size * sizeof(double)); // Allocate memory, out_buf is of type double
    if (!*out_buf) return LIBMATRIX_ERR_OOM; // Out of memory

    // Copy max(0, input element) into every element of the output
    for (size_t i = 0; i < *out_size; i++) {
        double value = *(in_buf+i);
        if (value > 0) {
            *(*out_buf+i) = value;
        } else {
            *(*out_buf+i) = 0;
        }
    }
    return LIBMATRIX_OK;
}

int libmatrix_relu_prime(double *buf, size_t size) {
    // Copy element > 0 into every element of the matrix
    for (size_t i = 0; i < size; i++) {
        double value = *(buf+i);
        if (value > 0) {
            *(buf+i) = 1;
        } else {
            *(buf+i) = 0;
        }
    }
    return LIBMATRIX_OK;
}

int libmatrix_softmax(double *buf, size_t size) {
    if (size < 1) { // Flattened matrix needs at least one element
        return LIBMATRIX_ERR_INPUT;
    }

    // Find max
    double max = *(buf); // Initialize with first element of matrix
    for (size_t i = 1; i < size; i++) { // We know the first element isn't larger than max, so start at i=1
        if (*(buf+i) > max) {
            max = *(buf+i);
        }
    }

    // Subtract max from every element, then apply exp on every element
    for (size_t i = 0; i < size; i++) {
        *(buf+i) = exp(*(buf+i) - max);
    }

    // Get sum of np.exp(z)
    double sum = *(buf);
    for (size_t i = 1; i < size; i++) { // We initialized with first element, so start at i=1
        sum += *(buf+i);
    }

    // Divide every element by sum
    for (size_t i = 0; i < size; i++) {
        *(buf+i) /= sum;
    }
    return LIBMATRIX_OK;
}

int libmatrix_predict(double *buf, size_t size, size_t *result) {
    if (size < 1) { // Flattened matrix needs at least one element
        return LIBMATRIX_ERR_INPUT;
    }

    *result = 0; // Initialize with first element of matrix
    for (size_t i = 1; i < size; i++) { // We know the first element isn't larger than max, so start at i=1
        if (*(buf+i) > *(buf+*result)) {
            *result = i;
        }
    }
    return LIBMATRIX_OK;
}

int libmatrix_transpose(double *in_buf, size_t in_rows, size_t in_cols, double **out_buf, size_t *out_rows, size_t *out_cols) {
    *out_rows = in_cols;
    *out_cols = in_rows;
    *out_buf = malloc(*out_rows * *out_cols * sizeof(double)); // Allocate memory, out_buf is of type double
    if (!*out_buf) return LIBMATRIX_ERR_OOM; // Out of memory

    for (size_t i = 0; i < *out_rows; i++) {
        for (size_t j = 0; j < *out_cols; j++) {
            *(*out_buf+*out_cols*i+j) = *(in_buf+in_cols*j+i); // Set [A.T]_ij to [A]_ji
        }
    }
    return LIBMATRIX_OK;
}
