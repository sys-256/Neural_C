#include <stdlib.h>
#include <math.h>

#include "libneural_helper.h"

uint8_t libneural_helper_generate_weights(size_t rows, size_t columns, double **output) {
    // The starting weights may be the same each time, so don't seed
    
    *output = malloc(rows * columns * sizeof(double)); // Allocate memory, output is of type double
    if (!*output) return LIBNEURAL_HELPER_ERR_OOM; // Out of memory

    // Initialize every value in the matrix with a random number in range [0, 1]
    for (size_t i = 0; i < rows * columns; i++) {
        *(*output+i) = (double) rand() / RAND_MAX;
    }
    return LIBNEURAL_HELPER_OK;
}

uint8_t libneural_helper_relu(double *input, size_t input_size, double **output) {
    *output = malloc(input_size * sizeof(double)); // Allocate memory, output is of type double
    if (!*output) return LIBNEURAL_HELPER_ERR_OOM; // Out of memory

    // Copy max(0, input element) into every element of the output
    for (size_t i = 0; i < input_size; i++) {
        double value = *(input+i);
        if (value > 0) {
            *(*output+i) = value;
        } else {
            *(*output+i) = 0;
        }
    }
    return LIBNEURAL_HELPER_OK;
}

uint8_t libneural_helper_relu_prime(double *input, size_t size) {
    // Copy element > 0 into every element of the matrix
    for (size_t i = 0; i < size; i++) {
        double value = *(input+i);
        if (value > 0) {
            *(input+i) = 1;
        } else {
            *(input+i) = 0;
        }
    }
    return LIBNEURAL_HELPER_OK;
}

uint8_t libneural_helper_softmax(double *input, size_t size) {
    if (size < 1) { // Flattened matrix needs at least one element
        return LIBNEURAL_HELPER_ERR_INPUT;
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
    return LIBNEURAL_HELPER_OK;
}

uint8_t libneural_helper_predict(double *input, size_t size, size_t *result) {
    if (size < 1) { // Flattened matrix needs at least one element
        return LIBNEURAL_HELPER_ERR_INPUT;
    }

    *result = 0; // Initialize with first element of matrix
    for (size_t i = 1; i < size; i++) { // We know the first element isn't larger than max, so start at i=1
        if (*(input+i) > *(input+*result)) {
            *result = i;
        }
    }
    return LIBNEURAL_HELPER_OK;
}
