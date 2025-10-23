#ifndef LIBNEURAL_HELPER_H
#define LIBNEURAL_HELPER_H

#include <stddef.h>
#include <stdint.h>

// Return codes
#define LIBNEURAL_HELPER_OK 0
#define LIBNEURAL_HELPER_ERR_OOM 1
#define LIBNEURAL_HELPER_ERR_INPUT 2

/**
 * Generates a flattened matrix of random weights
 * 
 * rows - The amount of rows in the matrix
 * columns - The amount of columns in the matrix
 * output - A pointer to where the matrix should be written to
 */
uint8_t libneural_helper_generate_weights(size_t rows, size_t columns, double **output);

/**
 * Applies ReLU on every element in a flattened matrix
 * 
 * input - The matrix to be read from
 * input_size - The number of elements in that matrix
 * output - A pointer to where the matrix should be written to
 */
uint8_t libneural_helper_relu(double *input, size_t input_size, double **output);

/**
 * Applies the derivative of ReLU on every element in a flattened matrix
 * 
 * input - The matrix to modify
 * size - The number of elements in that matrix
 */
uint8_t libneural_helper_relu_prime(double *input, size_t size);

/**
 * Applies softmax to a flattened matrix
 * 
 * input - The matrix to modify
 * size - The number of elements in the matrix to modify
 */
uint8_t libneural_helper_softmax(double *input, size_t size);

/**
 * Gets the index of the highest value in a flattened matrix
 * 
 * input - The matrix to be read from
 * size - The number of elements in that matrix
 * result - A pointer to where the highest index should be written to
 */
uint8_t libneural_helper_predict(double *input, size_t size, size_t *result);

#endif // LIBNEURAL_HELPER_H
