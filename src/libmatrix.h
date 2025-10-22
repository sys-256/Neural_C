#ifndef LIBMATRIX_H
#define LIBMATRIX_H

#include <stddef.h>
#include <stdint.h>

// Return codes
#define LIBMATRIX_OK 0
#define LIBMATRIX_ERR_OOM 1
#define LIBMATRIX_ERR_INPUT 2

/**
 * Generates a flattened matrix of random weights
 * 
 * rows - The amount of rows in the matrix
 * columns - The amount of columns in the matrix
 * out_buf - A pointer to the array to be written to
 * out_size - A pointer to the number of elements in the array to be written to
 */
int libmatrix_generate_weights(uint64_t rows, uint64_t columns, double **out_buf, size_t *out_size);

/**
 * Applies ReLU on every element in a flattened matrix
 * 
 * in_buf - The array to be read from
 * in_size - The number of elements in the array to be read from
 * out_buf - A pointer to the array to be written to
 * out_size - A pointer to the number of elements in the array to be written to
 */
int libmatrix_relu(double *in_buf, size_t in_size, double **out_buf, size_t *out_size);

/**
 * Applies the derivative of ReLU on every element in a flattened matrix
 * 
 * buf - The array to modify
 * size - The number of elements in the array to modify
 */
int libmatrix_relu_prime(double *buf, size_t size);

/**
 * Applies softmax to a flattened matrix
 * 
 * buf - The array to modify
 * size - The number of elements in the array to modify
 */
int libmatrix_softmax(double *buf, size_t size);

#endif // LIBMATRIX_H
