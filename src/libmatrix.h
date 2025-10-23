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
 * output - A pointer to where the matrix should be written to
 */
uint8_t libmatrix_generate_weights(uint64_t rows, uint64_t columns, double **output);

/**
 * Applies ReLU on every element in a flattened matrix
 * 
 * input - The matrix to be read from
 * input_size - The number of elements in that matrix
 * output - A pointer to where the matrix should be written to
 */
uint8_t libmatrix_relu(double *input, size_t input_size, double **output);

/**
 * Applies the derivative of ReLU on every element in a flattened matrix
 * 
 * input - The matrix to modify
 * size - The number of elements in that matrix
 */
uint8_t libmatrix_relu_prime(double *input, size_t size);

/**
 * Applies softmax to a flattened matrix
 * 
 * input - The matrix to modify
 * size - The number of elements in the matrix to modify
 */
uint8_t libmatrix_softmax(double *input, size_t size);

/**
 * Gets the index of the highest value in a flattened matrix
 * 
 * input - The matrix to be read from
 * size - The number of elements in that matrix
 * result - A pointer to where the highest index should be written to
 */
uint8_t libmatrix_predict(double *input, size_t size, size_t *result);

/**
 * Transposes the given matrix
 * 
 * input - The matrix to transpose
 * input_rows - The amount of rows in that matrix
 * input_cols - The amount of columns in that matrix
 * output - A pointer to where the transposed matrix should be written to
 */
uint8_t libmatrix_transpose(double *input, size_t input_rows, size_t input_cols, double **output);

/**
 * Applies the dot product on the given matrices
 * 
 * input1 - The matrix to multiply with
 * input1_rows - The amount of rows in that matrix
 * input1_cols - The amount of columns in that matrix which is the same as the amount of rows in the other
 * input2 - The other matrix to multiply with
 * input2_cols - The amount of columns in that matrix
 * output - A pointer to where the resulting matrix should be written to
 */
uint8_t libmatrix_dot_product(double *input1, size_t input1_rows, size_t input1_cols, double *input2, size_t input2_cols, double **output);

/**
 * Multiplies every value in the matrix by a given constant
 * 
 * input - The matrix to multiply with
 * input_size - The number of elements in that matrix
 * constant - The constant to multiply with
 * output - A pointer to where the resulting matrix should be written to
 */
uint8_t libmatrix_const_product(double *input, size_t input_size, double constant, double **output);

/**
 * Subtracts two given matrices
 * 
 * input1 - The first matrix
 * input2 - The matrix to subtract from the first
 * input_size - The amount of elements in the matrices
 * output - A pointer to where the resulting matrix should be written to
 */
uint8_t libmatrix_minus(double *input1, double *input2, size_t input_size, double **output);


#endif // LIBMATRIX_H
