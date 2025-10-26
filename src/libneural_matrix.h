#ifndef LIBNEURAL_MATRIX_H
#define LIBNEURAL_MATRIX_H

#include <stddef.h>
#include <stdint.h>

// Return codes
#define LIBNEURAL_MATRIX_OK 0
#define LIBNEURAL_MATRIX_ERR_OOM 1

/**
 * Transposes the given matrix
 *
 * input - The matrix to transpose
 * input_rows - The amount of rows in that matrix
 * input_cols - The amount of columns in that matrix
 * output - A pointer to where the transposed matrix should be written to
 */
uint8_t libneural_matrix_transpose(double *input, size_t input_rows, size_t input_cols, double **output);

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
uint8_t libneural_matrix_dot_product(double *input1, size_t input1_rows, size_t input1_cols, double *input2, size_t input2_cols, double **output);

/**
 * Multiplied every value in input1 by the corresponding element in input2
 *
 * input1 - The matrix that will be modified
 * input2 - The values that input1 will be multiplied by
 * input_size - The amount of elements in the matrices
 */
uint8_t libneural_matrix_hadamard_product(double *input1, double *input2, size_t input_size);

/**
 * Multiplies every value in the matrix by a given constant
 *
 * input - The matrix to multiply with
 * input_size - The number of elements in that matrix
 * constant - The constant to multiply with
 */
uint8_t libneural_matrix_const_product(double *input, size_t input_size, double constant);

/**
 * Subtracts two given matrices
 *
 * input1 - The first matrix
 * input2 - The matrix to subtract from the first
 * input_size - The amount of elements in the matrices
 */
uint8_t libneural_matrix_minus(double *input1, double *input2, size_t input_size);

#endif // LIBNEURAL_MATRIX_H
