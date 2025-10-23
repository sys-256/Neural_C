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
 * out_buf - A pointer to where the matrix should be written to
 * out_size - A pointer to where the number of elements in the matrix should be written to
 */
uint8_t libmatrix_generate_weights(uint64_t rows, uint64_t columns, double **out_buf, size_t *out_size);

/**
 * Applies ReLU on every element in a flattened matrix
 * 
 * in_buf - The matrix to be read from
 * in_size - The number of elements in the matrix to be read from
 * out_buf - A pointer to where the matrix should be written to
 * out_size - A pointer to where the number of elements in the matrix should to be written to
 */
uint8_t libmatrix_relu(double *in_buf, size_t in_size, double **out_buf, size_t *out_size);

/**
 * Applies the derivative of ReLU on every element in a flattened matrix
 * 
 * buf - The matrix to modify
 * size - The number of elements in the matrix to modify
 */
uint8_t libmatrix_relu_prime(double *buf, size_t size);

/**
 * Applies softmax to a flattened matrix
 * 
 * buf - The matrix to modify
 * size - The number of elements in the matrix to modify
 */
uint8_t libmatrix_softmax(double *buf, size_t size);

/**
 * Gets the index of the highest value in a flattened matrix
 * 
 * buf - The matrix to read from
 * size - The number of elements in the matrix to read from
 * result - A pointer to where the highest index should be written to
 */
uint8_t libmatrix_predict(double *buf, size_t size, size_t *result);

/**
 * Transposes the given matrix
 * 
 * in_buf - The matrix to transpose
 * in_rows - The amount of rows in that matrix
 * in_cols - The amount of columns in that matrix
 * out_buf - A pointer to where the transposed matrix should be written to
 * out_rows - A pointer to where the amount of rows in that matrix should be written to
 * out_cols - A pointer to where the amount of columns in that matrix should be written to
 */
uint8_t libmatrix_transpose(double *in_buf, size_t in_rows, size_t in_cols, double **out_buf, size_t *out_rows, size_t *out_cols);

/**
 * Applies the dot product on the given matrices
 * 
 * in1_buf - The matrix to multiply with
 * in1_rows - The amount of rows in that matrix
 * in1_cols - The amount of columns in that matrix
 * in2_buf - The other matrix to multiply with
 * in2_rows - The amount of rows in that matrix
 * in2_cols - The amount of columns in that matrix
 * out_buf - A pointer to where the resulting matrix should be written to
 * out_rows - A pointer to where the amount of rows in that matrix should be written to
 * out_cols - A pointer to where the amount of columns in that matrix should be written to
 */
uint8_t libmatrix_dot_product(double *in1_buf, size_t in1_rows, size_t in1_cols, double *in2_buf, size_t in2_rows, size_t in2_cols, double **out_buf, size_t *out_rows, size_t *out_cols);

/**
 * Multiplies every value in the matrix by a given constant
 * 
 * in_buf - The matrix to multiply with
 * in_size - The number of elements in the matrix to be read from
 * constant - The constant to multiply with
 * out_buf - A pointer to where the resulting matrix should be written to
 * out_size - A pointer to where the number of elements in the matrix should to be written to
 */
uint8_t libmatrix_const_product(double *in_buf, size_t in_size, double constant, double **out_buf, size_t *out_size);

/**
 * Subtracts two given matrices
 * 
 * in1_buf - The first matrix
 * in1_size - The amount of elements in that matrix
 * in2_buf - The matrix to subtract from the first
 * in2_size - The amount of elements in that matrix
 * out_buf - A pointer to where the resulting matrix should be written to
 * out_size - A pointer to where the amount of elements in that matrix should be written to
 */
uint8_t libmatrix_minus(double *in1_buf, size_t in1_size, double *in2_buf, size_t in2_size, double **out_buf, size_t *out_size);


#endif // LIBMATRIX_H
