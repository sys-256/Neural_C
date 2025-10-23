#include <stdlib.h>

#include "libneural_helper.h"
#include "libneural_matrix.h"
#include "libneural_train.h"

uint8_t libneural_train_feed_forward(double *input, double *w1, double *w2, size_t input_size, size_t hidden_size, size_t output_size, double **z1, double **a1, double **a2) {
    uint8_t error_code = libneural_matrix_dot_product(input, 1, input_size, w1, hidden_size, z1);
    if (error_code != LIBNEURAL_MATRIX_OK) return LIBNEURAL_TRAIN_ERR_OOM; // dot_product only returns OK or ERR_OOM
    
    error_code = libneural_helper_relu(*z1, hidden_size, a1);
    if (error_code != LIBNEURAL_HELPER_OK) return LIBNEURAL_TRAIN_ERR_OOM; // relu only returns OK or ERR_OOM
    
    error_code = libneural_matrix_dot_product(*a1, 1, hidden_size, w2, output_size, a2);
    if (error_code != LIBNEURAL_MATRIX_OK) return LIBNEURAL_TRAIN_ERR_OOM; // dot_product only returns OK or ERR_OOM
    
    error_code = libneural_helper_softmax(*a2, output_size);
    if (error_code != LIBNEURAL_HELPER_OK) {
        if (error_code == LIBNEURAL_HELPER_ERR_OOM) {
            return LIBNEURAL_TRAIN_ERR_OOM;
        } else { // Only 2 possible error codes (ERR_OOM, ERR_FORMAT)
            return LIBNEURAL_TRAIN_ERR_INPUT;
        }
    }
    return LIBNEURAL_TRAIN_OK;
}

// Without frees and error checks to make it more readable

// double *z1, *a1, *a2; // Get z1, a1, a2
// libneural_train_feed_forward(input, w1, w2, input_size, hidden_size, output_size, &z1, &a1, &a2);

// libneural_matrix_minus(a2, output, output_size); // dz2 = a2-y = (1, output_size), use a2 for dz2 (for less malloc)

// double *w2T;
// libneural_matrix_transpose(w2, hidden_size, output_size, &w2T); // w2T = w2.T = (output_size, hidden_size)

// double *dz1;
// libneural_matrix_dot_product(a2, 1, output_size, w2T, hidden_size, &dz1); // dz1 = dz2.dot(w2T)
// libneural_helper_relu_prime(z1, hidden_size); // z1 = ReLU(z1)
// libneural_matrix_hadamard_product(dz1, z1, hidden_size); // dz1 = dz1 * ReLU(z1) = (1, hidden_size)

// double *a1T;
// libneural_matrix_transpose(a1, 1, hidden_size, &a1T); // a1T = a1.T = (hidden_size, 1)

// double *dw2;
// libneural_matrix_dot_product(a1T, hidden_size, 1, a2, output_size, &dw2); // dw2 = a1T.dot(dz2) = (hidden_size, output_size)

// double *inputT;
// libneural_matrix_transpose(input, 1, input_size, &inputT); // inputT = input.T = (input_size, 1)

// double *dw1;
// libneural_matrix_dot_product(inputT, input_size, 1, dz1, hidden_size, &dw1); // dw1 = inputT.dot(dz1) = (input_size, hidden_size)

// libneural_matrix_const_product(dw2, hidden_size*output_size, alpha); // dw2 *= alpha
// libneural_matrix_const_product(dw1, input_size*hidden_size, alpha); // dw1 *= alpha
    
// libneural_matrix_minus(w2, dw2, hidden_size*output_size); // w2 -= dw2 (which was already multiplied by alpha)
// libneural_matrix_minus(w1, dw1, input_size*hidden_size); // w1 -= dw1 (which was already multiplied by alpha)

uint8_t libneural_train_back_propagation(double *input, double *w1, double *w2, double *output, size_t input_size, size_t hidden_size, size_t output_size, double alpha) {
    double *z1, *a1, *a2; // Get z1, a1, a2
    uint8_t error_code = libneural_train_feed_forward(input, w1, w2, input_size, hidden_size, output_size, &z1, &a1, &a2);
    if (error_code != LIBNEURAL_TRAIN_OK) return error_code; // Same error codes since same file

    libneural_matrix_minus(a2, output, output_size); // dz2 = a2-y = (1, output_size), use a2 for dz2, always returns OK so don't check for errors

    double *w2T;
    error_code = libneural_matrix_transpose(w2, hidden_size, output_size, &w2T); // w2T = w2.T = (output_size, hidden_size)
    if (error_code != LIBNEURAL_MATRIX_OK) return LIBNEURAL_TRAIN_ERR_OOM; // transpose only returns OK or ERR_OOM

    double *dz1;
    error_code = libneural_matrix_dot_product(a2, 1, output_size, w2T, hidden_size, &dz1); // dz1 = dz2.dot(w2T)
    if (error_code != LIBNEURAL_MATRIX_OK) return LIBNEURAL_TRAIN_ERR_OOM; // dot_product only returns OK or ERR_OOM
    free(w2T); // We don't need w2T after this, nor do we reassign it
    w2T = NULL;
    libneural_helper_relu_prime(z1, hidden_size); // z1 = ReLU(z1), always returns OK so don't check for errors
    libneural_matrix_hadamard_product(dz1, z1, hidden_size); // dz1 = dz1 * ReLU(z1) = (1, hidden_size), always returns OK so don't check for errors
    free(z1); // We don't need a1 after this, nor do we reassign it
    z1 = NULL;

    double *a1T;
    error_code = libneural_matrix_transpose(a1, 1, hidden_size, &a1T); // a1T = a1.T = (hidden_size, 1)
    if (error_code != LIBNEURAL_MATRIX_OK) return LIBNEURAL_TRAIN_ERR_OOM; // transpose only returns OK or ERR_OOM
    free(a1); // We don't need a1 after this, nor do we reassign it
    a1 = NULL;

    double *dw2;
    error_code = libneural_matrix_dot_product(a1T, hidden_size, 1, a2, output_size, &dw2); // dw2 = a1T.dot(dz2) = (hidden_size, output_size)
    if (error_code != LIBNEURAL_MATRIX_OK) return LIBNEURAL_TRAIN_ERR_OOM; // dot_product only returns OK or ERR_OOM
    free(a1T); // We don't need a1T after this, nor do we reassign it
    a1T = NULL;
    free(a2); // We don't need a2 (which is secretly dz2) after this, nor do we reassign it
    a2 = NULL;

    double *inputT;
    error_code = libneural_matrix_transpose(input, 1, input_size, &inputT); // inputT = input.T = (input_size, 1)
    if (error_code != LIBNEURAL_MATRIX_OK) return LIBNEURAL_TRAIN_ERR_OOM; // transpose only returns OK or ERR_OOM

    double *dw1;
    error_code = libneural_matrix_dot_product(inputT, input_size, 1, dz1, hidden_size, &dw1); // dw1 = inputT.dot(dz1) = (input_size, hidden_size)
    if (error_code != LIBNEURAL_MATRIX_OK) return LIBNEURAL_TRAIN_ERR_OOM; // dot_product only returns OK or ERR_OOM
    free(inputT); // We don't need a1T after this, nor do we reassign it
    inputT = NULL;
    free(dz1); // We don't need dz1 after this, nor do we reassign it
    dz1 = NULL;

    libneural_matrix_const_product(dw2, hidden_size*output_size, alpha); // always returns OK so don't check for errors
    libneural_matrix_const_product(dw1, input_size*hidden_size, alpha); // always returns OK so don't check for errors
    
    libneural_matrix_minus(w2, dw2, hidden_size*output_size); // always returns OK so don't check for errors
    libneural_matrix_minus(w1, dw1, input_size*hidden_size); // always returns OK so don't check for errors
    free(dw2); // We don't need dw2 after this, nor do we reassign it
    dw2 = NULL;
    free(dw1); // We don't need dw1 after this, nor do we reassign it
    dw1 = NULL;
    return LIBNEURAL_TRAIN_OK;
}
