#include <stdlib.h>

#include "libneural_helper.h"
#include "libneural_matrix.h"
#include "libneural_train.h"

uint8_t libneural_train_feed_forward(double *input, size_t input_size, double *w1, size_t hidden_size, double *w2, size_t output_size, double **z1, double **a1, double **a2) {
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
