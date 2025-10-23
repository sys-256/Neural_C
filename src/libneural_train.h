#ifndef LIBNEURAL_TRAIN_H
#define LIBNEURAL_TRAIN_H

#include <stddef.h>
#include <stdint.h>

// Return codes
#define LIBNEURAL_TRAIN_OK 0
#define LIBNEURAL_TRAIN_ERR_OOM 1
#define LIBNEURAL_TRAIN_ERR_INPUT 2

/**
 * Executes the feed forward algorithm
 * 
 * input - The values of the input layer
 * w1 - A input_size x hidden_size matrix of weights
 * w2 - A hidden_size x output_size matrix of weights
 * input_size - The amount of nodes in the input layer
 * hidden_size - The amount of nodes in the hidden layer
 * output_size - The amount of nodes in the output layer
 * z1 - A pointer to where the unactivated hidden layer should be written to
 * a1 - A pointer to where the activated hidden layer should be written to
 * a2 - A pointer to where the activated output layer should be written to
 */
uint8_t libneural_train_feed_forward(double *input, double *w1, double *w2, size_t input_size, size_t hidden_size, size_t output_size, double **z1, double **a1, double **a2);

/**
 * Executes the back propagation algorithm by updating the weights
 * 
 * input - The values of the input layer
 * w1 - A input_size x hidden_size matrix of weights
 * w2 - A hidden_size x output_size matrix of weights
 * output - The expected value of the output layer (one-hot encoded)
 * input_size - The amount of nodes in the input layer
 * hidden_size - The amount of nodes in the hidden layer
 * output_size - The amount of nodes in the output layer
 * alpha - The learning rate
 */
uint8_t libneural_train_back_propagation(double *input, double *w1, double *w2, double *output, size_t input_size, size_t hidden_size, size_t output_size, double alpha);

#endif // LIBNEURAL_TRAIN_H
