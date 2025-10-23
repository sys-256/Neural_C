#ifndef READ_IDX_H
#define READ_IDX_H

#include <stddef.h>
#include <stdint.h>

// Return codes
#define READ_IDX_OK 0
#define READ_IDX_ERR_IO 1
#define READ_IDX_ERR_FORMAT 2
#define READ_IDX_ERR_OOM 3

/**
 * Reads an IDX file
 * 
 * filename - The file to read from
 * out_buf - A pointer to the array to be written to
 * out_size - A pointer to the number of elements in the array to be written to
 */
uint8_t read_idx(char *filename, uint8_t **out_buf, size_t *out_size);

#endif // READ_IDX_H
