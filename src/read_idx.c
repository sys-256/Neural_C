#include "read_idx.h"
#include <stdio.h>
#include <stdlib.h>

uint8_t read_idx(char *filename, uint8_t **out_buf, size_t *out_size) {
    // Make sure a pointer for buffer and size were provided
    if (out_buf == NULL || out_size == NULL) return READ_IDX_ERR_FORMAT;

    // Try to open the file
    FILE *file = fopen(filename, "rb");
    if (!file) return READ_IDX_ERR_IO;

    uint32_t magic_byte;
    fread(&magic_byte, 4, 1, file); // Read 1 item of 4 bytes
    magic_byte = __builtin_bswap32(magic_byte); // Swap endianness
    
    if (magic_byte == 2051) { // If images
        uint32_t temp[3];
        fread(&temp, 4, 3, file); // Read 3 items of 4 bytes

        // Initialize variables
        uint32_t num_images = __builtin_bswap32(temp[0]);
        uint32_t image_height = __builtin_bswap32(temp[1]);
        uint32_t image_width = __builtin_bswap32(temp[2]);
        *out_size = num_images * image_height * image_width;

        *out_buf = malloc(*out_size); // Allocate buffer
        if (!*out_buf) return READ_IDX_ERR_OOM; // Out of memory

        fread(*out_buf, 1, *out_size, file); // Read all the pixels, each 1 byte
        fclose(file);
        return READ_IDX_OK;
    } else if (magic_byte == 2049) { // If labels
        uint32_t num_labels;
        fread(&num_labels, 4, 1, file); // Read 1 item of 4 bytes
        num_labels = __builtin_bswap32(num_labels); // Swap endianness
        *out_size = num_labels;

        *out_buf = malloc(*out_size * sizeof(uint8_t)); // Allocate buffer, out_buf is of type uint8_t
        if (!*out_buf) return READ_IDX_ERR_OOM; // Out of memory

        fread(*out_buf, 1, *out_size, file); // Read all the labels, each 1 byte
        fclose(file);
        return READ_IDX_OK;
    } else { // Invalid IDX file
        return READ_IDX_ERR_FORMAT;
    }
}