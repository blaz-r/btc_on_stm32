#ifndef IMAGES_H
#define IMAGES_H

#include <stdint.h>

#define IMAGE_W 96U
#define IMAGE_H 96U
#define IMAGE_C 3U
#define IMAGE_PAIR_COUNT 3U

typedef struct
{
    const uint8_t *a_rgb;
    const uint8_t *b_rgb;
    uint16_t sample_id;
} ImagePair;

extern const ImagePair image_pairs[IMAGE_PAIR_COUNT];

#endif
