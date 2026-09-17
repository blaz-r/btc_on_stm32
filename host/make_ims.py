"""Export selected OSC-D96 image pairs as C assets for the STM32 demo."""

from pathlib import Path
import os

import numpy as np
from PIL import Image


DATA_ROOT = Path(os.environ.get("CD_TEST_ROOT", "data"))
SAMPLE_IDS = (46, 226, 364)
PROJECT_ROOT = Path(__file__).resolve().parents[1] / "firmware" / "BTC_STM"
SOURCE_PATH = PROJECT_ROOT / "STM32CubeIDE" / "Appli" / "Application" / "User" / "Core" / "images.c"
HEADER_PATH = PROJECT_ROOT / "Appli" / "Core" / "Inc" / "images.h"


def load_rgb(split: str, sample_id: int) -> np.ndarray:
    image = np.asarray(
        Image.open(DATA_ROOT / split / f"{sample_id}.png").convert("RGB"),
        dtype=np.uint8,
    )
    assert image.shape == (96, 96, 3)
    return image


def write_array(output, name: str, image: np.ndarray) -> None:
    flat = image.reshape(-1)
    output.write(f"const uint8_t {name}[{len(flat)}] = {{\n")
    for offset in range(0, len(flat), 16):
        values = ", ".join(str(value) for value in flat[offset : offset + 16])
        output.write(f"    {values},\n")
    output.write("};\n\n")


def main() -> None:
    pairs = [(sample_id, load_rgb("A", sample_id), load_rgb("B", sample_id))
             for sample_id in SAMPLE_IDS]

    with SOURCE_PATH.open("w", newline="\n") as output:
        output.write('#include "images.h"\n\n')
        for sample_id, image_a, image_b in pairs:
            write_array(output, f"image_{sample_id}_a_rgb", image_a)
            write_array(output, f"image_{sample_id}_b_rgb", image_b)

        output.write("const ImagePair image_pairs[IMAGE_PAIR_COUNT] = {\n")
        for sample_id, _, _ in pairs:
            output.write(f"    {{image_{sample_id}_a_rgb, image_{sample_id}_b_rgb, {sample_id}U}},\n")
        output.write("};\n")

    HEADER_PATH.write_text(
        """#ifndef IMAGES_H
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
""",
        newline="\n",
    )


if __name__ == "__main__":
    main()
