from pathlib import Path
import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from stm_ai_runner import AiRunner

def basic():
    runner = AiRunner()

    runner.connect("serial:921600")
    runner.summary()

    runner.disconnect()

def infer_v1():
    runner = AiRunner()
    runner.connect("serial:921600")

    # mid-gray image pair
    x = np.zeros((1, 6, 96, 96), dtype=np.int8)

    outputs, profiler = runner.invoke([x])

    y = outputs[0]

    print("shape:", y.shape)
    print("dtype:", y.dtype)
    print("min/max:", y.min(), y.max())
    print("unique-ish:", np.unique(y)[:20])

    runner.disconnect()

def infer_real():
    data_root = Path("data")
    sample_id = 46

    image_a = np.asarray(
        Image.open(data_root / "A" / f"{sample_id}.png").convert("RGB")
    )
    image_b = np.asarray(
        Image.open(data_root / "B" / f"{sample_id}.png").convert("RGB")
    )
    label = np.asarray(
        Image.open(data_root / "label" / f"{sample_id}.png")
    )

    print("A:", image_a.shape, image_a.dtype)
    print("B:", image_b.shape, image_b.dtype)
    print("label:", label.shape, label.dtype)

    # Expect RGB 96x96 images
    assert image_a.shape == (96, 96, 3)
    assert image_b.shape == (96, 96, 3)

    # HWC -> CHW
    image_a = np.transpose(image_a, (2, 0, 1))  # (3, 96, 96)
    image_b = np.transpose(image_b, (2, 0, 1))  # (3, 96, 96)

    # Channel order:
    # [A_R, A_G, A_B, B_R, B_G, B_B]
    x_u8 = np.concatenate([image_a, image_b], axis=0)

    # Add batch dimension
    x_u8 = x_u8[None, ...]  # (1, 6, 96, 96)

    # ------------------------------------------------------------------
    # Quantize input
    #
    # From your runner.summary():
    #   QLinear(scale=1.0, zero_point=-128, int8)
    #
    # q = round(real / scale + zero_point)
    #   = pixel - 128
    # ------------------------------------------------------------------

    x_int8 = np.round(
        x_u8.astype(np.float32) / 1.0 - 128
    )

    x_int8 = np.clip(x_int8, -128, 127).astype(np.int8)

    print("input:", x_int8.shape, x_int8.dtype)
    print("input range:", x_int8.min(), x_int8.max())

    # ------------------------------------------------------------------
    # Run STM32N6
    # ------------------------------------------------------------------

    runner = AiRunner()
    runner.connect(os.environ.get("STM32_SERIAL", "serial:921600"))

    outputs, profiler = runner.invoke([x_int8])

    runner.disconnect()

    y_int8 = outputs[0]

    print("raw output:", y_int8.shape, y_int8.dtype)
    print("raw range:", y_int8.min(), y_int8.max())

    # ------------------------------------------------------------------
    # Dequantize output
    #
    # From runner.summary():
    #   QLinear(scale=0.062618285, zero_point=-69, int8)
    #
    # real = (q - zero_point) * scale
    # ------------------------------------------------------------------

    logits = (
                     y_int8.astype(np.float32) - (-69)
             ) * 0.062618285

    logits = logits[0, 0]  # (96, 96)

    print("logit range:", logits.min(), logits.max())

    # ------------------------------------------------------------------
    # Sigmoid
    # ------------------------------------------------------------------

    prob = 1.0 / (1.0 + np.exp(-logits))

    mask = prob >= 0.5

    # Make label binary for display/comparison
    if label.ndim == 3:
        label = label[..., 0]

    label_binary = label > 0

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    axes[0].imshow(image_a.transpose(1, 2, 0))
    axes[0].set_title("Image A")

    axes[1].imshow(image_b.transpose(1, 2, 0))
    axes[1].set_title("Image B")

    axes[2].imshow(label_binary, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth")

    im = axes[3].imshow(prob, cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title("STM32N6 Probability")

    axes[4].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[4].set_title("STM32N6 Prediction")

    for ax in axes:
        ax.axis("off")

    # Colorbar only for probability map
    fig.colorbar(
        im,
        ax=axes[3],
        fraction=0.046,
        pad=0.04,
        label="Change probability"
    )

    fig.suptitle(
        f"Change Detection — Sample {sample_id}",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # infer_v1()
    infer_real()
