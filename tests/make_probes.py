"""Builds the small probe images the browser test uploads.

    python tests/make_probes.py

Two kinds of probe, both of which have an exactly known tensor, so a browser result can be
compared with the ONNX model without any "resampling is allowed to differ" excuse:

  centre_crop_probe.png  288x96, black | check_image 96x96 | black. The centre square is exactly
                         check_image.jpg, and 96 -> 96 is not resampled, so the page must give
                         the same probabilities as check_expected.json. A page that cropped from
                         the corner or resized the whole strip would not.
  colour_probe_{1,2,3}.png  800x600 filled with one colour. A constant image stays constant under
                         any crop or resize, so the tensor is known exactly from the colour;
                         the three colours are red-, green- and blue-dominant and would disagree
                         if the channels were written in the wrong order.
"""
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")


def main():
    os.makedirs(OUT, exist_ok=True)
    leaf = Image.open("check_image.jpg").convert("RGB")
    assert leaf.size == (96, 96), leaf.size

    strip = Image.new("RGB", (288, 96), (0, 0, 0))
    strip.paste(leaf, (96, 0))
    strip.save(os.path.join(OUT, "centre_crop_probe.png"))

    for i, colour in enumerate([(190, 70, 60), (60, 170, 80), (70, 90, 200)], start=1):
        Image.fromarray(np.full((600, 800, 3), colour, np.uint8)).save(
            os.path.join(OUT, f"colour_probe_{i}.png"))
    print("wrote", ", ".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()
