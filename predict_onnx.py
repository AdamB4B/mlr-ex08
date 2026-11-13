"""One image through the exported ONNX model, with exactly the notebook's preprocessing.

    python predict_onnx.py check_image.jpg
    python predict_onnx.py some_photo.png --model export/leaf_model.onnx --info export/model_info.json

The preprocessing is the one of the notebook (Task 2 / the dataset): centre square crop,
resize to model_info's input_size, ToTensor (divide by 255), Normalize(mean, std),
channels first, one image per batch. It prints JSON so a test can compare with the page.
"""
import argparse
import json
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image


def preprocess(path, size, mean, std):
    image = Image.open(path).convert("RGB")
    side = min(image.size)                       # central square, like the page and the dataset
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    if image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return image, x.transpose(2, 0, 1)[None]     # HWC -> NCHW


def predict(path, model="leaf_model.onnx", info_path="model_info.json"):
    info = json.load(open(info_path, encoding="utf-8"))
    size = info["input_size"]
    image, x = preprocess(path, size, info["mean"], info["std"])
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    logits = session.run(None, {session.get_inputs()[0].name: x.astype(np.float32)})[0][0]
    e = np.exp(logits - logits.max())
    probabilities = e / e.sum()
    return {
        "image": path,
        "imageSize": list(image.size),
        "input": "1x3x%dx%d" % (size, size),
        "classes": info["classes"],
        "logits": [float(v) for v in logits],
        "probabilities": {c: float(p) for c, p in zip(info["classes"], probabilities)},
        "top": info["classes"][int(np.argmax(probabilities))],
        "topProbability": float(np.max(probabilities)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--model", default="leaf_model.onnx")
    parser.add_argument("--info", default="model_info.json")
    args = parser.parse_args()
    try:
        print(json.dumps(predict(args.image, args.model, args.info), indent=1))
    except FileNotFoundError as error:
        sys.exit(str(error))
