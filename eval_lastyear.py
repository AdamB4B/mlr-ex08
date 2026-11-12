# The starting point of REPORT.md, section 2: the exported model against last year's photographs.
# Needs only numpy, onnxruntime and Pillow — no PyTorch. Run:  python eval_lastyear.py
#
# It preprocesses exactly like index.html does: centre square crop, resize to the input size,
# divide by 255, standardize each channel with mean and std, channels first.

import json
import os
import urllib.request
import zipfile

import numpy as np
import onnxruntime as ort
from PIL import Image

ARCHIVE_ZIP = ("https://raw.githubusercontent.com/tomasvicar/MLR-public/"
               "master/exercises/data/ex08_leaves_images.zip")
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/leaves_old.zip"):
    urllib.request.urlretrieve(ARCHIVE_ZIP, "data/leaves_old.zip")
if not os.path.isdir("data/leaves_old"):
    with zipfile.ZipFile("data/leaves_old.zip") as archive:
        archive.extractall("data/leaves_old")      # unpacks to data/leaves_old/leaves/{train,val}

info = json.load(open("model_info.json", encoding="utf-8"))
SIZE, CLASSES = info["input_size"], info["classes"]
MEAN = np.array(info["mean"], np.float32)
STD = np.array(info["std"], np.float32)

session = ort.InferenceSession("leaf_model.onnx", providers=["CPUExecutionProvider"])
INPUT = session.get_inputs()[0].name


def probabilities(image):
    image = image.convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))    # centre square
    image = image.resize((SIZE, SIZE), Image.BILINEAR)
    x = np.asarray(image, np.float32) / 255.0                   # like ToTensor()
    x = (x - MEAN) / STD                                        # like Normalize()
    logits = session.run(None, {INPUT: x.transpose(2, 0, 1)[None]})[0][0]
    e = np.exp(logits - logits.max())
    return e / e.sum()


def score(folder, label):
    classes = sorted(c for c in CLASSES if os.path.isdir(os.path.join(folder, c)))
    table = np.zeros((len(classes), len(classes)), int)
    for c_true, cls in enumerate(classes):
        for name in sorted(os.listdir(os.path.join(folder, cls))):
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image = Image.open(os.path.join(folder, cls, name))
            table[c_true, int(probabilities(image).argmax())] += 1
    print(f"\n{label}: {np.trace(table)} / {table.sum()} = {np.trace(table) / table.sum():.3f}")
    print("truth \\ pred  " + " ".join(f"{c:>9s}" for c in classes))
    for i, cls in enumerate(classes):
        print(f"{cls:>11s}     " + " ".join(f"{v:9d}" for v in table[i]) +
              f"   | recall {table[i, i] / table[i].sum():.3f}")


for split in ("train", "val"):
    folder = f"data/leaves_old/leaves/{split}"
    if os.path.isdir(folder):
        score(folder, f"last year, {split} — other students, other phones")

print("\ncheck_image.jpg through this page's pipeline, against check_expected.json:")
expected = json.load(open("check_expected.json", encoding="utf-8"))
for cls, p in zip(CLASSES, probabilities(Image.open("check_image.jpg"))):
    print(f"  {cls:9s} {p:.4f}   notebook {expected[cls]:.4f}   difference {abs(p - expected[cls]):.1e}")
