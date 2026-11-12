# Scores the model that is on the page today on exactly the same three sets as task4_variants.py,
# so it can be the "as it stands" row of the table in REPORT.md.  Run: python eval_page_model.py
# The sets are built by the same code the runs use, and the model is fed the same way the page
# feeds it: centre crop, resize, /255, standardize, channels first, batch of one.
import json

import numpy as np
import onnxruntime as ort

import task4_variants as t

info = json.load(open("model_info.json", encoding="utf-8"))
CLASSES = info["classes"]
session = ort.InferenceSession("leaf_model.onnx", providers=["CPUExecutionProvider"])
INPUT = session.get_inputs()[0].name
paths = t.folders()


def score(dataset, label):
    loader = t.DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False)
    table = np.zeros((len(CLASSES), len(CLASSES)), int)
    for image, truth in loader:
        logits = session.run(None, {INPUT: image.numpy().astype("float32")})[0][0]
        table[int(truth), int(np.argmax(logits))] += 1
    total = table.sum()
    print(f"\n{label}: {np.trace(table)} / {total} = {np.trace(table) / total:.3f}")
    print("             " + " ".join(f"{c:>9s}" for c in CLASSES))
    for i, cls in enumerate(CLASSES):
        if table[i].sum() == 0:
            continue
        print(f"  {cls:>9s}   " + " ".join(f"{v:9d}" for v in table[i]) +
              f"   | recall {table[i, i] / table[i].sum():.3f}")
    return np.trace(table) / total


plain = t.plain_transform(96)
print("the model already on the page, leaf_model.onnx, trained in Task 2:")
score(t.LeafDataset(paths["val"], plain), "this year, val      (62 photos of your group's session)")
score(t.camera_like(96), "camera-like         (the same 62, cropped, rotated, recoloured)")
score(t.LeafDataset(paths["old"], plain), "last year, val      (120 photos, other students, other phones)")
