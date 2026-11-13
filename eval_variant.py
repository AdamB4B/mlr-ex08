"""Confusion matrices of one saved row, on the same three sets every row is scored on.

    python eval_variant.py "+ colour + geometric" 1

Reads the row from results.csv to find its weights, so the answer can be checked against the
failure the row was supposed to fix (apple against cherry).
"""
import sys

import numpy as np
import pandas as pd
import torch

import task4_variants as t


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "baseline (Task 2)"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    table = pd.read_csv("results.csv")
    hit = table[(table["variant"] == variant) & (table["seed"] == seed)]
    if hit.empty:
        sys.exit(f"{variant!r} seed {seed} is not in results.csv")
    path = f"weights-{hit.index[0]:02d}.pt"

    cfg = {**t.BASE, **t.VARIANTS[variant]}
    model = t.build_model(cfg)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    classes = t.LeafDataset(t.folders()["train"], t.plain_transform(cfg["size"])).classes

    def score(dataset, label):
        loader = t.DataLoader(dataset, batch_size=64, num_workers=0, pin_memory=False)
        matrix = np.zeros((len(classes), len(classes)), int)
        with torch.no_grad():
            for images, labels in loader:
                for truth, predicted in zip(labels.numpy(), model(images).argmax(1).numpy()):
                    matrix[truth, predicted] += 1
        print(f"\n{label}: {np.trace(matrix)} / {matrix.sum()} = {np.trace(matrix) / matrix.sum():.3f}")
        print("             " + " ".join(f"{c:>9s}" for c in classes))
        for i, cls in enumerate(classes):
            if matrix[i].sum():
                print(f"  {cls:>9s}   " + " ".join(f"{v:9d}" for v in matrix[i]) +
                      f"   | recall {matrix[i, i] / matrix[i].sum():.3f}")

    print(f"{variant}, seed {seed}, weights {path}")
    score(t.LeafDataset(t.folders()["val"], t.plain_transform(cfg["size"])), "this year, val ")
    score(t.camera_like(cfg["size"]), "camera-like    ")
    if t.folders()["old"]:
        score(t.LeafDataset(t.folders()["old"], t.plain_transform(cfg["size"])), "last year, val ")


if __name__ == "__main__":
    main()
