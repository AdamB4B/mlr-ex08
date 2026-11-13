"""Paired comparison of saved variants on the same images, with an exact McNemar test.

    python eval_paired.py "baseline (Task 2)" "+ colour + geometric" "+ all augmentation, cosine, 40 epochs"

Two accuracies on 62 images can differ by one image and look like a result. Comparing the two
models image by image removes that: the only images that matter are the ones where they
disagree, and the McNemar test asks whether the split of those disagreements is a coin flip.
The rows are found in results.csv, which also tells which weights-NN.pt belongs to each run.

For every seed the two runs share, and for every set (val, camera, last year), it prints

    b  = the first model right and the second wrong
    c  = the first model wrong and the second right
    p  = exact McNemar p-value (two-sided binomial on b and c)

p > 0.05 means the two models cannot be told apart on these images, whatever the accuracies say.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

import task4_variants as t


def weight_file(variant, seed):
    table = pd.read_csv("results.csv")
    hit = table[(table["variant"] == variant) & (table["seed"] == seed)]
    if hit.empty:
        sys.exit(f"{variant!r} seed {seed} is not in results.csv")
    path = f"weights-{hit.index[0]:02d}.pt"
    if not os.path.exists(path):
        sys.exit(f"{path} is missing")
    return path


def load(variant, seed):
    cfg = {**t.BASE, **t.VARIANTS[variant]}
    model = t.build_model(cfg)
    model.load_state_dict(torch.load(weight_file(variant, seed), map_location="cpu"))
    model.eval()
    return model, cfg


def correct(model, dataset):
    loader = t.DataLoader(dataset, batch_size=64, num_workers=0, pin_memory=False)
    hits = []
    with torch.no_grad():
        for images, labels in loader:
            predicted = model(images).argmax(dim=1)
            hits += (predicted == labels).tolist()
    return np.array(hits, dtype=bool)


def main():
    names = sys.argv[1:]
    if len(names) < 2:
        sys.exit(__doc__)
    first, others = names[0], names[1:]
    table = pd.read_csv("results.csv")
    seeds = sorted(set(table[table["variant"] == first]["seed"]) &
                   set.intersection(*[set(table[table["variant"] == n]["seed"]) for n in others]))
    print("paired McNemar test, first model =", first, "| seeds", seeds)

    for seed in seeds:
        model_a, cfg_a = load(first, seed)
        sets = {"val": t.LeafDataset(t.folders()["val"], t.plain_transform(cfg_a["size"])),
                "camera": t.camera_like(cfg_a["size"])}
        if t.folders()["old"]:
            sets["last year"] = t.LeafDataset(t.folders()["old"], t.plain_transform(cfg_a["size"]))
        hits_a = {name: correct(model_a, ds) for name, ds in sets.items()}
        for name in others:
            model_b, _ = load(name, seed)
            print(f"\nseed {seed}: {first}  vs  {name}")
            for set_name, ds in sets.items():
                hits_b = correct(model_b, ds)
                b = int((hits_a[set_name] & ~hits_b).sum())
                c = int((~hits_a[set_name] & hits_b).sum())
                p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
                print(f"  {set_name:10s} n={len(hits_a[set_name]):3d}  "
                      f"{hits_a[set_name].mean():.3f} vs {hits_b.mean():.3f}  "
                      f"only-first-right b={b:2d}  only-second-right c={c:2d}  "
                      f"p={p:.3f}  {'difference' if p < 0.05 else 'no difference at 5%'}")
    print("\nRead b and c, not the two accuracies: they are the images the models disagree on.")


if __name__ == "__main__":
    main()
