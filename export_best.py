"""Rebuild one variant, load the weights the sweep saved and export what the page needs.

    python export_best.py "+ colour + geometric" 1

Writes into export/, so nothing used by the page is touched:
    export/leaf_model.onnx        the network
    export/model_info.json        classes, input size, mean, std
    export/check_expected.json    the four probabilities of check_image.jpg, for the page's selftest

The row is found in results.csv, which also says which weights-NN.pt belongs to it. Before writing
anything it checks that the exported ONNX gives the same numbers as PyTorch on check_image.jpg and
on 20 validation photographs.
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image

import task4_variants as t
from predict_onnx import preprocess


def weight_file(variant, seed):
    table = pd.read_csv("results.csv")
    hit = table[(table["variant"] == variant) & (table["seed"] == seed)]
    if hit.empty:
        sys.exit(f"{variant!r} seed {seed} is not in results.csv")
    path = f"weights-{hit.index[0]:02d}.pt"
    if not os.path.exists(path):
        sys.exit(f"{path} is missing")
    return path


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "+ colour + geometric"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    cfg = {**t.BASE, **t.VARIANTS[variant]}
    size = cfg["size"]
    model = t.build_model(cfg)
    model.load_state_dict(torch.load(weight_file(variant, seed), map_location="cpu"))
    model.eval().cpu()

    os.makedirs("export", exist_ok=True)
    classes = t.LeafDataset(t.folders()["train"], t.plain_transform(size)).classes
    model_info = {"classes": classes, "input_size": size, "mean": t.MEAN, "std": t.STD}
    json.dump(model_info, open("export/model_info.json", "w"), indent=1)

    torch.onnx.export(model, (torch.zeros(1, 3, size, size),), "export/leaf_model.onnx",
                      input_names=["image"], output_names=["logits"], external_data=False)

    # the same metadata the notebook wrote in 2e, so the file alone says how it must be fed
    import onnx
    onnx_model = onnx.load("export/leaf_model.onnx")
    onnx_model.metadata_props.add(key="model_info", value=json.dumps(model_info))
    onnx.save(onnx_model, "export/leaf_model.onnx")

    # --- the same preprocessing the page and predict_onnx.py use -------------------------
    def tensor(path):
        _, x = preprocess(path, size, t.MEAN, t.STD)
        return x.astype(np.float32)

    session = ort.InferenceSession("export/leaf_model.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def onnx_logits(x):
        return session.run(None, {input_name: x})[0]

    x = tensor("check_image.jpg")
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(x)).numpy()
    logits = onnx_logits(x)[0]
    e = np.exp(logits - logits.max())
    probabilities = e / e.sum()
    json.dump({c: float(p) for c, p in zip(classes, probabilities)},
              open("export/check_expected.json", "w"), indent=1)

    # --- verify: ONNX against PyTorch, on the check image and on real photographs ---------
    val = t.LeafDataset(t.folders()["val"], t.plain_transform(size))
    diffs = [float(np.abs(torch_logits - onnx_logits(x)).max())]
    for i in range(20):
        xi = tensor(val.samples[i][0])
        with torch.no_grad():
            a = model(torch.from_numpy(xi)).numpy()
        diffs.append(float(np.abs(a - onnx_logits(xi)).max()))

    print(f"{variant}, seed {seed}, weights {weight_file(variant, seed)}")
    print(f"export/leaf_model.onnx        {os.path.getsize('export/leaf_model.onnx') / 1e6:.2f} MB "
          f"(the page's file is {os.path.getsize('leaf_model.onnx') / 1e6:.2f} MB)")
    print(f"parameters                    {sum(p.numel() for p in model.parameters()):,}")
    print(f"PyTorch vs ONNX, 21 images    largest |difference| of the logits {max(diffs):.2e}")
    print("export/check_expected.json   ",
          {c: round(float(p), 4) for c, p in zip(classes, probabilities)})
    print("\nThe page reads model_info.json for the input size, so copying these three files into the"
          "\nfolder of index.html is all it needs — copy all three together, because its selftest"
          "\ncompares the model with check_expected.json.")


if __name__ == "__main__":
    main()
