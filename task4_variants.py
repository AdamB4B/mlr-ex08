# Task 4 — one change at a time, measured against one fixed baseline.
#
# Run in Colab on a T4 (Runtime -> Change runtime type -> T4 GPU), after filling in GROUP_ZIP.
# Every variant is the baseline with ONE key changed, trained with SEEDS, and scored on three
# sets that are identical for every row:
#   val        your group's validation images (other students' leaves, your photo session)
#   camera     those same images, cropped/rotated/recoloured once with a fixed seed
#   last year  last year's public photos, which no variant trains on unless it says so
# The rows are printed and written to results.csv for REPORT.md.
#
# Note on size: every photograph we are given, this year's and last year's, is already
# 96 x 96, so a larger input can only interpolate. The size row is in the table to measure
# that, not because it is expected to help.

import copy
import json
import os
import urllib.request
import zipfile
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GROUP_ZIP = ""   # the teacher's shared Google Drive ZIP file link, the same as in Task 2
ARCHIVE_ZIP = ("https://raw.githubusercontent.com/tomasvicar/MLR-public/"
               "master/exercises/data/ex08_leaves_images.zip")
SEEDS = [0, 1]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
GROUP = "data/leaves"                       # this year: train/ val/ with one folder per class
OLD = "data/leaves_old/leaves"              # last year, unpacked one level deeper
RUN_ON = "group"                            # "group" needs GROUP_ZIP, "archive" uses only OLD


def folders():
    """The folders a run trains and scores on. With RUN_ON = "archive" the training data is
    last year's photographs and the val set is last year's val set, so the honest numbers are
    val and camera only — there is no third, unseen session left to score on."""
    if RUN_ON == "archive":
        return {"train": f"{OLD}/train", "val": f"{OLD}/val", "old": None}
    return {"train": f"{GROUP}/train", "val": f"{GROUP}/val", "old": f"{OLD}/val"}


def loader_kwargs():
    # Windows spawns dataloader workers instead of forking them, which is slower than just
    # reading the images in the main process; and pinning memory only helps a GPU
    return dict(batch_size=BASE["batch"], num_workers=2 if device.type == "cuda" else 0,
                pin_memory=device.type == "cuda")

# -------------------------------------------------------------------------------- data


def prepare_data():
    if not os.path.exists(f"{OLD}/train"):
        # the archive unpacks to leaves/train and leaves/val, so it has to land in its own folder
        os.makedirs(OLD, exist_ok=True)
        urllib.request.urlretrieve(ARCHIVE_ZIP, "data/leaves_old.zip")
        with zipfile.ZipFile("data/leaves_old.zip") as archive:
            archive.extractall("data/leaves_old")
        open("data/leaves_old/.archive_url", "w").write(ARCHIVE_ZIP)
    if RUN_ON == "archive":
        print(f"RUN_ON = archive: training and validating on last year's photos in {OLD}, "
              f"this year's group data is not used")
        return
    if not os.path.exists(f"{GROUP}/.group_zip_url"):
        # a ZIP you dropped in yourself wins, so you can hand over the data without a Drive link:
        # put the teacher's group ZIP next to this file as data/leaves.zip
        if os.path.exists("data/leaves.zip"):
            os.makedirs("data", exist_ok=True)
            with zipfile.ZipFile("data/leaves.zip") as archive:
                archive.extractall("data")
            open(f"{GROUP}/.group_zip_url", "w").write("local:data/leaves.zip")
        else:
            import gdown                         # only this branch needs it: %pip install gdown
            link = urlparse(GROUP_ZIP.strip())
            assert link.hostname == "drive.google.com", "paste the Drive file link into GROUP_ZIP"
            os.makedirs("data", exist_ok=True)
            gdown.download(url=GROUP_ZIP, output="data/leaves.zip", quiet=True)
            with zipfile.ZipFile("data/leaves.zip") as archive:
                archive.extractall("data")
            open(f"{GROUP}/.group_zip_url", "w").write(GROUP_ZIP)


def train_transform(size, augment):
    steps = []
    if "crop" in augment:                     # geometric: the leaf is not always centred
        steps += [transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
                  transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                  transforms.RandomRotation(20)]
    if "colour" in augment:                   # the phone changes light far more than shape
        steps += [transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


def plain_transform(size):
    # what the web page does: centre square crop, resize, /255, standardize
    def centre_square(img):
        side = min(img.size)
        return img.crop(((img.width - side) // 2, (img.height - side) // 2,
                         (img.width - side) // 2 + side, (img.height - side) // 2 + side))
    return transforms.Compose([transforms.Lambda(centre_square),
                               transforms.Resize((size, size)),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class LeafDataset(Dataset):
    """One folder per class; the class names are the sorted subfolder names (as in Task 3)."""

    def __init__(self, folder, transform):
        self.transform = transform
        self.classes = sorted(name for name in os.listdir(folder)
                              if os.path.isdir(os.path.join(folder, name)))
        self.samples = [(os.path.join(folder, c, f), i)
                        for i, c in enumerate(self.classes)
                        for f in sorted(os.listdir(os.path.join(folder, c)))
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), label


_CAMERA = {}


def camera_like(size, seed=7):
    # this year's validation photographs seen the way a phone camera sees them: cropped,
    # rotated and recoloured ONCE, with a fixed seed, so every variant is scored on
    # exactly the same pixels (built per input size, and cached)
    if size in _CAMERA:
        return _CAMERA[size]
    torch.manual_seed(seed)
    ds = LeafDataset(f"{folders()['val']}", transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)), transforms.RandomRotation(30),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)]))
    pairs = [ds[i] for i in range(len(ds))]
    _CAMERA[size] = TensorDataset(torch.stack([p[0] for p in pairs]),
                                  torch.tensor([p[1] for p in pairs]))
    return _CAMERA[size]


# ----------------------------------------------------------------------------- network


class LeafNet(nn.Module):
    """Task 3's network: four conv blocks, flatten, two linear layers."""

    def __init__(self, size=96, bn=False, gap=False, dropout=0.0):
        super().__init__()
        layers, in_ch, side = [], 3, size
        for ch in (16, 32, 64, 64):
            layers += [nn.Conv2d(in_ch, ch, 3, padding=1)]
            if bn:
                layers += [nn.BatchNorm2d(ch)]
            layers += [nn.ReLU(), nn.MaxPool2d(2)]
            in_ch, side = ch, side // 2
        self.features = nn.Sequential(*layers)
        self.gap = gap
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_ch if gap else in_ch * side * side, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.features(x)
        x = x.mean((2, 3)) if self.gap else x.flatten(1)
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


def build_model(cfg):
    if cfg["network"] == "mobilenet":
        model = models.mobilenet_v3_small(weights="DEFAULT")
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
        if cfg["freeze"]:                     # only the new last layer is allowed to learn
            for name, p in model.named_parameters():
                p.requires_grad = "classifier.3" not in name
    else:
        model = LeafNet(size=cfg["size"], bn=cfg["bn"], gap=cfg["gap"], dropout=cfg["dropout"])
    return model.to(device)


# --------------------------------------------------------------------------- training


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


def train_one(cfg, seed):
    torch.manual_seed(seed)
    size, paths = cfg["size"], folders()

    train_sets = [LeafDataset(paths["train"], train_transform(size, cfg["augment"]))]
    if cfg["extra_data"]:                     # last year's photographs added to training
        train_sets.append(LeafDataset(f"{OLD}/train", train_transform(size, cfg["augment"])))
    train_ds = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    kwargs = loader_kwargs()
    train_loader = DataLoader(train_ds, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), **kwargs)
    loaders = {"val": DataLoader(LeafDataset(paths["val"], plain_transform(size)), **kwargs),
               "camera": DataLoader(camera_like(size), **kwargs)}
    if paths["old"]:
        loaders["old"] = DataLoader(LeafDataset(paths["old"], plain_transform(size)), **kwargs)
    scored = ("val", "camera", "old") if paths["old"] else ("val", "camera")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    if cfg["class_weights"]:                  # the class the network keeps losing gets a louder loss
        labels = [s[1] for s in train_sets[0].samples]
        counts = np.bincount(labels, minlength=len(train_sets[0].classes)).astype("float64")
        weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device),
                                        label_smoothing=cfg["label_smoothing"])
        print("    class weights", weights.numpy().round(2))

    model = build_model(cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if cfg["optimizer"].startswith("sgd"):     # plain SGD is enough for a head on frozen features
        optimizer = torch.optim.SGD(trainable, lr=cfg["lr"], momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
                 if cfg["scheduler"] else None)

    best, best_state = -1.0, None
    for epoch in range(cfg["epochs"]):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()
        if scheduler:
            scheduler.step()
        acc = evaluate(model, loaders["val"])
        if cfg["best_epoch"] and acc > best:  # keep the weights of the best epoch, not of the last
            best, best_state = acc, copy.deepcopy(model.state_dict())
        if (epoch + 1) % 10 == 0 or epoch + 1 == cfg["epochs"]:
            print(f"    epoch {epoch + 1:3d}  training accuracy on val {acc:.3f}")
    if cfg["best_epoch"]:
        model.load_state_dict(best_state)
    return model, {name: evaluate(model, loaders[name]) for name in scored}


# ----------------------------------------------------------------------------- variants

BASE = dict(network="mobilenet", size=96, epochs=15, batch=32, lr=1e-3, optimizer="adam",
            weight_decay=0.0, scheduler=False, augment=(), freeze=False, bn=False, gap=False,
            dropout=0.0, class_weights=False, label_smoothing=0.0, best_epoch=False,
            extra_data=False)

VARIANTS = {
    # baseline: Task 2 exactly, so every row below is also comparable with your Task 2 run
    "baseline (Task 2)": {},
    "+ colour jitter": dict(augment=("colour",)),
    "+ geometric augmentation": dict(augment=("crop",)),
    "+ colour + geometric": dict(augment=("colour", "crop")),
    "+ frozen backbone, head only": dict(freeze=True, lr=0.1, optimizer="sgd_frozen", epochs=20),
    "+ backbone lr 1e-4, cosine, best epoch": dict(
        lr=1e-4, scheduler=True, epochs=30, best_epoch=True),
    "+ all augmentation, cosine, 40 epochs": dict(
        augment=("colour", "crop"), lr=1e-4, scheduler=True, epochs=40, best_epoch=True),
    "+ label smoothing": dict(label_smoothing=0.1),
    "+ weight decay 1e-4": dict(weight_decay=1e-4),
    "+ class weights": dict(class_weights=True),
    "+ last year's photos added to training": dict(extra_data=True),
    "+ last year, augmentation, cosine, best epoch": dict(
        extra_data=True, augment=("colour", "crop"), lr=1e-4, scheduler=True, epochs=40,
        best_epoch=True),
    "+ input 160 x 160": dict(size=160),
    # the network of Task 3, trained from scratch, for scale
    "Task 3 LeafNet from scratch": dict(network="leafnet"),
    "LeafNet + BN + augmentation": dict(network="leafnet", bn=True, augment=("colour", "crop")),
}

# edit this list to run fewer rows first; a row is ~1-3 minutes on a T4
RUN = list(VARIANTS)


def run_rows(run=RUN, seeds=SEEDS):
    prepare_data()
    has_old = folders()["old"] is not None
    columns = ["val", "camera", "last_year"] if has_old else ["val", "camera"]
    rows = []
    for name in run:
        cfg = {**BASE, **VARIANTS[name]}
        if not has_old and cfg["extra_data"]:
            print(f"\n{name}: skipped, last year's photos are the only data in this run")
            continue
        for seed in seeds:
            print(f"\n{name}  seed {seed}")
            model, acc = train_one(cfg, seed)
            row = dict(variant=name, seed=seed, val=round(acc["val"], 3),
                       camera=round(acc["camera"], 3))
            if has_old:
                row["last_year"] = round(acc["old"], 3)
            rows.append(row)
            print("  -> " + "   ".join(f"{k} {row[k]:.3f}" for k in columns))
            torch.save(model.state_dict(), f"weights-{len(rows) - 1:02d}.pt")
            pd.DataFrame(rows).to_csv("results.csv", index=False)   # save after every row
    table = pd.DataFrame(rows)
    table.to_csv("results.csv", index=False)
    print("\nmean over seeds:\n", table.groupby("variant")[columns].mean().round(3).to_string())
    return table


# ----------------------------------------------------------- export the best one to ONNX


def export_to_onnx(model, classes, size, path="leaf_model_best.onnx"):
    """Export as in 2e. index.html takes the input size and the classes from model_info.json,
    so replacing leaf_model.onnx and model_info.json by these two files is all the page needs."""
    model_info = {"classes": list(classes), "input_size": size, "mean": MEAN, "std": STD}
    model.eval().cpu()
    torch.onnx.export(model, (torch.zeros(1, 3, size, size),), path,
                      input_names=["image"], output_names=["logits"], external_data=False)
    with open("model_info_best.json", "w") as f:
        json.dump(model_info, f)
    print(f"{path} and model_info_best.json written; upload both to the repository, "
          f"renamed to leaf_model.onnx and model_info.json")


if __name__ == "__main__":
    run_rows()
