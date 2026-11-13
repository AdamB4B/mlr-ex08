"""Turns results.csv into the table of REPORT.md, with the noise verdict per row.

    python analyze_results.py [results.csv] [images in one val/camera set, default 62] [--seeds 0,1,2]

For every variant it prints the mean of the seeds, the individual runs under it, the change
against the baseline in percentage points of accuracy, and whether that change is larger than
the binomial noise of a 62-image set (2 sigma = 11.0 pp) with all seeds moving the same way.
"""
import argparse
import math

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("csv", nargs="?", default="results.csv")
parser.add_argument("n", nargs="?", type=int, default=62, help="images in one val/camera set")
parser.add_argument("--seeds", default="", help="only these seeds, e.g. 0,1,2")
args = parser.parse_args()
CSV, N = args.csv, args.n

table = pd.read_csv(CSV)
if args.seeds:
    table = table[table["seed"].isin([int(s) for s in args.seeds.split(",")])]
COLUMNS = ["val", "camera", "last_year"]
LABELS = {"val": "val", "camera": "camera", "last_year": "last year"}

for column in COLUMNS:
    table[column] = table[column].astype(float)

means = table.groupby("variant")[COLUMNS].mean()
seeds = table.groupby("variant")["seed"].apply(lambda s: sorted(s.tolist()))
base_name = [v for v in means.index if v.startswith("baseline")][0]
base = means.loc[base_name]

# one parameter count per architecture, so the table says what each model costs
params = {}
try:
    import task4_variants as t

    def count(name):
        cfg = {**t.BASE, **t.VARIANTS[name]}
        key = (cfg["network"], cfg["size"], cfg["gap"], cfg["bn"])
        if key not in params:
            params[key] = sum(p.numel() for p in t.build_model(cfg).parameters())
        return params[key]
except Exception as error:                                        # noqa: BLE001
    print("(no parameter counts:", error, ")\n")
    count = None

sigma = {c: 100 * math.sqrt(0.75 * 0.25 / (N if c != "last_year" else 120)) for c in COLUMNS}
print(f"one image in a {N}-image set is {100 / N:.2f} pp;  "
      f"2 sigma: val/camera {2 * sigma['val']:.1f} pp, last year {2 * sigma['last_year']:.1f} pp\n")

print("| Variant | Params | val | camera | last year | change vs baseline | larger than the noise? |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for name in means.index:
    row, runs = means.loc[name], table[table["variant"] == name]
    cells = {}
    for c in COLUMNS:
        values = " / ".join(f"{v:.3f}" for v in runs[c])
        cells[c] = f"**{row[c]:.3f}**<br>{values}"
    change = " ".join(f"`{LABELS[c]}` {100 * (row[c] - base[c]):+.1f} pp" for c in COLUMNS)
    if name == base_name:
        verdict = "—"
    else:
        parts = []
        for c in COLUMNS:
            difference = 100 * (row[c] - base[c])
            same_way = all((v > base[c]) == (difference > 0) for v in runs[c]) if difference else False
            if abs(difference) > 2 * sigma[c] and same_way:
                parts.append(f"`{LABELS[c]}` {'+' if difference > 0 else '−'}"
                             f"{abs(difference):.1f} pp in every seed")
        verdict = "; ".join(parts) if parts else "no — inside 2 sigma or seeds disagree"
    param_cell = f"{count(name) / 1e3:.0f} k" if count else "—"
    print(f"| {name} | {param_cell} | {cells['val']} | {cells['camera']} | "
          f"{cells['last_year']} | {change} | {verdict} |")

print("\nspread between seeds of the same variant (max − min), in accuracy:")
print(table.groupby("variant")[COLUMNS].agg(lambda s: s.max() - s.min()).round(3).to_string())
