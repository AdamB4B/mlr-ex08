# Reads results.csv and turns it into the REPORT.md table, with the noise verdict per row.
import math
import sys

import pandas as pd

csv = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
table = pd.read_csv(csv)
n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 62      # images in the set each column scores

means = table.groupby("variant")[["val", "camera", "last_year"]].mean()
spreads = table.groupby("variant")[["val", "camera", "last_year"]].agg(lambda s: s.max() - s.min())
base = means.loc[[i for i in means.index if i.startswith("baseline")][0]]
counts = table.groupby("variant").size()

print(f"seeds per row: {counts.min()}-{counts.max()}   "
      f"binomial 2 sigma: val/camera n={n_val} -> "
      f"{200 * math.sqrt(0.75 * 0.25 / n_val):.1f} pp,   last year n=120 -> "
      f"{200 * math.sqrt(0.55 * 0.45 / 120):.1f} pp\n")

print("| variant | seeds | val | camera | last year | best column, change vs baseline |")
print("| --- | --- | --- | --- | --- | --- |")
for name in means.index:
    row = means.loc[name]
    parts = []
    for col in ("val", "camera", "last_year"):
        d = 100 * (row[col] - base[col])
        parts.append(f"{col} {d:+.1f} pp")
    best = max(("camera", row["camera"] - base["camera"]),
               ("last_year", row["last_year"] - base["last_year"]), key=lambda p: p[1])
    print(f"| {name} | {counts[name]} | {row['val']:.3f} | {row['camera']:.3f} "
          f"| {row['last_year']:.3f} | {' / '.join(parts)} |")

print("\nrun-to-run spread of the SAME variant between seeds (max - min):")
print(spreads.round(3).to_string())
print("\nbaseline per seed:")
print(table[table["variant"].str.startswith("baseline")].to_string(index=False))
