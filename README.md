# MLR Exercise 8 — leaf classifier on the phone (Task 2) and the Task 4 experiments

Four species from a phone photograph: **apple, cherry, chestnut, maple**. The network is the
notebook's `mobilenet_v3_small`, fine-tuned on 244 photographs and exported to ONNX, and it runs
**in the browser** with ONNX Runtime Web (WebAssembly) — nothing is sent to a server.

This folder is self-contained: the page, the model, the reference files, the Task 4 experiments, the
tests and their recorded output.

* `index.html` — the whole application (HTML + CSS + JavaScript in one file), byte-identical to the
  page served at <https://adamb4b.github.io/mlr-ex08/>
* `leaf_model.onnx`, `model_info.json`, `check_image.jpg`, `check_expected.json` — the four files the
  notebook produced in Task 2, unmodified
* `ScreenShot.png` — the phone screenshot used in `REVIEW.md`
* `REVIEW.md` — the Task 2 review, with the numbers measured by the tests below
* `REPORT.md` — the Task 4 report
* `task4_variants.py`, `run_task4.py`, `analyze_results.py`, `eval_paired.py`, `eval_variant.py`,
  `eval_page_model.py`, `eval_lastyear.py`, `export_best.py` — the Task 4 experiments (not used by the
  page)
* `predict_onnx.py` — one image through `leaf_model.onnx` with the notebook's preprocessing
* `export/` — the improved Task 4 model (`leaf_model.onnx`, `model_info.json`, `check_expected.json`)
  and `export/site/`, a copy of the page that loads it
* `tests/` — the browser tests and their evidence (`tests/evidence/*.json`, `*.png`)
* `data/`, `weights-*.pt` — the photographs and the saved weights (kept locally, not part of the
  hand-in)
* `results.csv`, `analysis.txt`, `task4_run*.log` — what the experiments measured and the output they
  printed

## Requirements

Only needed to re-run the experiments and the tests; the page itself installs nothing.

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium webkit     # the two browsers the tests drive
```

## Run the page locally

Browsers refuse `fetch()` and the camera on `file://` pages, so serve the folder:

```powershell
python -m http.server 8000
```

then open <http://localhost:8000/index.html>. Camera access needs a secure context: `http://localhost`
counts, any other address must be `https://`.

The page loads three things from the jsDelivr CDN (ONNX Runtime Web, its WebAssembly binary, and the
QR encoder) and everything else from its own site. The model is ~6.3 MB and is cached by the browser
after the first load.

## Verify it yourself

The page has a built-in self test that runs `check_image.jpg` through exactly the same upload
pipeline and prints a JSON report:

<https://adamb4b.github.io/mlr-ex08/?selftest=1> — the title becomes `SELFTEST PASS` and the block at
the bottom starts with `SELFTEST_JSON=`.

The same checks, plus the ones the page cannot do itself, are automated:

```powershell
python tests/make_probes.py                              # the exact-tensor probe images
python tests/verify_page.py --browser chromium           # local copy of the page
python tests/verify_page.py --browser webkit             # Safari's engine, desktop
python tests/verify_page.py --browser chromium --url https://adamb4b.github.io/mlr-ex08/index.html
python tests/verify_page.py --browser chromium --fake-camera   # the live loop, fake video device
python tests/verify_errors.py                            # missing model, refused camera
```

Each run writes `tests/evidence/<browser>-<tag>.json` with every check, the full `SELFTEST_JSON`,
every network request and a screenshot. What was measured on 13 November 2026:

| Check | Result |
| --- | --- |
| `?selftest=1` vs `check_expected.json`, Chromium and WebKit, local and live | `SELFTEST PASS`, largest difference **1.25e-06** |
| the four probabilities to three decimals | apple 0.983, cherry 0.017, chestnut 0.000, maple 0.000 |
| the same numbers through the real **Upload image** button | apple 98.3 %, cherry 1.7 %, chestnut 0.0 %, maple 0.0 % |
| centre crop: a 288x96 strip whose middle square *is* `check_image.jpg` | identical probabilities (1.25e-06) |
| three constant-colour images (exact tensor, catches channel order) | largest difference **7.95e-06** |
| QR code: the page's own SVG screenshotted and decoded with zxing-cpp | decodes to `https://adamb4b.github.io/mlr-ex08/` |
| network, whole page load | 6 GET requests, 2 hosts (`adamb4b.github.io`, `cdn.jsdelivr.net`), no writes |
| live camera loop (fake device, Chromium) | 18–26 ms per frame, Stop releases the camera |
| missing `leaf_model.onnx` | "Could not load ./leaf_model.onnx (HTTP 404)", no probabilities shown |
| camera permission refused | the page explains it and the upload still classifies |

**Not tested:** a physical iPhone. The iOS claim rests on the code (no iOS-only feature, WebAssembly
SIMD, single-threaded WASM) and on WebKit — Safari's engine — passing every check on the desktop.
The only real-phone evidence in this hand-in is `ScreenShot.png`, taken on an Android phone.

## The QR code

It is drawn **in the browser** by `qrcode-generator` from the jsDelivr CDN — inline SVG, no QR-image
service, no request carrying the URL. The address it encodes is `APP_URL` at the top of the script in
`index.html`, set to the public address, so the QR is correct even when the file is opened from a
local server. `tests/verify_page.py` decodes the rendered code and fails if it is not exactly the
public URL.

## Task 4: the experiments

```powershell
python run_task4.py                    # every variant, three seeds -> results.csv + weights-NN.pt
python analyze_results.py              # the table with the change against the baseline
python eval_paired.py "baseline (Task 2)" "+ colour + geometric"
python eval_page_model.py              # the model that is on the page, on the same three sets
python export_best.py "+ colour + geometric" 1   # -> export/leaf_model.onnx + its own reference
```

Every variant is the Task 2 recipe with **one** key changed; the three scoring sets are built once
and shared by every row:

| Column | What it is |
| --- | --- |
| `val` | 62 of this year's validation photographs |
| `camera` | the same 62, cropped, rotated ±30° and recoloured once with fixed seed 7 |
| `last year` | 120 public photographs from last year's session |

Seeds are fixed per run (`torch.manual_seed`), and a repeated run of `baseline (Task 2)` seed 0 in a
fresh process reproduced 0.774 / 0.694 / 0.550 exactly. `results.csv` is the single source of every
number in `REPORT.md`.
