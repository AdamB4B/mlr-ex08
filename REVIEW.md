# REVIEW.md — Task 2, the leaf classifier on the phone

Page reviewed: <https://adamb4b.github.io/mlr-ex08/>
Model: `leaf_model.onnx` from the notebook (Task 2), unmodified. Browser tests run on 13 November
2026 in Chromium and in WebKit (Safari's engine) on the desktop; the phone evidence is the
screenshot below.

## 1. It runs on my phone

![Screenshot from my phone: the live camera on the Pages address, apple 99.6 %](ScreenShot.png)

The screenshot is the published address in the address bar, the camera picture of a leaf with the
dashed centre-crop square over it, and the classification under it: **apple — 99.6 %**, with the
four probability bars below.

| What I checked | What I saw |
| --- | --- |
| Phone and browser | Android phone, Chrome, 12 November 2026 |
| Did the page load from the public address? | Yes — the address bar shows the GitHub Pages address |
| Did the camera start? | Yes — a live leaf fills the stage, with the dashed centre-crop square over it |
| Did it classify the picture? | Yes: **apple — 99.6 %**, with the four probability bars under it |
| Did the page need any change to work on the phone? | No — the file on the phone is the same one the tests below pass |

How fast it feels on a phone is not something a screenshot records and I did not time it there; the
desktop test measures **18–26 ms** per inference, and the page never starts a second inference while
one is running.

What I did **not** test: an iPhone. This page uses no iOS-only or Chrome-only feature, and WebKit —
the engine of Safari — passes every check in §2 and §4 below, but "works in Safari's engine on a
desktop" is not the same claim as "works on an iPhone", so I am not making the second one.

## 2. The same numbers as the notebook

I uploaded `check_image.jpg` with the **Upload image** button and read the four probabilities off the
page. `check_expected.json`, written by the notebook when it saved the model, is next to them:

| Class | Notebook (`check_expected.json`) | My page | Difference |
| --- | --- | --- | --- |
| apple | 0.983 | 0.983 | < 0.001 |
| cherry | 0.017 | 0.017 | < 0.001 |
| chestnut | 0.000 | 0.000 | < 0.001 |
| maple | 0.000 | 0.000 | < 0.001 |

They agree to three decimals in both browsers, against both the local copy of the page and the
deployed one. The exact largest difference reported by the page's own self test is **1.25e-06**,
and the tensor it built was **`1x3x96x96`**. The self test is built in: open
<https://adamb4b.github.io/mlr-ex08/?selftest=1>, the title changes to `SELFTEST PASS` and the block
at the bottom starts with `SELFTEST_JSON=`.

Three decimals on one image can also come out right for the wrong reason, so two extra probes were
uploaded:

* **centre crop** — a 288 × 96 strip with `check_image.jpg` in the middle and black on both sides.
  The centre square is exactly the leaf and 96 → 96 is not resampled, so the page must give the same
  numbers as `check_expected.json`. It does (largest difference **1.25e-06**). A page that cropped
  from a corner, or resized the whole strip, would not.
* **constant colours** — three 800 × 600 images filled with one colour each. Any crop or resize
  leaves a constant image constant, so the tensor is known exactly from the colour; the three are
  red-, green- and blue-dominant, so a wrong channel order would change the answer. Largest
  difference from the ONNX model: **7.95e-06**.

All of it is recorded in `tests/evidence/*.json` (every check, the full `SELFTEST_JSON`, every
network request, screenshots), produced by `python tests/verify_page.py --browser chromium|webkit`,
locally and against the deployed address.

## 3. The preprocessing matches my model

From `model_info.json`: input size **96**, mean **[0.485, 0.456, 0.406]**, std
**[0.229, 0.224, 0.225]**, classes in this order **apple, cherry, chestnut, maple**. The page never
hardcodes these — it fetches `model_info.json` and uses `info.input_size`, `info.mean[c]`,
`info.std[c]` and `info.classes`.

Everything is in one function, `preprocess()`, and the same function serves the live camera and an
uploaded photo (`classify()` is its only caller):

| Step | Where it is in `index.html` | Does it match the notebook? |
| --- | --- | --- |
| central square crop of the frame | `const side = Math.min(sourceWidth, sourceHeight)`, then `sx`/`sy` place the crop in the middle | yes — the dataset photographs are already square 96 × 96, so the same rule changes nothing there; on a camera frame it discards the sides, which is the only way to feed a square model |
| resize to 96 × 96 | one `workCtx.drawImage(source, sx, sy, side, side, 0, 0, size, size)` — crop and resize together, `size` is `info.input_size` | yes; the model's input is 96 × 96 |
| pixels as RGB | `getImageData(0, 0, size, size)`, an RGBA array read as `rgba[i * 4 + c]` for c = 0,1,2, so alpha is dropped | yes, `Image.open(...).convert("RGB")` / `ToTensor()` |
| divide by 255 (`ToTensor`) | `rgba[i * 4 + c] / 255` | yes |
| standardize with mean and std (`Normalize`) | `( … / 255 − mean) / std`, with `mean`/`std` per channel from `model_info.json` | yes, `transforms.Normalize(mean, std)` with the same ImageNet numbers |
| channel order, shape `[1, 3, 96, 96]` | the channel loop is outside the pixel loop: `offset = c * area`, written as `data[offset + i]`, then `new ort.Tensor('float32', data, [1, 3, size, size])` | yes — PyTorch CHW, `ToTensor()` also puts channels first |
| softmax over the raw logits | `softmax()` after `session.run` returns the raw `logits` | yes — the model has no softmax layer, because `nn.CrossEntropyLoss` applies it during training |

The order is exactly the notebook's `transforms.Compose([ToTensor(), Normalize(mean, std)])`:
crop → resize → divide by 255 → subtract the mean and divide by the std per channel → channels
first. The probes in §2 are what turns this table from a reading of the code into a measurement.

## 4. Nothing leaves the phone

I captured every request the page makes — once with the local copy and once with the deployed
address, in both browsers. A normal page load makes **six GET requests** and nothing else:

| URL the page loads | Why | Does it receive my image? |
| --- | --- | --- |
| `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.min.js` | the library that runs the network | no — a fixed script file |
| `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort-wasm-simd-threaded.jsep.{mjs,wasm}` | its WebAssembly binary, loaded when the session is created | no |
| `https://cdn.jsdelivr.net/npm/qrcode-generator@1.4.4/qrcode.min.js` | draws the QR code locally as inline SVG | no |
| `./model_info.json`, `./leaf_model.onnx` | classes, input size, mean, std, and the model itself | no — the model comes **to** the phone (6.3 MB, cached) |

`?selftest=1` additionally fetches `./check_expected.json` and `./check_image.jpg` from the same
site. No request is anything other than GET, and no request body exists at all.

A camera frame goes `video` → `drawImage` into an offscreen canvas → a `Float32Array` → an
`ort.Tensor` → `session.run()` — all in the page's memory. An uploaded photo is handled the same way;
the browser creates a local `blob:` URL to display it, which is memory inside the page, not traffic
(the tests log it separately from the network requests for exactly this reason). The session is
created with `executionProviders: ['wasm']`, so the arithmetic is done by WebAssembly on the phone.

One deliberate restriction: `ort.env.wasm.numThreads = 1`. GitHub Pages sends no COOP/COEP headers,
so the page is not `crossOriginIsolated` and multi-threaded WebAssembly would fail there. One thread
is the correct setting, not a bug.

## What had to be fixed

* **In the page: nothing.** The deployed `index.html` passed every check above without a change —
  decoding, crop, resize, channel order, normalisation, softmax, the QR code and the live loop.
* **What was missing was proof.** The page had a self test, but `check_image.jpg` is already 96 × 96,
  so the crop and resize were never exercised by it, and the same test had been looked at only on a
  phone. This run adds `tests/`: the probes in §2, the same checks against the deployed address in
  Chromium and WebKit, the QR decoded from the rendered SVG, the full request log, and the two
  failure paths (a missing model, a refused camera). All **129** checks in `tests/evidence/` pass:
  95 on the page this review is about (17 × 4 — local and deployed, in Chromium and WebKit — plus
  20 for the live camera loop and 7 for the two failure paths), and 34 more on the improved model
  that Task 4 exports separately (`tests/summarise_evidence.py` prints the table).

## What I would still check on a real phone

Two things that no desktop test can answer: that the rear camera is chosen on an iPhone (the page
asks for `facingMode: { ideal: 'environment' }` and falls back to any camera), and how the page feels
when the leaf moves — the desktop fake-camera loop classifies every ~20 ms and never overlaps two
inferences, but that is a desktop CPU.
