# Leaf Classifier — MLR Exercise 8, Task 2

A static web page that classifies leaves (**apple, cherry, chestnut, maple**) with the
`leaf_model.onnx` network exported in the Exercise 8 notebook. The model runs **entirely in
the browser** with ONNX Runtime Web (WebAssembly) — on Android and on iPhone, with no app
to install and no server-side inference.

* `index.html` — the whole application (HTML + CSS + JavaScript in one file)
* `leaf_model.onnx` — the trained model (unmodified copy of the file from the notebook)
* `model_info.json` — class names, input size, mean and standard deviation
* `check_image.jpg`, `check_expected.json` — the notebook's reference image and probabilities

The notebook, the model and the reference files are not modified by this project.

## Public address (GitHub Pages)

<!-- ===== GITHUB PAGES URL ===== -->
**Public URL:** <https://adamb4b.github.io/mlr-ex08/>
<!-- ============================ -->

Published from the public repository [AdamB4B/mlr-ex08](https://github.com/AdamB4B/mlr-ex08)
using **Settings → Pages → Deploy from a branch → main / (root)**. No collaborators have been added.

## Run it locally

Browsers refuse `fetch()` and camera access on `file://` pages, so serve the folder over HTTP:

```powershell
# in this folder
python -m http.server 8000
```

Then open <http://localhost:8000/index.html>.

*Model loading needs an internet connection* (ONNX Runtime Web and the QR encoder come from the
jsDelivr CDN). Camera access additionally needs a **secure context**: `http://localhost` is accepted
by browsers, any other address must be `https://`.

To try it on your phone before publishing, both devices must be on the same network and you need
HTTPS for the camera — the simplest correct path is to publish on GitHub Pages. Uploading a photo
works over plain HTTP on a LAN.

### Verify the numbers yourself

Open <http://localhost:8000/index.html?selftest=1>. The page runs `check_image.jpg` through the
same pipeline as an uploaded photo and prints a JSON report containing `logits`, `probabilities`,
`tensorShape`, a per-class comparison with `check_expected.json` and an overall `pass` flag.

Alternatively just click **Upload image** and choose `check_image.jpg`.

## Configure the QR code after publishing

The QR code at the bottom of the page is generated **in the browser** (library
`qrcode-generator`, inline SVG). It never uses an external QR-image service.

In `index.html`, near the top of the `<script>` block:

```js
const APP_URL = "";
```

* **Easiest — leave it empty.** When the page is served from `*.github.io`, the QR code
  automatically encodes the page's own public address. So after you publish, it is already correct.
* **Or set it explicitly**, which also makes the QR code correct in local testing:

  ```js
  const APP_URL = "https://USERNAME.github.io/mlr-ex08/";
  ```

Commit and push again after changing it. If neither is available, the QR panel says
“QR code not configured yet” instead of showing a code that points to the wrong place.
Scan the code with your phone and check that the encoded address matches the address bar.

## How the preprocessing matches the notebook

`model_info.json`: `input_size = 96`, `classes = [apple, cherry, chestnut, maple]`,
`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`.

The notebook runs `transforms.ToTensor()` followed by `transforms.Normalize(mean, std)` on
photographs that were already centre-cropped and resized to 96 × 96. The page reproduces this
in the function **`preprocess()`** in `index.html`, in this order:

1. central square crop of the frame/photo (`side = min(width, height)`),
2. resize to 96 × 96 with `drawImage(...)` on an offscreen canvas (`imageSmoothingQuality = 'high'`),
3. pixels read with `getImageData()`, alpha ignored → RGB,
4. each value divided by 255 (like `ToTensor()`),
5. `(value − mean[c]) / std[c]` per channel (like `Normalize()`),
6. written **channel-first** as `[1, 3, 96, 96]` — index `c*96*96 + y*96 + x`, matching PyTorch CHW.

The model's output is the raw score vector `logits` (shape `[1, 4]`); the page applies **softmax**
in `softmax()` and prints the classes in the order given by `model_info.json`.
The exact same `preprocess()`/`classify()` functions serve the live camera and the photo upload.

## Camera and performance notes

* Rear camera preferred (`facingMode: { ideal: 'environment' }`), front camera as browser fallback.
* The live loop uses `requestVideoFrameCallback` where available, a timer otherwise, classifies at
  most every 200 ms and never lets two inferences overlap (`busy` flag).
* `ort.env.wasm.numThreads = 1` is deliberate: GitHub Pages sends no COOP/COEP headers, so the page
  is not `crossOriginIsolated` and multi-threaded WebAssembly would fail there.
* First load downloads ~6 MB of model plus the ONNX Runtime Web WASM binary; both are then cached
  by the browser. WebAssembly SIMD is required — Safari 16.4+ / iOS 16.4+ and current Chrome/Firefox.
* The camera is released on **Stop camera**, when a photo is uploaded and when the page is left.

## Privacy

No image, frame or photo leaves the device. The only network requests are the two CDN libraries
(jsDelivr) and the model/reference files from this same site. No analytics, no tracking, no backend.
