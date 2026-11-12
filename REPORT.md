# REPORT.md — Task 4, making the leaf classifier better

> **Write this file yourself.** The notebook says the report is yours; the agent may only write the
> code. What is below is a skeleton: the `TODO` cells are for the numbers your runs produce, and the
> two tables under "Where we start" and "How big is the noise" are real measurements that were
> already made and can be checked by the commands at the end. Delete the `> note:` lines before
> you push.

Goal: the best classifier **on the phone camera**, not the best number on the validation folder.
Every change below is one change at a time against one fixed baseline, on the same images, with a
fixed seed and repeated seeds.

---

## 1. The baseline and how it is measured

Baseline: `TODO` — the Task 2 model (`mobilenet_v3_small`, whole network fine-tuned, Adam 1e-3,
15 epochs) / the Task 3 `LeafNet`. Write which one, because every row is compared with it.

Code that produced the table: `task4_variants.py`, `RUN = TODO`, seeds `TODO`.

Each variant is scored on three sets, all identical for every row:

| Column | What it is | Why it is there |
| --- | --- | --- |
| **val** | this year's validation photographs | the number the notebook prints |
| **camera** | the same photographs cropped, rotated and recoloured once, fixed seed 7 | the closest thing we have to the phone camera without taking new photos |
| **last year** | last year's public photos, 30 per species, other students, other phones | the only set nobody in the group has ever photographed |

## 2. Where we start (measured before any change)

The exported `leaf_model.onnx` from Task 2, run unchanged through ONNX Runtime on last year's
photographs — i.e. the model that is on my page today, tested on leaves nobody in my group
photographed. 820 images, of which the model has seen none.

| Set | Accuracy |
| --- | --- |
| last year, val (120 images) | **0.642** (77 / 120) |
| last year, train (700 images) | 0.624 (437 / 700) |

| truth \ pred | apple | cherry | chestnut | maple | recall |
| --- | --- | --- | --- | --- | --- |
| apple | 24 | 3 | 2 | 1 | 0.800 |
| cherry | **22** | 4 | 0 | 4 | **0.133** |
| chestnut | 0 | 2 | 22 | 6 | 0.733 |
| maple | 3 | 0 | 0 | 27 | 0.900 |

So the classifier is not weak everywhere: **cherry is being read as apple**, 22 times out of 30.
`TODO` — write why you think that is, and which of your changes is aimed at it.

## 3. How big is the noise

The validation set has 120 images, so **one image is 0.83 percentage points**. With 120 images and
an accuracy around 0.64, the binomial spread of a single measurement is about **±4.4 pp for 1σ,
±8.8 pp for 2σ**. To call a 5 pp improvement real you would need about 370 images, not 120.

| n | one image | 1σ around 0.64 | 2σ around 0.64 |
| --- | --- | --- | --- |
| 120 (val, camera, last-year-val) | 0.83 pp | 4.4 pp | 8.8 pp |
| 700 (last-year-train) | 0.14 pp | 1.8 pp | 3.6 pp |

`TODO` — say in your own words: which rows of your table are therefore **not** a difference, only
a different roll of the same dice, and how repeating the seeds helped you decide.

## 4. What I tried

One row per change; for a row with two seeds write both numbers, not only the mean.

| Variant | one change from the baseline | val | camera | last year | larger than the noise? | kept? |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | — | `TODO` | `TODO` | `TODO` | — | — |
| + colour jitter | `ColorJitter(0.3,0.3,0.3,0.05)` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| + geometric augmentation | crop / flips / ±20° | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| + colour + geometric | both | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| + frozen backbone | only the new last layer trains | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| + lr 1e-4, cosine, 30 epochs, best epoch | smaller steps, longer | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| + augmentation, cosine, 40 epochs | combination | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| + label smoothing | 0.1 | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| + weight decay | AdamW 1e-4 | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| + class weights | cherry gets a louder loss | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| + last year's photos in training | +700 images | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| + input 160 × 160 | larger tensor | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| Task 3 `LeafNet` from scratch | no ImageNet | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| `LeafNet` + BN + augmentation | Task 3 with both | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |

> note: two rows are already expected to be honest negatives. The photographs we are given, this
> year's and last year's, are **already 96 × 96**, so a 160 × 160 input can only interpolate what
> is there. Write that down as a measured negative rather than dropping the row.

## 5. What each change brought

`TODO` — one short paragraph per change that mattered: what you expected, what came out, whether
the seeds agreed. Quote the table, do not repeat it.

## 6. What did **not** help

`TODO` — this is a required part of the report. Include at least the changes that made the
validation number go up while the camera and last-year numbers did not, and say what you conclude
from that.

## 7. The model I would put on the phone

`TODO` — name the variant, give its camera and last-year accuracy against the baseline, and say
what it costs: file size `TODO` MB, how long the first load and one classification take on your
phone `TODO`.

Exported with `export_to_onnx()` as in 2e; `index.html` reads the input size and the class names
from `model_info.json`, so `leaf_model.onnx` and `model_info.json` are the only two files the page
needs. Before replacing the model on the page: `TODO` — say what you checked on the phone with the
new model, and note that `check_expected.json` no longer belongs to it and has to be regenerated.

## 8. What I would try next

`TODO` — two or three sentences.

---

## Appendix — how to repeat these numbers

```powershell
# the starting point of section 2: the exported model against last year's photographs
python eval_lastyear.py
```

```python
# the table of section 4
# Colab, T4 GPU
import task4_variants as t
t.GROUP_ZIP = "https://drive.google.com/file/d/.../view?usp=sharing"
t.run_rows(t.RUN[:6], seeds=[0, 1])     # writes results.csv
```
