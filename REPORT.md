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

**Baseline: the Task 2 recipe, retrained** — `mobilenet_v3_small` pretrained on ImageNet, last layer
replaced by 4 outputs, the **whole** network fine-tuned with Adam at lr 1e-3, batch 32, 15 epochs,
no augmentation. It is retrained here instead of reusing `leaf_model.onnx` so that the baseline and
every variant come from the same code with the same seeds; §2 shows the model that is on the page
today, and the two agree (0.774 on this year's validation set for the page model, 0.742 / 0.774 /
0.790 for the retrained baseline over three seeds).

Code: `task4_variants.py`, `RUN = all 13 variants`, `SEEDS = [0, 1, 2]`, run on the CPU, about
40 minutes. One change from the baseline per row. The data and the three scoring sets are the same
for every row:

| Column | What it is | Why it is there |
| --- | --- | --- |
| **val** | 62 of this year's validation photographs | the number the notebook prints, and the least reliable one |
| **camera** | the same 62 photographs, cropped, rotated ±30° and recoloured once with fixed seed 7 | the closest thing we have to the phone camera without going out and taking new photos |
| **last year** | 120 public photographs from last year — other students, other phones, other weather | the only set this year's model has never had any contact with |

The `camera` and `last year` columns are generated identically for every variant, so a row that
wins on them really generalises better; a row that wins only on `val` has mostly learned more about
62 photographs.

## 2. Where we start (measured before any change)

`eval_page_model.py` scores the model that is on the page today — `leaf_model.onnx` from Task 2,
unmodified — on the same three sets that every variant below is scored on. Nothing was retrained
for this table; it is the model as it stands.

| Set | Accuracy | What fails |
| --- | --- | --- |
| this year, val (62) | **0.774** | apple 0.588, cherry 0.632 recall; chestnut and maple 1.000 |
| camera-like (the same 62) | **0.661** | apple recall falls to **0.235** — 13 of 17 apples become cherry |
| last year, val (120) | **0.642** | cherry recall falls to **0.133** — 22 of 30 cherries become apple |

| last year, truth \ pred | apple | cherry | chestnut | maple |
| --- | --- | --- | --- | --- |
| apple | 24 | 3 | 2 | 1 |
| cherry | **22** | 4 | 0 | 4 |
| chestnut | 0 | 2 | 22 | 6 |
| maple | 3 | 0 | 0 | 27 |

The classifier is not weak everywhere. **Chestnut and maple are never confused with anything** on
this year's photos, and every mistake in all three tables is between **apple and cherry** — in one
direction on this year's photos, in the other direction on last year's. The two species differ by
the fine serration of the edge and by the pattern of the veins, and on a 96 × 96 photograph taken
hand-held, in a different light, that difference is what disappears first: the same six leaves that
are 0.588 apple on a flat validation photo are 0.235 apple once the picture is cropped, rotated and
recoloured.

So the improvements in this report are not judged by whether they raise the overall number, but by
whether they fix **apple against cherry** and survive the camera column.

## 3. How big is the noise

This year's data is small: **244 training and 62 validation photographs** (train: cherry 75,
apple 68, maple 60, chestnut 41 — val: 19, 17, 15, 11). So on the `val` column **one image is
1.6 percentage points**, and a single measurement near 75 % accuracy has a spread of about
**±5.5 pp for 1σ, ±11 pp for 2σ**. A difference between two variants has to be bigger than that to
be a difference at all, which is why every row below is repeated with three seeds.

| n | one image | 1σ around 0.75 | 2σ around 0.75 |
| --- | --- | --- | --- |
| 62 (val, camera) | 1.61 pp | 5.5 pp | 11.0 pp |
| 120 (last year, val) | 0.83 pp | 4.0 pp | 7.9 pp |
| 700 (last year, train) | 0.14 pp | 1.3 pp | 2.7 pp |

To call a 5 pp improvement real you would need about 260 images, not 62. Averaging three seeds
removes the part of the noise that comes from the training run itself — shuffling, dropout, random
augmentation — but it cannot remove the part that comes from only having 62 photographs to test on,
because every seed is tested on the same 62.

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
