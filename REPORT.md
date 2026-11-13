# REPORT.md — Task 4, what actually improved the leaf classifier

Everything below was measured in this folder: `results.csv` holds every run, `weights-NN.pt` the
weights of row NN, `analysis.txt` is the table generator's output and `tests/evidence/` the browser
checks. Nothing here is copied from an earlier attempt, and no number is an estimate.

## 1. Where the work started

The model that is on the page (Task 2) is a fine-tuned `mobilenet_v3_small` with a four-class head.
Task 4 asks whether a different recipe is measurably better. The starting point — the row called
`baseline (Task 2)` — is that same recipe retrained here, not the file from Task 2:

| | value |
| --- | --- |
| network | `torchvision.models.mobilenet_v3_small(weights="DEFAULT")`, last layer replaced by `Linear(576, 4)` |
| parameters | 1,521,956 |
| input | 96 × 96, `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` |
| loss / optimiser | `CrossEntropyLoss`, Adam, lr 1e-3, batch 32 |
| epochs | 15 |
| training data | 244 photographs (68 apple, 75 cherry, 41 chestnut, 60 maple) |
| augmentation in the baseline | none — resize + normalize only, like Task 2 |

The retrained baseline is not the same file as `leaf_model.onnx`, and it must not be: the notebook
trained on Colab, with a different order of random-number use and no separate seed for the shuffle,
so a second training of "the same" recipe is a different model. That is exactly why every variant
here is compared with a baseline retrained **in the same conditions**, and not with the Task 2 file.

The Task 2 model, scored on the same three sets for reference (`eval_page_model.py`):

| model | val (62) | camera-like (62) | last year (120) |
| --- | --- | --- | --- |
| `leaf_model.onnx` from Task 2 | 0.774 | 0.661 | 0.642 |
| retrained baseline, mean of 6 seeds | 0.769 | 0.613 | 0.520 |

(The three-seed baseline means in §4 are 0.769 / 0.645 / 0.567; §6 explains why the six-seed
`camera` and `last year` numbers are lower — the three extra seeds happened to be weak, which is the
point of running them.)

## 2. The three scoring sets, and the one that is honest

| Column | What it is | Why it is there |
| --- | --- | --- |
| `val` | the 62 validation photographs of this year's session (17 apple, 19 cherry, 11 chestnut, 15 maple) | the split the notebook used |
| `camera` | the same 62 photographs, each turned once into a phone-like picture: `RandomResizedCrop(size, scale=(0.6, 1.0))`, `RandomRotation(30)`, `ColorJitter(0.4, 0.4, 0.4, 0.1)`, with `torch.manual_seed(7)` | closer to the deployed situation: a leaf held in front of a camera, not scanned |
| `last year` | 120 photographs from last year's session — other students, other phones, other light | the only set not produced by this year's group |

The camera set is built **once**, with a fixed seed, and the identical tensors are used for every
row, so comparing two rows is a comparison on the same 62 pictures. It is not, however, an
independent set: it is a transformation of the `val` photographs, so `val` and `camera` share the
same leaves and the same mistakes.

There is a second, more serious caveat that the table below cannot show. The camera set is built
with the **same family of transforms** that the augmented variants train with (`RandomResizedCrop`,
`RandomRotation`, `ColorJitter`). A model trained with them is being tested with its own training
distribution, so its `camera` advantage is partly memorised augmentation, not generalisation.
`last year` is the only column where that cannot happen, which is why the final choice is checked
on it as well, and why §6 reports its paired test separately instead of trusting the `camera` gain.

## 3. Noise: how large a difference means anything

Validation accuracy is a proportion over a fixed set of images, so it moves in steps:

* one image in the 62-image sets is **1.61 pp**; in the 120-image set, **0.83 pp**;
* the binomial standard deviation near 75 % accuracy is 5.5 pp for 62 images and 4.0 pp for 120;
* two standard deviations — the bar used in the table — are **11.0 pp** and **7.9 pp**.

So two runs whose accuracies differ by five points on 62 images have not been shown to differ at
all. Every row of the table therefore reports, next to the mean, whether the change is larger than
2 σ **and** in the same direction in every seed. The stronger test — models compared image by image,
exact McNemar — is in §6.

## 4. Every run

Three seeds (0, 1, 2) per variant; each row is the mean of the three, with the individual runs under
it. "Change vs baseline" is in percentage points of accuracy. All rows share everything not named in
the change column: 244 photographs, Adam 1e-3, batch 32, 15 epochs, 96 × 96.

| Variant | Params | val | camera | last year | change vs baseline | larger than the noise? |
| --- | --- | --- | --- | --- | --- | --- |
| + all augmentation, cosine, 40 epochs | 1522 k | **0.839**<br>0.855 / 0.823 / 0.839 | **0.774**<br>0.790 / 0.806 / 0.726 | **0.542**<br>0.525 / 0.583 / 0.517 | `val` +7.0 pp `camera` +12.9 pp `last year` -2.5 pp | `camera` +12.9 pp in every seed |
| + backbone lr 1e-4, cosine, best epoch | 1522 k | **0.817**<br>0.806 / 0.806 / 0.839 | **0.618**<br>0.597 / 0.629 / 0.629 | **0.497**<br>0.483 / 0.492 / 0.517 | `val` +4.8 pp `camera` -2.7 pp `last year` -6.9 pp | no — inside 2 sigma or seeds disagree |
| + class weights | 1522 k | **0.806**<br>0.839 / 0.806 / 0.774 | **0.629**<br>0.613 / 0.645 / 0.629 | **0.544**<br>0.508 / 0.475 / 0.650 | `val` +3.8 pp `camera` -1.6 pp `last year` -2.2 pp | no — inside 2 sigma or seeds disagree |
| + class weights aimed at cherry (x2) | 1522 k | **0.791**<br>0.694 / 0.839 / 0.839 | **0.613**<br>0.581 / 0.677 / 0.581 | **0.519**<br>0.425 / 0.550 / 0.583 | `val` +2.2 pp `camera` -3.2 pp `last year` -4.7 pp | no — inside 2 sigma or seeds disagree |
| **+ colour + geometric** | 1522 k | **0.887**<br>0.790 / 0.935 / 0.935 | **0.795**<br>0.790 / 0.806 / 0.790 | **0.653**<br>0.658 / 0.683 / 0.617 | `val` +11.8 pp `camera` +15.0 pp `last year` +8.6 pp | **all three columns, in these three seeds — but see §6, where three fresh seeds pull the means down** |
| + colour jitter | 1522 k | **0.758**<br>0.742 / 0.790 / 0.742 | **0.634**<br>0.565 / 0.661 / 0.677 | **0.583**<br>0.542 / 0.633 / 0.575 | `val` -1.1 pp `camera` -1.1 pp `last year` +1.7 pp | no — inside 2 sigma or seeds disagree |
| + dropout 0.5 in the classifier | 1522 k | **0.833**<br>0.806 / 0.855 / 0.839 | **0.688**<br>0.661 / 0.677 / 0.726 | **0.533**<br>0.508 / 0.533 / 0.558 | `val` +6.5 pp `camera` +4.3 pp `last year` -3.4 pp | no — inside 2 sigma or seeds disagree |
| + frozen backbone, head only | 1522 k | **0.554**<br>0.484 / 0.629 / 0.548 | **0.409**<br>0.403 / 0.452 / 0.371 | **0.347**<br>0.283 / 0.417 / 0.342 | `val` -21.5 pp `camera` -23.7 pp `last year` -21.9 pp | **all three, −21 to −24 pp in every seed** |
| + geometric augmentation | 1522 k | **0.839**<br>0.839 / 0.806 / 0.871 | **0.667**<br>0.742 / 0.677 / 0.581 | **0.536**<br>0.575 / 0.500 / 0.533 | `val` +7.0 pp `camera` +2.1 pp `last year` -3.1 pp | no — inside 2 sigma or seeds disagree |
| + input 160 × 160 | 1522 k | **0.747**<br>0.823 / 0.806 / 0.613 | **0.618**<br>0.645 / 0.629 / 0.581 | **0.492**<br>0.467 / 0.508 / 0.500 | `val` -2.1 pp `camera` -2.7 pp `last year` -7.5 pp | no — inside 2 sigma or seeds disagree |
| + label smoothing | 1522 k | **0.785**<br>0.790 / 0.758 / 0.806 | **0.656**<br>0.645 / 0.661 / 0.661 | **0.539**<br>0.500 / 0.567 / 0.550 | `val` +1.6 pp `camera` +1.0 pp `last year` -2.8 pp | no — inside 2 sigma or seeds disagree |
| + last year's photos added to training | 1522 k | **0.785**<br>0.710 / 0.806 / 0.839 | **0.688**<br>0.613 / 0.790 / 0.661 | **0.889**<br>0.883 / 0.883 / 0.900 | `val` +1.6 pp `camera` +4.3 pp `last year` +32.2 pp | `last year` +32.2 pp — **not a fair column, see below** |
| + last year, augmentation, cosine, best epoch | 1522 k | **0.823**<br>0.823 / 0.839 / 0.806 | **0.817**<br>0.806 / 0.823 / 0.823 | **0.909**<br>0.917 / 0.917 / 0.892 | `val` +5.4 pp `camera` +17.2 pp `last year` +34.2 pp | `camera` +17.2 pp; `last year` — **not a fair column** |
| + weight decay 1e-4 | 1522 k | **0.769**<br>0.742 / 0.742 / 0.823 | **0.661**<br>0.613 / 0.629 / 0.742 | **0.528**<br>0.525 / 0.450 / 0.608 | `val` +0.0 pp `camera` +1.6 pp `last year` -3.9 pp | no — inside 2 sigma or seeds disagree |
| LeafNet + BN + augmentation | 209 k | **0.753**<br>0.758 / 0.726 / 0.774 | **0.635**<br>0.565 / 0.694 / 0.645 | **0.478**<br>0.542 / 0.417 / 0.475 | `val` -1.6 pp `camera` -1.1 pp `last year` -8.9 pp | `last year` −8.9 pp in every seed |
| LeafNet + GAP + BN + augmentation | 65 k | **0.645**<br>0.694 / 0.613 / 0.629 | **0.548**<br>0.645 / 0.565 / 0.435 | **0.414**<br>0.442 / 0.442 / 0.358 | `val` -12.3 pp `camera` -9.7 pp `last year` -15.3 pp | `val` −12.3 pp, `last year` −15.3 pp |
| LeafNet + global average pooling | 65 k | **0.624**<br>0.565 / 0.710 / 0.597 | **0.355**<br>0.323 / 0.355 / 0.387 | **0.397**<br>0.433 / 0.350 / 0.408 | `val` -14.5 pp `camera` -29.0 pp `last year` -17.0 pp | **all three, −14 to −29 pp in every seed** |
| Task 3 LeafNet from scratch | 208 k | **0.699**<br>0.710 / 0.694 / 0.694 | **0.376**<br>0.387 / 0.435 / 0.306 | **0.386**<br>0.442 / 0.333 / 0.383 | `val` -6.9 pp `camera` -26.9 pp `last year` -18.1 pp | `camera` −26.9 pp, `last year` −18.1 pp |
| **baseline (Task 2)** | 1522 k | **0.769**<br>0.774 / 0.806 / 0.726 | **0.645**<br>0.694 / 0.645 / 0.597 | **0.567**<br>0.550 / 0.550 / 0.600 | — | — |

**The two `last year` rows are not comparable with the rest.** They train on last year's 700
training photographs, so scoring them on last year's 120 validation photographs measures how well
the network fits a session it has seen, not how well it generalises to a new one. Their +32 and
+34 pp say that last year's photographs are easy to memorise, nothing more. Read their `val` and
`camera` columns (which are fair) and ignore `last year` there. This is also why the variant that
wins on the `camera` column overall, 0.817, is **not** the model exported at the end.

### Spread between seeds of the same recipe

Max − min over every run of that recipe in `results.csv`; the first three rows have six runs, the
rest three.

| Variant | val | camera | last year |
| --- | --- | --- | --- |
| baseline (Task 2) | 0.080 | 0.146 | 0.208 |
| + colour + geometric | 0.161 | 0.032 | 0.200 |
| + all augmentation, cosine, 40 epochs | 0.049 | 0.080 | 0.091 |
| + geometric augmentation | 0.065 | 0.161 | 0.075 |
| + colour jitter | 0.048 | 0.112 | 0.091 |
| + input 160 × 160 | 0.210 | 0.064 | 0.041 |

Choosing the seed after seeing the score is worth up to 16 pp on `val` — as large as the effects
being studied. Any single-run comparison would have been a coin toss; this is the strongest argument
for several seeds per row and for the paired test in §6.

## 5. The change that worked: colour + geometric augmentation, and why

`+ colour + geometric` is the Task 2 recipe with exactly one key changed: the training transform
gains `RandomResizedCrop(96, scale=(0.6, 1.0))`, `RandomHorizontalFlip()`, `RandomVerticalFlip()`,
`RandomRotation(20)` and `ColorJitter(0.3, 0.3, 0.3, 0.05)`. Same network, same optimiser, same 15
epochs, same 244 photographs.

Over the first three seeds it beat the baseline on all three columns, and the smallest gain (val,
+11.8 pp) is above 2 σ; it was also the only fair variant that improved `last year` by more than
2 σ (+8.6 pp). §6 shows what three more seeds do to that picture.

The mechanism is visible in the confusion matrices (`eval_variant.py`), seed 1 — the deployed
model against the baseline of the same seed:

| Recall | baseline | + colour + geometric |
| --- | --- | --- |
| val apple | 0.647 (11/17) | 0.824 (14/17) |
| val cherry | 0.789 (15/19) | 0.947 (18/19) |
| camera apple | **0.118 (2/17)** | **0.765 (13/17)** |
| camera cherry | 0.842 (16/19) | 0.789 (15/19) |
| last year apple | 0.867 (26/30) | 0.700 (21/30) |
| last year cherry | **0.267 (8/30)** | **0.567 (17/30)** |
| last year chestnut | **0.333 (10/30)** | **0.867 (26/30)** |
| last year maple | 0.733 (22/30) | 0.600 (18/30) |

The baseline's failure is concentrated, not spread over the matrix. Under a phone-like crop it
loses 15 of 17 apples: the crop cuts off the fruit's edge and the network calls the remaining leaf
cherry (12 of the 17 apples go to cherry). On last year's photographs it calls two thirds of the
cherries something else and files half the chestnuts under maple.

Augmentation does not make the network smarter; it stops it from keying on the exact framing and
colour balance of this year's 244 photographs. The trade is visible too: the augmented model gives
up five of last year's 30 apples (21 instead of 26) and keeps 9 more cherries, 16 more chestnuts and
loses 4 maples — a net of 16 images out of 120. On the camera set, the crop-induced
apple→cherry errors drop from 12 to 2.

Neither half does this alone:

| Recipe | val | camera | last year |
| --- | --- | --- | --- |
| + colour jitter only | 0.758 | 0.634 | 0.583 |
| + geometric augmentation only | 0.839 | 0.667 | 0.536 |
| + both | **0.887** | **0.795** | **0.653** |

Geometric augmentation alone raises `val` by 7 pp but does not touch the camera column (0.667 vs
0.645, inside noise); colour jitter alone is flat everywhere. The two together are worth more than
the sum of their parts, which is the standard finding for augmentation: colour changes teach
invariance to the phone's white balance, cropping teaches invariance to framing, and a network needs
both before either shows up on a new session.

## 6. Is the difference real? Three more seeds, and paired tests

Three seeds show a direction; they cannot size an effect, because one seed is worth up to 16 pp on
`val` (§4). So the three decisive rows were retrained with three fresh seeds (3, 4, 5) — same
recipes, same sets, different initialisation and shuffle — and the comparison below is over **six
runs per model**.

| Column | baseline | + colour + geometric | difference | 2 σ | verdict |
| --- | --- | --- | --- | --- | --- |
| val | 0.769 | 0.838 | +7.0 pp | 11.0 pp | positive, below what 62 images can resolve |
| camera | 0.613 | 0.793 | **+18.0 pp** | 11.0 pp | clearly real |
| last year | 0.520 | 0.590 | +7.1 pp | 7.9 pp | positive, just below the bar |

Two things change when the seeds are added. The three-seed means in §4 (0.887 / 0.795 / 0.653) were
the optimistic end of the spread: with six seeds the recipe is **0.838 / 0.793 / 0.590**. And the gap
on the augmented `camera` set actually grows, because the baseline's extra seeds were weak (0.548,
0.548). The honest statement is therefore: a large, solid gain when the leaf is photographed the way
the page photographs it, and a positive but unresolved ~7 pp gain on untouched photographs.

Accuracy also throws away the pairing. On the images where the two models disagree, does one win
more often than a coin flip? `eval_paired.py` runs the exact McNemar test per seed (`b` = baseline
right and variant wrong, `c` = the other way round).

| seed | val | camera | last year |
| --- | --- | --- | --- |
| 0 | 0.774 vs 0.790 (b 5, c 6) | 0.694 vs 0.790 (b 5, c 11) | 0.550 vs 0.658 (b 9, c 22, **p 0.029**) |
| 1 | 0.806 vs 0.935 (b 1, c 9, **p 0.021**) | 0.645 vs 0.806 (b 6, c 16, p 0.052) | 0.550 vs 0.683 (b 17, c 33, **p 0.033**) |
| 2 | 0.726 vs 0.935 (b 1, c 14, **p 0.001**) | 0.597 vs 0.790 (b 5, c 17, **p 0.017**) | 0.600 vs 0.617 (b 10, c 12, p 0.832) |
| 3 | 0.758 vs 0.806 (b 3, c 6, p 0.508) | 0.645 vs 0.806 (b 1, c 11, **p 0.006**) | 0.508 vs 0.483 (b 23, c 20, p 0.761) |
| 4 | 0.758 vs 0.790 (b 9, c 11, p 0.824) | 0.548 vs 0.790 (b 7, c 22, **p 0.008**) | 0.392 vs 0.508 (b 17, c 31, p 0.059) |
| 5 | 0.790 vs 0.774 (b 7, c 6, p 1.000) | 0.548 vs 0.774 (b 4, c 18, **p 0.004**) | 0.517 vs 0.592 (b 18, c 27, p 0.233) |

On `camera` the improvement is significant in four seeds of six, and the discordant images favour
the augmented model in all six (c runs from 11 to 22, b from 1 to 7). On `last year` it is
significant in two seeds, favours the augmented model in five of six, and is slightly behind in one
(seed 3). On `val` it is significant in two seeds of six and inside noise in the rest, slightly
behind in one (seed 5). The fair summary: **the camera-set gain is established; the +7 pp on
untouched photographs is consistent in direction but this sample cannot confirm it.**

`+ colour + geometric` vs `+ all augmentation, cosine, 40 epochs` (the other strong row), six seeds:

| | mean over 6 seeds | paired test |
| --- | --- | --- |
| val | 0.838 vs 0.836 | significant in one seed (1); in three seeds more discordant images go the other way |
| camera | **0.793 vs 0.761** | never significant; favourable in three seeds, tied in the other three (b/c 10/6, 8/3, 7/4; 6/6, 6/6, 7/7) |
| last year | **0.590 vs 0.554** | never significant; favourable in the first three seeds, behind in the last three (b/c 27/11, 27/15, 27/15, 22/26, 16/24, 11/13) |

So the 40-epoch cosine schedule buys nothing over the plain 15-epoch recipe: the means are level on
`val` and behind on the other two, and no per-seed difference is significant. On three seeds its
`val` deficit looked like −4.8 pp and its `last year` deficit like −11.1 pp; on six seeds those are
−0.2 pp and −3.6 pp. The direction survived, the size did not — another reason to treat the
three-seed means in §4 as a first look, not as a measurement.

## 7. The choice

**Exported model: `+ colour + geometric`, seed 1** — `export/leaf_model.onnx` (6.37 MB),
`export/model_info.json`, `export/check_expected.json`.

| Criterion | Why this row |
| --- | --- |
| a real improvement | the largest gain of any fair variant: +18.0 pp on `camera` over six seeds, positive on that column in all six |
| generalises | best `last year` among the fair rows over six seeds (0.590 vs 0.520), and in five of six seeds it wins more discordant images than it loses |
| not fooled by the metric | does not need the augmented `camera` column to win — it is also ahead on the honest one |
| cost | 1.52 M parameters, 15 epochs (~0.9 min per run on this CPU), same file size as the current page model |
| simplest explanation | it is the plain Task 2 recipe with a stronger training transform; no new loss, no new optimiser, nothing to tune at inference |

Rejected, in order of how tempting they look in §4:

* **`+ last year, augmentation, cosine, best epoch`** — 0.817 camera, the best number in the table.
  Its `last year` score (0.909) is not evidence of generalisation: it trained on that session. Its
  fair columns are 0.823 `val` and 0.817 `camera` — the camera number is 2.2 pp above the plain
  recipe, which is inside the noise of a 62-image set — for 40 epochs and 944 photographs instead of
  15 and 244. Deploying it would also make the app depend on last year's data, which is not the point
  of the exercise.
* **`+ all augmentation, cosine, 40 epochs`** — over six seeds 0.836 / 0.761 / 0.554: level with the
  plain recipe on `val` and behind on `camera` and `last year`, for three times the training time.
* **`+ input 160 × 160`** — more compute per frame on the phone *and* worse on every column; the
  dataset is 96 × 96 photographs, so upsampling adds no information.
* **`+ class weights` and `+ class weights aimed at cherry (×2)`** — the two attempts to attack the
  apple/cherry confusion directly, both flat or worse. The loss weighting changes which class the
  network prefers, not which features it uses; with a confusion that comes from cropping off the
  fruit, it cannot help.
* **`LeafNet` variants, including `+ global average pooling`** — the Task 3 network from scratch is
  26.9 pp worse on `camera` than the pretrained baseline. GAP, batch norm and augmentation all
  recover some of it and none comes close; 208 k parameters cannot replace ImageNet pretraining on
  244 photographs. This is a useful negative result, not a failure of the experiment.

### Seed choice, stated plainly

The exported weights are seed 1, the best of the six on both `camera` (0.806) and `last year`
(0.683). Choosing a seed after looking at the scores is a small optimism — seed 1 against seed 0 is
three photographs out of 120, and against the six-seed mean it is eleven — so the number to quote for
this recipe is the **mean over six seeds (0.838 / 0.793 / 0.590)**, not the deployed seed's score.
The deployed file is one seed only because the page runs one network.

### Checked in a real browser before being called finished

The exported model was put next to a copy of `index.html` and run in Chromium and WebKit
(`tests/evidence/*-local-site.json`): its own `?selftest=1` reports `SELFTEST PASS` with a largest
difference of **1.00e-06** between the page and the new `check_expected.json`, the upload button
gives the same numbers as the bars, the centre-crop and constant-colour probes are exact, and every
request is still a GET. So the improved model is a drop-in replacement: copy
`export/leaf_model.onnx`, `export/model_info.json` and `export/check_expected.json` together into
the folder of `index.html`. The three must move together, because the page's self test compares the
model with `check_expected.json`.

## 8. What did not help, and what I would do next

Every one of these was inside 2 σ on all three columns: dropout 0.5 (+6.5 `val`, −3.4 `last year`),
label smoothing (+1.6, −2.8), weight decay (+0.0, −3.9), 160 × 160 (−2.1, −7.5), and the two class
weightings. Frozen backbone was actively harmful (−21.5 pp `val`): with 244 photographs the head
alone cannot relearn what the frozen features discard.

Three things I would do next, in the order I would spend the time:

1. **More data, from more phones.** The table says the recipe is not the bottleneck — the 244
   photographs are. Adding last year's session to training raised the score on that session from
   0.567 to 0.889, i.e. the features transfer, the data does not exist. Collecting ~50 photographs
   per class from several phones, with the leaf at the distance it is actually photographed, is
   worth more than any further hyper-parameter.
2. **A test set that is independent by construction.** The 62-image sets can only resolve changes of
   about 11 pp. A held-out session of 300+ photographs would make §6 decisive instead of "two seeds
   of six", and would let the final choice be made on `last year` alone.
3. **Then, and only then, a bigger backbone.** `mobilenet_v3_small` is not the limit yet: the same
   recipe with `mobilenet_v3_large` or a ConvNeXt-tiny head is a ten-minute experiment once there is
   data to train it on, and it would cost one more second per frame in the browser.

Two smaller things are worth a line: the `camera` column should be rebuilt without the transforms
that training uses (e.g. only perspective and lighting), otherwise it will keep rewarding the
augmented rows; and the 96 × 96 input should stay until the photographs are collected at a higher
resolution.

## Appendix — how every number was produced

```powershell
# data (not part of the hand-in): this year's ZIP and last year's public archive into data/
python run_task4.py                        # 19 variants x 3 seeds -> results.csv, weights-NN.pt
                                           # safe to re-run: rows already measured are skipped
python -c "import task4_variants as t; t.run_rows(['baseline (Task 2)', '+ colour + geometric', '+ all augmentation, cosine, 40 epochs'], seeds=[3, 4, 5])"
python analyze_results.py results.csv 62 --seeds 0,1,2 > analysis.txt   # the 3-seed table above
python analyze_results.py                            # the same table with every run in results.csv
python eval_paired.py "baseline (Task 2)" "+ colour + geometric"
python eval_variant.py "+ colour + geometric" 1
python eval_page_model.py                  # the Task 2 model on the same three sets
python export_best.py "+ colour + geometric" 1
python tests/verify_page.py --browser chromium --folder export/site
python tests/verify_page.py --browser webkit   --folder export/site
```

* `results.csv` — 66 rows: the 19 variants × 3 seeds, plus seeds 3–5 for the three decisive rows in
  §6. The row number is the weight file (`weights-07.pt` is row 7), and rows are never rewritten,
  so every number above can be traced to a file.
* Determinism: `torch.manual_seed(seed)` seeds the model, the shuffle and the augmentation, and the
  camera set is fixed with `torch.manual_seed(7)`. Running the two training commands above, in that
  order, regenerates this table — I checked that `baseline (Task 2)` seed 0 comes out identical in a
  fresh process. One honest caveat: the *first* training run of a process also builds the camera set,
  which draws from the same generator, so a row belongs to the sequence that produced it; re-running
  a single row on its own can move it by a few points. Every number in this report is therefore
  quoted from `results.csv`, not from ad-hoc reruns.
* Cost: the 57-row sweep took 73 minutes of wall clock on this machine (CPU, 12 torch threads), with
  the browser tests running at the same time, and the nine extra rows another ~14 minutes. Median per
  run: 0.8 min (15 epochs, MobileNet), 1.5 min (30 epochs), 2.4 min (40 epochs), 2.2 min (15 epochs
  on 944 photographs), 7.6 min (40 epochs on 944 photographs). The four LeafNet variants are
  0.4–0.7 min each.
* The notebook `Copy of ex08_student.ipynb` was not modified. Task 4 lives in `task4_variants.py`
  (recipes), `run_task4.py` (the sweep), `analyze_results.py`, `eval_paired.py`, `eval_variant.py`,
  `eval_page_model.py` and `export_best.py`, so a single run can be repeated without re-executing the
  notebook's demonstration cells. The Task 3 network is inside `task4_variants.py` as `LeafNet`, with
  the same layer sizes as the notebook.
