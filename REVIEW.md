# REVIEW.md — Task 2

Reviewed page: <https://adamb4b.github.io/mlr-ex08/>


## 1. It runs on my phone

![Screenshot from my phone: the live camera on the Pages address, apple 99.6 %](ScreenShot.png)

| What I checked | What I saw |
| --- | --- |
| Phone and browser | The app opens in the browser |
| Did the page load, did the camera ask for permission? | The Page loads and it asks for camera permission |
| Did it classify the live picture, and how fast? | It classify the live picture almost instantly |
| Which class won on a real leaf, and did it change when I moved the phone? | It changed for a little bit but after it it stabilaze |
| Did I have to fix anything for it to work? | No, everything worked properly |

## 2. The same numbers as the notebook

| Class | Notebook (`check_expected.json`) | My page | Difference |
| --- | --- | --- | --- |
| apple | 0.983 | 0.983 | none |
| cherry | 0.017 | 0.017 | none |
| chestnut | 0.000 | 0.000 | none |
| maple | 0.000 | 0.000 | none |

## 3. The preprocessing matches my model

> note: open `index.html` and find the answers by yourself, with the line numbers. `model_info.json`
> and the `transform` in the notebook are what you are comparing against.

From `model_info.json`: input size `TODO`, mean `TODO`, std `TODO`, classes in this order `TODO`.

| Step | Where it is in `index.html` (function, line) | Does it match the notebook? |
| --- | --- | --- |
| central square crop of the frame | `TODO` | `TODO` |
| resize to 96 × 96 | `TODO` | `TODO` |
| pixels read out, alpha dropped | `TODO` | `TODO` |
| divided by 255 (`ToTensor`) | `TODO` | `TODO` |
| standardized with mean and std (`Normalize`) | `TODO` | `TODO` |
| channel order in the tensor (CHW, shape `[1,3,96,96]`) | `TODO` | `TODO` |
| softmax over the raw logits | `TODO` | `TODO` |

Order of the steps compared with `transforms.Compose([ToTensor(), Normalize(mean, std)])`: `TODO`

## 4. Nothing leaves the phone

> note: DevTools → Network (Chrome) or the browser's equivalent, then take a photo. What is
> actually sent, and where?

| URL the page loads | Why it is needed | Does it receive my image? |
| --- | --- | --- |
| `TODO` | `TODO` | `TODO` |
| `TODO` | `TODO` | `TODO` |
| `TODO` | `TODO` | `TODO` |

Did any request carry image data (look at the request payload, not only the URL)? `TODO`
Is the model downloaded to the phone and run there? `TODO`

## What the agent had to fix

> note: what was wrong, what you told it to change, what changed. If nothing was wrong, say so.

* `TODO`
