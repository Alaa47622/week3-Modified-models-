# Water Segmentation — 12-Channel Sentinel-2: Scratch U-Net vs Pretrained U-Net

Binary water segmentation on 12-band multispectral GeoTIFF images with PNG masks.
This project compares two models under identical conditions:

| | Model A (Week 1) | Model B (Week 2) |
|---|---|---|
| Architecture | U-Net, random initialisation | U-Net with ImageNet-pretrained **ResNet-34** encoder |
| Input | 12 channels | 12 channels (first conv adapted from 3 → 12) |
| Training | End-to-end | Two-phase fine-tuning (frozen warm-up → full fine-tune) |

## Pipeline

```
12-channel Multispectral Images
            │
            ▼
      Dataset + Masks
            │
      ┌─────┴─────┐
      ▼           ▼
 Scratch U-Net   Pretrained U-Net (ResNet-34)
                       │
                 First Conv → 12 channels
                       │
      └─────┬─────┘
            ▼
       Fine-tuning
            │
            ▼
       Test Evaluation  →  IoU · F1 · Precision · Recall
            │
            ▼
      Model Comparison → Final Findings
```

## How the pretrained model accepts 12 channels

ImageNet encoders expect 3-channel RGB. Only the first convolution depends on the channel count
(`64×3×7×7`), so only that layer is replaced (`64×12×7×7`); all deeper layers keep their pretrained weights.

1. Every input channel starts from the **mean of the three pretrained RGB filters**.
2. The true pretrained R, G, B filters are copied onto bands **B4, B3, B2**.
3. All weights are scaled by **3/12** to keep the first layer's activation magnitude comparable.

## Fine-tuning strategy

| Phase | Epochs | Trainable | Learning rate |
|---|---|---|---|
| 1 — warm-up | 5 | decoder, head, new first conv (+BN) | 1e-3 |
| 2 — fine-tune | 30 | whole network | encoder 1e-4, decoder 3e-4 |

The scratch model gets the same 35-epoch budget.

## Fair-comparison protocol

- Same train/val/test split (seed 42), same train-only normalisation, same augmentation
  (flips and 90° rotations), same loss (0.5·BCE + 0.5·Dice).
- Best checkpoint chosen by **validation IoU**.
- Probability threshold chosen on the **validation set only**, per model.
- Test set evaluated once per model after all choices are fixed.

## Results

Dataset: **16 labelled images**, split **11 train / 2 validation / 3 test**.

### Validation (threshold tuned on validation)

| Model | IoU |
|---|---|
| Scratch U-Net | 0.6978 |
| Pretrained U-Net (ResNet-34) | 0.5566 |

### Held-out test

| Model | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| Scratch U-Net | 0.3725 | 0.5428 | _fill from `comparison_test.csv`_ | _fill_ |
| Pretrained U-Net (ResNet-34) | 0.0892 | _fill_ | _fill_ | _fill_ |

> The full table with all metrics is saved by the notebook as `comparison_test.csv`,
> `comparison_val.csv` and `results.json`. Paste the missing values above from there.

Validation IoU gain (pretrained − scratch): **−0.1412** (tuned thresholds), **−0.1268** (best epoch, threshold 0.5).

![Comparison plots](comparison_plots.png)
![Prediction comparison](prediction_comparison.png)

## Findings

- **In this run the pretrained model did not outperform the scratch model**, so the
  "measurably higher validation IoU" criterion is not yet met.
- **The dataset is very small.** With 2 validation and 3 test images, a single image can move IoU by
  0.2 or more. The scratch model alone drops from 0.70 (validation) to 0.37 (test), which shows how noisy
  these estimates are. Differences between models are not statistically reliable at this size.
- **The pretrained model received few weight updates.** With 11 training images and batch size 4 there are
  3 steps per epoch (~100 total), of which only ~15 are in the warm-up phase. A ~24M-parameter model with a
  randomly initialised decoder needs more updates than that to converge.
- **Domain gap.** ImageNet features were learned on RGB; 9 of the 12 bands (NIR, red-edge, SWIR) are
  unfamiliar to the encoder and must be learned by the first convolution.
- Images are resized to 128×128, which can blur small water bodies.

### Planned improvements

1. Train on **random crops** (hundreds of patches per epoch) instead of whole resized images.
2. Replace the single split with **k-fold cross-validation** and report mean ± std for both models.
3. Increase warm-up and total epochs for the pretrained model.
4. Try other encoders (ResNet-18, EfficientNet) and add water indices (NDWI/MNDWI) as extra channels.
5. Inspect label quality and class balance (water fraction per image).

## Repository contents

```
water_segmentation_week2_pretrained_vs_scratch.ipynb   # full pipeline
README.md
comparison_plots.png
prediction_comparison.png
comparison_val.csv / comparison_test.csv / results.json
```

## Reproduce

1. Open the notebook in Google Colab (GPU runtime recommended).
2. Run all cells. Dependencies not preinstalled in Colab: `rasterio`, `gdown`.
3. Requires internet access to download the dataset and the ImageNet ResNet-34 weights.

Dataset layout expected: `dataset/images/*.tif` (12 bands) and `dataset/labels/*.png`.
