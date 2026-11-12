# Model card: skin-lesion classifier with conformal prediction sets

Version 1.0, 15 November 2026. Mohammad Rababah. Self-directed project, public repository.

## Model details

- **Architecture:** ResNet-18 with ImageNet weights (torchvision `IMAGENET1K_V1`), final
  layer replaced for 7 classes, all layers fine-tuned. Input: 64 × 64 RGB images, with
  ImageNet normalisation inside the model.
- **Training:** Adam, learning rate 3e-4, batch size 64, cross-entropy weighted by
  inverse class frequency, early stopping on validation balanced accuracy (patience 5).
  Seed 0; best epoch 9.
- **Post-processing:** temperature scaling (T = 1.87, fitted on validation); conformal
  prediction sets (hybrid method, alpha = 0.10); a referral rule on the sets.
- **Weights:** not published. `make reproduce` retrains them from the fixed seeds.
- **Code licence:** MIT.

## Intended use

- **Intended:** a public demonstration of calibrated probabilities, conformal prediction
  sets and a referral rule, on a standard benchmark; research and teaching.
- **Not intended:** any clinical use, including diagnosis, triage or screening, or any
  decision about a person. Not validated by clinicians. Not for commercial use (the data
  licence is non-commercial).

## Data

DermaMNIST from MedMNIST v2, derived from HAM10000: dermatoscope images, downsampled by
MedMNIST, in seven diagnostic classes. Official splits: 7,007 training, 1,003 validation
and 2,005 test images. Licence CC BY-NC 4.0.

| Class | Abbreviation | Test images |
| --- | --- | --- |
| Actinic keratoses and intraepithelial carcinoma | akiec | 66 |
| Basal cell carcinoma | bcc | 103 |
| Benign keratosis-like lesions | bkl | 220 |
| Dermatofibroma | df | 23 |
| Melanoma | mel | 223 |
| Melanocytic nevi | nv | 1,341 |
| Vascular lesions | vasc | 29 |

The data are strongly imbalanced: two thirds of the images are nevi.

## Evaluation protocol

All choices were frozen on 12 November 2026, before the test set was first used: the
model, the temperature, the conformal method and alpha, and the referral rule. The test
set was then evaluated once. Intervals: 2,000 bootstrap resamples of the test images for
accuracy, ECE and NLL; 1,000 random 50/50 calibration and evaluation splits for conformal
coverage.

## Metrics (test set, 2,005 images)

| Metric | Value |
| --- | --- |
| Balanced accuracy | 0.715 (95% CI 0.676 to 0.754) |
| ECE, before and after temperature scaling | 0.115, then 0.023 (95% CI 0.020 to 0.045) |
| NLL, before and after temperature scaling | 0.757, then 0.625 (95% CI 0.587 to 0.665) |
| Conformal coverage, hybrid, alpha = 0.10 | 0.921 (95% of splits: 0.894 to 0.942) |
| Melanoma coverage, hybrid | 0.90 |
| Mean prediction-set size | 1.80 classes |
| Referred to a clinician | 59% |
| Non-benign lesions auto-accepted | 3.2% |

Per-class coverage with the hybrid sets: akiec 0.91, bcc 0.91, bkl 0.73, df 0.78,
mel 0.90, nv 0.96, vasc 0.97.

## Limitations and risks

- **Not safe for clinical use.** At the chosen operating point, 3.2% of non-benign
  lesions are auto-accepted as benign. In practice that would be false reassurance about
  a cancer.
- **Coverage is guaranteed only for data like the calibration data.** Under blur, low
  resolution or brightness changes, hybrid coverage fell to 0.75 to 0.82, and melanoma
  coverage to as low as 0.31. Under heavy blur and low resolution, the sets also got
  smaller, so the system looked more certain while being more wrong. Any use outside
  these exact conditions needs an input-quality check, drift monitoring, or
  recalibration on new data.
- **Benign classes have no per-class guarantee.** By design, the hybrid guarantees 90%
  per class only for the three non-benign classes. Benign keratosis-like lesions are
  covered 73% of the time.
- **Rare classes are small:** 23 dermatofibroma and 29 vascular test images, so their
  per-class numbers are noisy. Full per-class (Mondrian) sets need at least 9
  calibration images per class at alpha = 0.10.
- **Optimism from the split:** the official DermaMNIST splits put images of the same
  lesion in more than one split (Abhishek, Jain and Hamarneh, 2024), so the model may
  have seen other images of some test lesions during training, and every score here is
  likely optimistic. DermaMNIST-C, their corrected version, keeps each lesion in one
  split; re-running on it is the next step.
- **Shortcuts:** dermatoscope frame artefacts and image borders may act as shortcuts.
  Not tested.
- **Population:** HAM10000 comes from two clinical sites, in Austria and Australia, and,
  as is widely reported, contains mostly lighter skin types. Performance on other
  populations is unknown.
- **Resolution:** 64 × 64 pixels discards detail a dermatologist would use.
- **Benign grouping:** which classes count as benign for the referral rule is a
  modelling assumption, not medical advice.

## Reproduce

`make reproduce` regenerates every number and figure in this card and in the README.
See the README for details and run times.

## Citation

- Yang et al., "MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D
  biomedical image classification", Scientific Data, 2023.
- Tschandl, Rosendahl and Kittler, "The HAM10000 dataset, a large collection of
  multi-source dermatoscopic images of common pigmented skin lesions", Scientific
  Data, 2018.
- Abhishek, Jain and Hamarneh, "Investigating the Quality of DermaMNIST and Fitzpatrick17k
  Dermatological Image Datasets", arXiv:2401.14497, 2024.
