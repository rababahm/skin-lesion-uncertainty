# A skin-lesion classifier that knows when to say "I don't know"

![CI](https://github.com/rababahm/skin-lesion-uncertainty/actions/workflows/ci.yml/badge.svg)

**Result:** one sentence with the main number and its 95% interval.

<!-- ![Main figure](reports/figures/main.png) -->

## The question

Can a small image classifier give probabilities you can trust, and prediction sets
that contain the true diagnosis 90% of the time, even for rare classes?

## Baselines (validation split, 28 × 28 pixels)

Stratified 5-fold cross-validation on the training split (mean ± standard deviation),
then one score on the validation split. Class weights are balanced in all models.

| Model | CV balanced accuracy | CV macro-F1 | Val balanced accuracy | Val macro-F1 |
| --- | --- | --- | --- | --- |
| Majority class | 0.143 ± 0.000 | 0.115 ± 0.000 | 0.143 | 0.115 |
| Logistic regression (C = 0.1) | 0.414 ± 0.032 | 0.338 ± 0.021 | 0.429 | 0.324 |
| Gradient boosting | 0.343 ± 0.021 | 0.376 ± 0.025 | 0.373 | 0.407 |

Logistic regression leads on balanced accuracy (it catches more of the rare classes);
gradient boosting leads on macro-F1 (fewer false alarms). Reproduce with `make baselines`.

## CNN (validation split, 28 × 28 pixels)

A small convolutional network (41,351 parameters), trained with Adam and early stopping
on validation balanced accuracy (patience 5). Three seeds per setting, mean ± standard
deviation.

| Model | Val balanced accuracy |
| --- | --- |
| CNN, class weights on | 0.542 ± 0.047 |
| CNN, class weights off | 0.413 ± 0.016 |
| Logistic regression (baseline above) | 0.429 |

Class weights are what make the CNN beat the baseline: without them it predicts the
common class too often. Each run's score is its best epoch, chosen on validation, so
these numbers are optimistic; the test split is used once, at the end.

Reproduce: run `uv run python -m skin_lesion_uncertainty.train --seed S` for seeds 0 to 2,
with and without `--no-class-weights`, then `uv run python -m skin_lesion_uncertainty.report`.

## Calibration (validation split)

Top-label calibration: confidence is the largest predicted probability. ECE uses 15
equal-width bins. Temperature scaling fits one parameter T by minimising the negative
log-likelihood (NLL); it changes the probabilities but never the predicted class.

| Model | Balanced acc. | T | ECE before | ECE after | NLL before | NLL after |
| --- | --- | --- | --- | --- | --- | --- |
| Class weights on | 0.587 | 1.24 | 0.083 | 0.054 | 1.029 | 1.013 |
| Class weights off | 0.431 | 0.89 | 0.046 | 0.031 | 0.710 | 0.707 |

Seed 0 shown. Over three seeds, validation balanced accuracy is 0.542 ± 0.047 with class
weights and 0.413 ± 0.016 without. T is fitted on the validation set, so the "after"
values are slightly optimistic.

![Reliability diagram](reports/figures/reliability_validation.png)

Class weights buy balanced accuracy at the cost of calibration. The weighted model is
over-confident: at about 83% confidence it is right 64% of the time, and the fitted
temperature is above 1 (T = 1.24). Temperature scaling lowers ECE from 0.083 to 0.054,
but NLL barely moves, because the weights shift each class's probabilities by a different
factor and a single temperature cannot undo that. The unweighted model is already close
to calibrated. With 1,003 validation images, even a perfectly calibrated model would show
an ECE of about 0.03 (simulated), so differences below about 0.05 are within noise.

Reproduce: train seed 0 with `--save-logits artifacts/cw_on_seed0` (and with
`--no-class-weights --save-logits artifacts/cw_off_seed0`), then
`uv run python -m skin_lesion_uncertainty.calibration`.

## How to run

```bash
git clone git@github.com:rababahm/skin-lesion-uncertainty.git
cd skin-lesion-uncertainty
make reproduce
```

Runs on a laptop CPU. Seeds are fixed.

## Data

DermaMNIST, part of [MedMNIST](https://medmnist.com/): dermatoscope images in seven
diagnostic categories, derived from the HAM10000 dataset. Licence CC BY-NC 4.0
(non-commercial). Not for clinical use. The images are downloaded by the `medmnist`
package and never committed.

![Training images per class](reports/figures/class_counts.png)

![Example training images for each class](reports/figures/examples.png)

Please cite:

- Yang et al., "MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D
  biomedical image classification", Scientific Data, 2023.
- Tschandl, Rosendahl and Kittler, "The HAM10000 dataset, a large collection of
  multi-source dermatoscopic images of common pigmented skin lesions", Scientific
  Data, 2018.

## Licence

Code under MIT. See `LICENSE`.
