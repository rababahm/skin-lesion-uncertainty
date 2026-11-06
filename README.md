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
