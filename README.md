# A skin-lesion classifier that knows when to say "I don't know"

![CI](https://github.com/rababahm/skin-lesion-uncertainty/actions/workflows/ci.yml/badge.svg)

**Result:** one sentence with the main number and its 95% interval.

![Main figure](reports/figures/main.png)

## The question

Can a small image classifier give probabilities you can trust, and prediction sets
that contain the true diagnosis 90% of the time, even for rare classes?

## What this shows

- What the result means.
- What it does not show (limits, assumptions).

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

Please cite:

- Yang et al., "MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D
  biomedical image classification", Scientific Data, 2023.
- Tschandl, Rosendahl and Kittler, "The HAM10000 dataset, a large collection of
  multi-source dermatoscopic images of common pigmented skin lesions", Scientific
  Data, 2018.

## Licence

Code under MIT. See `LICENSE`.
