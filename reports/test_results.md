Test set: 2005 images; T = 1.87 (fitted on validation)

| Metric (95% bootstrap interval) | Before scaling | After scaling |
| --- | --- | --- |
| Balanced accuracy | 0.715 [0.676, 0.754] | unchanged |
| ECE | 0.115 [0.101, 0.133] | 0.023 [0.020, 0.045] |
| NLL | 0.757 [0.691, 0.829] | 0.625 [0.587, 0.665] |

Conformal sets, alpha = 0.1, 1000 random 50/50 splits of the test set

| Method | Coverage (95% of splits) | Melanoma | Set size | Referred | Missed non-benign |
| --- | --- | --- | --- | --- | --- |
| standard | 0.900 (0.872 to 0.926) | 0.80 | 1.46 | 49% | 7.4% |
| mondrian | 0.903 (0.874 to 0.927) | 0.90 | 2.05 | 67% | 2.9% |
| hybrid | 0.921 (0.894 to 0.942) | 0.90 | 1.80 | 59% | 3.2% |

Per-class coverage:
  standard  akiec 0.83  bcc 0.78  bkl 0.73  df 0.78  mel 0.80  nv 0.96  vasc 0.97
  mondrian  akiec 0.91  bcc 0.91  bkl 0.90  df 0.93  mel 0.90  nv 0.90  vasc 0.93
  hybrid    akiec 0.91  bcc 0.91  bkl 0.73  df 0.78  mel 0.90  nv 0.96  vasc 0.97

| alpha | Method | Coverage | Set size | Referred | Missed non-benign |
| --- | --- | --- | --- | --- | --- |
| 0.02 | standard | 0.980 | 2.60 | 71% | 0.7% |
| 0.02 | hybrid | 0.987 | 3.77 | 100% | 0.0% |
| 0.05 | standard | 0.951 | 1.90 | 61% | 3.3% |
| 0.05 | hybrid | 0.961 | 2.58 | 76% | 0.7% |
| 0.10 | standard | 0.900 | 1.46 | 49% | 7.4% |
| 0.10 | hybrid | 0.921 | 1.80 | 59% | 3.2% |
| 0.15 | standard | 0.850 | 1.21 | 37% | 12.9% |
| 0.15 | hybrid | 0.874 | 1.46 | 50% | 6.4% |
| 0.20 | standard | 0.801 | 1.07 | 30% | 18.9% |
| 0.20 | hybrid | 0.827 | 1.25 | 42% | 9.5% |

Saved reports/figures/referral_tradeoff.png
