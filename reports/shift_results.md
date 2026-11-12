Calibrated on clean test images, evaluated on shifted ones; alpha = 0.1

| Shift | Level | Balanced acc. | ECE after T | Coverage, standard | Coverage, hybrid | Melanoma, hybrid | Set size, hybrid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none |  | 0.715 | 0.023 | 0.900 | 0.920 | 0.90 | 1.79 |
| blur | sigma 0.5 | 0.711 | 0.025 | 0.903 | 0.920 | 0.90 | 1.78 |
| blur | sigma 1 | 0.550 | 0.051 | 0.867 | 0.898 | 0.83 | 1.77 |
| blur | sigma 2 | 0.343 | 0.152 | 0.771 | 0.816 | 0.41 | 1.49 |
| brightness | x 0.5 | 0.205 | 0.085 | 0.775 | 0.823 | 0.67 | 1.75 |
| brightness | x 0.75 | 0.528 | 0.037 | 0.871 | 0.901 | 0.89 | 1.90 |
| brightness | x 1.25 | 0.621 | 0.040 | 0.851 | 0.871 | 0.81 | 2.12 |
| brightness | x 1.5 | 0.491 | 0.077 | 0.721 | 0.754 | 0.81 | 2.78 |
| resolution | 32 px | 0.479 | 0.081 | 0.838 | 0.882 | 0.70 | 1.70 |
| resolution | 16 px | 0.333 | 0.154 | 0.757 | 0.803 | 0.31 | 1.46 |
