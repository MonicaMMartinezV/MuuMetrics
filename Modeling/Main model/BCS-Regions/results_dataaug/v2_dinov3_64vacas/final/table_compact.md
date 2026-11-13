| Policy | Description | LOCO MAE | LOCO ΔMAE vs A0 [95 % CI] | 5-fold×10 MAE | 5-fold×10 ΔMAE vs A0 [95 % CI] | Assessment |
|---|---|---|---|---|---|---|
| A0 | No augmentation (baseline) | 0.452 | — | 0.464 | — | Reference |
| A1 | Horizontal flip | 0.446 | -0.006 [-0.017, +0.008] | 0.464 | -0.000 [-0.009, +0.011] | Lower MAE, n.s. |
| A3 | Rotation ±5° | 0.456 | +0.003 [-0.006, +0.013] | 0.473 | +0.009 [-0.000, +0.019] | Higher MAE, n.s. |
| A4 | Translation ±3 %, scale ±3 % | 0.448 | -0.004 [-0.015, +0.007] | 0.466 | +0.002 [-0.010, +0.014] | No consistent change |
| B-BR05 | Brightness ±5 % | 0.453 | +0.000 [-0.001, +0.002] | 0.466 | +0.002 [-0.001, +0.005] | Higher MAE, n.s. |
| B-BR10 | Brightness ±10 % | 0.454 | +0.002 [-0.000, +0.003] | 0.469 | +0.005 [-0.000, +0.010] | Higher MAE, n.s. |
| B-CT05 | Contrast ±5 % | 0.452 | -0.000 [-0.001, +0.001] | 0.467 | +0.003 [-0.000, +0.006] | No consistent change |
| B-CT10 | Contrast ±10 % | 0.453 | +0.001 [-0.000, +0.002] | 0.469 | +0.005 [+0.000, +0.010] | Higher MAE† |
| A2 | Brightness + contrast ±10 % | 0.454 | +0.002 [-0.000, +0.004] | 0.469 | +0.005 [-0.000, +0.011] | Higher MAE, n.s. |
| A5 | A1 + A2 + A3 + A4 | 0.444 | -0.008 [-0.018, +0.003] | 0.460 | -0.004 [-0.013, +0.007] | Lower MAE, n.s. |
| A6 | A5 + Gaussian noise | 0.442 | -0.011 [-0.021, +0.002] | 0.459 | -0.006 [-0.015, +0.005] | Lower MAE, n.s. |
| Nested | A0/A1 chosen on training cows (δ = 0; A1 in 0/64 folds) | 0.452 | +0.000 [+0.000, +0.000] | — | — | Identical to A0 (A0 chosen in every fold) |

n.s.: the 95 % CI of ΔMAE includes 0 in all four analyses (photo- and cow-level MAE, LOCO and 5-fold × 10). †: the 95 % CI excludes 0 in at least one of the four analyses; not adjusted for multiple comparisons.
