| Policy | Description | LOCO MAE | LOCO ΔMAE vs A0 [95 % CI] | 5-fold×10 MAE | 5-fold×10 ΔMAE vs A0 [95 % CI] | Assessment |
|---|---|---|---|---|---|---|
| A0 | No augmentation (baseline) | 0.583 | — | 0.629 | — | Reference |
| A1 | Horizontal flip | 0.576 | -0.007 [-0.021, +0.007] | 0.623 | -0.005 [-0.018, +0.007] | Lower MAE, n.s. |
| A3 | Rotation ±5° | 0.609 | +0.026 [+0.008, +0.047] | 0.649 | +0.020 [-0.004, +0.047] | Higher MAE† |
| A4 | Translation ±3 %, scale ±3 % | 0.598 | +0.016 [-0.009, +0.044] | 0.644 | +0.016 [-0.010, +0.043] | Higher MAE, n.s. |
| B-BR05 | Brightness ±5 % | 0.588 | +0.006 [-0.001, +0.012] | 0.638 | +0.009 [+0.000, +0.018] | Higher MAE† |
| B-BR10 | Brightness ±10 % | 0.589 | +0.007 [-0.001, +0.015] | 0.639 | +0.010 [-0.001, +0.020] | Higher MAE, n.s. |
| B-CT05 | Contrast ±5 % | 0.589 | +0.006 [+0.001, +0.012] | 0.639 | +0.011 [+0.002, +0.020] | Higher MAE† |
| B-CT10 | Contrast ±10 % | 0.588 | +0.005 [-0.001, +0.012] | 0.638 | +0.010 [-0.000, +0.019] | Higher MAE, n.s. |
| A2 | Brightness + contrast ±10 % | 0.590 | +0.007 [-0.001, +0.016] | 0.640 | +0.011 [+0.000, +0.023] | Higher MAE† |
| A5 | A1 + A2 + A3 + A4 | 0.599 | +0.016 [-0.003, +0.033] | 0.645 | +0.017 [-0.001, +0.035] | Higher MAE, n.s. |
| A6 | A5 + Gaussian noise | 0.601 | +0.018 [-0.003, +0.038] | 0.649 | +0.020 [+0.002, +0.039] | Higher MAE† |
| Nested | A0/A1 chosen on training cows (δ = 0; A1 in 37/37 folds) | 0.576 | -0.007 [-0.021, +0.007] | — | — | Lower MAE, n.s. |

n.s.: the 95 % CI of ΔMAE includes 0 in all four analyses (photo- and cow-level MAE, LOCO and 5-fold × 10). †: the 95 % CI excludes 0 in at least one of the four analyses; not adjusted for multiple comparisons.
