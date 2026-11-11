# Etapa C: selección anidada A0/A1 (LOCO externo, LOCO interno por vaca)

Selección δ=0 (primario): A0 en 0 folds, A1 en 37 folds de 37.  
Selección δ=0.01 (sensibilidad): A0 en 31, A1 en 6.  
Control: predicciones finales A0 vs results/A0 máx |Δ| = 0.00e+00; A1 vs results/A1 máx |Δ| = 0.00e+00.

| Procedimiento | MAE fotos [IC 95 %] | MAE vaca [IC 95 %] | RMSE | R² fotos | R² vacas | Bias | ≤0.5 | ΔMAE fotos vs A0 [IC 95 %] | ΔMAE vaca vs A0 [IC 95 %] | Vacas mejor/peor |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 (canónico) | 0.5826 [0.4736, 0.6928] | 0.5454 [0.4646, 0.6332] | 0.6819 | 0.549 | 0.519 | +0.0911 | 55.8% | — | — | — |
| A1 siempre | 0.5758 [0.4640, 0.6894] | 0.5387 [0.4516, 0.6326] | 0.6775 | 0.555 | 0.517 | +0.0954 | 54.7% | -0.0068 [-0.0213, +0.0074] | -0.0067 [-0.0247, +0.0107] | 19/18 |
| Anidado δ=0 (primario) | 0.5758 [0.4640, 0.6894] | 0.5387 [0.4516, 0.6326] | 0.6775 | 0.555 | 0.517 | +0.0954 | 54.7% | -0.0068 [-0.0213, +0.0074] | -0.0067 [-0.0247, +0.0107] | 19/18 |
| Anidado δ=0.01 (sensibilidad) | 0.5889 [0.4777, 0.7017] | 0.5530 [0.4702, 0.6429] | 0.6891 | 0.540 | 0.503 | +0.0951 | 55.3% | +0.0063 [+0.0009, +0.0132] | +0.0076 [+0.0014, +0.0155] | 0/6 |
