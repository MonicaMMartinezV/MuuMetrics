# Análisis para el artículo (firma {"backbone": "vit_large_patch16_dinov3.lvd1689m", "size": 320, "layer": 21, "regions": ["hips", "rump", "tailhead", "thigh_L", "thigh_R"], "seg_md5": "cb2c6bf80c3da295dc95fe92364b3e72"})

## 1. Error por BCS (LOCO)

| BCS | vacas | fotos | MAE | MAE vaca | sesgo | pred. media |
|---|---|---|---|---|---|---|
| 2.00 | 2 | 50 | 0.616 | 0.665 | +0.609 | 2.61 |
| 2.25 | 4 | 77 | 0.494 | 0.418 | +0.462 | 2.71 |
| 2.50 | 8 | 51 | 0.475 | 0.504 | +0.415 | 2.92 |
| 2.75 | 4 | 24 | 0.664 | 0.569 | +0.648 | 3.40 |
| 3.00 | 3 | 14 | 0.388 | 0.401 | +0.374 | 3.37 |
| 3.25 | 1 | 7 | 0.380 | 0.380 | -0.251 | 3.00 |
| 3.50 | 8 | 39 | 0.273 | 0.343 | +0.097 | 3.60 |
| 3.75 | 5 | 21 | 0.244 | 0.238 | +0.072 | 3.82 |
| 4.00 | 3 | 100 | 0.482 | 0.439 | -0.470 | 3.53 |
| 4.25 | 10 | 69 | 0.292 | 0.269 | -0.118 | 4.13 |
| 4.50 | 18 | 117 | 0.346 | 0.369 | -0.254 | 4.25 |
| 4.75 | 11 | 56 | 0.398 | 0.489 | -0.325 | 4.42 |
| 5.00 | 2 | 45 | 0.319 | 0.275 | -0.285 | 4.71 |

Bland-Altman por vaca x BCS (n=79): sesgo -0.020, LoA [-0.912, +0.872], pendiente del sesgo con el nivel -0.334; r 0.868, pendiente pred~real 0.634, MAE 0.359, |dif| <= 0.25: 43%, <= 0.5: 70%.
Por foto (n=670): sesgo -0.007, LoA [-1.040, +1.026].

## 2. Cámaras (fotos {np.str_('0'): 432, np.str_('4'): 238}, vacas {np.str_('0'): 66, np.str_('4'): 55}; LOCO por vaca)

| prueba | entrena | fotos | vacas | MAE | MAE vaca | r | MAE vaca (entrena todo) | dif vs misma cámara [IC95] |
|---|---|---|---|---|---|---|---|---|
| cam0 | cam0 | 432 | 66 | 0.426 | 0.448 | 0.825 | 0.426 | +0.000 [+0.000, +0.000] |
| cam0 | cam4 | 432 | 66 | 0.453 | 0.437 | 0.804 | 0.426 | -0.011 [-0.056, +0.034] |
| cam4 | cam0 | 238 | 55 | 0.488 | 0.458 | 0.857 | 0.367 | +0.054 [-0.013, +0.120] |
| cam4 | cam4 | 238 | 55 | 0.425 | 0.404 | 0.874 | 0.367 | +0.000 [+0.000, +0.000] |

## 3. Tiempo

Estabilidad (vaca x BCS con fotos en >= 2 días, predicción media por día): vaca_bcs_con_2_dias 59, sd_entre_dias_mediana 0.251, sd_entre_dias_media 0.273, dif_dias_consecutivos_mediana 0.227, sesgo_por_vaca_abs_medio 0.381, icc1_predicciones 0.856, sd_entre_vacas_pred 0.644

| vaca | BCS | desde | hasta | fotos | pred. media | pred. sd |
|---|---|---|---|---|---|---|
| 8758 | 3.50 | 2025-07-15 | 2025-07-15 | 1 | 3.44 | 0.00 |
| 8758 | 4.00 | 2025-06-01 | 2025-06-23 | 11 | 3.95 | 0.19 |
| 8758 | 4.25 | 2025-05-31 | 2025-05-31 | 2 | 3.93 | 0.43 |
| 9008 | 2.25 | 2025-06-24 | 2025-06-26 | 2 | 2.09 | 0.28 |
| 9008 | 4.50 | 2025-05-26 | 2025-06-08 | 10 | 3.89 | 0.60 |
