> Nota: la sección "2. Cámaras" no es válida: cam0 y cam4 son la misma cámara (EXPERIMENTOS [31], corrección). No se usa en el paper.

# Análisis para el artículo (firma {"backbone": "vit_large_patch16_dinov3.lvd1689m", "size": 320, "layer": 21, "regions": ["hips", "rump", "tailhead", "thigh_L", "thigh_R"], "seg_md5": "cb2c6bf80c3da295dc95fe92364b3e72"})

## 1. Error por BCS (LOCO)

| BCS | vacas | fotos | MAE | MAE vaca | sesgo | pred. media |
|---|---|---|---|---|---|---|
| 2.00 | 1 | 33 | 0.487 | 0.487 | +0.445 | 2.44 |
| 2.25 | 4 | 71 | 0.546 | 0.460 | +0.531 | 2.78 |
| 2.50 | 6 | 47 | 0.567 | 0.627 | +0.567 | 3.07 |
| 2.75 | 3 | 23 | 0.748 | 0.714 | +0.733 | 3.48 |
| 3.00 | 2 | 12 | 0.369 | 0.288 | +0.359 | 3.36 |
| 3.25 | 1 | 7 | 0.424 | 0.424 | -0.290 | 2.96 |
| 3.50 | 6 | 34 | 0.244 | 0.251 | +0.070 | 3.57 |
| 3.75 | 3 | 22 | 0.499 | 0.416 | +0.494 | 4.24 |
| 4.00 | 4 | 66 | 0.365 | 0.340 | -0.328 | 3.67 |
| 4.25 | 9 | 45 | 0.331 | 0.285 | -0.190 | 4.06 |
| 4.50 | 17 | 94 | 0.395 | 0.392 | -0.322 | 4.18 |
| 4.75 | 9 | 48 | 0.427 | 0.539 | -0.390 | 4.36 |
| 5.00 | 2 | 40 | 0.295 | 0.255 | -0.275 | 4.72 |

Bland-Altman por vaca x BCS (n=67): sesgo -0.043, LoA [-0.978, +0.892], pendiente del sesgo con el nivel -0.375; r 0.846, pendiente pred~real 0.596, MAE 0.387, |dif| <= 0.25: 39%, <= 0.5: 66%.
Por foto (n=542): sesgo +0.039, LoA [-1.021, +1.099].

## 2. Cámaras (fotos {np.str_('0'): 336, np.str_('4'): 206}, vacas {np.str_('0'): 57, np.str_('4'): 49}; LOCO por vaca)

| prueba | entrena | fotos | vacas | MAE | MAE vaca | r | MAE vaca (entrena todo) | dif vs misma cámara [IC95] |
|---|---|---|---|---|---|---|---|---|
| cam0 | cam0 | 336 | 57 | 0.462 | 0.498 | 0.800 | 0.450 | +0.000 [+0.000, +0.000] |
| cam0 | cam4 | 336 | 57 | 0.453 | 0.458 | 0.810 | 0.450 | -0.040 [-0.100, +0.016] |
| cam4 | cam0 | 206 | 49 | 0.493 | 0.438 | 0.860 | 0.392 | +0.004 [-0.063, +0.073] |
| cam4 | cam4 | 206 | 49 | 0.449 | 0.434 | 0.861 | 0.392 | +0.000 [+0.000, +0.000] |

## 3. Tiempo

Estabilidad (vaca x BCS con fotos en >= 2 días, predicción media por día): vaca_bcs_con_2_dias 55, sd_entre_dias_mediana 0.233, sd_entre_dias_media 0.274, dif_dias_consecutivos_mediana 0.234, sesgo_por_vaca_abs_medio 0.419, icc1_predicciones 0.856, sd_entre_vacas_pred 0.610

| vaca | BCS | desde | hasta | fotos | pred. media | pred. sd |
|---|---|---|---|---|---|---|
| 8758 | 3.50 | 2025-07-15 | 2025-07-15 | 1 | 3.54 | 0.00 |
| 8758 | 4.00 | 2025-06-01 | 2025-06-23 | 11 | 4.01 | 0.19 |
| 8758 | 4.25 | 2025-05-31 | 2025-05-31 | 2 | 3.92 | 0.36 |
| 9008 | 2.25 | 2025-06-24 | 2025-06-26 | 2 | 2.06 | 0.19 |
| 9008 | 4.50 | 2025-05-26 | 2025-06-08 | 10 | 3.74 | 0.50 |
