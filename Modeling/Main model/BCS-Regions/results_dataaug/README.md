# Resultados del estudio de data augmentation

El mismo estudio (políticas A0–A6, etapa B y selección anidada A0/A1 de la etapa C;
código `augment.py`, `ablation.py`, `nested.py`, `paper_figure.py`) corrido sobre dos
versiones del modelo. Los archivos se copiaron sin modificar desde `work/aug/`.

| Carpeta | Modelo | Datos | Fecha | Resultado |
|---|---|---|---|---|
| [`v1_dinov2_37vacas/`](v1_dinov2_37vacas/README.md) | `model/main` @ `47b515d`: DINOv2-L/14 378 px, último bloque, 6 regiones | 37 vacas, 369 fotos | 2026-11-09/10 | Ninguna política mejora; la selección anidada elige A1 (flip) en 37/37 folds sin mejora demostrable |
| [`v2_dinov3_64vacas/`](v2_dinov3_64vacas/) | `model/main` @ `b9e47b0`: DINOv3-L/16 384 px, bloque 21, 6 regiones, segmentador seg3 (md5 `cb2c6bf8`) | 64 vacas, 542 fotos (`Dataset_Fr_BCS_v2`) | 2026-11-11/12 | Ninguna política mejora por vaca; la selección anidada elige A0 en 64/64 folds (= A0). EXPERIMENTOS [29] |

En `v2_dinov3_64vacas/` también están `controls.json` (A0 reproduce el modelo canónico,
dif. máx. 5e-5; A0-DUP idéntico a A0) y `audit.json` (114 folds recalculados desde cero:
0 particiones y 0 alphas distintos). Su `nested_A0A1/report.md` es el reporte que genera
`nested.py`; `report_aggregate.md` solo existe en v1.

La etapa C de v2 se cortó por un reinicio de la máquina (2026-11-11 16:42, fold 23/64) y se
repitió completa con `nested.py run --overwrite` el 2026-11-12.
