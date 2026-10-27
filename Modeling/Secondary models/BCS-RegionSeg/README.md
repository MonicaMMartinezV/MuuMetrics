# BCS-RegionSeg — segmentación de regiones de BCS con traslape

Prueba de concepto para segmentar automáticamente las 6 regiones anatómicas usadas en la
evaluación de condición corporal (BCS) a partir de `Dataset/BCS_masking.sam2` (6 imágenes,
máscaras SAM2/RLE exportadas de Roboflow).

Regiones (orden canónico de canales): `loin` (lomo), `hips` (cadera), `rump` (grupa/isquiones),
`tailhead` (base de cola), `thigh_L`, `thigh_R` (muslos). Las regiones se traslapan
(~45 % de los píxeles etiquetados pertenecen a 2+ regiones), por eso se usa una cabeza
**multi-etiqueta (6 sigmoides)** en lugar de softmax.

## Resultados (leave-one-image-out, IoU en la imagen no vista)

| Modelo | Lomo | Cadera | Grupa | Base cola | Muslo izq. | Muslo der. | mIoU |
|---|---|---|---|---|---|---|---|
| DINOv2-B (congelado) + FPN multi-etiqueta | 0.76 | 0.63 | 0.78 | 0.48 | 0.65 | 0.64 | **0.657** |
| ConvNeXt-T (fine-tuned) + FPN multi-etiqueta | 0.73 | 0.71 | 0.70 | 0.53 | 0.67 | 0.57 | **0.650** |
| SAM2-Hiera-B+ (congelado) + FPN multi-etiqueta | 0.71 | 0.60 | 0.63 | 0.35 | 0.64 | 0.55 | 0.579 |
| YOLO11s-seg (instancias) | 0.72 | 0.67 | 0.68 | 0.26 | 0.60 | 0.46 | 0.565 |

Detalle por imagen en `results/`. Comparación visual en `figures/comparison_all.jpg`
(columnas: GT | ConvNeXt | DINOv2 | SAM2 | YOLO).

Conclusiones:
- DINOv2 y ConvNeXt empatan y son complementarios (DINOv2 mejor en poses giradas,
  ConvNeXt en vacas pequeñas/lejanas) → candidato: ensamble de ambos.
- SAM2 sigue bordes y manchas Holstein, no la anatomía; YOLO-seg da máscaras en bloque y
  pierde la base de cola. No recomendados para regiones sin contorno.
- Con 6 imágenes todo sobreajusta: el siguiente paso es etiquetar 30–50 imágenes **con clase
  asignada** en Roboflow (el export actual no trae clase y el orden de anotaciones varía).

## Archivos

- `rle.py` — decodificador COCO RLE comprimido (sin pycocotools).
- `label.py` — asigna clase a cada máscara por geometría → `labels.json` (verificado a mano).
- `prep.py` — construye `data.npz` (imágenes + máscaras 6×H×W).
- `train_ml.py` — entrenamiento multi-etiqueta LOO. Args: `<encoder timm> <iters> [freeze]`.
- `yolo_loo.py` — YOLO11-seg LOO.
- `viz.py`, `viz_pred.py` — visualización de anotaciones y predicciones.

## Cómo correr

Todo corre en la imagen `muumetrics-tracking` (torch + timm + ultralytics, GPU). Los scripts
esperan el directorio de trabajo montado en `/w` y el dataset en `/d`:

```bash
W=<carpeta de trabajo con estos scripts y labels.json>
DS=Dataset/BCS_masking.sam2/train
RUN="docker run --rm --gpus all -u $(id -u):$(id -g) -e HOME=/w -e HF_HOME=/w/.hf -e YOLO_CONFIG_DIR=/w/.yolo \
     -v $W:/w -v $PWD/$DS:/d:ro --entrypoint python muumetrics-tracking:latest"

$RUN /w/prep.py
$RUN /w/train_ml.py convnext_tiny.fb_in22k 1500
$RUN /w/train_ml.py vit_base_patch14_dinov2.lvd142m 1500 freeze
$RUN /w/train_ml.py sam2_hiera_base_plus.fb_r896 1500 freeze
$RUN /w/yolo_loo.py yolo11s-seg.pt 150
$RUN /w/viz_pred.py
```
