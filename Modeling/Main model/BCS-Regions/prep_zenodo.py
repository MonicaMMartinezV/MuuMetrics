# =============================================================
# Nombre del archivo: prep_zenodo.py
#
# Descripción:
#   Prepara el dataset de Zenodo (RGB-D BCS, sala rotativa) con el formato
#   de Dataset_Fr_BCS para usarlo con bcs_regions.py.
#
#   En cada foto salen 2-3 vacas; la etiquetada es la del centro (la del
#   establo que mira la cámara). Por foto:
#     1. Mask R-CNN (torchvision, COCO) detecta las vacas (clase 'cow').
#     2. Se elige la detección cuyo centro horizontal está más cerca del
#        centro de la imagen (y que cruza el centro).
#     3. Se recorta su caja (+ margen) a resolución original.
#   Salida: <out>/z<animal_id>_<BCS>/<carpeta>_<hora>.jpg
#           <out>/_recortes.csv  (caja elegida, score, n vacas, motivo si se descarta)
#
# Uso (Docker, el Dataset se monta solo lectura -> salida en otro volumen):
#   docker compose run --rm -e TORCH_HOME=/work/.torch \
#     -v <Dataset>/Dataset_Zenodo_BCS:/out --entrypoint python bcs prep_zenodo.py \
#     --src /data/Zenodo_RGBD_BCS --out /out [--date 06.12.2024] [--max 0]
# =============================================================
import argparse, csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

COW = 21          # índice COCO de 'cow'
MARGIN = 0.03     # margen relativo alrededor de la caja

def merged(boxes: np.ndarray) -> np.ndarray:
    """True para cajas que contienen otra vaca detectada completa (varias vacas fusionadas en una caja)."""
    n = len(boxes); out = np.zeros(n, bool)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for j in range(n):
        for k in range(n):
            if k == j or area[k] > 0.7 * area[j]: continue
            iw = min(boxes[j, 2], boxes[k, 2]) - max(boxes[j, 0], boxes[k, 0])
            ih = min(boxes[j, 3], boxes[k, 3]) - max(boxes[j, 1], boxes[k, 1])
            if iw > 0 and ih > 0 and iw * ih / area[k] > 0.9 and (boxes[k, 2] - boxes[k, 0]) < 0.8 * (boxes[j, 2] - boxes[j, 0]):
                out[j] = True; break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/Zenodo_RGBD_BCS")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--date", default="06.12.2024")
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--max", type=int, default=0, help="máximo de fotos (0 = todas)")
    a = ap.parse_args(); src, out = Path(a.src), Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in csv.DictReader(open(src / "labels.csv")) if r["data"] == a.date]
    if a.max: rows = rows[:a.max]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    net = maskrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=a.score).to(dev).eval()

    log, kept = [], 0
    for i, r in enumerate(rows):
        rel = Path(*r["RGB_path"].replace("\\", "/").split("/")).with_suffix(".jpg")
        img = Image.open(src / "rgb" / rel).convert("RGB"); W, H = img.size
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255).to(dev)
        with torch.no_grad():
            det = net([x])[0]
        keep = det["labels"] == COW
        boxes, scores = det["boxes"][keep].cpu().numpy(), det["scores"][keep].cpu().numpy()
        entry = {"foto": str(rel), "vaca": r["animal_id"], "bcs": r["BCS"], "n_vacas": len(boxes)}
        cx = (boxes[:, 0] + boxes[:, 2]) / 2 if len(boxes) else np.array([])
        crosses = (boxes[:, 0] < W / 2) & (boxes[:, 2] > W / 2) if len(boxes) else np.array([], bool)
        crosses &= ~merged(boxes)
        if not crosses.any():
            entry["motivo"] = "sin vaca en el centro"; log.append(entry); continue
        j = np.where(crosses)[0][np.argmin(np.abs(cx[crosses] - W / 2))]
        x0, y0, x1, y1 = boxes[j]; mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
        box = (int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(W, x1 + mx)), int(min(H, y1 + my)))
        frame = rel.parts[1]
        dst = out / f"z{r['animal_id']}_{float(r['BCS']):g}" / f"{frame}_{rel.stem}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.crop(box).save(dst, quality=92); kept += 1
        entry.update({"score": round(float(scores[j]), 3), "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                      "centro_x_rel": round(float(cx[j] / W), 3), "recorte": str(dst.relative_to(out)), "motivo": ""})
        log.append(entry)
        if (i + 1) % 200 == 0: print(f"  {i + 1}/{len(rows)}  guardadas {kept}", flush=True)

    keys = ["foto", "vaca", "bcs", "n_vacas", "score", "centro_x_rel", "x0", "y0", "x1", "y1", "recorte", "motivo"]
    with open(out / "_recortes.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(log)
    print(f"[zenodo] {kept}/{len(rows)} recortes -> {out} | descartadas: {len(rows) - kept}")

if __name__ == "__main__":
    main()
