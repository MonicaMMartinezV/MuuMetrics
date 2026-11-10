# =============================================================
# Nombre del archivo: prep_unict_bcs.py
#
# Descripción:
#   Prepara las fotos UNICT (BCS Database, iplab.dmi.unict.it/legacy/BCS)
#   con el formato de Dataset_Fr_BCS para bcs_regions.py.
#   Por foto (704x576 cenital, cola a la derecha):
#     1. Giro 90° horario (cola abajo, como el robot).
#     2. Segmentador nuevo (robot + UNICT) sobre la foto completa.
#     3. Recorte = caja de la unión de las 6 regiones (prob > 0.5) + margen.
#   BCS = promedio de los 2 evaluadores (BCS1, BCS2 en el nombre del .txt).
#   Salida: <out>/u<vaca>_<BCS>/<foto>.jpg  +  <out>/_recortes.csv
#
# Uso (Docker, imagen muumetrics-seg):
#   python prep_unict_bcs.py --src /data/UNICT_raw --labels /data/Dataset_UNICT_seg/bcs_unict_207.csv \
#       --out /out --seg_weights /work/seg/final_all40_unict40.pt
# =============================================================
import argparse, csv
from pathlib import Path
import numpy as np, torch
from PIL import Image
from segmenter import load_segmenter, segment, REGIONS

MARGIN = 0.05

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/UNICT_raw")
    ap.add_argument("--labels", default="/data/Dataset_UNICT_seg/bcs_unict_207.csv")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--seg_weights", default="/work/seg/final_all40_unict40.pt")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); net = load_segmenter(a.seg_weights, dev)
    log = []
    for r in csv.DictReader(open(a.labels)):
        f = Path(a.src, "cows", f"{r['foto']}_cow.jpg"); e = {"foto": r["foto"], "vaca": r["vaca"], "bcs": r["bcs"]}
        if not f.exists(): e["motivo"] = "sin imagen"; log.append(e); continue
        img = np.asarray(Image.open(f).convert("RGB").transpose(Image.Transpose.ROTATE_270))  # 90° horario
        p = segment(net, img, dev) > 0.5; H, W = img.shape[:2]
        area = {k: round(float(p[i].mean()), 4) for i, k in enumerate(REGIONS)}
        missing = [k for k, v in area.items() if v == 0]
        u = p.any(0); ys, xs = np.nonzero(u)
        if not len(xs): e["motivo"] = "sin regiones"; log.append(e); continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
        box = (int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(W, x1 + mx + 1)), int(min(H, y1 + my + 1)))
        dst = out / f"u{r['vaca']}_{float(r['bcs']):g}" / f"{r['foto']}.jpg"; dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).crop(box).save(dst, quality=95)
        e.update({"x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3], "regiones_vacias": ";".join(missing),
                  "recorte": str(dst.relative_to(out)), "motivo": "", **{f"area_{k}": v for k, v in area.items()}})
        log.append(e)
    keys = ["foto", "vaca", "bcs", "recorte", "motivo", "regiones_vacias", "x0", "y0", "x1", "y1"] + [f"area_{k}" for k in REGIONS]
    with open(out / "_recortes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(log)
    ok = [e for e in log if not e.get("motivo")]
    print(f"[unict] {len(ok)}/{len(log)} recortes -> {out} | vacas: {len({e['vaca'] for e in ok})} | "
          f"con región vacía: {sum(1 for e in ok if e['regiones_vacias'])} | descartadas: {[(e['foto'], e['motivo']) for e in log if e.get('motivo')]}")

if __name__ == "__main__":
    main()
