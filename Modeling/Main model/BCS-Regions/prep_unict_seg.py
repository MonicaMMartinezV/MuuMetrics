# =============================================================
# Nombre del archivo: prep_unict_seg.py
#
# Descripción:
#   Prepara el export de Roboflow "BCS Dataset 2-vista_superior.coco"
#   (43 fotos cenitales UNICT 704x576, polígonos de 6 regiones) como datos
#   de entrenamiento del segmentador (segmenter.py).
#   Por foto:
#     1. Polígonos COCO -> 6 máscaras binarias (se traslapan) en el orden
#        de segmenter.REGIONS: loin, hips, rump, tailhead, thigh_L, thigh_R.
#          costillas->loin, Ileon->hips, Isquion->rump, base_cola->tailhead,
#          muslo_izq/muslo_der -> thigh_L/thigh_R (o al revés con --swap-thighs).
#     2. Giro 90° horario (cola abajo, como el robot).
#   Correcciones de etiqueta (clase repetida en la misma foto, ver FIXES).
#   Salida: <out>/<split>/<vaca>_<resto>.jpg  +  .npz (masks uint8 [6,h,w] 0/255, classes)
#           <out>/_log.csv
#
# Uso:
#   python3 prep_unict_seg.py --src "<Dataset>/BCS Dataset 2-vista_superior.coco" \
#       --out <Dataset>/Dataset_UNICT_seg
# =============================================================
import argparse, csv, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

REGIONS = ["loin", "hips", "rump", "tailhead", "thigh_L", "thigh_R"]
CAT2REG = {"costillas": "loin", "Ileon": "hips", "Isquion": "rump", "base_cola": "tailhead",
           "muslo_izq": "thigh_L", "muslo_der": "thigh_R"}
# (prefijo del archivo, caja COCO redondeada) -> clase correcta
FIXES = {("714_4_5_20_184_cow", (532, 160, 67, 85)): "base_cola",
         ("44_4_7_20_3_cow", (15, 195, 292, 301)): "costillas"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--swap-thighs", action="store_true", help="muslo_izq->thigh_R y muslo_der->thigh_L")
    a = ap.parse_args(); src, out = Path(a.src), Path(a.out)
    c2r = dict(CAT2REG)
    if a.swap_thighs: c2r["muslo_izq"], c2r["muslo_der"] = "thigh_R", "thigh_L"
    log = []
    for split in ["train", "valid", "test"]:
        d = json.load(open(src / split / "_annotations.coco.json"))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        anns = {}
        for an in d["annotations"]: anns.setdefault(an["image_id"], []).append(an)
        (out / split).mkdir(parents=True, exist_ok=True)
        for im in d["images"]:
            fn, W, H = im["file_name"], im["width"], im["height"]
            stem = fn.split("_jpg.rf.")[0]
            masks = np.zeros((6, H, W), np.uint8); seen, fixed = [], []
            for an in anns.get(im["id"], []):
                name = cats[an["category_id"]]
                key = (stem, tuple(int(round(v)) for v in an["bbox"]))
                if key in FIXES: fixed.append(f"{name}->{FIXES[key]}"); name = FIXES[key]
                k = REGIONS.index(c2r[name]); seen.append(name)
                canvas = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(canvas)
                for poly in an["segmentation"]:
                    if len(poly) >= 6: dr.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], fill=255)
                masks[k] = np.maximum(masks[k], np.asarray(canvas))
            dup = sorted({s for s in seen if seen.count(s) > 1})
            empty = [REGIONS[k] for k in range(6) if not masks[k].any()]
            img = Image.open(src / split / fn).convert("RGB").transpose(Image.Transpose.ROTATE_270)  # 90° horario
            masks = np.ascontiguousarray(np.rot90(masks, k=-1, axes=(1, 2)))
            name = stem.removesuffix("_cow")
            img.save(out / split / f"{name}.jpg", quality=95)
            np.savez_compressed(out / split / f"{name}.npz", masks=masks, classes=np.array(REGIONS))
            log.append({"split": split, "foto": name, "vaca": name.split("_")[0], "original": fn,
                        "correcciones": ";".join(fixed), "duplicadas": ";".join(dup), "vacias": ";".join(empty)})
    with open(out / "_log.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(log[0])); wr.writeheader(); wr.writerows(log)
    bad = [r for r in log if r["duplicadas"] or r["vacias"]]
    print(f"[unict-seg] {len(log)} fotos -> {out} | con problemas: {len(bad)}")
    for r in bad: print("  ", r)

if __name__ == "__main__":
    main()
