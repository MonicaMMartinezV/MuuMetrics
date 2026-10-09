#!/usr/bin/env python3
# =============================================================
# Nombre del archivo: assign_ids.py
# Descripción: Asigna IDs nuevos de vaca (0, 1, 2, ...) SIN usar los
#   registros de ordeño (cows24 / patadas). Solo usa la apariencia visual.
#
#   1) Embedding por recorte con ConvNeXt preentrenado (ImageNet, sin entrenar).
#   2) Un embedding por tracklet (promedio de sus recortes).
#   3) Clustering aglomerativo por similitud coseno (centroides), con la regla:
#      dos tracklets que comparten una foto NO pueden ser la misma vaca.
#   4) Cada cluster = un ID nuevo. Se ordenan de mayor a menor número de fotos.
#
# Salidas (en --out):
#   detections_ids.csv : detections.csv + columna cow_id
#   cows_summary.csv   : un renglón por ID (nº detecciones, fotos, tracklets)
#   montages/cow_XX.jpg: muestra de recortes por ID, para revisión manual
#
# Advertencia: es agrupamiento sin etiquetas. El umbral se elige a ojo con
# --sweep y la validación real es manual.
# =============================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw


def embed_crops(crops_dir, det_ids, device):
    model = timm.create_model("convnext_base", pretrained=True, num_classes=0).to(device).eval()
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    files = {int(p.name.split("_")[0]): p for p in Path(crops_dir).glob("*.jpg")}
    ids = [i for i in det_ids if i in files]
    feats = {}
    with torch.no_grad():
        for s in range(0, len(ids), 32):
            batch = ids[s:s + 32]
            x = torch.stack([tf(Image.open(files[i]).convert("RGB")) for i in batch]).to(device)
            f = torch.nn.functional.normalize(model(x), dim=1).cpu().numpy()
            for i, v in zip(batch, f):
                feats[i] = v
    print(f"Embeddings calculados: {len(feats)} de {len(det_ids)} detecciones")
    return feats


def build_tracklet_vectors(df, feats):
    """Promedio normalizado de los embeddings de cada tracklet."""
    tids, vecs, img_sets = [], [], []
    for tid, g in df.groupby("tracklet_id"):
        g = g[g.det_id.isin(feats.keys())]
        if g.empty:
            continue
        v = np.mean([feats[d] for d in g.det_id], axis=0)
        tids.append(tid)
        vecs.append(v / np.linalg.norm(v))
        img_sets.append(set(g.image))
    return tids, np.array(vecs), img_sets


def cluster(vecs, img_sets, threshold):
    """Aglomerativo por centroides. Respeta la regla de 'misma foto => distinta vaca'."""
    n = len(vecs)
    V = vecs.copy()
    S = V @ V.T
    np.fill_diagonal(S, -np.inf)

    # conflicto[i, j] = True si i y j comparten alguna foto
    C = np.zeros((n, n), dtype=bool)
    by_image = {}
    for idx, imgs in enumerate(img_sets):
        for im in imgs:
            by_image.setdefault(im, []).append(idx)
    for members in by_image.values():
        for a in members:
            for b in members:
                C[a, b] = True
    np.fill_diagonal(C, False)

    active = np.ones(n, dtype=bool)
    members = [[i] for i in range(n)]
    while True:
        M = np.where(active[:, None] & active[None, :] & ~C, S, -np.inf)
        k = int(np.argmax(M))
        i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < threshold:
            break
        ni, nj = len(members[i]), len(members[j])
        V[i] = (V[i] * ni + V[j] * nj)
        V[i] /= np.linalg.norm(V[i])
        members[i] += members[j]
        members[j] = []
        active[j] = False
        C[i] = C[i] | C[j]
        C[:, i] = C[i]
        C[i, i] = False
        S[i] = V @ V[i]
        S[:, i] = S[i]
        S[i, i] = -np.inf

    labels = np.full(n, -1)
    clusters = [members[i] for i in range(n) if active[i]]
    clusters.sort(key=len, reverse=True)
    for cow_id, mem in enumerate(clusters):
        for m in mem:
            labels[m] = cow_id
    return labels, len(clusters)


def sweep(vecs, img_sets, thresholds):
    print("umbral  -> nº de IDs")
    for t in thresholds:
        _, k = cluster(vecs, img_sets, t)
        print(f"{t:.2f}    -> {k}")


def make_montages(df, crops_dir, out_dir, per_cow=16):
    mdir = out_dir / "montages"
    mdir.mkdir(parents=True, exist_ok=True)
    files = {int(p.name.split("_")[0]): p for p in Path(crops_dir).glob("*.jpg")}
    cell = 160
    for cow_id, g in df.groupby("cow_id"):
        picks = g.sample(min(per_cow, len(g)), random_state=0).det_id.tolist()
        cols = 4
        rows = int(np.ceil(len(picks) / cols))
        sheet = Image.new("RGB", (cols * cell, rows * (cell + 18)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, d in enumerate(picks):
            if d not in files:
                continue
            im = Image.open(files[d]).convert("RGB").resize((cell, cell))
            x, y = (k % cols) * cell, (k // cols) * (cell + 18)
            sheet.paste(im, (x, y + 18))
            draw.text((x + 4, y + 3), f"ID {cow_id} · det {d}", fill="black")
        sheet.save(mdir / f"cow_{cow_id:02d}.jpg", quality=88)
    print(f"Montajes en: {mdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Carpeta con detections.csv y crops/")
    ap.add_argument("--threshold", type=float, default=0.6, help="Similitud coseno mínima para unir dos grupos")
    ap.add_argument("--sweep", action="store_true", help="Solo imprime nº de IDs para varios umbrales")
    args = ap.parse_args()

    run = Path(args.run)
    df = pd.read_csv(run / "detections.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", device)

    feats = embed_crops(run / "crops", df.det_id.tolist(), device)
    tids, vecs, img_sets = build_tracklet_vectors(df, feats)
    print(f"Tracklets con embedding: {len(tids)}")

    if args.sweep:
        sweep(vecs, img_sets, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        return

    labels, k = cluster(vecs, img_sets, args.threshold)
    tid_to_cow = {tid: int(lab) for tid, lab in zip(tids, labels)}
    df["cow_id"] = df.tracklet_id.map(tid_to_cow)
    df = df.dropna(subset=["cow_id"]).astype({"cow_id": int})

    summary = df.groupby("cow_id").agg(
        n_detecciones=("det_id", "size"),
        n_fotos=("image", "nunique"),
        n_tracklets=("tracklet_id", "nunique"),
        camaras=("cam", lambda s: ",".join(sorted(set(s)))),
        primera=("ts", "min"),
        ultima=("ts", "max"),
    ).reset_index()

    df.to_csv(run / "detections_ids.csv", index=False)
    summary.to_csv(run / "cows_summary.csv", index=False)
    make_montages(df, run / "crops", run)

    print(f"Umbral {args.threshold:.2f} -> {k} IDs nuevos")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
