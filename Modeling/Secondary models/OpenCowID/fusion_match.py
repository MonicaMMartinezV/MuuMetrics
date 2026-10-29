# =============================================================
# Nombre del archivo: fusion_match.py
# Descripción: Cruce ByID -> CAETEC combinando dos señales (al estilo
#   WildFusion):
#     - global: similitud coseno de MegaDescriptor-L-384 (prototipo por vaca/ID)
#     - local:  ALIKED + LightGlue + RANSAC; nº de puntos correspondientes
#               (inliers) entre fotos, media de los 3 mejores pares
#   Solo usa fotos de CAETEC tomadas desde atrás (vistas.csv de
#   view_filter.py, prob >= --min_rear), que es lo que ve la cámara del
#   robot. Candidatos por vaca: IDs en ordeño todos sus días con fotos.
#   Fusión: suma de z-scores de ambas señales entre los candidatos.
#
#   Controles (para saber si cada señal sirve en estas fotos):
#     - AUC local en ByID: pares misma vaca/otro día vs vacas distintas
#     - AUC local en CAETEC: pares mismo ID vs IDs distintos
#     - anclas: posición del ID del robot sugerido confiable
#
# Uso (en la imagen muumetrics-opencowid, con GPU):
#   python fusion_match.py --byid_dir /data/ByID --caetec_dir /data/opencowid/caetec_crops_448 \
#       --views_csv /data/opencowid/views/vistas.csv --ordeno_dir /data/GeneralidadesOrdeno \
#       --robot_csv /data/byid_vs_robot/vacas_byid.csv --out_dir /data/opencowid/fusion
# =============================================================

import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import ALIKED, LightGlue
from lightglue.utils import load_image, rbd

from match_byid_caetec import (DEVICE, embed, load_anchors, load_byid, load_megadescriptor,
                               load_presence, proto, review_sheet)

N_BYID, N_CAETEC = 4, 3


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


class LocalMatcher:
    def __init__(self, max_kp=1024, long_side=512):
        self.ext = ALIKED(max_num_keypoints=max_kp).eval().to(DEVICE)
        self.lg = LightGlue(features="aliked").eval().to(DEVICE)
        self.long_side, self.cache = long_side, {}

    @torch.no_grad()
    def feats(self, path):
        if path not in self.cache:
            img = load_image(path, resize=self.long_side).to(DEVICE)
            self.cache[path] = self.ext.extract(img)
        return self.cache[path]

    @torch.no_grad()
    def inliers(self, a, b):
        fa, fb = self.feats(a), self.feats(b)
        m = rbd(self.lg({"image0": fa, "image1": fb}))["matches"]
        if len(m) < 8:
            return 0
        pa = rbd(fa)["keypoints"][m[:, 0]].cpu().numpy()
        pb = rbd(fb)["keypoints"][m[:, 1]].cpu().numpy()
        try:
            _, mask = cv2.findFundamentalMat(pa, pb, cv2.USAC_MAGSAC, 3.0, 0.999, 10000)
        except cv2.error:  # casos degenerados (puntos casi colineales)
            return 0
        return int(mask.sum()) if mask is not None else 0

    def score(self, paths_a, paths_b):
        s = sorted((self.inliers(a, b) for a in paths_a for b in paths_b), reverse=True)
        return float(np.mean(s[:3])) if s else 0.0


def spread(paths, n):
    """n fotos repartidas en el tiempo (los nombres empiezan con la fecha)."""
    paths = sorted(paths)
    if len(paths) <= n:
        return paths
    return [paths[int(i)] for i in np.linspace(0, len(paths) - 1, n)]


def zs(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--byid_dir", required=True)
    ap.add_argument("--caetec_dir", required=True)
    ap.add_argument("--views_csv", required=True)
    ap.add_argument("--min_rear", type=float, default=0.6)
    ap.add_argument("--ordeno_dir", required=True)
    ap.add_argument("--robot_csv", default=None)
    ap.add_argument("--anchor_support", type=float, default=0.6)
    ap.add_argument("--n_control", type=int, default=300, help="Pares positivos/negativos por control de AUC")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    random.seed(0)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Datos
    byid = load_byid(args.byid_dir)
    byid_by_cow = defaultdict(list)
    for r in byid:
        byid_by_cow[r["cow"]].append(str(r["path"]))
    rear = defaultdict(list)
    for r in csv.DictReader(open(args.views_csv)):
        if r["vista"] == "atras" and float(r["atras"]) >= args.min_rear:
            rear[r["id_robot"]].append((float(r["atras"]), str(Path(args.caetec_dir) / r["id_robot"] / r["foto"])))
    rear = {k: [p for _, p in sorted(v, reverse=True)] for k, v in rear.items()}
    presence = load_presence(args.ordeno_dir)
    anchors, sugg = load_anchors(args.robot_csv, args.anchor_support)
    print(f"ByID: {len(byid_by_cow)} vacas | CAETEC desde atrás (p>={args.min_rear}): "
          f"{sum(map(len, rear.values()))} fotos, {len(rear)} IDs | anclas: {len(anchors)}")

    # Señal global
    net, tf = load_megadescriptor()
    all_c = [p for ps in rear.values() for p in ps]
    Ec = dict(zip(all_c, embed(net, tf, all_c)))
    all_b = [p for ps in byid_by_cow.values() for p in ps]
    Eb = dict(zip(all_b, embed(net, tf, all_b)))
    del net
    torch.cuda.empty_cache()
    P_id = {k: proto(np.stack([Ec[p] for p in v])) for k, v in rear.items()}
    P_cow = {k: proto(np.stack([Eb[p] for p in v])) for k, v in byid_by_cow.items()}

    # Controles de la señal local
    lm = LocalMatcher()
    multi_day = {c: ps for c, ps in byid_by_cow.items() if len({Path(p).name[:10] for p in ps}) > 1}
    pos, neg = [], []
    cows = list(multi_day)
    for _ in range(args.n_control):
        c = random.choice(cows)
        a = random.choice(multi_day[c])
        b = random.choice([p for p in multi_day[c] if Path(p).name[:10] != Path(a).name[:10]])
        pos.append(lm.inliers(a, b))
        c2 = random.choice([x for x in cows if x != c])
        neg.append(lm.inliers(a, random.choice(byid_by_cow[c2])))
    ctrl = {"auc_local_ByID_otro_dia": auc(pos, neg), "inliers_med_pos_ByID": float(np.median(pos)),
            "inliers_med_neg_ByID": float(np.median(neg))}
    multi = [k for k, v in rear.items() if len(v) > 1]
    pos, neg = [], []
    for _ in range(args.n_control):
        k = random.choice(multi)
        a, b = random.sample(rear[k], 2)
        pos.append(lm.inliers(a, b))
        k2 = random.choice([x for x in rear if x != k])
        neg.append(lm.inliers(a, random.choice(rear[k2])))
    ctrl.update({"auc_local_CAETEC": auc(pos, neg), "inliers_med_pos_CAETEC": float(np.median(pos)),
                 "inliers_med_neg_CAETEC": float(np.median(neg))})
    print("Controles:", ctrl, flush=True)

    # Cruce
    results = {}
    for n, (cow, paths) in enumerate(sorted(byid_by_cow.items())):
        days = {Path(p).name[:10] for p in paths}
        cands = [k for k in rear if all(k in presence.get(d, {k}) for d in days)]
        q = spread(paths, N_BYID)
        g = [float(P_cow[cow] @ P_id[k]) for k in cands]
        l = [lm.score(q, rear[k][:N_CAETEC]) for k in cands]
        f = zs(g) + zs(l)
        results[cow] = {k: {"global": g[i], "local": l[i], "fusion": float(f[i])} for i, k in enumerate(cands)}
        print(f"[{n + 1}/{len(byid_by_cow)}] vaca {cow}: {len(cands)} candidatos", flush=True)

    def ranking(cow, key):
        return sorted(results[cow], key=lambda k: -results[cow][k][key])

    summary = {"controles": ctrl, "anclas": {}}
    for key in ("global", "local", "fusion"):
        pos = {c: (ranking(c, key).index(rid) + 1 if rid in results[c] else None) for c, rid in anchors.items()}
        valid = [p for p in pos.values() if p]
        summary["anclas"][key] = {"con_ID_en_candidatos": len(valid),
                                  "top1": sum(p == 1 for p in valid), "top5": sum(p <= 5 for p in valid),
                                  "posicion_mediana": float(np.median(valid)) if valid else None,
                                  "candidatos_medianos": float(np.median([len(results[c]) for c in anchors])),
                                  "detalle": {str(c): {"id_robot": anchors[c], "posicion": p} for c, p in pos.items()}}
    with open(out / "resumen.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Anclas:", {k: {kk: vv for kk, vv in v.items() if kk != "detalle"} for k, v in summary["anclas"].items()})

    with open(out / "candidatos.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ID ByID", "Fotos ByID", "ID robot sugerido (dataset_del)", "Apoyo sugerido", "Candidatos"]
                   + [f"top{k + 1} (fusión)" for k in range(5)] + ["Inliers top1", "Posición del sugerido (fusión)",
                                                                    "ID confirmado", "Notas"])
        for cow in sorted(results):
            rk = ranking(cow, "fusion")
            rid, sup = sugg.get(cow, ("", ""))
            w.writerow([cow, len(byid_by_cow[cow]), rid, sup, len(rk)]
                       + [f"{k} ({results[cow][k]['fusion']:.2f})" for k in rk[:5]] + [""] * (5 - len(rk[:5]))
                       + [round(results[cow][rk[0]]["local"], 1) if rk else "",
                          rk.index(rid) + 1 if rid in rk else "", "", ""])

    (out / "hojas").mkdir(exist_ok=True)
    for cow in results:
        rk = ranking(cow, "fusion")[:3]
        review_sheet(cow, spread(byid_by_cow[cow], 5), [(k, results[cow][k]["fusion"]) for k in rk],
                     rear, out / "hojas" / f"vaca_{cow:02d}.jpg")
    print(f"Listo: {out}")


if __name__ == "__main__":
    main()
