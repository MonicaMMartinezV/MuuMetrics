# =============================================================
# Nombre del archivo: reid_head.py
# Descripción: Re-identificación ByID (cámara del robot, salida del VMS 3)
#   -> fotos de celular (CAETEC + "IDs identificados como vigentes"),
#   entrenada con pares de la misma vaca en ambos dominios:
#     - fotos de ByID con ID del robot "seguro" (match_byid_robot_visits.py)
#     - fotos de celular recortadas (crop_caetec.py) con su vista (view_filter.py)
#
#   Embeddings de MegaDescriptor-L-384 (congelado, promedio con volteo
#   horizontal) y tres métodos encima:
#     crudo     similitud coseno directa
#     centrado  se resta la media de cada dominio (robot / celular)
#     cabeza    proyección lineal (inicia en identidad) + clasificador
#               ArcFace sobre IDs, con los dominios pesados por igual
#
#   Evaluación (5 folds sobre los IDs seguros de ByID): las fotos del robot
#   de los IDs de prueba no se usan para entrenar; cada ID de prueba
#   (prototipo de sus fotos ByID) se busca entre las fotos de celular de
#   sus candidatos: vacas del grupo 3 (= VMS 3) todos los días con foto.
#   Se comparan entrenar/galería con solo fotos traseras vs todas.
#
#   Con el mejor método, entrenado con todos los IDs seguros, propone IDs
#   para las carpetas de ByID sin ID seguro. A las de sin_match se les
#   quitan los IDs del registro de ordeños con hora (si fueran una de
#   esas vacas, el horario ya las habría ligado).
#
#   Hubs: opcionalmente se aplica CSLS (se resta a cada ID de la galería
#   su similitud media con las carpetas de ByID más parecidas).
#
#   Propuestas (propuestas.csv) para subir carpetas a seguros/dudosos: ver
#   propose(). La columna "Nivel final" se llena a mano tras revisar las
#   hojas y se conserva entre corridas; match_byid_robot_visits.py la usa.
#
#   Salida en <out_dir>: cv.csv, cv_consultas.csv (para calibrar el margen),
#   candidatos.csv, propuestas.csv, hojas/vaca_XX.jpg y emb_*.npz (caché).
#
# Uso (en la imagen muumetrics-opencowid, con GPU):
#   python reid_head.py --byid_dir /data/ByID --grouped_csv /data/robot_visits/agrupado/resumen.csv \
#       --blocks_csv /data/robot_visits/bloques_robot.csv --ordeno_dir /data/GeneralidadesOrdeno \
#       --phone /data/opencowid/caetec_crops_448:/data/opencowid/views/vistas.csv \
#               /data/opencowid/vigentes_crops_448:/data/opencowid/views_vigentes/vistas.csv \
#       --out_dir /data/opencowid/reid_v2
# =============================================================

import argparse, csv, re
import datetime as dt
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from match_byid_caetec import DEVICE, load_megadescriptor, proto, review_sheet

MIN_REAR = 0.6


# ---------- datos ----------
def load_phone(specs):
    rows = []
    for spec in specs:
        crops, views = spec.split(":")
        for r in csv.DictReader(open(views)):
            rows.append({"id": r["id_robot"], "path": str(Path(crops) / r["id_robot"] / r["foto"]),
                         "rear": r["vista"] == "atras" and float(r["atras"]) >= MIN_REAR})
    return rows


def load_byid(byid_dir, grouped_csv):
    rows = []
    for r in csv.DictReader(open(grouped_csv, encoding="utf-8-sig")):
        p = Path(byid_dir) / r["ID ByID"] / r["Imagen"]
        rows.append({"folder": r["ID ByID"], "path": str(p), "day": dt.date.fromisoformat(r["Imagen"][:10]),
                     "id": r["Carpeta destino"] if r["Nivel"] == "seguros" else None, "level": r["Nivel"]})
    return rows


def load_group3(ordeno_dir):
    g3 = defaultdict(set)
    for f in Path(ordeno_dir).glob("*.csv"):
        d = dt.datetime.strptime(re.search(r"_(\d{8})", f.name).group(1), "%Y%m%d").date()
        for r in csv.DictReader(open(f, encoding="latin1"), delimiter=";"):
            if r["Número del grupo"] == "3":
                g3[d].add(r["Número del animal"])
    return g3


def load_counts(ordeno_dir, lag=dt.timedelta(days=2)):
    """("Número del animal", día) -> "Ordeños Ayer"; el archivo del día D reporta el día D-2."""
    cnt = {}
    for f in Path(ordeno_dir).glob("*.csv"):
        d = dt.datetime.strptime(re.search(r"_(\d{8})", f.name).group(1), "%Y%m%d").date() - lag
        for r in csv.DictReader(open(f, encoding="latin1"), delimiter=";"):
            if r["Ordeños Ayer"].isdigit():
                cnt[(r["Número del animal"], d)] = int(r["Ordeños Ayer"])
    return cnt


def propose(ranks, time_ids, top_k=3, strong=0.05, weak=0.02):
    """Reglas para subir carpetas a seguros/dudosos a partir de `ranks` (carpeta -> [(ID, puntaje)]):
      - Con 1 ráfaga ligada por horario (dudosos): si ese ID queda en el top-k de apariencia,
        son dos evidencias independientes que coinciden -> seguros.
      - Sin horario: asignación uno a uno (húngaro) para que un ID no sea varias carpetas.
        top-1 = asignado = mejor carpeta de ese ID (mutuo) y margen >= strong -> seguros;
        top-1 = asignado y margen >= weak -> dudosos."""
    from scipy.optimize import linear_sum_assignment
    free = [c for c in ranks if not time_ids.get(c) and ranks[c]]
    ids = sorted({x for c in free for x, _ in ranks[c]})
    M = np.full((len(free), len(ids)), -1e3)
    for i, c in enumerate(free):
        for x, sc in ranks[c]:
            M[i, ids.index(x)] = sc
    rows, cols = linear_sum_assignment(-M)
    assigned = {free[i]: ids[j] for i, j in zip(rows, cols) if M[i, j] > -1e3}
    best_folder = {x: max((c for c in free if any(y == x for y, _ in ranks[c])),
                          key=lambda c: dict(ranks[c])[x]) for x in ids}
    out = {}
    for c, rk in ranks.items():
        top = [x for x, _ in rk]
        margin = rk[0][1] - rk[1][1] if len(rk) > 1 else 0.0
        row = {"top1": top[0] if top else "", "margen": round(margin, 3), "asignado": assigned.get(c, ""),
               "mutuo": "sí" if top and best_folder.get(top[0]) == c else "no", "nivel": "", "id": "", "motivo": "",
               "pos_horario": ""}
        if time_ids.get(c):
            pos = {t: top.index(t) + 1 for t in time_ids[c] if t in top}
            row["pos_horario"] = "; ".join(f"{t}: {pos.get(t, '-')}" for t in sorted(time_ids[c]))
            hits = [t for t, k in pos.items() if k <= top_k]
            if len(hits) == 1:
                row.update(nivel="seguros", id=hits[0], motivo=f"horario (1 ráfaga) + apariencia top{pos[hits[0]]}")
        elif top and assigned.get(c) == top[0]:
            if row["mutuo"] == "sí" and margin >= strong:
                row.update(nivel="seguros", id=top[0], motivo=f"apariencia (mutuo, margen {margin:.3f})")
            elif margin >= weak:
                row.update(nivel="dudosos", id=top[0], motivo=f"apariencia (margen {margin:.3f})")
        out[c] = row
    return out


def candidates(days, ids, g3):
    """IDs en el grupo 3 alguno de los archivos d-1..d+2 de cada día con foto (el reporte va desfasado)."""
    ok = set(ids)
    for d in days:
        near = set().union(*(g3.get(d + dt.timedelta(k), set()) for k in range(-1, 3)))
        if near:
            ok &= near
    return sorted(ok)


# ---------- embeddings ----------
@torch.no_grad()
def embed_all(paths, cache):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        known = dict(zip(z["paths"], range(len(z["paths"]))))
        if all(p in known for p in paths):
            idx = [known[p] for p in paths]
            return z["e"][idx], z["ef"][idx]
    net, tf = load_megadescriptor()
    net.eval().to(DEVICE)
    E, EF = [], []
    for i in range(0, len(paths), 16):
        x = torch.stack([tf(ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for p in paths[i:i + 16]]).to(DEVICE)
        E.append(F.normalize(net(x).float(), dim=1).cpu().numpy())
        EF.append(F.normalize(net(torch.flip(x, dims=[3])).float(), dim=1).cpu().numpy())
    E, EF = np.concatenate(E), np.concatenate(EF)
    np.savez(cache, paths=np.array(paths), e=E, ef=EF)
    return E, EF


def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# ---------- cabeza ----------
def train_head(X, y, w, epochs=300, lr=1e-3, scale=30.0, margin=0.2, reg=1e-2, seed=0):
    """Proyección lineal (identidad al inicio) + ArcFace; X ya normalizado, w = peso por muestra."""
    torch.manual_seed(seed)
    X, y, w = (torch.tensor(a).to(DEVICE) for a in (X.astype(np.float32), y, w.astype(np.float32)))
    D, n = X.shape[1], int(y.max()) + 1
    W = torch.eye(D, device=DEVICE).requires_grad_()
    C = torch.stack([F.normalize(X[y == c].mean(0), dim=0) for c in range(n)]).clone().requires_grad_()
    opt = torch.optim.Adam([W, C], lr=lr)
    eye = torch.eye(D, device=DEVICE)
    for _ in range(epochs):
        z = F.normalize(F.dropout(X, 0.1) @ W, dim=1)
        cos = z @ F.normalize(C, dim=1).T
        theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        logits = scale * torch.where(F.one_hot(y, n).bool(), torch.cos(theta + margin), cos)
        loss = (F.cross_entropy(logits, y, reduction="none") * w).sum() / w.sum() + reg * ((W - eye) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    Wf = W.detach().cpu().numpy()
    return lambda Z: l2n(Z @ Wf)


# ---------- evaluación ----------
def hub_penalty(gallery, folder_protos, k=5):
    """CSLS: similitud media de cada ID de la galería con sus k carpetas de ByID más parecidas.
    Restarla evita que unos pocos IDs ("hubs") queden primero para muchas carpetas."""
    F_ = np.stack(folder_protos)
    return {c: float(np.sort(F_ @ g)[-k:].mean()) for c, g in gallery.items()}


def rank_query(qvec, gallery, cands, pen=None):
    ids = [c for c in cands if c in gallery]
    sims = np.array([gallery[c] @ qvec - (pen[c] if pen else 0.0) for c in ids])
    order = np.argsort(-sims)
    return [(ids[k], float(sims[k])) for k in order]


def run_cv(phone, byid, Pe, Pc, Bc, g3, folds=5, seed=0):
    """Pe: embeddings crudos de celular; Pc/Bc: Split centrados por dominio (celular / ByID)."""
    sure = sorted({r["id"] for r in byid if r["id"]})
    rng = np.random.default_rng(seed)
    order = rng.permutation(sure)
    fold_of = {c: k % folds for k, c in enumerate(order)}
    Braw = l2n(np.stack([r["e"] for r in byid]))
    folders = defaultdict(list)
    for i, r in enumerate(byid):
        if r["folder"].isdigit():
            folders[r["folder"]].append(i)
    results, details = [], []
    for train_set in ("atras", "todas"):
        for gal_set in ("atras", "todas"):
            pm = [r["rear"] or gal_set == "todas" for r in phone]
            for method in ("crudo", "centrado", "cabeza"):
                if method != "cabeza" and train_set == "todas":
                    continue  # sin entrenamiento no depende del conjunto de entrenamiento
                recs = {"no": [], "csls": []}
                for k in range(folds):
                    test = [c for c in sure if fold_of[c] == k]
                    if method == "cabeza":
                        f = fit_head(phone, byid, Pc, Bc, train_set, exclude=set(test))
                        P, B = f(Pc.avg), f(Bc.avg)
                    elif method == "centrado":
                        P, B = Pc.avg, Bc.avg
                    else:
                        P, B = Pe, Braw
                    gallery = protos([r["id"] for r in phone], P, pm)
                    pen = hub_penalty(gallery, [proto(B[ix]) for ix in folders.values()])
                    for c in test:
                        idx = [i for i, r in enumerate(byid) if r["id"] == c]
                        cands = candidates({byid[i]["day"] for i in idx}, gallery.keys(), g3)
                        if c not in cands:
                            continue
                        for hub in recs:
                            rk = rank_query(proto(B[idx]), gallery, cands, pen if hub == "csls" else None)
                            ids = [x for x, _ in rk]
                            recs[hub].append({"id": c, "rank": ids.index(c) + 1, "n": len(rk),
                                              "margen": rk[0][1] - rk[1][1]})
                for hub, rs in recs.items():
                    cfg = {"entrenamiento": train_set if method == "cabeza" else "-", "galeria": gal_set,
                           "metodo": method, "hubs": hub}
                    ranks = [r["rank"] for r in rs]
                    results.append({**cfg, "consultas": len(rs),
                                    "top1": float(np.mean([r == 1 for r in ranks])), "top5": float(np.mean([r <= 5 for r in ranks])),
                                    "posicion_mediana": float(np.median(ranks)),
                                    "candidatos_medianos": float(np.median([r["n"] for r in rs])),
                                    "azar_top1": float(np.mean([1 / r["n"] for r in rs]))})
                    details += [{**cfg, **r} for r in rs]
                    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in results[-1].items()}, flush=True)
    return results, details


def protos(ids, E, mask):
    by = defaultdict(list)
    for i, (c, m) in enumerate(zip(ids, mask)):
        if m:
            by[c].append(i)
    return {c: proto(E[ix]) for c, ix in by.items()}


def fit_head(phone, byid, Pc, Bc, train_set, exclude=()):
    """Entrena con celular (traseras o todas) + ByID seguros fuera de `exclude`; original y volteada como muestras."""
    pi = [i for i, r in enumerate(phone) if r["rear"] or train_set == "todas"]
    bi = [i for i, r in enumerate(byid) if r["id"] and r["id"] not in exclude]
    labels = sorted({phone[i]["id"] for i in pi} | {byid[i]["id"] for i in bi})
    lab = {c: k for k, c in enumerate(labels)}
    X = np.concatenate([Pc["e"][pi], Pc["ef"][pi], Bc["e"][bi], Bc["ef"][bi]])
    y = np.array([lab[phone[i]["id"]] for i in pi] * 2 + [lab[byid[i]["id"]] for i in bi] * 2)
    w = np.concatenate([np.full(2 * len(pi), 0.5 / len(pi)), np.full(2 * len(bi), 0.5 / max(1, len(bi)))])
    return train_head(l2n(X), y, w)


class Split(dict):
    """Embeddings centrados de un dominio: e y ef (volteada) para entrenar, avg para comparar."""
    def __init__(self, e, ef, mean):
        super().__init__(e=l2n(e - mean), ef=l2n(ef - mean))
        self.avg = l2n(self["e"] + self["ef"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--byid_dir", required=True)
    ap.add_argument("--grouped_csv", required=True)
    ap.add_argument("--blocks_csv", required=True)
    ap.add_argument("--ordeno_dir", required=True)
    ap.add_argument("--bursts_csv", required=True, help="rafagas.csv de match_byid_robot_visits.py")
    ap.add_argument("--phone", nargs="+", required=True, help="<carpeta recortes>:<vistas.csv> (uno o varios)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / "hojas").mkdir(parents=True, exist_ok=True)

    phone, byid, g3 = load_phone(args.phone), load_byid(args.byid_dir, args.grouped_csv), load_group3(args.ordeno_dir)
    log_ids = {r["ID animal"] for r in csv.DictReader(open(args.blocks_csv, encoding="utf-8-sig")) if r["Usado"] == "sí"}
    print(f"Celular: {len(phone)} fotos, {len({r['id'] for r in phone})} IDs ({sum(r['rear'] for r in phone)} traseras) | "
          f"ByID: {len(byid)} fotos, {sum(bool(r['id']) for r in byid)} con ID seguro "
          f"({len({r['id'] for r in byid if r['id']})} IDs)", flush=True)

    pe, pef = embed_all([r["path"] for r in phone], out / "emb_celular.npz")
    be, bef = embed_all([r["path"] for r in byid], out / "emb_byid.npz")
    for r, e, ef in zip(byid, be, bef):
        r["e"] = l2n(e + ef)
    Pe = l2n(pe + pef)
    Pc = Split(pe, pef, np.concatenate([pe, pef]).mean(0))
    Bc = Split(be, bef, np.concatenate([be, bef]).mean(0))
    cv, details = run_cv(phone, byid, Pe, Pc, Bc, g3)
    for name, rows in (("cv.csv", cv), ("cv_consultas.csv", details)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    best = max(cv, key=lambda r: (r["top1"], r["top5"], -r["posicion_mediana"]))
    keys = ("entrenamiento", "galeria", "metodo", "hubs")
    calib = [(r["margen"], r["rank"] == 1) for r in details if all(r[k] == best[k] for k in keys)]
    print("Mejor:", best, flush=True)

    def cv_acc(m):
        """Aciertos top-1 en el CV entre las consultas con margen >= m."""
        hit = [ok for mm, ok in calib if mm >= m]
        return f"{sum(hit)}/{len(hit)}" if hit else "0/0"

    # Modelo final con todos los IDs seguros
    pm = [r["rear"] or best["galeria"] == "todas" for r in phone]
    if best["metodo"] == "cabeza":
        f = fit_head(phone, byid, Pc, Bc, best["entrenamiento"])
        P, B = f(Pc.avg), f(Bc.avg)
    elif best["metodo"] == "centrado":
        P, B = Pc.avg, Bc.avg
    else:
        P, B = Pe, l2n(np.stack([r["e"] for r in byid]))
    gallery = protos([r["id"] for r in phone], P, pm)
    phone_paths = defaultdict(list)
    for r, m in zip(phone, pm):
        if m:
            phone_paths[r["id"]].append(r["path"])

    folders = defaultdict(list)
    for i, r in enumerate(byid):
        folders[r["folder"]].append(i)
    sure_folders = {r["folder"] for r in byid if r["id"]}
    targets = [c for c in folders if c.isdigit() and c not in sure_folders]
    cnt = load_counts(args.ordeno_dir)
    per_day = defaultdict(lambda: defaultdict(int))
    for r in csv.DictReader(open(args.bursts_csv, encoding="utf-8-sig")):
        per_day[r["ID ByID"]][dt.date.fromisoformat(r["Fecha/hora"][:10])] += 1
    time_ids = defaultdict(set)  # IDs ligados por horario a una sola ráfaga de la carpeta
    for r in csv.DictReader(open(args.grouped_csv, encoding="utf-8-sig")):
        if "con hora" in r["Motivo"] and r["Nivel"] != "seguros":
            time_ids[r["ID ByID"]].add(r["Carpeta destino"])
    ranks = {}
    pen = hub_penalty(gallery, [proto(B[ix]) for c, ix in folders.items() if c.isdigit()]) if best["hubs"] == "csls" else None
    with open(out / "candidatos.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["ID ByID", "Nivel", "Fotos", "Días", "Candidatos"] + [f"top{k + 1}" for k in range(5)]
                   + ["Puntaje top1", "Margen top1-top2", "Aciertos CV con margen >= este", "ID robot confirmado", "Notas"])
        for c in sorted(targets, key=int):
            idx = folders[c]
            level = "horario (1 ráfaga)" if time_ids.get(c) else "sin horario"
            cands = candidates({byid[i]["day"] for i in idx}, gallery.keys(), g3)
            if not time_ids.get(c):  # si fuera una vaca del registro, el horario ya la habría ligado
                cands = [x for x in cands if x not in log_ids]
            # La vaca tuvo que ordeñarse al menos tantas veces como ráfagas tiene la carpeta ese día
            cands = [x for x in cands if all(cnt.get((x, d), 9) >= k for d, k in per_day[c].items())]
            rk = rank_query(proto(B[idx]), gallery, cands, pen)
            ranks[c] = rk
            margin = rk[0][1] - rk[1][1] if len(rk) > 1 else 0.0
            w.writerow([c, level, len(idx), len({byid[i]["day"] for i in idx}), len(rk)]
                       + [f"{x} ({s:.3f})" for x, s in rk[:5]] + [""] * (5 - len(rk[:5]))
                       + [round(rk[0][1], 3) if rk else "", round(margin, 3), cv_acc(margin), "", ""])
            review_sheet(int(c), sorted(byid[i]["path"] for i in idx), rk, phone_paths, out / "hojas" / f"vaca_{int(c):02d}.jpg")

    props = propose(ranks, time_ids)
    final_col = "Nivel final (a mano: seguros/dudosos/no)"
    prev = {}  # se conservan las decisiones tomadas a mano en una corrida anterior
    if (out / "propuestas.csv").exists():
        prev = {r["ID ByID"]: (r[final_col], r["Notas"])
                for r in csv.DictReader(open(out / "propuestas.csv", encoding="utf-8-sig"))}
    with open(out / "propuestas.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["ID ByID", "IDs por horario (1 ráfaga)", "Posición en apariencia", "Candidatos", "top1", "Margen",
                    "Asignación uno a uno", "Mutuo", "Nivel propuesto", "ID propuesto", "Motivo", final_col, "Notas"])
        for c in sorted(props, key=int):
            p = props[c]
            w.writerow([c, "; ".join(sorted(time_ids.get(c, ()))), p["pos_horario"], len(ranks[c]), p["top1"], p["margen"],
                        p["asignado"], p["mutuo"], p["nivel"], p["id"], p["motivo"], *prev.get(c, ("", ""))])
    print("Propuestas:", dict(Counter(p["nivel"] or "-" for p in props.values())))
    print(f"Listo: {out}")


if __name__ == "__main__":
    main()
