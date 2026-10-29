# =============================================================
# Nombre del archivo: last_search_byid.py
# Descripción: Última búsqueda de IDs para las carpetas de ByID que aún no
#   están en Dataset_Fr (IDs confirmados a mano), solo por apariencia y sin
#   marco de tiempo: no se usa la hora/día de las fotos para filtrar.
#
#   - Etiquetas ByID: Dataset_Fr/<ID>/ByID<carpeta>__<foto>.jpg.
#   - Galería: fotos de celular (CAETEC + vigentes) de los IDs que NO están
#     confirmados (una carpeta sin confirmar no puede ser una vaca ya confirmada).
#   - Modelo: MegaDescriptor congelado + cabeza ArcFace (reid_head.py),
#     entrenada con celular + todas las fotos de Dataset_Fr.
#   - Validación: deja fuera una vaca confirmada a la vez (cabeza entrenada sin
#     ella) y la busca entre los IDs no confirmados + ella. Así se mide qué tan
#     confiable es el top-1/top-5 y con qué margen.
#   - GeneralidadesOrdeno solo como filtro opcional sin fechas: IDs que
#     estuvieron en el grupo 3 (VMS 3) en algún archivo de mayo-julio. Se
#     usa solo si mejora la validación.
#
# Salida en <out_dir> (Dataset/LastSearchByID): validacion.csv,
#   validacion_resumen.csv, candidatos.csv (top-10 por carpeta), resumen.csv
#   (top-3, margen, confianza, asignación 1 a 1) y hojas/carpeta_<XX>.jpg.
#
# Uso (imagen muumetrics-opencowid, con GPU; Dataset montado en /data):
#   python last_search_byid.py --byid_dir /data/ByID --fr_dir /data/Dataset_Fr --ordeno_dir /ordeno \
#       --phone /data/opencowid/caetec_crops_448:/data/opencowid/views/vistas.csv \
#               /data/opencowid/vigentes_crops_448:/data/opencowid/views_vigentes/vistas.csv \
#       --emb_dir /data/opencowid/reid_v2 --out_dir /data/LastSearchByID
# =============================================================

import argparse, csv, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageOps
from scipy.optimize import linear_sum_assignment
from torchvision import transforms

from match_byid_caetec import DEVICE, proto
from reid_head import Split, embed_all, hub_penalty, l2n, load_phone, protos, train_head

NOGROUP = "No encontre grupo"
TOPN = 10


# ---------- otros extractores (--backbone) ----------
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p, self.eps = nn.Parameter(torch.ones(1) * p), eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), x.shape[-2:]).pow(1.0 / self.p).flatten(1)


class MiewID(nn.Module):
    """MiewID-msv3 (conservationxlabs, HF) sin transformers: EfficientNetV2-M + GeM + BatchNorm, 2152-d."""
    def __init__(self, weights):
        super().__init__()
        from safetensors.torch import load_file
        self.backbone = timm.create_model("efficientnetv2_rw_m", pretrained=False, num_classes=0)
        self.backbone.global_pool = GeM()
        self.bn = nn.BatchNorm1d(2152)
        sd = {k: v for k, v in load_file(weights).items() if not k.startswith("final.")}
        self.load_state_dict(sd)

    def forward(self, x):
        return self.bn(self.backbone(x).flatten(1))


class OpenCows(nn.Module):
    """ResNet50 de Andrew et al. 2020 (OpenCows2020, CWOA/MetricLearningIdentification): fc 1000 -> embedding 128."""
    def __init__(self, weights):
        super().__init__()
        self.net = torchvision.models.resnet50()
        self.fc_embedding = nn.Linear(1000, 128)
        sd = torch.load(weights, map_location="cpu", weights_only=False)["model_state"]
        self.net.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc_")})
        self.fc_embedding.load_state_dict({"weight": sd["fc_embedding.weight"], "bias": sd["fc_embedding.bias"]})

    def forward(self, x):
        return self.fc_embedding(self.net(x))


def letterbox(size):
    """Como loadResizeImage del repo original: lado mayor a `size`, centrado en fondo negro, valores 0-255."""
    def f(im):
        r = size / max(im.size)
        im = im.resize((int(im.width * r), int(im.height * r)), Image.LANCZOS)
        bg = Image.new("RGB", (size, size))
        bg.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
        return torch.from_numpy(np.array(bg, dtype=np.float32)).permute(2, 0, 1)
    return f


def load_backbone(name, models_dir):
    if name == "miewid":
        tf = transforms.Compose([transforms.Resize((440, 440)), transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return MiewID(Path(models_dir) / "miewid_msv3.safetensors"), tf
    if name.startswith("opencows_"):
        return OpenCows(Path(models_dir) / "opencows2020" / f"{name.split('_', 1)[1]}.pkl"), letterbox(224)
    raise ValueError(name)


@torch.no_grad()
def embed_backbone(paths, cache, name, models_dir, bs=16):
    """Como reid_head.embed_all (original + volteada, normalizados) para otro extractor."""
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        known = dict(zip(z["paths"], range(len(z["paths"]))))
        if all(p in known for p in paths):
            idx = [known[p] for p in paths]
            return z["e"][idx], z["ef"][idx]
    net, tf = load_backbone(name, models_dir)
    net.eval().to(DEVICE)
    E, EF = [], []
    for i in range(0, len(paths), bs):
        x = torch.stack([tf(ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for p in paths[i:i + bs]]).to(DEVICE)
        E.append(F.normalize(net(x).float(), dim=1).cpu().numpy())
        EF.append(F.normalize(net(torch.flip(x, dims=[3])).float(), dim=1).cpu().numpy())
    E, EF = np.concatenate(E), np.concatenate(EF)
    np.savez(cache, paths=np.array(paths), e=E, ef=EF)
    return E, EF


def load_byid(byid_dir, fr_dir):
    """Todas las fotos de ByID con su carpeta y, si está en Dataset_Fr, su ID confirmado."""
    conf = {}
    for p in Path(fr_dir).glob("*/*"):
        conf[p.name.split("__", 1)[1]] = p.parent.name
    rows = []
    for p in sorted(Path(byid_dir).glob("*/*.jpg")):
        folder = p.parent.name
        rows.append({"folder": folder, "path": str(p), "name": p.name, "id": conf.get(p.name)})
    return rows


def ever_group3(ordeno_dir):
    ids = set()
    for f in Path(ordeno_dir).glob("*.csv"):
        for r in csv.DictReader(open(f, encoding="latin1"), delimiter=";"):
            if r["Número del grupo"] == "3":
                ids.add(r["Número del animal"])
    return ids


def queries(byid):
    """Carpetas sin ninguna foto confirmada -> índices. 'No encontre grupo': cada foto sin confirmar es su propia consulta."""
    conf_folders = {r["folder"] for r in byid if r["id"] and r["folder"] != NOGROUP}
    q = defaultdict(list)
    for i, r in enumerate(byid):
        if r["folder"] == NOGROUP:
            if not r["id"]:
                q[f"NE_{Path(r['name']).stem}"].append(i)
        elif r["folder"] not in conf_folders:
            q[r["folder"]].append(i)
    return dict(q)


def fit(phone, byid, Pc, Bc, exclude=()):
    """Cabeza ArcFace con celular (todas las vistas) + ByID confirmadas fuera de `exclude`."""
    pi = list(range(len(phone)))
    bi = [i for i, r in enumerate(byid) if r["id"] and r["id"] not in exclude]
    lab = {c: k for k, c in enumerate(sorted({phone[i]["id"] for i in pi} | {byid[i]["id"] for i in bi}))}
    X = np.concatenate([Pc["e"][pi], Pc["ef"][pi], Bc["e"][bi], Bc["ef"][bi]])
    y = np.array([lab[phone[i]["id"]] for i in pi] * 2 + [lab[byid[i]["id"]] for i in bi] * 2)
    w = np.concatenate([np.full(2 * len(pi), 0.5 / len(pi)), np.full(2 * len(bi), 0.5 / len(bi))])
    f = train_head(l2n(X), y, w)
    return f(Pc.avg), f(Bc.avg)


def rank(q, gallery, cands, pen=None):
    sc = sorted(((c, float(gallery[c] @ q) - (pen[c] if pen else 0.0)) for c in cands if c in gallery), key=lambda x: -x[1])
    return sc


def validate(phone, byid, Pc, Bc, g3, mask_sets):
    conf = sorted({r["id"] for r in byid if r["id"]})
    gal_ids = {r["id"] for r in phone}
    rows = []
    for c in conf:
        if c not in gal_ids:
            continue
        P, B = fit(phone, byid, Pc, Bc, exclude={c})
        idx = [i for i, r in enumerate(byid) if r["id"] == c]
        others = [i for i, r in enumerate(byid) if not r["id"] or r["id"] != c]
        q = proto(B[idx])
        for gname, mask in mask_sets.items():
            gallery = protos([r["id"] for r in phone], P, mask)
            # hubs: carpetas de ByID que no son esta vaca
            fold = defaultdict(list)
            for i in others:
                fold[byid[i]["folder"] + (byid[i]["name"] if byid[i]["folder"] == NOGROUP else "")].append(i)
            pen = hub_penalty(gallery, [proto(B[ix]) for ix in fold.values()])
            base = [x for x in gallery if x not in conf or x == c]
            for filt in ("sin_filtro", "grupo3"):
                cands = base if filt == "sin_filtro" else [x for x in base if x in g3]
                if c not in cands:
                    continue
                for hub in ("no", "csls"):
                    rk = rank(q, gallery, cands, pen if hub == "csls" else None)
                    ids = [x for x, _ in rk]
                    rows.append({"vaca": c, "fotos_byid": len(idx), "galeria": gname, "filtro": filt, "hubs": hub,
                                 "posicion": ids.index(c) + 1, "candidatos": len(rk), "top1": ids[0],
                                 "margen": round(rk[0][1] - rk[1][1], 4)})
        print("validada", c, flush=True)
    return rows


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["galeria"], r["filtro"], r["hubs"])].append(r)
    out = []
    for (g, f, h), rs in sorted(by.items()):
        pos = np.array([r["posicion"] for r in rs])
        out.append({"galeria": g, "filtro": f, "hubs": h, "vacas": len(rs), "top1": round(float((pos == 1).mean()), 3),
                    "top3": round(float((pos <= 3).mean()), 3), "top5": round(float((pos <= 5).mean()), 3),
                    "top10": round(float((pos <= 10).mean()), 3), "posicion_mediana": float(np.median(pos)),
                    "candidatos_medianos": float(np.median([r["candidatos"] for r in rs]))})
    return out


def thumb(p, W, H):
    im = Image.open(p).convert("RGB")
    im.thumbnail((W, H))
    bg = Image.new("RGB", (W, H), "#ddd")
    bg.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    return bg


def sheet(path, title_q, qpaths, cand_rows, ref_paths, n=6, W=190, H=220):
    rr = [(title_q, qpaths[:n], "#222")] + [(t, ref_paths.get(c, [])[:n], "#555") for t, c in cand_rows]
    im = Image.new("RGB", (n * (W + 6) + 6, len(rr) * (H + 30) + 6), "white")
    d = ImageDraw.Draw(im)
    for j, (title, ps, col) in enumerate(rr):
        y = 6 + j * (H + 30)
        d.rectangle([0, y, im.width, y + 22], fill=col)
        d.text((8, y + 5), title, fill="white")
        for k, p in enumerate(ps):
            im.paste(thumb(p, W, H), (6 + k * (W + 6), y + 26))
    im.save(path, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--byid_dir", required=True)
    ap.add_argument("--fr_dir", required=True)
    ap.add_argument("--ordeno_dir", required=True)
    ap.add_argument("--phone", nargs="+", required=True)
    ap.add_argument("--emb_dir", required=True, help="carpeta con la caché emb_celular.npz / emb_byid.npz")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--backbone", default="megadescriptor",
                    help="megadescriptor | miewid | opencows_RTL_50-50_fold-0 | opencows_SoftmaxRTL_50-50_fold-1; "
                         "varios unidos con + se concatenan (promedio de similitudes)")
    ap.add_argument("--models_dir", default="/data/opencowid/models")
    ap.add_argument("--validate_only", action="store_true", help="solo la validación, en <out_dir>/modelos/<backbone>")
    args = ap.parse_args()
    out = Path(args.out_dir)
    if args.validate_only:
        out = out / "modelos" / args.backbone
    (out if args.validate_only else out / "hojas").mkdir(parents=True, exist_ok=True)

    phone = load_phone(args.phone)
    byid = load_byid(args.byid_dir, args.fr_dir)
    g3 = ever_group3(args.ordeno_dir)
    emb = Path(args.emb_dir)
    feats = []
    for name in args.backbone.split("+"):
        if name == "megadescriptor":
            feats.append(embed_all([r["path"] for r in phone], emb / "emb_celular.npz") +
                         embed_all([r["path"] for r in byid], emb / "emb_byid_all.npz"))
        else:
            feats.append(embed_backbone([r["path"] for r in phone], emb / f"emb_celular_{name}.npz", name, args.models_dir) +
                         embed_backbone([r["path"] for r in byid], emb / f"emb_byid_all_{name}.npz", name, args.models_dir))
    pe, pef, be, bef = (l2n(np.concatenate([f[k] for f in feats], axis=1)) for k in range(4))
    Pc = Split(pe, pef, np.concatenate([pe, pef]).mean(0))
    Bc = Split(be, bef, np.concatenate([be, bef]).mean(0))
    masks = {"atras": [r["rear"] for r in phone], "todas": [True] * len(phone)}

    # ---- validación con las vacas confirmadas ----
    val = validate(phone, byid, Pc, Bc, g3, masks)
    summ = summarize(val)
    for name, data in (("validacion.csv", val), ("validacion_resumen.csv", summ)):
        with open(out / name, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    for s in summ:
        print(s)
    best = max(summ, key=lambda s: (s["top1"], s["top5"], -s["posicion_mediana"]))
    print("mejor configuración:", best)
    if args.validate_only:
        return
    bv = [r for r in val if (r["galeria"], r["filtro"], r["hubs"]) == (best["galeria"], best["filtro"], best["hubs"])]
    ok_m = sorted(r["margen"] for r in bv if r["posicion"] == 1)
    bad_m = sorted(r["margen"] for r in bv if r["posicion"] != 1)
    # margen por encima del cual en validación no hubo errores de top-1
    m_high = max(bad_m) + 1e-4 if bad_m else 0.0

    # ---- búsqueda final ----
    conf = {r["id"] for r in byid if r["id"]}
    P, B = fit(phone, byid, Pc, Bc)
    gallery = protos([r["id"] for r in phone], P, masks[best["galeria"]])
    Q = queries(byid)
    fold = defaultdict(list)
    for i, r in enumerate(byid):
        fold[r["folder"] + (r["name"] if r["folder"] == NOGROUP else "")].append(i)
    pen = hub_penalty(gallery, [proto(B[ix]) for ix in fold.values()]) if best["hubs"] == "csls" else None
    cands = [x for x in gallery if x not in conf and (best["filtro"] == "sin_filtro" or x in g3)]
    ranks = {q: rank(proto(B[ix]), gallery, cands, pen) for q, ix in Q.items()}

    # asignación uno a uno (pista: dos carpetas distintas suelen ser vacas distintas)
    qs, ids = list(ranks), sorted(cands)
    M = np.array([[dict(ranks[q])[x] for x in ids] for q in qs])
    rr, cc = linear_sum_assignment(-M)
    assigned = {qs[i]: ids[j] for i, j in zip(rr, cc)}
    best_q = {x: qs[int(np.argmax(M[:, j]))] for j, x in enumerate(ids)}

    ref = defaultdict(list)
    for r in phone:
        if r["rear"] or best["galeria"] == "todas":
            ref[r["id"]].append(r["path"])
    cand_rows, summ_rows = [], []
    for q, rk in sorted(ranks.items(), key=lambda kv: (not kv[0].isdigit(), int(kv[0]) if kv[0].isdigit() else 0, kv[0])):
        margin = rk[0][1] - rk[1][1]
        top = rk[0][0]
        conf_lvl = "alta" if margin >= m_high and best_q[top] == q and assigned.get(q) == top else \
                   "media" if assigned.get(q) == top else "baja"
        for k, (x, s) in enumerate(rk[:TOPN]):
            cand_rows.append({"carpeta": q, "posicion": k + 1, "id": x, "score": round(s, 4),
                              "en_grupo3": "sí" if x in g3 else "no"})
        summ_rows.append({"carpeta": q, "fotos": len(Q[q]), "top1": top, "top2": rk[1][0], "top3": rk[2][0],
                          "margen": round(margin, 4), "top1_mutuo": "sí" if best_q[top] == q else f"no (mejor: {best_q[top]})",
                          "asignado_1a1": assigned.get(q, ""), "confianza": conf_lvl, "hoja": f"hojas/carpeta_{q}.jpg"})
        qpaths = sorted(byid[i]["path"] for i in Q[q])
        sheet(out / "hojas" / f"carpeta_{q}.jpg", f"ByID {q}  ({len(Q[q])} fotos)  confianza {conf_lvl}", qpaths,
              [(f"#{k+1}  ID {x}  score {s:.3f}{'  (asignado 1a1)' if assigned.get(q) == x else ''}", x)
               for k, (x, s) in enumerate(rk[:5])], ref)
    for name, data in (("candidatos.csv", cand_rows), ("resumen.csv", summ_rows)):
        with open(out / name, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"margen sin errores en validación: {m_high:.4f} | aciertos top1 márgenes: {ok_m} | errores: {bad_m}")
    print(f"consultas: {len(Q)} | candidatos: {len(cands)} | listo: {out}")


if __name__ == "__main__":
    main()
