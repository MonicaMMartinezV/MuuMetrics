# =============================================================
# Nombre del archivo: match_byid_caetec.py
# Descripción: Propone, para cada vaca de ByID (fotos dorsales del robot,
#   may-jul 2025), los IDs del robot más parecidos entre las fotos de
#   confirmación de CAETEC (celular, oct 2026, recortadas con
#   crop_caetec.py), comparando embeddings de re-identificación.
#
#   Modelos (--models, uno o varios):
#     opencowid=<pesos.pth>  ResNet50 de OpenCowID entrenado con texturas sintéticas
#     megadescriptor         MegaDescriptor-L-384 (re-ID de fauna, timm / HF hub)
#
#   Para cada modelo:
#     1. Control dentro de ByID: cada foto de un día posterior contra el
#        prototipo (media de embeddings) del primer día de cada vaca.
#     2. Control dentro de CAETEC: cada foto contra prototipos; el de su
#        propio ID se arma solo con fotos de otras personas.
#     3. Cruce ByID -> CAETEC: prototipo de la vaca ByID contra los IDs de
#        CAETEC que estaban en ordeño todos sus días con fotos (según
#        Generalidades de Ordeño). Top-k por similitud coseno.
#     4. Anclas: vacas ByID con ID del robot sugerido confiable
#        (dataset_del.csv, >= --anchor_support de sus fotos y sin días
#        incompatibles): posición de ese ID en la lista del modelo.
#   El azar da top-1 = 1/N y posición media ~N/2.
#
#   Salida: candidatos.csv, resumen.json y hojas de revisión por vaca
#   (fotos ByID arriba, fotos de los 3 mejores candidatos abajo).
#
# Uso (en la imagen muumetrics-opencowid, con GPU):
#   python match_byid_caetec.py --byid_dir /data/ByID --caetec_dir /data/opencowid/caetec_crops \
#       --ordeno_dir /data/GeneralidadesOrdeno --robot_csv /data/byid_vs_robot/vacas_byid.csv \
#       --models opencowid=/data/opencowid/models/best.pth megadescriptor --out_dir /data/opencowid/match
# =============================================================

import argparse, csv, json, re, sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms

sys.path.insert(0, "/opt/OpenCowID")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOPK = 5


# ---------- modelos ----------
def load_opencowid(weights):
    from models.resnet import Orig_resnet_with_projection
    from utils import Custom_resize_transform
    net = Orig_resnet_with_projection()
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    tf = transforms.Compose([Custom_resize_transform(), transforms.ToTensor(),
                             transforms.Normalize((0.2032, 0.1978, 0.2084), (0.3502, 0.3455, 0.3491))])
    return net, tf


def load_megadescriptor():
    import timm
    net = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", num_classes=0, pretrained=True)
    tf = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(),
                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    return net, tf


@torch.no_grad()
def embed(net, tf, paths, bs=32):
    net.eval().to(DEVICE)
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([tf(ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for p in paths[i:i + bs]])
        e = net(x.to(DEVICE)).float()
        out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
    return np.concatenate(out)


def proto(E):
    m = E.mean(0)
    return m / np.linalg.norm(m)


# ---------- datos ----------
def load_byid(byid_dir):
    rows = []
    for p in sorted(Path(byid_dir).glob("*/*.jpg")):
        if p.parent.name.isdigit() and " - Copy" not in p.name:
            rows.append({"cow": int(p.parent.name), "path": p, "day": p.name[:10]})
    return rows


def load_caetec(caetec_dir):
    return [{"id": p.parent.name, "path": p, "person": p.name.split("__")[0]}
            for p in sorted(Path(caetec_dir).glob("*/*.jpg"))]


def load_presence(ordeno_dir):
    pres = {}
    for f in Path(ordeno_dir).glob("*.csv"):
        d = datetime.strptime(re.search(r"_(\d{8})", f.name).group(1), "%Y%m%d").date()
        pres[d.isoformat()] = {r["Número del animal"] for r in csv.DictReader(open(f, encoding="latin1"), delimiter=";")
                               if r["Número del animal"]}
    return pres


def load_anchors(robot_csv, min_support, min_photos=3):
    anchors = {}
    if not robot_csv or not Path(robot_csv).exists():
        return anchors, {}
    sugg = {}
    for r in csv.DictReader(open(robot_csv, encoding="utf-8-sig")):
        if not r["ID ByID"].isdigit() or not r["ID robot sugerido"]:
            continue
        cow, sup = int(r["ID ByID"]), float(r["% fotos que lo apoyan"] or 0)
        sugg[cow] = (r["ID robot sugerido"], sup)
        if sup >= min_support and int(r["Fotos"]) >= min_photos and not r["Días incompatibles del sugerido (Generalidades)"]:
            anchors[cow] = r["ID robot sugerido"]
    return anchors, sugg


# ---------- evaluación ----------
def eval_byid(byid, E):
    by_cow = defaultdict(list)
    for i, r in enumerate(byid):
        by_cow[r["cow"]].append(i)
    gal, queries = {}, []
    for cow, idx in by_cow.items():
        days = sorted({byid[i]["day"] for i in idx})
        if len(days) < 2:
            continue
        gal[cow] = proto(E[[i for i in idx if byid[i]["day"] == days[0]]])
        queries += [(i, cow) for i in idx if byid[i]["day"] != days[0]]
    cows = list(gal)
    P = np.stack([gal[c] for c in cows])
    ranks = [int((P @ E[i] > P[cows.index(c)] @ E[i]).sum()) for i, c in queries]
    return {"vacas": len(cows), "consultas": len(queries), "top1": float(np.mean([r == 0 for r in ranks])),
            "top5": float(np.mean([r < 5 for r in ranks])), "azar_top1": 1 / len(cows)}


def eval_caetec(caetec, E):
    by_id = defaultdict(list)
    for i, r in enumerate(caetec):
        by_id[r["id"]].append(i)
    ids = list(by_id)
    P = np.stack([proto(E[by_id[c]]) for c in ids])
    ranks = []
    for c, idx in by_id.items():
        persons = {caetec[i]["person"] for i in idx}
        if len(persons) < 2:
            continue
        for i in idx:
            others = [j for j in idx if caetec[j]["person"] != caetec[i]["person"]]
            sims = P @ E[i]
            sims[ids.index(c)] = proto(E[others]) @ E[i]
            ranks.append(int((sims > sims[ids.index(c)]).sum()))
    return {"ids": len(ids), "consultas": len(ranks), "top1": float(np.mean([r == 0 for r in ranks])) if ranks else None,
            "top5": float(np.mean([r < 5 for r in ranks])) if ranks else None, "azar_top1": 1 / len(ids)}


def cross_match(byid, Eb, caetec, Ec, presence):
    cow_idx, id_idx = defaultdict(list), defaultdict(list)
    for i, r in enumerate(byid):
        cow_idx[r["cow"]].append(i)
    for i, r in enumerate(caetec):
        id_idx[r["id"]].append(i)
    ids = sorted(id_idx)
    P = np.stack([proto(Ec[id_idx[c]]) for c in ids])
    out = {}
    for cow, idx in cow_idx.items():
        days = {byid[i]["day"] for i in idx}
        allowed = [k for k, c in enumerate(ids) if all(c in presence.get(d, {c}) for d in days)]
        sims = P[allowed] @ proto(Eb[idx])
        order = np.argsort(-sims)
        out[cow] = [(ids[allowed[k]], float(sims[k])) for k in order]
    return out


def anchor_ranks(ranking, anchors):
    res = {}
    for cow, rid in anchors.items():
        lst = [c for c, _ in ranking.get(cow, [])]
        res[cow] = (lst.index(rid) + 1 if rid in lst else None, len(lst))
    return res


# ---------- hojas ----------
def fit(path, w, h):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((w, h))
    c = Image.new("RGB", (w, h), "white")
    c.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return c


def review_sheet(cow, byid_paths, cands, caetec_by_id, out_path, n=5, W=200, H=230):
    rows = [("ByID " + str(cow), byid_paths[:n])] + \
           [(f"#{k + 1}  ID {c}  sim {s:.3f}", caetec_by_id[c][:n]) for k, (c, s) in enumerate(cands[:3])]
    sheet = Image.new("RGB", (n * (W + 6) + 6, len(rows) * (H + 30) + 6), "white")
    d = ImageDraw.Draw(sheet)
    for r, (title, ps) in enumerate(rows):
        y = 6 + r * (H + 30)
        d.rectangle([0, y, sheet.width, y + 22], fill="#222" if r == 0 else "#555")
        d.text((8, y + 5), title, fill="white")
        for k, p in enumerate(ps):
            sheet.paste(fit(p, W, H), (6 + k * (W + 6), y + 26))
    sheet.save(out_path, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--byid_dir", required=True)
    ap.add_argument("--caetec_dir", required=True)
    ap.add_argument("--ordeno_dir", required=True)
    ap.add_argument("--robot_csv", default=None, help="vacas_byid.csv de build_byid_robot_match.py (para las anclas)")
    ap.add_argument("--anchor_support", type=float, default=0.6)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sheets_model", default=None, help="Modelo para las hojas de revisión (por defecto el primero).")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    byid, caetec = load_byid(args.byid_dir), load_caetec(args.caetec_dir)
    presence = load_presence(args.ordeno_dir)
    anchors, sugg = load_anchors(args.robot_csv, args.anchor_support)
    print(f"ByID: {len(byid)} fotos, {len({r['cow'] for r in byid})} vacas | CAETEC: {len(caetec)} recortes, "
          f"{len({r['id'] for r in caetec})} IDs | anclas: {len(anchors)}")

    summary, rankings = {}, {}
    for spec in args.models:
        name, _, weights = spec.partition("=")
        net, tf = load_opencowid(weights) if name == "opencowid" else load_megadescriptor()
        Eb = embed(net, tf, [r["path"] for r in byid])
        Ec = embed(net, tf, [r["path"] for r in caetec])
        del net
        torch.cuda.empty_cache()
        ranking = cross_match(byid, Eb, caetec, Ec, presence)
        ar = anchor_ranks(ranking, anchors)
        valid = [r for r, _ in ar.values() if r is not None]
        summary[name] = {
            "dentro_ByID": eval_byid(byid, Eb),
            "dentro_CAETEC": eval_caetec(caetec, Ec),
            "anclas": {"n": len(anchors), "con_ID_en_CAETEC": len(valid),
                       "top1": float(np.mean([r == 1 for r in valid])) if valid else None,
                       "top5": float(np.mean([r <= 5 for r in valid])) if valid else None,
                       "posicion_mediana": float(np.median(valid)) if valid else None,
                       "candidatos_medianos": float(np.median([n for _, n in ar.values()])) if ar else None,
                       "detalle": {str(c): {"id_robot": anchors[c], "posicion": r, "de": n} for c, (r, n) in ar.items()}},
        }
        rankings[name] = ranking
        print(name, json.dumps({k: v for k, v in summary[name].items() if k != "anclas"}, ensure_ascii=False))
        print(name, "anclas:", {k: v for k, v in summary[name]["anclas"].items() if k != "detalle"})

    with open(out / "resumen.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    names = list(rankings)
    with open(out / "candidatos.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        head = ["ID ByID", "Fotos ByID", "ID robot sugerido (dataset_del)", "Apoyo sugerido", "Candidatos posibles"]
        for n in names:
            head += [f"{n} top{k + 1}" for k in range(TOPK)] + [f"{n} posición del sugerido"]
        w.writerow(head + ["ID confirmado", "Notas"])
        for cow in sorted(rankings[names[0]]):
            rid, sup = sugg.get(cow, ("", ""))
            row = [cow, sum(r["cow"] == cow for r in byid), rid, sup, len(rankings[names[0]][cow])]
            for n in names:
                lst = rankings[n][cow]
                row += [f"{c} ({s:.3f})" for c, s in lst[:TOPK]] + [""] * (TOPK - len(lst[:TOPK]))
                ids = [c for c, _ in lst]
                row.append(ids.index(rid) + 1 if rid in ids else "")
            w.writerow(row + ["", ""])

    sm = args.sheets_model or names[0]
    caetec_by_id = defaultdict(list)
    for r in caetec:
        caetec_by_id[r["id"]].append(r["path"])
    byid_by_cow = defaultdict(list)
    for r in byid:
        byid_by_cow[r["cow"]].append(r["path"])
    (out / "hojas").mkdir(exist_ok=True)
    for cow, cands in rankings[sm].items():
        review_sheet(cow, byid_by_cow[cow], cands, caetec_by_id, out / "hojas" / f"vaca_{cow:02d}.jpg")
    print(f"Listo: {out / 'candidatos.csv'}, {out / 'resumen.json'}, hojas en {out / 'hojas'} (modelo {sm})")


if __name__ == "__main__":
    main()
