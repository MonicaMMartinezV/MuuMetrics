# =============================================================
# Nombre del archivo: exp_desc2.py
#
# Descripción:
#   Experimento [24]: tres ideas para el descriptor, sin tocar el modelo
#   final (seg3 + DINOv2-L 224 px / bloque 17 + ridge, 64 vacas).
#
#   A. Backbones nuevos (congelados, mismas máscaras, promedio por región):
#        dinov3_224   vit_large_patch16_dinov3.lvd1689m   224 px (14x14)
#        dinov3_384   vit_large_patch16_dinov3.lvd1689m   384 px (24x24)
#        siglip2_256  vit_large_patch16_siglip_256.v2_webli 256 px (16x16)
#      bloques 8..24 (LOCO para todos; rep5x10 para el mejor de cada uno).
#   B. Otra forma de resumir cada región (DINOv2-L 224 / bloque 17):
#        media (actual) | sub2x2 (media en 4 cuadrantes de la región,
#        partida en su centroide) | media+desv | media+máx
#   C. Medidas de forma de las máscaras (área, caja, centroide, convexidad,
#      compacidad por región; perfil de ancho de la silueta; cocientes
#      cadera/lomo, grupa/cadera, separación de muslos) -> ~75 columnas.
#        forma sola | DINOv2 + forma (peso del bloque de forma elegido por
#        CV interna junto con alpha, porque el ridge estandariza columnas
#        y 75 columnas contra 6150 casi no pesarían).
#
#   Todas las variantes usan la misma rejilla de alphas (más amplia que la
#   del modelo, porque sub2x2 tiene 4x columnas) y se comparan contra la
#   base evaluada igual. Diferencia pareada de MAE por vaca (IC95 bootstrap).
#
#   Pasos:  extract (GPU) -> <work>/exp_desc2/*.npz
#           eval    (CPU) -> <work>/exp_desc2/resultados.csv, preds_loco.csv
# Uso: docker compose run --rm --entrypoint python bcs exp_desc2.py extract
#      docker compose run --rm --entrypoint python bcs exp_desc2.py eval
# =============================================================
import argparse, csv, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import timm
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_erosion
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import bcs_regions as br
from data import load_records
from segmenter import REGIONS

BACKS = {"dinov3_224": ("vit_large_patch16_dinov3.lvd1689m", 224),
         "dinov3_384": ("vit_large_patch16_dinov3.lvd1689m", 384),
         "siglip2_256": ("vit_large_patch16_siglip_256.v2_webli", 256)}
BLOCKS = list(range(8, 25))                                   # 1-based
ALPHAS = (1, 10, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 1e6)
SHAPE_W = (0.0, 1.0, 3.0, 10.0, 30.0)

# ---------------- extracción ----------------
def token_weights(masks, size, g):
    m = F.interpolate(torch.from_numpy((masks > 0.5).astype(np.float32))[None], size=(size, size), mode="nearest")
    return F.adaptive_avg_pool2d(m, (g, g))[0].reshape(6, -1)            # [6, g*g]

def prep(img, size, mean, std):
    x = np.asarray(Image.fromarray(img).resize((size, size), Image.BICUBIC)) / 255.0
    return torch.from_numpy(((x - np.array(mean)) / np.array(std)).transpose(2, 0, 1)).float()[None]

@torch.no_grad()
def cmd_extract(a):
    work, dev = Path(a.work_dir), br.device(); out = work / "exp_desc2"; out.mkdir(exist_ok=True)
    recs = load_records(a.data_dir)
    masks = [np.load(br.mask_file(work, r))["masks"].astype(np.float32) / 255.0 for r in recs]
    meta = dict(y=np.array([r["bcs"] for r in recs]), vaca=np.array([r["vaca"] for r in recs]), name=np.array([r["name"] for r in recs]))
    # A. backbones nuevos, promedio por región en cada bloque
    for key, (name, size) in BACKS.items():
        f = out / f"feats_{key}.npz"
        if f.exists(): print(f"[extract] {f.name} ya existe"); continue
        bb = timm.create_model(name, pretrained=True, num_classes=0, img_size=size).to(dev).eval()
        cfg = timm.data.resolve_data_config({}, model=bb); ps = bb.patch_embed.patch_size[0]; g = size // ps
        nb = len(bb.blocks); blocks = [b for b in BLOCKS if b <= nb]; R, M, t0 = [], [], time.time()
        for i, r in enumerate(recs):
            x = prep(br.read_rgb(r["path"]), size, cfg["mean"], cfg["std"]).to(dev)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                outs = bb.forward_intermediates(x, indices=[b - 1 for b in blocks], norm=True, output_fmt="NCHW", intermediates_only=True)
            w = token_weights(masks[i], size, g).to(dev); tot = w.sum(1, keepdim=True); miss = tot[:, 0] < 0.5; regs = []
            for sp in outs:
                tok = sp[0].float().reshape(sp.shape[1], -1).T
                v = (w @ tok) / tot.clamp_min(1e-6); v[miss] = 0; regs.append(v.cpu().numpy())
            R.append(np.stack(regs).astype(np.float16)); M.append(miss.float().cpu().numpy())
            if (i + 1) % 100 == 0: print(f"  {key} {i + 1}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
        np.savez(f, reg=np.stack(R), miss=np.stack(M), blocks=np.array(blocks), **meta)
        print(f"[extract] {key}: {nb} bloques, malla {g}x{g}, media {cfg['mean']} -> {f.name} ({time.time() - t0:.0f}s)", flush=True)
        del bb; torch.cuda.empty_cache()
    # B. tokens del modelo actual (DINOv2-L, br.SIZE, bloque br.LAYER) + pesos por token
    f = out / "tok_actual.npz"
    if not f.exists():
        bb = br.load_backbone(dev); g = br.SIZE // 14; T, W = [], []
        for i, r in enumerate(recs):
            x = prep(br.read_rgb(r["path"]), br.SIZE, br.MEAN, br.STD).to(dev)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                sp = bb.forward_intermediates(x, indices=[br.LAYER], norm=True, output_fmt="NCHW", intermediates_only=True)[0]
            T.append(sp[0].float().reshape(sp.shape[1], -1).T.cpu().numpy().astype(np.float16)); W.append(token_weights(masks[i], br.SIZE, g).numpy())
        np.savez(f, tok=np.stack(T), w=np.stack(W), g=g, **meta); print(f"[extract] tokens actuales -> {f.name}", flush=True)
    # C. forma
    f = out / "forma.npz"
    if not f.exists():
        S = np.stack([shape_feats(m) for m in masks]); np.savez(f, X=S, **meta); print(f"[extract] forma {S.shape} -> {f.name}", flush=True)

def shape_feats(masks):
    """Medidas de forma de las 6 máscaras (coordenadas de la foto original, sin aplastar)."""
    B = masks > 0.5; U = B.any(0); ys, xs = np.nonzero(U)
    if len(ys) == 0: return np.zeros(6 * 9 + 12 + 5)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1; Wd, Ht, AU = x1 - x0, y1 - y0, U.sum()
    feats, widths = [], {}
    for k in range(6):
        b = B[k]; yy, xx = np.nonzero(b)
        if len(yy) < 10: feats += [0] * 8 + [1]; widths[REGIONS[k]] = 0.0; continue
        area = len(yy); bw, bh = (xx.max() - xx.min() + 1) / Wd, (yy.max() - yy.min() + 1) / Ht
        per = (b & ~binary_erosion(b)).sum()
        edge = np.nonzero(b & ~binary_erosion(b)); pts = np.c_[edge[1], edge[0]]
        try: hull = ConvexHull(pts).volume
        except Exception: hull = area
        feats += [area / AU, bw, bh, (xx.mean() - x0) / Wd, (yy.mean() - y0) / Ht, area / max(hull, 1), per ** 2 / (4 * np.pi * area),
                  np.std((xx - xx.mean()) / Wd), 0]
        widths[REGIONS[k]] = bw
    prof = [U[int(y0 + (y1 - y0) * q)].sum() / Wd for q in np.linspace(0.04, 0.96, 12)]   # perfil de ancho de la silueta
    tl, tr = B[REGIONS.index("thigh_L")], B[REGIONS.index("thigh_R")]
    sep = 0.0
    if tl.any() and tr.any(): sep = (np.nonzero(tr)[1].mean() - np.nonzero(tl)[1].mean()) / Wd
    try: uh = AU / ConvexHull(np.c_[xs, ys][::7]).volume
    except Exception: uh = 1.0
    ratios = [widths["hips"] / max(widths["loin"], 1e-3), widths["rump"] / max(widths["hips"], 1e-3), sep, uh, Ht / Wd]
    return np.array(feats + prof + ratios, dtype=np.float64)

# ---------------- evaluación ----------------
def splits_for(y, cows, seeds):
    uc = np.unique(cows); strata = np.round(y * 2).astype(int)
    return {"loco": [list(GroupKFold(len(uc)).split(y, y, cows))],
            "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(y, strata, cows)) for s in range(seeds)]}

def fp_blocks(Xa, Xb, ytr, gtr, Xa_te, Xb_te):
    """Ridge sobre [Xa estandarizada, w * Xb estandarizada]; (w, alpha) por GroupKFold(4) interna. Xb=None -> ridge normal."""
    ws = SHAPE_W if Xb is not None else (0.0,)
    def design(ia, ib, ja, jb, w):
        A, B, _, _ = br.standardize(ia, ja)
        if Xb is None or w == 0: return A, B
        C, D, _, _ = br.standardize(ib, jb); return np.hstack([A, w * C]), np.hstack([B, w * D])
    err = {}
    for i, v in GroupKFold(4).split(Xa, ytr, gtr):
        for w in ws:
            A, B = design(Xa[i], None if Xb is None else Xb[i], Xa[v], None if Xb is None else Xb[v], w)
            for al, (p, _) in br.ridge_dual(A, B, ytr[i], ALPHAS).items(): err.setdefault((w, al), []).append(np.abs(p - ytr[v]))
    w, al = min(err, key=lambda k: np.mean(np.concatenate(err[k])))
    A, B = design(Xa, Xb, Xa_te, Xb_te, w)
    return br.ridge_dual(A, B, ytr, [al])[al][0], (w, al)

def run(Xa, y, cows, schemes, Xb=None, only=None):
    res, ch = {}, []
    for sch, reps in schemes.items():
        if only and sch not in only: continue
        P = []
        for spl in reps:
            p = np.zeros(len(y))
            for tr, te in spl:
                p[te], c = fp_blocks(Xa[tr], None if Xb is None else Xb[tr], y[tr], cows[tr], Xa[te], None if Xb is None else Xb[te]); ch.append(c)
            P.append(p)
        res[sch] = np.mean(P, 0)
    return res, ch

def flat(reg, miss): return np.concatenate([reg.astype(np.float64).reshape(len(reg), -1), miss], 1)

def pool_variants(tok, w, g):
    """tok [N,P,D] float16, w [N,6,P] -> dict variante -> X."""
    N, P, D = tok.shape; ii, jj = np.divmod(np.arange(P), g); out = {k: [] for k in ["media", "sub2x2", "media+desv", "media+máx"]}
    for n in range(N):
        T = tok[n].astype(np.float64); W = w[n]; tot = W.sum(1); miss = (tot < 0.5).astype(float)
        mu = (W @ T) / np.maximum(tot, 1e-6)[:, None]; mu[tot < 0.5] = 0
        sd = np.sqrt(np.maximum((W @ T ** 2) / np.maximum(tot, 1e-6)[:, None] - mu ** 2, 0)); sd[tot < 0.5] = 0
        mx = np.stack([T[W[r] > 0.5].max(0) if (W[r] > 0.5).any() else np.zeros(D) for r in range(6)])
        q, qm = [], []
        for r in range(6):
            if tot[r] < 0.5: q += [np.zeros(D)] * 4; qm += [1] * 4; continue
            ci, cj = (W[r] * ii).sum() / tot[r], (W[r] * jj).sum() / tot[r]
            for a_ in (ii < ci, ii >= ci):
                for b_ in (jj < cj, jj >= cj):
                    wq = W[r] * a_ * b_; s = wq.sum()
                    q.append((wq @ T) / s if s >= 0.25 else np.zeros(D)); qm.append(float(s < 0.25))
        out["media"].append(np.r_[mu.ravel(), miss]); out["sub2x2"].append(np.r_[np.ravel(q), qm])
        out["media+desv"].append(np.r_[mu.ravel(), sd.ravel(), miss]); out["media+máx"].append(np.r_[mu.ravel(), mx.ravel(), miss])
    return {k: np.array(v) for k, v in out.items()}

def cmd_eval(a):
    out = Path(a.work_dir) / "exp_desc2"; t0 = time.time(); rows, preds = [], {}
    z = np.load(out / "tok_actual.npz"); y, cows, names = z["y"], z["vaca"], z["name"]; uc = np.unique(cows)
    S = splits_for(y, cows, a.seeds); rng = np.random.RandomState(0)

    def report(label, res, extra=""):
        for sch, p in res.items():
            m = br.metrics(p, y, cows); e = p - y
            rows.append({"variante": label, "esquema": sch, **{k: round(float(v), 4) for k, v in m.items()},
                         "sesgo_flacas": round(float(e[y < 3].mean()), 3), "sesgo_gordas": round(float(e[y >= 4.5].mean()), 3), "nota": extra})
            print(f"  {label:28s} {sch:7s} MAE={m['mae']:.3f} R2={m['r2_fotos']:.2f} R2v={m['r2_vacas']:.2f} ±0.5={m['acc_05']:.0%} "
                  f"MAE vaca={m['mae_vaca']:.3f} sesgo flacas={e[y < 3].mean():+.2f} gordas={e[y >= 4.5].mean():+.2f} {extra} ({time.time() - t0:.0f}s)", flush=True)
        if "loco" in res: preds[label] = res["loco"]

    # B. pooling del modelo actual (incluye la base)
    print("[B] resumen por región (DINOv2-L actual)", flush=True)
    PV = pool_variants(z["tok"], z["w"], int(z["g"]))
    for k, X in PV.items():
        res, ch = run(X, y, cows, S); report(f"pool {k}", res, f"alpha moda {max(set(c[1] for c in ch), key=[c[1] for c in ch].count)}")
    base = PV["media"]

    # C. forma
    print("[C] forma", flush=True)
    Sh = np.load(out / "forma.npz")["X"]; Sh = Sh[:, Sh.std(0) > 0]
    res, _ = run(Sh, y, cows, S); report("forma sola", res, f"{Sh.shape[1]} cols")
    res, ch = run(base, y, cows, S, Xb=Sh); ws = [c[0] for c in ch]
    report("media + forma", res, f"peso forma moda {max(set(ws), key=ws.count)}")

    # A. backbones nuevos: barrido de bloques con LOCO, rep5x10 para el mejor
    for key in BACKS:
        f = out / f"feats_{key}.npz"
        if not f.exists(): continue
        zb = np.load(f); assert (zb["name"] == names).all(); best = None
        print(f"[A] {key}", flush=True)
        for j, b in enumerate(zb["blocks"]):
            res, _ = run(flat(zb["reg"][:, j], zb["miss"]), y, cows, S, only=["loco"]); mae = np.abs(res["loco"] - y).mean()
            report(f"{key} bloque {b}", res)
            if best is None or mae < best[0]: best = (mae, j, b)
        res, _ = run(flat(zb["reg"][:, best[1]], zb["miss"]), y, cows, S, only=["rep5x10"]); report(f"{key} bloque {best[2]}", res, "(mejor bloque LOCO)")

    # diferencias pareadas por vaca contra la base (LOCO)
    ref = np.array([np.abs(preds["pool media"] - y)[cows == u].mean() for u in uc])
    print("\nMAE por vaca vs base (pool media), LOCO, IC95 bootstrap:")
    with open(out / "pareado.csv", "w", newline="") as fh:
        wtr = csv.writer(fh); wtr.writerow(["variante", "dif", "ic_lo", "ic_hi", "mejoran"])
        for k, p in preds.items():
            if k == "pool media": continue
            d = np.array([np.abs(p - y)[cows == u].mean() for u in uc]) - ref
            bs = [d[rng.randint(0, len(uc), len(uc))].mean() for _ in range(5000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
            wtr.writerow([k, round(d.mean(), 4), round(lo, 4), round(hi, 4), int((d < 0).sum())])
            print(f"  {k:28s} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  mejoran {(d < 0).sum()}/{len(uc)}")
    with open(out / "resultados.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0])); wtr.writeheader(); wtr.writerows(rows)
    with open(out / "preds_loco.csv", "w", newline="") as fh:
        wtr = csv.writer(fh); wtr.writerow(["foto", "vaca", "bcs_real"] + list(preds))
        for i in range(len(y)): wtr.writerow([names[i], cows[i], y[i]] + [round(float(preds[k][i]), 4) for k in preds])
    print(f"-> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "eval"])
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS_v2")
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args(); {"extract": cmd_extract, "eval": cmd_eval}[a.step](a)
