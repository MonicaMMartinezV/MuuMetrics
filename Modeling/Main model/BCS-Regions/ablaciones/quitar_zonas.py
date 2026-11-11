# =============================================================
# Nombre del archivo: ablaciones/quitar_zonas.py
#
# Descripción:
#   Las 6 regiones se traslapan; aquí se parten en 7 zonas exclusivas (cada
#   píxel en una sola, primera regla que se cumple): base de cola, grupa∩muslo,
#   grupa sola, cadera∩muslo, cadera sola, muslo solo, lomo. Mismos tokens de
#   DINOv3-L (bloque br.LAYER) que bcs_regions.describe(), promediados por zona.
#   Con 37 vacas (2026-11-05) quitar "cadera sola" (-0.030*) y "muslo solo"
#   (-0.016) bajaba el error: son las zonas "rojas". Aquí se prueba quitarlas
#   con las 64 vacas, mismo ridge del modelo y validación por vaca.
#   Como esas zonas se eligieron antes de tener las vacas 9000-9027, el MAE en
#   esas 27 vacas nuevas es una prueba no sesgada de la elección.
#   Salidas: <work>/ablaciones/{feat_zonas.npz, quitar_zonas.csv}
#
# Uso (GPU solo la primera vez):
#   docker compose run --rm --entrypoint python bcs ablaciones/quitar_zonas.py
# =============================================================
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bcs_regions as br
from comun import splits_for, fit_apply, tokens, grid, modelo, cache_ok, guardar_cache
from data import load_records

L, H, R, T, TH = 1, 2, 4, 8, 16 | 32          # bits de loin, hips, rump, tailhead, thigh_L|thigh_R
RULES = [("base_cola", lambda c: c & T), ("grupa_muslo", lambda c: (c & R) and (c & TH)), ("grupa_sola", lambda c: c & R),
         ("cadera_muslo", lambda c: (c & H) and (c & TH)), ("cadera_sola", lambda c: c & H), ("muslo_solo", lambda c: c & TH),
         ("lomo", lambda c: c & L)]
ZN = [n for n, _ in RULES]; ROJAS = ["cadera_sola", "muslo_solo"]; D = None   # dimensión del backbone: de features.npz

def assign():
    a = np.full(64, -1)
    for c in range(1, 64):
        for k, (_, f) in enumerate(RULES):
            if f(c): a[c] = k; break
    return a

@torch.no_grad()
def compute(work, recs, dst, firma):
    dev = br.device(); bb = br.load_backbone(dev); G = grid(bb)
    bits = torch.tensor([1, 2, 4, 8, 16, 32], dtype=torch.long); a = assign(); Wc = torch.zeros(len(ZN), 64)
    for c in range(64):
        if a[c] >= 0: Wc[a[c], c] = 1
    X = np.zeros((len(recs), len(ZN) * D + len(ZN)), np.float32)
    for i, r in enumerate(recs):
        img = br.read_rgb(r["path"])
        tok = tokens(bb, img, dev)[1].cpu()
        m = np.load(br.mask_file(work, r))["masks"].astype(np.float32) / 255.0
        mb = F.interpolate(torch.from_numpy((m > 0.5).astype(np.float32))[None], size=(br.SIZE, br.SIZE), mode="nearest")[0]
        code = (mb.long() * bits[:, None, None]).sum(0)
        fr = F.adaptive_avg_pool2d(F.one_hot(code, 64).permute(2, 0, 1).float()[None], (G, G))[0].reshape(64, -1)
        w = Wc @ fr; tot = w.sum(1, keepdim=True); v = (w @ tok) / tot.clamp_min(1e-6)
        miss = tot[:, 0] < 0.5; v[miss] = 0
        X[i, :len(ZN) * D] = v.reshape(-1).numpy(); X[i, len(ZN) * D:] = miss.float().numpy()
        if (i + 1) % 100 == 0: print(f"  {i + 1}/{len(recs)}", flush=True)
    guardar_cache(dst, firma, X=X, name=np.array([r["name"] for r in recs]))

def cols(keep):
    idx = [ZN.index(z) for z in keep]
    return np.concatenate([np.arange(i * D, (i + 1) * D) for i in idx] + [len(ZN) * D + np.array(idx)])

def main(a):
    work = Path(a.work_dir); out = work / "ablaciones"; out.mkdir(exist_ok=True)
    global D
    X6, y, cows, names, firma, _, D = modelo(work); recs = load_records(a.data_dir)
    assert [r["name"] for r in recs] == list(names), "features.npz no coincide con los datos"
    f = out / "feat_zonas.npz"
    if not cache_ok(f, firma): compute(work, recs, f, firma)
    XZ = np.load(f)["X"].astype(np.float64); schemes = splits_for(y, cows, a.seeds)
    keys = sorted(set(zip(cows, y))); nuevas = np.array([c.startswith("90") for c, _ in keys])
    V = {"modelo (features.npz)": X6, "7_zonas": XZ}
    for z in ZN: V[f"7_zonas_sin_{z}"] = XZ[:, cols([k for k in ZN if k != z])]
    V["7_zonas_sin_rojas (cadera_sola+muslo_solo)"] = XZ[:, cols([k for k in ZN if k not in ROJAS])]
    rows, ce = [], {}; rng = np.random.RandomState(0); t0 = time.time()
    for v, Xv in V.items():
        r = {"variante": v}
        for sch, reps in schemes.items():
            P = []
            for splits in reps:
                p = np.zeros(len(y))
                for tr, te in splits: p[te] = fit_apply(Xv[tr], y[tr], cows[tr], Xv[te])
                P.append(p)
            P = np.mean(P, 0); m = br.metrics(P, y, cows)
            r.update({f"mae_{sch}": round(float(m["mae"]), 4), f"r2_fotos_{sch}": round(float(m["r2_fotos"]), 4), f"r2_vacas_{sch}": round(float(m["r2_vacas"]), 4)})
            if sch == "rep5x10": ce[v] = np.array([abs(P[(cows == c) & (y == b)].mean() - b) for c, b in keys])
        rows.append(r)
    base = ce["modelo (features.npz)"]
    for r in rows:
        e = ce[r["variante"]]; d = e - base
        for tag, sel in (("todas", np.ones(len(d), bool)), ("nuevas", nuevas)):
            dd = d[sel]; bs = [dd[rng.randint(0, len(dd), len(dd))].mean() for _ in range(5000)]
            r[f"mae_vaca_{tag}"] = round(float(e[sel].mean()), 4); r[f"dif_{tag}"] = round(float(dd.mean()), 4)
            r[f"ic95_{tag}"] = f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]"
    with open(out / "quitar_zonas.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{'variante':44s} {'MAE loco':>8s} {'MAE 5x10':>8s} {'R2vac':>6s} | dif. por vaca×BCS vs 6 regiones (5x10): todas (n={len(keys)})  |  27 vacas nuevas (n={nuevas.sum()})")
    for r in rows:
        print(f"{r['variante']:44s} {r['mae_loco']:8.3f} {r['mae_rep5x10']:8.3f} {r['r2_vacas_rep5x10']:6.3f} | "
              f"{r['dif_todas']:+.3f} {r['ic95_todas']}  |  {r['dif_nuevas']:+.3f} {r['ic95_nuevas']}")
    print(f"-> {out / 'quitar_zonas.csv'} ({time.time() - t0:.0f}s)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS_v2")
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--seeds", type=int, default=10)
    main(ap.parse_args())
