# =============================================================
# Nombre del archivo: exp_head.py
#
# Descripción:
#   Experimento [17]: cabeza no lineal en lugar del ridge, con los
#   descriptores del modelo actual (<work>/features.npz: seg3 + DINOv2-L,
#   64 vacas). No toca el modelo final.
#
#   Cabezas (todas eligen sus hiperparámetros con GroupKFold(4) por vaca
#   DENTRO del train de cada fold, minimizando el MAE por foto, como el ridge):
#     ridge   bcs_regions.fit_predict (referencia)
#     krr     kernel ridge RBF; alpha x gamma (gamma relativo a 1/dims)
#     mlp     6150 -> 256 -> 1 (GELU, dropout 0.5), AdamW lr 1e-3, batch 64,
#             weight decay y número de épocas elegidos dentro del fold;
#             predicción = promedio de 5 semillas
#     ridge+mlp  promedio simple de las dos predicciones
#
# Uso: docker compose run --rm --entrypoint python bcs exp_head.py
# =============================================================
import argparse, csv, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import bcs_regions as br

DEV = br.device()
EPOCHS = [5, 10, 25, 50, 100]
WDS = [0.1, 1.0, 10.0]
KRR_ALPHA, KRR_GAMMA = [0.01, 0.1, 1.0, 10.0], [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]

def std(tr, te):
    mu, sd = tr.mean(0), tr.std(0); sd[sd == 0] = 1
    return (tr - mu) / sd, (te - mu) / sd

# ---------- MLP ----------
def mlp_curve(A, ytr, B, wd, seed):
    """Entrena hasta max(EPOCHS) y devuelve {épocas: predicción en B} en cada punto de control."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(A.shape[1], 256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, 1)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    xa = torch.tensor(A, dtype=torch.float32, device=DEV); xb = torch.tensor(B, dtype=torch.float32, device=DEV)
    ym = ytr.mean(); yt = torch.tensor(ytr - ym, dtype=torch.float32, device=DEV)[:, None]
    g = torch.Generator(device="cpu").manual_seed(seed); out = {}
    for ep in range(1, max(EPOCHS) + 1):
        net.train()
        for idx in torch.randperm(len(xa), generator=g).split(64):
            loss = nn.functional.smooth_l1_loss(net(xa[idx]), yt[idx], beta=0.5)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in EPOCHS:
            net.eval()
            with torch.no_grad(): out[ep] = net(xb)[:, 0].cpu().numpy() + ym
    return out

def mlp_fit_predict(Xtr, ytr, gtr, Xte, seeds=5):
    err = {(wd, ep): [] for wd in WDS for ep in EPOCHS}
    for i, v in GroupKFold(4).split(Xtr, ytr, gtr):
        A, B = std(Xtr[i], Xtr[v])
        for wd in WDS:
            for ep, p in mlp_curve(A, ytr[i], B, wd, 0).items(): err[(wd, ep)].append(np.abs(p - ytr[v]))
    wd, ep = min(err, key=lambda k: np.mean(np.concatenate(err[k])))
    A, B = std(Xtr, Xte)
    return np.mean([mlp_curve(A, ytr, B, wd, s)[ep] for s in range(seeds)], 0), (wd, ep)

# ---------- Kernel ridge RBF ----------
def sqdist(A, B): return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0)

def krr_predict(A, ytr, B, al, gm):
    g = gm / A.shape[1]; K = np.exp(-g * sqdist(A, A)); Kb = np.exp(-g * sqdist(B, A)); ym = ytr.mean()
    return Kb @ np.linalg.solve(K + al * np.eye(len(A)), ytr - ym) + ym

def krr_fit_predict(Xtr, ytr, gtr, Xte):
    err = {(al, gm): [] for al in KRR_ALPHA for gm in KRR_GAMMA}
    for i, v in GroupKFold(4).split(Xtr, ytr, gtr):
        A, B = std(Xtr[i], Xtr[v])
        for k in err: err[k].append(np.abs(krr_predict(A, ytr[i], B, *k) - ytr[v]))
    k = min(err, key=lambda k: np.mean(np.concatenate(err[k]))); A, B = std(Xtr, Xte)
    return krr_predict(A, ytr, B, *k), k

HEADS = {"ridge": lambda *a: br.fit_predict(*a), "krr": krr_fit_predict, "mlp": mlp_fit_predict}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="/work"); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--heads", default="ridge,krr,mlp")
    a = ap.parse_args(); work = Path(a.work_dir); out = work / "exp_head"; out.mkdir(exist_ok=True)
    X, y, cows, names = br.load_features(work); uc = np.unique(cows); strata = np.round(y * 2).astype(int)
    schemes = {"loco": [list(GroupKFold(len(uc)).split(X, y, cows))],
               "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, strata, cows)) for s in range(a.seeds)]}
    print(f"{len(y)} fotos, {len(uc)} vacas, {X.shape[1]} dims, dispositivo {DEV}", flush=True)
    P = {}; sel = {}
    for h in a.heads.split(","):
        for sch, reps in schemes.items():
            t0 = time.time(); runs = []
            for splits in reps:
                p = np.zeros(len(y))
                for tr, te in splits:
                    p[te], s = HEADS[h](X[tr], y[tr], cows[tr], X[te]); sel.setdefault((h, sch), []).append(s)
                runs.append(p)
            P[(h, sch)] = np.mean(runs, 0)
            print(f"  {h:6s} {sch:7s} listo ({time.time() - t0:.0f}s) | elegidos: "
                  f"{sorted(set(map(str, sel[(h, sch)])), key=lambda s: -list(map(str, sel[(h, sch)])).count(s))[:3]}", flush=True)
    if "ridge" in a.heads and "mlp" in a.heads:
        for sch in schemes: P[("ridge+mlp", sch)] = (P[("ridge", sch)] + P[("mlp", sch)]) / 2
    rows = []
    print()
    for (h, sch), p in P.items():
        m = br.metrics(p, y, cows); e = p - y
        r = {"cabeza": h, "esquema": sch, **{k: round(float(v), 4) for k, v in m.items()},
             "sesgo_flacas": round(float(e[y < 3].mean()), 3), "sesgo_gordas": round(float(e[y >= 4.5].mean()), 3)}
        if h != "ridge":
            base = P[("ridge", sch)]
            d = np.array([np.abs(p - y)[cows == c].mean() - np.abs(base - y)[cows == c].mean() for c in uc])
            rng = np.random.RandomState(0); bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(5000)]
            r.update(dif_mae_vaca=round(float(d.mean()), 4), dif_lo=round(float(np.percentile(bs, 2.5)), 4),
                     dif_hi=round(float(np.percentile(bs, 97.5)), 4), mejoran=int((d < 0).sum()))
        rows.append(r)
        extra = f" | vs ridge {r['dif_mae_vaca']:+.3f} [{r['dif_lo']:+.3f}, {r['dif_hi']:+.3f}] mejoran {r['mejoran']}/{len(uc)}" if h != "ridge" else ""
        print(f"{h:10s} {sch:7s} MAE={m['mae']:.3f} R2={m['r2_fotos']:.2f} R2v={m['r2_vacas']:.2f} ±0.5={m['acc_05']:.0%} "
              f"MAEvaca={m['mae_vaca']:.3f} flacas={r['sesgo_flacas']:+.2f} gordas={r['sesgo_gordas']:+.2f}{extra}")
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(out / "preds_loco.csv", "w", newline="") as fh:
        hs = [h for (h, s) in P if s == "loco"]; w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real"] + hs)
        for i in range(len(y)): w.writerow([names[i], cows[i], y[i]] + [round(float(P[(h, "loco")][i]), 4) for h in hs])
    print(f"-> {out}")

if __name__ == "__main__":
    main()
