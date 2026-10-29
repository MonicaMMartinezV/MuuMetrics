# =============================================================
# Nombre del archivo: region_transformer.py
#
# Descripción:
#   Cabeza transformer sobre los 6 tokens de región de un backbone
#   congelado (features cacheadas por backbones_frozen.py /
#   backbones_sweep.py), con el mismo protocolo que las demás cabezas:
#   mismos folds por vaca y validación interna GroupKFold por vaca
#   dentro del train de cada fold.
#
#   tokens de región -> proyección + embedding de región -> 1 capa de
#   self-attention entre regiones -> attention pooling con una query
#   aprendida (sus pesos = importancia de cada región por foto) -> BCS.
#   Las regiones ausentes se enmascaran en la atención.
#
# Salidas (<out_dir>): preds_B_<bb>_regiones_transformer.csv,
#   attn_<bb>.csv (peso de cada región por foto), y reconstruye
#   metrics.csv / summary.csv.
# =============================================================
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupKFold

from filmModel import load_records
from backbones_frozen import load_folds
from backbones_sweep import write_preds, rebuild_tables

REGIONS = ["loin", "hips", "rump", "tailhead", "thigh_L", "thigh_R"]
FEATURE_FILES = {"dinov2": "features_dinov2_378.npz"}  # el resto: features_<bb>.npz

class RegionTransformer(nn.Module):
    def __init__(self, c_in, d=64, heads=4, drop=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(c_in, d))
        self.reg_emb = nn.Parameter(torch.zeros(6, d))
        self.enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=drop, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x, absent):  # x [B,6,C], absent [B,6] bool
        h = self.enc(self.proj(x) + self.reg_emb, src_key_padding_mask=absent)
        z, w = self.pool(self.query.expand(len(x), -1, -1), h, h, key_padding_mask=absent)
        return self.head(z[:, 0])[:, 0], w[:, 0]  # w [B,6], promedio de cabezas

def fit_predict(hp, Xtr, Atr, ytr, Xte, Ate, seeds=(0,)):
    mu, sd = Xtr.mean((0, 1), keepdims=True), Xtr.std((0, 1), keepdims=True) + 1e-6
    tr, te = torch.tensor((Xtr - mu) / sd, dtype=torch.float32), torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    atr, ate = torch.tensor(Atr > 0.5), torch.tensor(Ate > 0.5)
    ym, ys = ytr.mean(), ytr.std() + 1e-6; yt = torch.tensor((ytr - ym) / ys, dtype=torch.float32)
    preds, ws = [], []
    for s in seeds:
        torch.manual_seed(s); net = RegionTransformer(tr.shape[2], drop=hp["drop"])
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=hp["wd"])
        for _ in range(hp["epochs"]):
            net.train()
            for idx in torch.randperm(len(tr)).split(32):
                a = atr[idx].clone()
                a |= (torch.rand_like(a, dtype=torch.float32) < 0.15)          # quita regiones al azar (aumento)
                a[a.all(1)] = atr[idx][a.all(1)]                               # nunca todas
                p, _ = net(tr[idx], a); loss = F.mse_loss(p, yt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p, w = net(te, ate)
        preds.append(p.numpy() * ys + ym); ws.append(w.numpy())
    return np.mean(preds, 0), np.mean(ws, 0)

GRID = [{"wd": wd, "epochs": ep, "drop": 0.3} for wd in (1e-2, 1e-1) for ep in (40, 120)]

def main(a):
    t0 = time.time(); out = Path(a.out_dir)
    records = load_records(a.csv, a.images_dir, only_valid=True)
    y = np.array([r["bcs"] for r in records]); cows = np.array([r["vaca"] for r in records])
    folds = load_folds(Path(a.cv_dir), records)
    summ = list(csv.DictReader(open(out / "summary.csv")))
    cand = [s for s in summ if s["experimento"].endswith("_regiones_ridge")]
    cand.sort(key=lambda s: float(s["mae_mean"]))
    bbs = [s["experimento"].split("_")[1] for s in cand[:a.top]]
    print("backbones elegidos (mejor ridge con regiones):", bbs, flush=True)
    hp_file = out / "hparams.json"; hparams = json.load(open(hp_file))
    for bb in bbs:
        f = np.load(out / FEATURE_FILES.get(bb, f"features_{bb}.npz"))
        X = f["reg"].reshape(len(records), 6, -1); A = f["absent"]
        exp = f"B_{bb}_regiones_transformer"; p = np.zeros(len(records)); W = np.zeros((len(records), 6)); hparams[exp] = []
        for k, (tr, te) in enumerate(folds):
            tr, te = np.array(tr), np.array(te)
            best, best_mae = None, np.inf
            for hp in GRID:                                     # validación interna por vaca
                err = []
                for itr, iva in GroupKFold(4).split(tr, groups=cows[tr]):
                    pp, _ = fit_predict(hp, X[tr][itr], A[tr][itr], y[tr][itr], X[tr][iva], A[tr][iva])
                    err.append(np.abs(pp - y[tr][iva]))
                mae = float(np.mean(np.concatenate(err)))
                if mae < best_mae: best, best_mae = hp, mae
            p[te], W[te] = fit_predict(best, X[tr], A[tr], y[tr], X[te], A[te], seeds=(0, 1, 2))
            hparams[exp].append({"fold": k, **best, "inner_mae": round(best_mae, 4)})
            print(f"[{bb}] fold {k}: {best} inner={best_mae:.3f} test MAE={np.abs(p[te]-y[te]).mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
        write_preds(out, exp, records, folds, cows, y, p)
        with open(out / f"attn_{bb}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["foto", "vaca", "fold"] + REGIONS)
            for k, (_, te) in enumerate(folds):
                for i in te: w.writerow([Path(records[i]["path"]).name, cows[i], k] + [round(float(v), 4) for v in W[i]])
        print(f"[{bb}] {exp}: MAE={np.abs(p - y).mean():.3f} | atención media por región: "
              + " ".join(f"{r}={v:.2f}" for r, v in zip(REGIONS, W.mean(0))), flush=True)
    json.dump(hparams, open(hp_file, "w"), indent=1)
    rebuild_tables(out, records, folds, y)
    print(f"TRANSFORMER DONE {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset_frbcs_del.csv")
    ap.add_argument("--images_dir", default="Dataset_Fr_BCS")
    ap.add_argument("--cv_dir", default="outputs_frbcs_cv")
    ap.add_argument("--out_dir", default="outputs_backbones_cv")
    ap.add_argument("--top", type=int, default=2)
    main(ap.parse_args())
