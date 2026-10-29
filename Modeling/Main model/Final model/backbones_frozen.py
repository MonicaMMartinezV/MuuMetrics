# =============================================================
# Nombre del archivo: backbones_frozen.py
#
# Descripción:
#   Comparación de backbones para estimar BCS solo con imagen, con
#   los mismos 5 folds agrupados por vaca de filmModel.py
#   (<cv_dir>/nofilm/fold*/split.json) y las mismas etiquetas
#   corregidas (dataset_frbcs_del.csv, valido=True).
#
#   E0  ConvNeXt-Base fine-tuned: se leen sus preds_test.npz.
#   E1  DINOv2 ViT-B/14 congelado a 378 px: CLS + promedio de patches.
#   E2  DINOv2 ViT-B/14 congelado a 378 px: masked average pooling por
#       región (6 máscaras del segmentador reducidas a la malla 27x27,
#       cada patch pesa la fracción de su área dentro de la región);
#       región ausente -> vector en cero + indicador.
#   Cabezas para E1/E2: ridge, MLP (MSE) y MLP con salida ordinal CORN.
#   Hiperparámetros: validación interna GroupKFold por vaca dentro del
#   train (train+val) de cada fold; las vacas de test no se tocan.
#
# Salidas (<out_dir>): features_*.npz, preds_<exp>.csv,
#   metrics.csv, summary.csv, hparams.json
# =============================================================
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from filmModel import load_records

BCS_MIN, BCS_STEP, N_CLS = 2.0, 0.25, 13  # escala 2.00 ... 5.00
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

# Datos y folds
def load_folds(cv_dir: Path, records):
    folds = []
    for k in range(5):
        s = json.load(open(cv_dir / "nofilm" / f"fold{k}" / "split.json"))["vacas"]
        pool, test = set(s["train"] + s["val"]), set(s["test"])
        folds.append(([i for i, r in enumerate(records) if r["vaca"] in pool],
                      [i for i, r in enumerate(records) if r["vaca"] in test]))
    tested = sorted(i for _, te in folds for i in te)
    assert tested == list(range(len(records))), "los folds de test no cubren cada foto exactamente una vez"
    return folds

def mask_path(img_path: str, masks_dir: Path) -> Path:
    p = Path(img_path)
    return masks_dir / p.parent.name / (p.stem + ".npz")

# Features DINOv2 congelado
@torch.no_grad()
def extract(records, masks_dir: Path, size: int, device):
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                              img_size=size).to(device).eval()
    g = size // 14
    glob_f, reg_f, absent = [], [], []
    for r in records:
        img = Image.open(r["path"]).convert("RGB").resize((size, size), Image.BICUBIC)  # aplasta a cuadrado: la vaca completa entra
        x = torch.from_numpy(((np.asarray(img) / 255.0 - MEAN) / STD).transpose(2, 0, 1)).float()[None].to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            tok = model.forward_features(x)[0].float()          # [1+g*g, 768], ya normalizado
        cls, patches = tok[0], tok[model.num_prefix_tokens:]
        glob_f.append(torch.cat([cls, patches.mean(0)]).cpu().numpy())

        m = np.load(mask_path(r["path"], masks_dir))["masks"].astype(np.float32) / 255.0   # [6,h,w] probabilidades
        m = torch.from_numpy((m > 0.5).astype(np.float32))[None]
        m = F.interpolate(m, size=(size, size), mode="nearest")
        w = F.avg_pool2d(m, 14)[0].reshape(6, -1).to(device)     # [6, g*g] fracción del patch en la región
        tot = w.sum(1, keepdim=True)
        vec = (w @ patches) / tot.clamp_min(1e-6)
        miss = (tot[:, 0] < 0.5)                                  # menos de medio patch -> región ausente
        vec[miss] = 0
        reg_f.append(vec.reshape(-1).cpu().numpy()); absent.append(miss.float().cpu().numpy())
    return np.stack(glob_f), np.stack(reg_f), np.stack(absent)

# Cabezas
class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=256, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, d_out))
    def forward(self, x): return self.net(x)

def to_rank(y): return np.clip(np.round((y - BCS_MIN) / BCS_STEP), 0, N_CLS - 1).astype(int)

def corn_loss(logits, rank):
    """CORN (Shi et al., 2021): la tarea k solo usa ejemplos con rango > k-1."""
    loss, n = 0.0, 0
    for k in range(logits.shape[1]):
        sel = rank > k - 1
        if sel.sum() == 0: continue
        loss = loss + F.binary_cross_entropy_with_logits(logits[sel, k], (rank[sel] > k).float(), reduction="sum")
        n += int(sel.sum())
    return loss / max(n, 1)

def corn_predict(logits):
    return (torch.cumprod(torch.sigmoid(logits), 1) > 0.5).sum(1)

def fit_predict(kind, hp, Xtr, ytr, Xte, seed=0):
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if kind == "ridge":
        return Ridge(alpha=hp["alpha"]).fit(Xtr, ytr).predict(Xte)
    torch.manual_seed(seed)
    xt, xe = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
    corn = kind == "mlp_corn"
    net = MLP(xt.shape[1], N_CLS - 1 if corn else 1)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=hp["wd"])
    mu, sd = ytr.mean(), ytr.std() + 1e-6
    yt = torch.tensor(to_rank(ytr)) if corn else torch.tensor((ytr - mu) / sd, dtype=torch.float32)
    for _ in range(hp["epochs"]):
        net.train(); opt.zero_grad()
        out = net(xt)
        loss = corn_loss(out, yt) if corn else F.mse_loss(out[:, 0], yt)
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        out = net(xe)
    return (BCS_MIN + BCS_STEP * corn_predict(out).numpy()) if corn else out[:, 0].numpy() * sd + mu

GRIDS = {"ridge": [{"alpha": a} for a in (1, 10, 100, 1e3, 1e4, 1e5)],
         "mlp_mse": [{"wd": wd, "epochs": ep} for wd in (1e-4, 1e-1) for ep in (100, 300)],
         "mlp_corn": [{"wd": wd, "epochs": ep} for wd in (1e-4, 1e-1) for ep in (100, 300)]}

def inner_select(kind, X, y, groups):
    gkf = GroupKFold(n_splits=4)
    best, best_mae = None, np.inf
    for hp in GRIDS[kind]:
        err = []
        for tr, va in gkf.split(X, y, groups):
            err.append(np.abs(fit_predict(kind, hp, X[tr], y[tr], X[va]) - y[va]))
        mae = float(np.mean(np.concatenate(err)))
        if mae < best_mae: best, best_mae = hp, mae
    return best, best_mae

# Métricas
def metrics(pred, y, ref):
    rd = np.clip(np.round(pred / BCS_STEP) * BCS_STEP, BCS_MIN, BCS_MIN + BCS_STEP * (N_CLS - 1))
    return {"mae": float(np.abs(pred - y).mean()), "rmse": float(np.sqrt(((pred - y) ** 2).mean())),
            "acc_0": float(np.mean(np.isclose(rd, y))), "acc_025": float(np.mean(np.abs(rd - y) <= 0.25 + 1e-9)),
            "acc_05": float(np.mean(np.abs(rd - y) <= 0.5 + 1e-9)), "mae_media": float(np.abs(ref - y).mean())}

def main(a):
    t0 = time.time()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    records = load_records(a.csv, a.images_dir, only_valid=True)
    y = np.array([r["bcs"] for r in records]); cows = np.array([r["vaca"] for r in records])
    folds = load_folds(Path(a.cv_dir), records)
    print(f"{len(records)} fotos, {len(set(cows))} vacas, folds de test: {[len(te) for _, te in folds]}")

    feat_file = out / f"features_dinov2_{a.size}.npz"
    if feat_file.exists():
        f = np.load(feat_file); G, Rg, A = f["glob"], f["reg"], f["absent"]
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        G, Rg, A = extract(records, Path(a.masks_dir), a.size, dev)
        np.savez_compressed(feat_file, glob=G, reg=Rg, absent=A, paths=np.array([r["path"] for r in records]))
    print(f"features: global {G.shape}, regiones {Rg.shape}, regiones ausentes por clase {A.sum(0).astype(int).tolist()} ({time.time()-t0:.0f}s)")
    feats = {"E1_dinov2_global": G, "E2_dinov2_regiones": np.hstack([Rg, A])}

    preds = {}
    # E0: ConvNeXt-Base fine-tuned ya entrenado (mismos folds)
    idx = {Path(r["path"]).name: i for i, r in enumerate(records)}
    p0 = np.full(len(records), np.nan)
    for k in range(5):
        z = np.load(Path(a.cv_dir) / "nofilm" / f"fold{k}" / "preds_test.npz")
        for pth, pr in zip(z["paths"], z["preds"]): p0[idx[Path(str(pth)).name]] = pr
    assert not np.isnan(p0).any()
    preds["E0_convnext_base_ft"] = p0

    hparams = {}
    for fname, X in feats.items():
        for kind in ("ridge", "mlp_mse", "mlp_corn"):
            exp = f"{fname}_{kind}"; p = np.zeros(len(records)); hparams[exp] = []
            for k, (tr, te) in enumerate(folds):
                tr, te = np.array(tr), np.array(te)
                hp, inner_mae = inner_select(kind, X[tr], y[tr], cows[tr])
                p[te] = fit_predict(kind, hp, X[tr], y[tr], X[te])
                hparams[exp].append({"fold": k, **hp, "inner_mae": round(inner_mae, 4)})
            preds[exp] = p
            print(f"{exp}: listo ({time.time()-t0:.0f}s)")

    rows, summ = [], []
    for exp, p in preds.items():
        with open(out / f"preds_{exp}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["foto", "vaca", "fold", "bcs_real", "bcs_pred"])
            for k, (_, te) in enumerate(folds):
                for i in te: w.writerow([Path(records[i]["path"]).name, cows[i], k, y[i], round(float(p[i]), 4)])
        per = []
        for k, (tr, te) in enumerate(folds):
            m = metrics(p[te], y[te], y[tr].mean()); per.append(m); rows.append({"experimento": exp, "fold": k, **m})
        summ.append({"experimento": exp, **{f"{key}_mean": float(np.mean([m[key] for m in per])) for key in per[0]},
                     **{f"{key}_sd": float(np.std([m[key] for m in per])) for key in per[0]}})
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
        w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
    summ.sort(key=lambda s: s["mae_mean"])
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0])); w.writeheader()
        w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for s in summ])
    json.dump(hparams, open(out / "hparams.json", "w"), indent=1)
    print(f"\n{'experimento':38s} {'MAE':>12s} {'RMSE':>6s} {'acc0':>5s} {'±.25':>5s} {'±.5':>5s} {'MAE media':>10s}")
    for s in summ:
        print(f"{s['experimento']:38s} {s['mae_mean']:.3f}±{s['mae_sd']:.3f} {s['rmse_mean']:6.3f} {s['acc_0_mean']:5.2f} "
              f"{s['acc_025_mean']:5.2f} {s['acc_05_mean']:5.2f} {s['mae_media_mean']:10.3f}")
    print(f"Total {time.time()-t0:.0f}s -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset_frbcs_del.csv")
    ap.add_argument("--images_dir", default="Dataset_Fr_BCS")
    ap.add_argument("--masks_dir", default="Dataset_Fr_BCS_masks")
    ap.add_argument("--cv_dir", default="outputs_frbcs_cv")
    ap.add_argument("--out_dir", default="outputs_backbones_cv")
    ap.add_argument("--size", type=int, default=378)
    main(ap.parse_args())
