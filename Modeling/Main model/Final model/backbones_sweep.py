# =============================================================
# Nombre del archivo: backbones_sweep.py
#
# Descripción:
#   Barrido de backbones congelados para BCS solo con imagen, con el
#   mismo protocolo que E1/E2 de backbones_frozen.py (mismos folds por
#   vaca, mismas cabezas ridge / MLP / CORN, misma validación interna).
#   Para cada backbone:
#     global   = [resumen global, promedio de patches]; el resumen es el
#                CLS (DINOv2/DINOv3), el attention pooling (SigLIP) o el
#                max-pooling del mapa (encoder del segmentador, ConvNeXt).
#     regiones = promedio de patches dentro de cada una de las 6 máscaras
#                (reducidas a la malla de patches del backbone, peso =
#                fracción del patch dentro de la región) + indicador de
#                región ausente.
#   Al final reconstruye metrics.csv y summary.csv con TODOS los
#   preds_*.csv de <out_dir> (E0, E1, E2 y el barrido).
# =============================================================
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import timm
from PIL import Image

from filmModel import load_records
from backbones_frozen import load_folds, mask_path, inner_select, fit_predict, metrics, MEAN, STD

# nombre corto -> (modelo timm, tamaño de entrada, tipo)
BACKBONES = {
    "dinov2L":   ("vit_large_patch14_dinov2.lvd142m", 378, "vit"),
    "dinov3B":   ("vit_base_patch16_dinov3.lvd1689m", 384, "vit"),
    "siglip2B":  ("vit_base_patch16_siglip_384.v2_webli", 384, "vit"),
    "segencB":   ("vit_base_patch14_dinov2.lvd142m", 378, "segenc"),
    "convnextB": ("convnext_base.fb_in22k_ft_in1k", 384, "cnn"),
    "dinov2Lreg": ("vit_large_patch14_reg4_dinov2.lvd142m", 378, "vit"),
    "dinov2g":   ("vit_giant_patch14_dinov2.lvd142m", 378, "vit"),
}

def build(spec, segenc_weights):
    name, size, kind = spec
    if kind == "segenc":  # encoder DINOv2 del segmentador de regiones (pesos de final_all40.pt)
        m = timm.create_model(name, pretrained=False, features_only=True, dynamic_img_size=True, out_indices=(2, 5, 8, 11))  # mismos índices que al entrenar
        sd = torch.load(segenc_weights, map_location="cpu")
        m.load_state_dict({k[2:]: v for k, v in sd.items() if k.startswith("e.")})
        return m
    kw = {"img_size": size} if kind == "vit" else {}
    return timm.create_model(name, pretrained=True, num_classes=0, **kw)

@torch.no_grad()
def tokens(model, kind, x):
    """-> (resumen global [C], mapa de patches [C, gh, gw])"""
    if kind == "vit":
        t = model.forward_features(x)
        n = model.num_prefix_tokens; p = t[0, n:]
        g = int(round(p.shape[0] ** 0.5)); fmap = p.T.reshape(-1, g, g)
        summ = t[0, 0] if n > 0 else model.forward_head(t, pre_logits=True)[0]
        return summ, fmap
    fmap = (model(x)[-1] if kind == "segenc" else model.forward_features(x))[0]
    return fmap.amax((1, 2)), fmap

@torch.no_grad()
def extract(records, masks_dir, spec, segenc_weights, device):
    name, size, kind = spec
    model = build(spec, segenc_weights).to(device).eval()
    G, Rg, A = [], [], []
    for r in records:
        img = Image.open(r["path"]).convert("RGB").resize((size, size), Image.BICUBIC)  # igual que E1/E2: imagen completa aplastada
        x = torch.from_numpy(((np.asarray(img) / 255.0 - MEAN) / STD).transpose(2, 0, 1)).float()[None].to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            summ, fmap = tokens(model, kind, x)
        summ, fmap = summ.float(), fmap.float()
        C, gh, gw = fmap.shape; patches = fmap.reshape(C, -1).T
        G.append(torch.cat([summ, patches.mean(0)]).cpu().numpy())
        m = np.load(mask_path(r["path"], masks_dir))["masks"].astype(np.float32) / 255.0
        m = F.interpolate(torch.from_numpy((m > 0.5).astype(np.float32))[None], size=(size, size), mode="nearest")
        w = F.adaptive_avg_pool2d(m, (gh, gw))[0].reshape(6, -1).to(device)
        tot = w.sum(1, keepdim=True); vec = (w @ patches) / tot.clamp_min(1e-6)
        miss = tot[:, 0] < 0.5; vec[miss] = 0
        Rg.append(vec.reshape(-1).cpu().numpy()); A.append(miss.float().cpu().numpy())
    del model; torch.cuda.empty_cache()
    return np.stack(G), np.stack(Rg), np.stack(A)

def write_preds(out, exp, records, folds, cows, y, p):
    with open(out / f"preds_{exp}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["foto", "vaca", "fold", "bcs_real", "bcs_pred"])
        for k, (_, te) in enumerate(folds):
            for i in te: w.writerow([Path(records[i]["path"]).name, cows[i], k, y[i], round(float(p[i]), 4)])

def rebuild_tables(out, records, folds, y):
    """metrics.csv / summary.csv desde todos los preds_*.csv (mismo cálculo de backbones_frozen.py)."""
    name_idx = {Path(r["path"]).name: i for i, r in enumerate(records)}
    rows, summ = [], []
    for f in sorted(out.glob("preds_*.csv")):
        exp = f.stem[len("preds_"):]; p = np.full(len(records), np.nan)
        for r in csv.DictReader(open(f)): p[name_idx[r["foto"]]] = float(r["bcs_pred"])
        per = []
        for k, (tr, te) in enumerate(folds):
            m = metrics(p[te], y[te], y[tr].mean()); per.append(m); rows.append({"experimento": exp, "fold": k, **m})
        summ.append({"experimento": exp, **{f"{key}_mean": float(np.mean([m[key] for m in per])) for key in per[0]},
                     **{f"{key}_sd": float(np.std([m[key] for m in per])) for key in per[0]}})
    rnd = lambda d: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(map(rnd, rows))
    summ.sort(key=lambda s: s["mae_mean"])
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0])); w.writeheader(); w.writerows(map(rnd, summ))
    print(f"\n{'experimento':40s} {'MAE':>12s} {'RMSE':>6s} {'acc0':>5s} {'±.25':>5s} {'±.5':>5s} {'MAE media':>10s}")
    for s in summ:
        print(f"{s['experimento']:40s} {s['mae_mean']:.3f}±{s['mae_sd']:.3f} {s['rmse_mean']:6.3f} {s['acc_0_mean']:5.2f} "
              f"{s['acc_025_mean']:5.2f} {s['acc_05_mean']:5.2f} {s['mae_media_mean']:10.3f}", flush=True)

def main(a):
    t0 = time.time(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    records = load_records(a.csv, a.images_dir, only_valid=True)
    y = np.array([r["bcs"] for r in records]); cows = np.array([r["vaca"] for r in records])
    folds = load_folds(Path(a.cv_dir), records)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp_file = out / "hparams.json"; hparams = json.load(open(hp_file)) if hp_file.exists() else {}
    for short in a.backbones.split(","):
        spec = BACKBONES[short]; ff = out / f"features_{short}.npz"
        if ff.exists():
            f = np.load(ff); G, Rg, A = f["glob"], f["reg"], f["absent"]
        else:
            G, Rg, A = extract(records, Path(a.masks_dir), spec, a.segenc_weights, dev)
            np.savez_compressed(ff, glob=G, reg=Rg, absent=A)
        print(f"[{short}] features global {G.shape}, regiones {Rg.shape} ({time.time()-t0:.0f}s)", flush=True)
        for fname, X in (("global", G), ("regiones", np.hstack([Rg, A]))):
            for kind in a.heads.split(","):
                exp = f"B_{short}_{fname}_{kind}"; p = np.zeros(len(records)); hparams[exp] = []
                for k, (tr, te) in enumerate(folds):
                    tr, te = np.array(tr), np.array(te)
                    hp, inner = inner_select(kind, X[tr], y[tr], cows[tr])
                    p[te] = fit_predict(kind, hp, X[tr], y[tr], X[te]); hparams[exp].append({"fold": k, **hp, "inner_mae": round(inner, 4)})
                write_preds(out, exp, records, folds, cows, y, p)
                print(f"[{short}] {exp}: MAE={np.abs(p - y).mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(hparams, open(hp_file, "w"), indent=1)
        rebuild_tables(out, records, folds, y)
    print(f"SWEEP DONE {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset_frbcs_del.csv")
    ap.add_argument("--images_dir", default="Dataset_Fr_BCS")
    ap.add_argument("--masks_dir", default="Dataset_Fr_BCS_masks")
    ap.add_argument("--cv_dir", default="outputs_frbcs_cv")
    ap.add_argument("--out_dir", default="outputs_backbones_cv")
    ap.add_argument("--segenc_weights", default="final_all40.pt")
    ap.add_argument("--backbones", default="segencB,dinov2L,dinov3B,siglip2B,convnextB")
    ap.add_argument("--heads", default="ridge,mlp_mse,mlp_corn")
    main(ap.parse_args())
