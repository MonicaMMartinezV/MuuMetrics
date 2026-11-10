# =============================================================
# Nombre del archivo: ablation.py
#
# Descripción:
#   Estudio de ablación de data augmentation sobre el modelo canónico
#   (bcs_regions.py, que NO se modifica). Solo lee el caché canónico
#   (<work>/masks, <work>/features.npz) y escribe todo en <work>/aug.
#
#   Diseño:
#   - Test: siempre las filas originales del caché canónico (sin TTA nuevo).
#   - Train: por cada foto original, la fila original + K variantes
#     aumentadas, cada una con peso 1/(K+1). El peso total de cada foto
#     es 1, igual que en A0.
#   - Ridge ponderado: min_w,b  sum_i w_i (y_i - b - a_i.w)^2 + alpha |w|^2,
#     con estandarización ponderada (media y sd con los mismos pesos).
#     Con copias idénticas (A0-DUP) es exactamente el problema de A0.
#   - alpha: en cada fold se usa el que select_alpha() canónico elige con
#     las filas ORIGINALES del train de ese fold. A0 usa fit_predict() sin cambios.
#   - Particiones: las mismas que cmd_evaluate (LOCO y 5 folds x 10 semillas).
#
# Uso (dentro del contenedor):
#   python ablation.py run --policy A0 A0-DUP     # controles
#   python ablation.py run --policy A1 A2 A3      # requiere controles aprobados
#   python ablation.py run --policy all           # batería completa (controles primero)
#   python ablation.py report                     # tabla + gráficos de lo ya corrido
#   python ablation.py checkflip --n 20           # flip: máscaras transformadas vs segmentador
# =============================================================
import argparse, csv, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import augment as AG
from data import load_records
from bcs_regions import (device, load_backbone, describe, read_rgb, mask_file, load_features, select_alpha,
                         fit_predict, metrics, BACKBONE, SIZE, ALPHAS)
from segmenter import REGIONS

ORDER = ["A0", "A0-DUP", "A1", "A2", "A3", "A4", "A5", "A6"]
CONTROLS = ["A0", "A0-DUP"]
TOL_DUP = 1e-6          # |pred A0-DUP - pred A0| máximo aceptado
TOL_CANON = 6e-5        # A0 vs preds_*.csv canónicos (guardados con 4 decimales)
N_BOOT = 5000

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def jdump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2, ensure_ascii=False)

# ---------- Datos canónicos ----------
def load_canon(a):
    work = Path(a.work_dir); X, y, cows, names = load_features(work)
    recs = {r["name"]: r for r in load_records(a.data_dir, verbose=False)}
    if set(recs) != set(names): raise SystemExit("features.npz no coincide con el dataset: corre el pipeline canónico.")
    return X, y, cows, names, [recs[n] for n in names]

def schemes(X, y, cows, seeds):
    """Exactamente las particiones de cmd_evaluate (bcs_regions.py)."""
    uc = np.unique(cows); strata = np.round(y * 2).astype(int)
    return {"loco": [list(GroupKFold(len(uc)).split(X, y, cows))],
            "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, strata, cows)) for s in range(seeds)]}

# ---------- Descriptores aumentados ----------
def feat_file(a, pol): return Path(a.out_dir) / "features" / f"{pol}.npz"

def cmd_extract(a, pol):
    out = feat_file(a, pol)
    if AG.POLICIES[pol] is None or (out.exists() and not a.overwrite): return
    X, y, cows, names, recs = load_canon(a); dev = device(); bb = load_backbone(dev)
    work, K, spec = Path(a.work_dir), AG.K_VARIANTS, AG.POLICIES[pol]
    Xa = np.zeros((len(recs), K, X.shape[1]), np.float32); rows = []; t0 = time.time()
    qc = Path(a.out_dir) / "qc" / pol; qc.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(recs):
        img = read_rgb(r["path"]); m = np.load(mask_file(work, r))["masks"].astype(np.float32) / 255.0
        for k in range(K):
            e = AG.resolve(spec, AG.sample_params(r["name"], k)); im2, m2 = AG.apply(img, m, e)
            Xa[i, k] = describe(bb, im2, m2, dev)
            b = m2 > 0.5; xs = np.arange(m2.shape[2])
            cx = [float((b[c].sum(0) * xs).sum() / max(b[c].sum(), 1)) for c in (4, 5)]
            rows.append({"foto": r["name"], "vaca": r["vaca"], "k": k, **{kk: e[kk] for kk in
                         ("flip", "brightness", "contrast", "angle", "tx", "ty", "scale", "noise")},
                         "noise_seed": e["noise_seed"], "thighL_left_of_thighR": cx[0] < cx[1],
                         "regiones_ausentes": int(Xa[i, k, -6:].sum())})
            if i < 4 and k == 0: save_qc(qc / f"{i}_{Path(r['name']).stem}.jpg", img, m, im2, m2)
        if (i + 1) % 50 == 0: print(f"  [{pol}] {i + 1}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, Xaug=Xa, name=names, policy=json.dumps(spec), k=K, base_seed=AG.BASE_SEED)
    with open(out.with_suffix(".params.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    bad = sum(not r["thighL_left_of_thighR"] for r in rows)
    print(f"[extract {pol}] {Xa.shape} -> {out} | variantes con thigh_L a la derecha de thigh_R: {bad} ({time.time() - t0:.0f}s)")

def save_qc(path, img, m, im2, m2):
    """Lado a lado: original | aumentada, con thigh_L en rojo y thigh_R en azul."""
    def tint(im, ms):
        x = im.astype(np.float32).copy(); h, w = im.shape[:2]
        for c, col in ((4, (255, 0, 0)), (5, (0, 80, 255))):
            mm = np.asarray(Image.fromarray((ms[c] * 255).astype(np.uint8)).resize((w, h))) > 127
            x[mm] = 0.55 * x[mm] + 0.45 * np.array(col)
        return x.astype(np.uint8)
    a, b = tint(img, m), tint(im2, m2); Image.fromarray(np.concatenate([a, b], 1)).resize(
        ((a.shape[1] + b.shape[1]) // 3, a.shape[0] // 3)).save(path, quality=85)

# ---------- Ridge ponderado ----------
def fit_weighted(Xtr, ytr, wtr, Xte, alpha):
    """Ridge con pesos por fila y estandarización ponderada; igual a fit_predict cuando los pesos son 1 y alpha coincide."""
    sw = wtr.sum(); mu = wtr @ Xtr / sw
    sd = np.sqrt(wtr @ (Xtr - mu) ** 2 / sw); sd[sd <= 1e-12 * np.maximum(1, np.abs(mu))] = 1
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd; yb = wtr @ ytr / sw; s = np.sqrt(wtr)
    As, ys = A * s[:, None], (ytr - yb) * s
    dual = np.linalg.solve(As @ As.T + alpha * np.eye(len(As)), ys)
    return B @ (As.T @ dual) + yb

def alphas_for(a, X, y, cows, sch):
    """alpha canónico por fold (select_alpha con filas originales del train), en caché junto con las vacas de test."""
    f = Path(a.out_dir) / "splits.json"
    reg = json.load(open(f)) if f.exists() else None
    if reg and reg.get("seeds") == a.seeds and reg.get("features_sha256") == sha256(Path(a.work_dir) / "features.npz"):
        return reg
    reg = {"seeds": a.seeds, "features_sha256": sha256(Path(a.work_dir) / "features.npz"), "alphas_grid": list(ALPHAS),
           "vacas": sorted(np.unique(cows).tolist()), "schemes": {}}
    for name, reps in sch.items():
        reg["schemes"][name] = [[{"test_vacas": sorted(np.unique(cows[te]).tolist()), "n_train": int(len(tr)),
                                  "n_test": int(len(te)), "alpha": float(select_alpha(X[tr], y[tr], cows[tr]))}
                                 for tr, te in splits] for splits in reps]
        print(f"  [alpha] {name}: {sum(len(s) for s in reps)} folds", flush=True)
    jdump(reg, f); return reg

# ---------- Evaluación ----------
def predict_policy(a, pol, X, y, cows, sch, reg):
    spec = AG.POLICIES[pol]; out = {}
    Xa = None if spec is None else np.load(feat_file(a, pol))["Xaug"].astype(np.float64)
    K = 0 if Xa is None else Xa.shape[1]
    for name, reps in sch.items():
        P = []
        for r, splits in enumerate(reps):
            p = np.zeros(len(y))
            for f, (tr, te) in enumerate(splits):
                al = reg["schemes"][name][r][f]["alpha"]
                if spec is None:
                    p[te], al0 = fit_predict(X[tr], y[tr], cows[tr], X[te])
                    if al0 != al: raise SystemExit(f"alpha distinto al canónico en {name}/{r}/{f}: {al0} vs {al}")
                else:
                    Xtr = np.concatenate([X[tr], Xa[tr].reshape(-1, X.shape[1])])
                    ytr = np.concatenate([y[tr], np.repeat(y[tr], K)])
                    p[te] = fit_weighted(Xtr, ytr, np.full(len(ytr), 1 / (K + 1)), X[te], al)
            P.append(p)
        out[name] = np.array(P)
    return out

def summarize(P, y, cows):
    p = P.mean(0); m = metrics(p, y, cows)
    m["bias"] = float((p - y).mean()); m["mae_reps_sd"] = float(np.std([np.abs(q - y).mean() for q in P]))
    return p, m

def cmd_evaluate(a, pol):
    X, y, cows, names, _ = load_canon(a); sch = schemes(X, y, cows, a.seeds); reg = alphas_for(a, X, y, cows, sch)
    if pol not in CONTROLS: require_controls(a)
    t0 = time.time(); P = predict_policy(a, pol, X, y, cows, sch, reg); res = Path(a.out_dir) / "results" / pol
    res.mkdir(parents=True, exist_ok=True); rows = []
    for name, Ps in P.items():
        p, m = summarize(Ps, y, cows); np.save(res / f"preds_{name}_reps.npy", Ps)
        rows.append({"esquema": name, **{k: round(float(v), 4) for k, v in m.items()}, "fotos": len(y), "vacas": len(np.unique(cows))})
        with open(res / f"preds_{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real", "bcs_pred"])
            w.writerows([[n, c, yy, round(float(pp), 4)] for n, c, yy, pp in zip(names, cows, y, p)])
    with open(res / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    code = {f: sha256(Path(__file__).parent / f) for f in ("ablation.py", "augment.py", "bcs_regions.py", "segmenter.py", "data.py")}
    jdump({"policy": pol, "spec": AG.POLICIES[pol], "k_variants": 0 if AG.POLICIES[pol] is None else AG.K_VARIANTS,
           "row_weight": None if AG.POLICIES[pol] is None else 1 / (AG.K_VARIANTS + 1), "base_seed": AG.BASE_SEED,
           "cv_seeds": list(range(a.seeds)), "bootstrap": {"n": N_BOOT, "seed": 0, "unit": "vaca"},
           "backbone": BACKBONE, "size": SIZE, "regions": REGIONS, "features_sha256": reg["features_sha256"],
           "aug_features_sha256": None if AG.POLICIES[pol] is None else sha256(feat_file(a, pol)),
           "code_sha256": code, "torch": torch.__version__, "python": platform.python_version(),
           "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
           "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}, res / "config.json")
    for r in rows:
        print(f"[{pol:6s} {r['esquema']:7s}] MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} R2f={r['r2_fotos']:.3f} R2v={r['r2_vacas']:.3f} "
              f"bias={r['bias']:+.4f} ±0.5={r['acc_05']:.1%} MAEvaca={r['mae_vaca']:.4f} ({time.time() - t0:.0f}s)")

# ---------- Controles ----------
def check_controls(a):
    X, y, cows, names, _ = load_canon(a); R = Path(a.out_dir) / "results"; rep = {}
    for name in ("loco", "rep5x10"):
        p0 = np.load(R / "A0" / f"preds_{name}_reps.npy"); pd = np.load(R / "A0-DUP" / f"preds_{name}_reps.npy")
        canon = {row["foto"]: float(row["bcs_pred"]) for row in csv.DictReader(open(Path(a.work_dir) / f"preds_{name}.csv"))}
        c = np.array([canon[n] for n in names])
        rep[name] = {"max_abs_A0_vs_canonico": float(np.abs(p0.mean(0) - c).max()),
                     "max_abs_A0DUP_vs_A0": float(np.abs(pd - p0).max())}
    xa = np.load(feat_file(a, "A0-DUP"))["Xaug"].astype(np.float64)
    rep["max_abs_features_A0DUP_vs_canonico"] = float(np.abs(xa - X[:, None]).max())
    rep["ok_canonico"] = all(rep[s]["max_abs_A0_vs_canonico"] <= TOL_CANON for s in ("loco", "rep5x10"))
    rep["ok_dup"] = all(rep[s]["max_abs_A0DUP_vs_A0"] <= TOL_DUP for s in ("loco", "rep5x10"))
    rep["tolerancias"] = {"canonico": TOL_CANON, "dup": TOL_DUP}; rep["ok"] = rep["ok_canonico"] and rep["ok_dup"]
    jdump(rep, Path(a.out_dir) / "controls.json"); print("[controles]", json.dumps(rep, indent=1))
    if not rep["ok"]: raise SystemExit("Controles NO aprobados: no se evalúan políticas de augmentation.")

def require_controls(a):
    f = Path(a.out_dir) / "controls.json"
    if not f.exists() or not json.load(open(f)).get("ok"):
        raise SystemExit("Faltan controles aprobados: corre primero  run --policy A0 A0-DUP")

# ---------- Reporte ----------
def paired(p, p0, y, cows, rng):
    uc = np.unique(cows); idx = [np.where(cows == c)[0] for c in uc]; n = np.array([len(i) for i in idx])
    se = np.array([np.abs(p - y)[i].sum() for i in idx]); se0 = np.array([np.abs(p0 - y)[i].sum() for i in idx])
    b = rng.randint(0, len(uc), (N_BOOT, len(uc))); dc = se / n - se0 / n
    bp = (se - se0)[b].sum(1) / n[b].sum(1); bc = dc[b].mean(1); ma = se[b].sum(1) / n[b].sum(1)
    q = lambda v: np.percentile(v, [2.5, 97.5])
    return {"mae_ic95": q(ma), "d_mae": (se - se0).sum() / n.sum(), "d_mae_ic95": q(bp),
            "d_mae_vaca": dc.mean(), "d_mae_vaca_ic95": q(bc), "vacas_mejor": int((dc < -1e-9).sum()), "vacas_peor": int((dc > 1e-9).sum())}

def cmd_report(a):
    X, y, cows, names, _ = load_canon(a); R = Path(a.out_dir) / "results"; rep = Path(a.out_dir) / "report"
    pols = [p for p in ORDER if (R / p / "preds_loco_reps.npy").exists()]
    if "A0" not in pols: raise SystemExit("Falta A0.")
    rows = []
    for name in ("loco", "rep5x10"):
        p0 = np.load(R / "A0" / f"preds_{name}_reps.npy").mean(0)
        for pol in pols:
            p, m = summarize(np.load(R / pol / f"preds_{name}_reps.npy"), y, cows)
            d = paired(p, p0, y, cows, np.random.RandomState(0))
            rows.append({"esquema": name, "politica": pol, "mae": m["mae"], "mae_lo": d["mae_ic95"][0], "mae_hi": d["mae_ic95"][1],
                         "mae_vaca": m["mae_vaca"], "mae_vaca_lo": m["mae_vaca_ic95_lo"], "mae_vaca_hi": m["mae_vaca_ic95_hi"],
                         "rmse": m["rmse"], "r2_fotos": m["r2_fotos"], "r2_vacas": m["r2_vacas"], "bias": m["bias"],
                         "acc_05": m["acc_05"], "d_mae": d["d_mae"], "d_mae_lo": d["d_mae_ic95"][0], "d_mae_hi": d["d_mae_ic95"][1],
                         "d_mae_vaca": d["d_mae_vaca"], "d_mae_vaca_lo": d["d_mae_vaca_ic95"][0], "d_mae_vaca_hi": d["d_mae_vaca_ic95"][1],
                         "vacas_mejor": d["vacas_mejor"], "vacas_peor": d["vacas_peor"], "mae_reps_sd": m["mae_reps_sd"]})
    rep.mkdir(parents=True, exist_ok=True)
    with open(rep / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
        w.writerows([{k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for r in rows])
    f3 = lambda v: f"{v:.3f}"; ci = lambda r, k: f"{r[k]:+.3f} [{r[k + '_lo']:+.3f}, {r[k + '_hi']:+.3f}]"
    md = ["| Esquema | Política | MAE fotos [IC 95 %] | MAE vaca [IC 95 %] | RMSE | R² fotos | R² vacas | Bias | ≤0.5 | ΔMAE fotos vs A0 [IC 95 %] | ΔMAE vaca vs A0 [IC 95 %] | Vacas mejor/peor |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['esquema']} | {r['politica']} | {f3(r['mae'])} [{f3(r['mae_lo'])}, {f3(r['mae_hi'])}] | "
                  f"{f3(r['mae_vaca'])} [{f3(r['mae_vaca_lo'])}, {f3(r['mae_vaca_hi'])}] | {f3(r['rmse'])} | {r['r2_fotos']:.2f} | "
                  f"{r['r2_vacas']:.2f} | {r['bias']:+.3f} | {r['acc_05']:.0%} | " +
                  ("—" if r["politica"] == "A0" else ci(r, "d_mae")) + " | " + ("—" if r["politica"] == "A0" else ci(r, "d_mae_vaca")) +
                  f" | {'—' if r['politica'] == 'A0' else str(r['vacas_mejor']) + '/' + str(r['vacas_peor'])} |")
    (rep / "comparison.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
    plot(rows, [p for p in pols if p != "A0"], rep)

def plot(rows, pols, rep):
    if not pols: return
    import os; os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")       # el usuario del contenedor no tiene $HOME
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    series = [("d_mae", "MAE por foto", "#2a78d6", "o"), ("d_mae_vaca", "MAE promedio por vaca", "#eb6834", "s")]
    fig, axs = plt.subplots(1, 2, figsize=(9, 1.1 + 0.55 * len(pols)), sharey=True, facecolor=surf)
    for ax, name, title in zip(axs, ("loco", "rep5x10"), ("Una vaca fuera (LOCO)", "5 folds por vaca × 10 semillas")):
        ax.set_facecolor(surf); rs = {r["politica"]: r for r in rows if r["esquema"] == name}
        for j, (k, lab, col, mk) in enumerate(series):
            yy = np.arange(len(pols)) + (j - 0.5) * 0.28
            v = np.array([rs[p][k] for p in pols]); lo = np.array([rs[p][k + "_lo"] for p in pols]); hi = np.array([rs[p][k + "_hi"] for p in pols])
            ax.hlines(yy, lo, hi, color=col, lw=2, zorder=2)
            ax.plot(v, yy, mk, color=col, ms=7, mec=surf, mew=1.5, zorder=3, label=lab)
        ax.axvline(0, color=ink2, lw=1, zorder=1)
        ax.set_title(title, color=ink, fontsize=10, loc="left"); ax.grid(axis="x", color=grid, lw=0.8); ax.set_axisbelow(True)
        ax.set_xlabel("Δ MAE vs A0 (BCS)  ·  < 0 = mejor que A0", color=ink2, fontsize=9)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(grid); ax.tick_params(colors=ink2, labelsize=9, length=0)
    axs[0].set_yticks(np.arange(len(pols)), pols); axs[0].invert_yaxis()
    fig.suptitle("Diferencia pareada de MAE respecto a A0 (IC 95 %, bootstrap por vaca)", color=ink, fontsize=11, x=0.02, ha="left")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=8, labelcolor=ink2, bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"): fig.savefig(rep / f"delta_mae_vs_A0.{ext}", dpi=200, facecolor=surf)
    print(f"[report] -> {rep / 'delta_mae_vs_A0.png'}")

# ---------- Verificación del flip ----------
def cmd_checkflip(a):
    """Compara las máscaras del flip+intercambio con lo que el segmentador predice sobre la foto volteada."""
    from segmenter import load_segmenter, segment
    X, y, cows, names, recs = load_canon(a); dev = device(); net = load_segmenter(a.seg_weights, dev); work = Path(a.work_dir)
    rng = np.random.RandomState(0); sel = rng.choice(len(recs), min(a.n, len(recs)), replace=False)
    e = AG.resolve({"flip": 1.0}, AG.sample_params("x", 0)); iou = {c: [] for c in REGIONS}; iou_noswap = {c: [] for c in REGIONS}
    for i in sel:
        r = recs[i]; img = read_rgb(r["path"]); m = np.load(mask_file(work, r))["masks"].astype(np.float32) / 255.0
        im2, m2 = AG.apply(img, m, e); m_noswap = m[:, :, ::-1]
        s = segment(net, im2, dev); s = torch.nn.functional.interpolate(torch.from_numpy(s)[None], size=m2.shape[1:], mode="area")[0].numpy()
        for j, c in enumerate(REGIONS):
            for store, mm in ((iou, m2), (iou_noswap, m_noswap)):
                A, B = mm[j] > 0.5, s[j] > 0.5; store[c].append((A & B).sum() / max((A | B).sum(), 1))
    rep = {"n": len(sel), "iou_flip_con_intercambio": {c: round(float(np.mean(v)), 3) for c, v in iou.items()},
           "iou_flip_sin_intercambio": {c: round(float(np.mean(v)), 3) for c, v in iou_noswap.items()}}
    jdump(rep, Path(a.out_dir) / "checkflip.json"); print("[checkflip]", json.dumps(rep, indent=1))

# ---------- CLI ----------
def expand(pols): return ORDER if pols == ["all"] else pols

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ablación de data augmentation sobre el modelo canónico de BCS por regiones")
    ap.add_argument("step", choices=["run", "extract", "evaluate", "controls", "report", "checkflip"])
    ap.add_argument("--policy", nargs="+", default=["all"], help=f"políticas: {' '.join(ORDER)} o all")
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS")
    ap.add_argument("--work_dir", default="/work", help="caché canónico (solo lectura)")
    ap.add_argument("--out_dir", default="/work/aug", help="salidas de la ablación")
    ap.add_argument("--seg_weights", default="/weights/final_all40.pt")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n", type=int, default=20, help="checkflip: fotos a revisar")
    ap.add_argument("--overwrite", action="store_true", help="recalcular descriptores aumentados")
    a = ap.parse_args(); Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    if Path(a.out_dir).resolve() == Path(a.work_dir).resolve(): raise SystemExit("--out_dir no puede ser el caché canónico.")
    pols = expand(a.policy)
    for p in pols:
        if p not in AG.POLICIES: raise SystemExit(f"Política desconocida: {p}")
    if a.step == "extract":
        for p in pols: cmd_extract(a, p)
    elif a.step == "evaluate":
        for p in pols: cmd_evaluate(a, p)
    elif a.step == "controls": check_controls(a)
    elif a.step == "report": cmd_report(a)
    elif a.step == "checkflip": cmd_checkflip(a)
    else:
        ctrl = [p for p in CONTROLS if p in pols]; rest = [p for p in pols if p not in CONTROLS]
        for p in ctrl: cmd_extract(a, p); cmd_evaluate(a, p)
        if set(CONTROLS) <= set(ctrl): check_controls(a)
        if rest: require_controls(a)
        for p in rest: cmd_extract(a, p); cmd_evaluate(a, p)
        cmd_report(a)
