# =============================================================
# Nombre del archivo: analisis/analisis_paper.py
#
# Descripción:
#   Análisis para el artículo con el modelo de esta rama (bcs_regions.py,
#   comprobando la firma de features.npz). Todo por vaca, con IC por
#   bootstrap de vacas (5000):
#     1. error_bcs     error y sesgo por valor de BCS (predicciones LOCO) y
#                      Bland-Altman por vaca y por foto
#     2. camaras       generalización entre cámaras: LOCO por vaca entrenando
#                      solo con fotos de una cámara y probando en la otra,
#                      contra entrenar con la misma cámara (mismas fotos de
#                      prueba, comparación pareada por vaca)
#     3. tiempo        estabilidad de la predicción de una misma vaca entre
#                      días (sin cambio de BCS) y si sigue los cambios de BCS
#                      de las vacas calificadas más de una vez (8758, 9008)
#   Salidas: <work>/analisis/*.csv, resumen.md y figures/analisis_*.{pdf,png}
#
# Uso: docker compose run --rm --entrypoint python bcs analisis/analisis_paper.py
# =============================================================
import argparse, csv, json, re, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "ablaciones"))
import bcs_regions as br                    # noqa: E402
from comun import modelo                    # noqa: E402

N_BOOT = 5000
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8,
                     "axes.labelsize": 8, "legend.fontsize": 7, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})

def fecha(n): return re.search(r"(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}-\d{2}", n).group(1)
def camara(n): return re.search(r"_cam(\d+)_", n).group(1)

def boot_ci(f, cows, rng, n=N_BOOT):
    """IC95 de un estadístico f(índice de vacas) remuestreando vacas."""
    uc = np.unique(cows); b = [f(uc[rng.randint(0, len(uc), len(uc))]) for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])

def loco_preds(X, y, cows, mask_tr=None, mask_te=None):
    """LOCO por vaca; opcionalmente el train usa solo filas mask_tr y se predicen solo filas mask_te."""
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(len(np.unique(cows))).split(X, y, cows):
        if mask_tr is not None: tr = tr[mask_tr[tr]]
        if mask_te is not None: te = te[mask_te[te]]
        if len(te) == 0 or len(np.unique(cows[tr])) < 8: continue
        p[te] = br.fit_predict(X[tr], y[tr], cows[tr], X[te])[0]
    return p

# ---------- 1. Error por BCS y Bland-Altman ----------
def error_bcs(p, y, cows, out, figs, rng):
    filas = []
    for b in np.unique(y):
        m = y == b; e = p[m] - y[m]; cs = np.unique(cows[m])
        filas.append({"bcs": b, "vacas": len(cs), "fotos": int(m.sum()), "mae": float(np.abs(e).mean()),
                      "mae_vaca": float(np.mean([np.abs(e[cows[m] == c]).mean() for c in cs])), "sesgo": float(e.mean()),
                      "pred_media": float(p[m].mean())})
    with open(out / "error_por_bcs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0])); w.writeheader(); w.writerows([{k: round(v, 4) if isinstance(v, float) else v for k, v in f.items()} for f in filas])
    # por vaca x BCS (la unidad de etiqueta): media de predicciones vs etiqueta
    keys = sorted({(c, b) for c, b in zip(cows, y)}); gp = np.array([p[(cows == c) & (y == b)].mean() for c, b in keys]); gy = np.array([b for _, b in keys])
    d = gp - gy; mu, sd = d.mean(), d.std(ddof=1); loa = (mu - 1.96 * sd, mu + 1.96 * sd)
    pend = np.polyfit((gp + gy) / 2, d, 1)[0]                                   # tendencia del sesgo con el nivel
    r = np.corrcoef(gp, gy)[0, 1]; slope = np.polyfit(gy, gp, 1)[0]
    ba = {"unidad": "vaca x BCS", "n": len(keys), "sesgo": mu, "loa": loa, "pendiente_sesgo_vs_media": pend, "r": r,
          "pendiente_pred_vs_real": slope, "mae": float(np.abs(d).mean()), "dentro_025": float(np.mean(np.abs(d) <= 0.25 + 1e-9)),
          "dentro_05": float(np.mean(np.abs(d) <= 0.5 + 1e-9))}
    e = p - y; ba_f = {"unidad": "foto", "n": len(y), "sesgo": float(e.mean()), "loa": (float(e.mean() - 1.96 * e.std(ddof=1)), float(e.mean() + 1.96 * e.std(ddof=1)))}
    # figura: (a) predicho vs real por vaca x BCS, (b) Bland-Altman
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"wspace": 0.32})
    lo, hi = 1.8, 5.2
    ax[0].plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls="--", zorder=0)
    ax[0].fill_between([lo, hi], [lo - 0.25, hi - 0.25], [lo + 0.25, hi + 0.25], color="0.9", zorder=0, lw=0)
    ax[0].scatter(gy, gp, s=np.clip([((cows == c) & (y == b)).sum() for c, b in keys], 1, 48) * 3 + 6, facecolor="#0072B2", alpha=0.55, edgecolor="white", lw=0.4)
    ax[0].plot([lo, hi], np.polyval(np.polyfit(gy, gp, 1), [lo, hi]), color="#D55E00", lw=1.0)
    ax[0].set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Reference BCS", ylabel="Predicted BCS (cow mean)", aspect="equal")
    ax[0].text(0.04, 0.96, f"$r$ = {r:.2f}\nMAE = {ba['mae']:.2f}\nslope = {slope:.2f}", transform=ax[0].transAxes, va="top")
    ax[0].set_title("(a)", loc="left", fontweight="bold")
    m = (gp + gy) / 2
    ax[1].scatter(m, d, s=14, facecolor="#0072B2", alpha=0.55, edgecolor="white", lw=0.4)
    for v, ls in ((mu, "-"), (loa[0], "--"), (loa[1], "--")): ax[1].axhline(v, color="#D55E00" if ls == "-" else "0.45", lw=0.9, ls=ls)
    ax[1].axhline(0, color="0.75", lw=0.6)
    ax[1].text(5.17, mu, f"bias {mu:+.2f}", va="bottom", ha="right"); ax[1].text(5.17, loa[1], f"+1.96 SD {loa[1]:+.2f}", va="bottom", ha="right")
    ax[1].text(5.17, loa[0], f"$-$1.96 SD {loa[0]:+.2f}", va="top", ha="right")
    ax[1].set(xlim=(lo, hi), xlabel="Mean of predicted and reference BCS", ylabel="Predicted $-$ reference")
    ax[1].set_title("(b)", loc="left", fontweight="bold")
    fig.savefig(figs / "analisis_bland_altman.pdf", bbox_inches="tight"); fig.savefig(figs / "analisis_bland_altman.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    return filas, ba, ba_f

# ---------- 2. Cámaras ----------
def camaras(X, y, cows, names, out, rng):
    cam = np.array([camara(n) for n in names]); cams = sorted(set(cam)); filas = []
    P = {(a, b): loco_preds(X, y, cows, cam == a, cam == b) for a in cams for b in cams}
    pall = loco_preds(X, y, cows)
    for b in cams:                                                             # fotos de prueba de la cámara b
        te = cam == b
        for a in cams:
            p, q = P[(a, b)], P[(b, b)]                                         # entrenar con a vs con la misma cámara b
            ok = te & ~np.isnan(p) & ~np.isnan(q)
            uc = np.unique(cows[ok]); dc = np.array([np.abs(p - y)[ok & (cows == c)].mean() - np.abs(q - y)[ok & (cows == c)].mean() for c in uc])
            mv = lambda pp: float(np.mean([np.abs(pp - y)[ok & (cows == c)].mean() for c in uc]))
            bs = dc[rng.randint(0, len(uc), (N_BOOT, len(uc)))].mean(1)
            filas.append({"prueba": f"cam{b}", "entrena": f"cam{a}", "fotos": int(ok.sum()), "vacas": len(uc), "mae": float(np.abs(p - y)[ok].mean()),
                          "mae_vaca": mv(p), "r": float(np.corrcoef(p[ok], y[ok])[0, 1]), "mae_vaca_todo": mv(pall),
                          "dif_vs_misma": float(dc.mean()), "ic_lo": float(np.percentile(bs, 2.5)), "ic_hi": float(np.percentile(bs, 97.5))})
    with open(out / "camaras.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0])); w.writeheader(); w.writerows([{k: round(v, 4) if isinstance(v, float) else v for k, v in f.items()} for f in filas])
    return filas, {c: int((cam == c).sum()) for c in cams}, {c: len(np.unique(cows[cam == c])) for c in cams}

# ---------- 3. Tiempo ----------
def tiempo(p, y, cows, names, out, figs, rng):
    dia = np.array([fecha(n) for n in names])
    # estabilidad: vaca x BCS con >= 2 días; promedio de predicción por día
    filas = []; dif = []
    for c, b in sorted({(c, b) for c, b in zip(cows, y)}):
        m = (cows == c) & (y == b); ds = sorted(set(dia[m]))
        if len(ds) < 2: continue
        pd = np.array([p[m & (dia == d)].mean() for d in ds])
        filas.append({"vaca": c, "bcs": b, "dias": len(ds), "sd_entre_dias": float(pd.std(ddof=1)), "rango": float(pd.max() - pd.min()),
                      "error_medio": float(pd.mean() - b)})
        dif += list(np.abs(np.diff(pd)))
    sd_dias = np.array([f["sd_entre_dias"] for f in filas]); err = np.array([f["error_medio"] for f in filas])
    # ICC(1) de las predicciones por foto, agrupando por vaca x BCS (cuánto de la variación es entre animales)
    g = np.array([f"{c}|{b}" for c, b in zip(cows, y)]); ug = np.unique(g); k = np.array([(g == u).sum() for u in ug])
    mg = np.array([p[g == u].mean() for u in ug]); msb = (k * (mg - p.mean()) ** 2).sum() / (len(ug) - 1)
    msw = sum(((p[g == u] - p[g == u].mean()) ** 2).sum() for u in ug) / (len(p) - len(ug)); k0 = (len(p) - (k ** 2).sum() / len(p)) / (len(ug) - 1)
    icc = (msb - msw) / (msb + (k0 - 1) * msw)
    est = {"vaca_bcs_con_2_dias": len(filas), "sd_entre_dias_mediana": float(np.median(sd_dias)), "sd_entre_dias_media": float(sd_dias.mean()),
           "dif_dias_consecutivos_mediana": float(np.median(dif)), "sesgo_por_vaca_abs_medio": float(np.abs(err).mean()), "icc1_predicciones": float(icc),
           "sd_entre_vacas_pred": float(mg.std(ddof=1))}
    with open(out / "estabilidad.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0])); w.writeheader(); w.writerows([{kk: round(v, 4) if isinstance(v, float) else v for kk, v in f.items()} for f in filas])
    # cambios de BCS: vacas con más de una calificación
    cambios = []
    for c in np.unique(cows):
        bs = sorted(set(y[cows == c]))
        if len(bs) < 2: continue
        for b in bs:
            m = (cows == c) & (y == b); ds = sorted(set(dia[m]))
            cambios.append({"vaca": c, "bcs": b, "desde": ds[0], "hasta": ds[-1], "fotos": int(m.sum()), "pred_media": float(p[m].mean()),
                            "pred_sd": float(p[m].std(ddof=1)) if m.sum() > 1 else 0.0})
    with open(out / "cambios_bcs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cambios[0])); w.writeheader(); w.writerows([{kk: round(v, 4) if isinstance(v, float) else v for kk, v in f.items()} for f in cambios])
    # figura: trayectoria de las vacas con cambio
    vs = sorted({c["vaca"] for c in cambios}); fig, axs = plt.subplots(1, len(vs), figsize=(3.4 * len(vs), 2.5), squeeze=False)
    import datetime as dt
    for ax, v in zip(axs[0], vs):
        m = cows == v; ds = sorted(set(dia[m])); t = [dt.date.fromisoformat(d) for d in ds]
        ax.scatter([dt.date.fromisoformat(d) for d in dia[m]], p[m], s=8, color="#0072B2", alpha=0.45, lw=0, label="Prediction (photo)")
        ax.plot(t, [p[m & (dia == d)].mean() for d in ds], color="#0072B2", lw=0.9, label="Prediction (daily mean)")
        ax.step(t, [y[m & (dia == d)][0] for d in ds], where="post", color="#D55E00", lw=1.4, label="Reference BCS")
        ax.set_title(f"Cow {v}", loc="left"); ax.set_ylabel("BCS"); ax.set_ylim(1.8, 5.2); ax.tick_params(axis="x", rotation=30, labelsize=7)
    axs[0][0].legend(frameon=False, loc="lower left")
    fig.savefig(figs / "analisis_cambios_bcs.pdf", bbox_inches="tight"); fig.savefig(figs / "analisis_cambios_bcs.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    return est, cambios

def main(a):
    work = Path(a.work_dir); out = work / "analisis"; out.mkdir(parents=True, exist_ok=True); figs = RAIZ / "figures"; rng = np.random.RandomState(0)
    X, y, cows, names, firma, _, _ = modelo(work); print(f"[modelo] {firma}", flush=True)
    p = loco_preds(X, y, cows)
    canon = {r["foto"]: float(r["bcs_pred"]) for r in csv.DictReader(open(work / "preds_loco.csv"))}
    assert np.abs(p - np.array([canon[n] for n in names])).max() < 6e-5, "LOCO no coincide con preds_loco.csv del modelo"
    eb, ba, ba_f = error_bcs(p, y, cows, out, figs, rng)
    cam, nf, nv = camaras(X, y, cows, names, out, rng)
    est, cambios = tiempo(p, y, cows, names, out, figs, rng)
    L = [f"# Análisis para el artículo (firma {json.dumps(firma)})", "", "## 1. Error por BCS (LOCO)", "",
         "| BCS | vacas | fotos | MAE | MAE vaca | sesgo | pred. media |", "|---|---|---|---|---|---|---|"]
    L += [f"| {f['bcs']:.2f} | {f['vacas']} | {f['fotos']} | {f['mae']:.3f} | {f['mae_vaca']:.3f} | {f['sesgo']:+.3f} | {f['pred_media']:.2f} |" for f in eb]
    L += ["", f"Bland-Altman por vaca x BCS (n={ba['n']}): sesgo {ba['sesgo']:+.3f}, LoA [{ba['loa'][0]:+.3f}, {ba['loa'][1]:+.3f}], "
          f"pendiente del sesgo con el nivel {ba['pendiente_sesgo_vs_media']:+.3f}; r {ba['r']:.3f}, pendiente pred~real {ba['pendiente_pred_vs_real']:.3f}, "
          f"MAE {ba['mae']:.3f}, |dif| <= 0.25: {ba['dentro_025']:.0%}, <= 0.5: {ba['dentro_05']:.0%}.",
          f"Por foto (n={ba_f['n']}): sesgo {ba_f['sesgo']:+.3f}, LoA [{ba_f['loa'][0]:+.3f}, {ba_f['loa'][1]:+.3f}].", "",
          f"## 2. Cámaras (fotos {nf}, vacas {nv}; LOCO por vaca)", "",
          "| prueba | entrena | fotos | vacas | MAE | MAE vaca | r | MAE vaca (entrena todo) | dif vs misma cámara [IC95] |", "|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {f['prueba']} | {f['entrena']} | {f['fotos']} | {f['vacas']} | {f['mae']:.3f} | {f['mae_vaca']:.3f} | {f['r']:.3f} | {f['mae_vaca_todo']:.3f} | "
          f"{f['dif_vs_misma']:+.3f} [{f['ic_lo']:+.3f}, {f['ic_hi']:+.3f}] |" for f in cam]
    L += ["", "## 3. Tiempo", "", "Estabilidad (vaca x BCS con fotos en >= 2 días, predicción media por día): " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in est.items()), "",
          "| vaca | BCS | desde | hasta | fotos | pred. media | pred. sd |", "|---|---|---|---|---|---|---|"]
    L += [f"| {c['vaca']} | {c['bcs']:.2f} | {c['desde']} | {c['hasta']} | {c['fotos']} | {c['pred_media']:.2f} | {c['pred_sd']:.2f} |" for c in cambios]
    (out / "resumen.md").write_text("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="/work")
    main(ap.parse_args())
