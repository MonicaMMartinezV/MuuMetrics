# =============================================================
# Nombre del archivo: mejoras/seleccion_anidada.py
#
# Descripción:
#   Selección anidada por vaca de las mejoras candidatas del modelo
#   (EXPERIMENTOS [26] y [27]), para decidir sin sesgo de selección si
#   alguna se adopta:
#     regiones   6 (base) | 5 sin lomo                      ([27]: -0.014 por vaca)
#     resolución 384 px (base) | 320 px                     ([26]: mejora en 4 métricas, IC al borde)
#   -> 4 configuraciones. El resto del modelo no cambia (bloque 21,
#   promedio por región + indicadores de ausencia, ridge de br.fit_predict).
#
#   Diseño (pre-registrado en <out>/prereg.json ANTES de evaluar):
#   - Externo: los esquemas de 'evaluate' (LOCO y 5 folds x 10 semillas).
#   - Interno: dentro del train externo, StratifiedGroupKFold(5) por vaca;
#     se elige la configuración con menor MAE POR VACA en la validación
#     interna. Empate o diferencia < DELTA -> se queda la base (sesgo a
#     favor de no cambiar el modelo).
#   - La configuración elegida se reentrena con todo el train externo y
#     predice las vacas de test, que no participan en nada de lo anterior.
#   - Además: cada configuración FIJA evaluada por separado (no anidado),
#     solo como referencia.
#   - Comparación: MAE por vaca de lo anidado vs base, bootstrap pareado
#     por vaca (5000).
#
#   Descriptores: 384 px = features.npz del modelo (comprobando su firma);
#   320 px = mismas piezas de bcs_regions.py con SIZE=320, en caché con su
#   propia firma (<out>/feat_320.npz). Quitar una región = quitar sus D
#   columnas y su indicador de ausencia (el descriptor es por región).
#
# Uso: docker compose run --rm --entrypoint python bcs mejoras/seleccion_anidada.py
# =============================================================
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch, timm
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "ablaciones"))
import bcs_regions as br                                   # noqa: E402
from data import load_records                              # noqa: E402
from comun import modelo, splits_for, cache_ok, guardar_cache   # noqa: E402

CONFIGS = {"base": (br.SIZE, None), "sin_lomo": (br.SIZE, "loin"), "320": (320, None), "320_sin_lomo": (320, "loin")}
BASE = "base"
DELTA = 0.0           # la base solo se reemplaza si otra configuración es estrictamente mejor en la validación interna
N_BOOT = 5000

def descriptores_a(work: Path, out: Path, size: int, f_base: dict, recs, dev):
    """Descriptores del modelo a otra resolución, con las mismas piezas de br (patch_tokens, region_weights, pool_regions)."""
    f = {**f_base, "size": size}; path = out / f"feat_{size}.npz"
    if cache_ok(path, f): return np.load(path)["X"].astype(np.float64)
    old = br.SIZE; br.SIZE = size
    try:
        bb = br.load_backbone(dev); X = []; t0 = time.time()
        for i, r in enumerate(recs):
            m = np.load(br.mask_file(work, r))["masks"].astype(np.float32) / 255.0
            X.append(br.describe(bb, br.read_rgb(r["path"]), m, dev))
            if (i + 1) % 100 == 0: print(f"  [{size} px] {i + 1}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
        del bb; torch.cuda.empty_cache()
    finally:
        br.SIZE = old
    X = np.stack(X).astype(np.float32); guardar_cache(path, f, X=X); return X.astype(np.float64)

def columnas(regions, D, quitar):
    """Índices de columnas sin la región `quitar` (sus D promedios y su indicador)."""
    R = len(regions); keep = [i for i, r in enumerate(regions) if r != quitar]
    return np.concatenate([np.arange(i * D, (i + 1) * D) for i in keep] + [np.array([R * D + i for i in keep])])

def mae_vaca(p, y, cows):
    return float(np.mean([np.abs(p - y)[cows == c].mean() for c in np.unique(cows)]))

def interna(tr, y, cows, seed):
    strata = np.round(y[tr] * 2).astype(int)
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(tr, strata, cows[tr]))

def elegir(Xs, tr, y, cows, seed):
    """Validación interna por vaca -> (configuración elegida, MAE por vaca interno de cada una)."""
    folds = interna(tr, y, cows, seed); err = {}
    for name, X in Xs.items():
        p = np.zeros(len(tr))
        for a, v in folds: p[v] = br.fit_predict(X[tr[a]], y[tr[a]], cows[tr[a]], X[tr[v]])[0]
        err[name] = mae_vaca(p, y[tr], cows[tr])
    mejor = min(err, key=err.get)
    return (mejor if err[mejor] < err[BASE] - DELTA else BASE), err

def pareado(p, p0, y, cows, rng):
    uc = np.unique(cows); d = np.array([np.abs(p - y)[cows == c].mean() - np.abs(p0 - y)[cows == c].mean() for c in uc])
    bs = d[rng.randint(0, len(uc), (N_BOOT, len(uc)))].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"d_mae_vaca": float(d.mean()), "ic95": [float(lo), float(hi)], "vacas_mejor": int((d < -1e-9).sum()), "vacas_peor": int((d > 1e-9).sum())}

def main(a):
    work, out = Path(a.work_dir), Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); dev = br.device()
    X384, y, cows, names, f_base, regions, D = modelo(work)
    recs = {r["name"]: r for r in load_records(a.data_dir, verbose=False)}; recs = [recs[n] for n in names]
    X320 = descriptores_a(work, out, 320, f_base, recs, dev)
    full = {br.SIZE: X384, 320: X320}
    Xs = {name: (full[s] if q is None else full[s][:, columnas(regions, D, q)]) for name, (s, q) in CONFIGS.items()}
    for n, X in Xs.items(): print(f"[config] {n:14s} {X.shape}", flush=True)

    pr = {"fecha": time.strftime("%Y-%m-%d %H:%M:%S"), "configs": {k: {"size": s, "quitar": q} for k, (s, q) in CONFIGS.items()},
          "base": BASE, "criterio": "menor MAE por vaca en validación interna StratifiedGroupKFold(5) por vaca dentro del train externo",
          "regla": f"se cambia la base solo si otra es mejor por más de {DELTA}", "externo": ["loco", f"rep5x{a.seeds}"],
          "primaria": "MAE por vaca, anidado vs base, bootstrap pareado por vaca (5000)",
          "firma_modelo": f_base, "nota": "candidatas elegidas tras ver [26] y [27] en estas 64 vacas: lo anidado corrige la selección DENTRO de estas 4, no la elección de las 4."}
    (out / "prereg.json").write_text(json.dumps(pr, indent=2, ensure_ascii=False))

    sch = splits_for(y, cows, a.seeds); res = {}; t0 = time.time()
    for esquema, reps in sch.items():
        P = {k: [] for k in list(Xs) + ["anidado"]}; elecciones = []
        for r, splits in enumerate(reps):
            p = {k: np.zeros(len(y)) for k in P}
            for k, (tr, te) in enumerate(splits):
                for name, X in Xs.items(): p[name][te] = br.fit_predict(X[tr], y[tr], cows[tr], X[te])[0]
                sel, err = elegir(Xs, tr, y, cows, seed=1000 * r + k)
                p["anidado"][te] = p[sel][te]; elecciones.append({"rep": r, "fold": k, "elegida": sel, **{f"int_{n}": round(e, 4) for n, e in err.items()}})
            for n in P: P[n].append(p[n])
            print(f"  [{esquema}] repetición {r + 1}/{len(reps)} ({time.time() - t0:.0f}s)", flush=True)
        P = {n: np.mean(v, 0) for n, v in P.items()}; rng = np.random.RandomState(0)
        tabla = {}
        for n, p in P.items():
            m = {k: float(v) for k, v in br.metrics(p, y, cows).items()}
            tabla[n] = {**m, "r_pearson": float(np.corrcoef(p, y)[0, 1]), **({} if n == BASE else {"vs_base": pareado(p, P[BASE], y, cows, rng)})}
        cuenta = {c: sum(e["elegida"] == c for e in elecciones) for c in CONFIGS}
        res[esquema] = {"metricas": tabla, "elecciones": cuenta, "detalle": elecciones}
        print(f"\n=== {esquema}: elecciones {cuenta}")
        for n, m in tabla.items():
            vs = m.get("vs_base"); txt = "" if vs is None else f"  dif vaca {vs['d_mae_vaca']:+.4f} [{vs['ic95'][0]:+.4f}, {vs['ic95'][1]:+.4f}] ({vs['vacas_mejor']}/{vs['vacas_peor']})"
            print(f"  {n:14s} MAE {m['mae']:.4f}  MAE vaca {m['mae_vaca']:.4f}  R2 {m['r2_fotos']:.3f}  r {m['r_pearson']:.3f}{txt}")
        for n, p in P.items(): np.save(out / f"preds_{esquema}_{n}.npy", p)
    (out / "resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"-> {out} ({time.time() - t0:.0f}s)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS_v2")
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--out_dir", default="/work/mejoras/seleccion_anidada")
    main(ap.parse_args())
