# =============================================================
# Nombre del archivo: nested.py
#
# Descripción:
#   Etapa C (exploratoria): validación cruzada anidada por vaca para un
#   PROCEDIMIENTO que elige entre A0 (sin augmentation) y A1 (flip, p=0.5,
#   K=3, semilla 0) usando solo vacas de entrenamiento.
#
#   Externo : LOCO, los mismos N folds que cmd_evaluate (GroupKFold(N), N vacas).
#   Interno : LOCO sobre las N-1 vacas de train externo (N-1 folds).
#             En cada fold interno: alpha = select_alpha() canónico sobre las
#             filas originales del train interno; A0 = fit_predict() canónico;
#             A1 = fit_weighted() (peso 1/(K+1) por fila) con ese mismo alpha.
#   Criterio: MAE promedio por vaca en las N-1 vacas internas.
#   Regla   : A1 si MAE_vaca(A1) < MAE_vaca(A0) - delta; si no, A0.
#             delta primario = 0; sensibilidad pre-registrada delta = 0.01.
#   Final   : política elegida reentrenada con las N-1 vacas (alpha canónico
#             de ese train) -> una predicción por foto de la vaca excluida.
#
#   No modifica nada del modelo canónico ni de work/aug/{features,results,report*}.
#   Escribe solo en <out>/nested_A0A1/.
#
# Uso (dentro del contenedor):
#   python nested.py check    # verificación de fugas, sin ajustar modelos
#   python nested.py run      # evaluación + reporte
#   python nested.py report   # regenerar el reporte desde los resultados
# =============================================================
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold

import augment as AG
from bcs_regions import select_alpha, fit_predict, metrics
from ablation import load_canon, fit_weighted, feat_file, paired, sha256, jdump, N_BOOT

POLICY_A1 = "A1"
DELTAS = {"primario": 0.0, "sensibilidad": 0.01}      # fijados antes de correr (ver prereg.json)

def mae_vaca(p, y, cows):
    return float(np.mean([np.abs(p - y)[cows == c].mean() for c in np.unique(cows)]))

def outer_splits(X, y, cows):
    return list(GroupKFold(len(np.unique(cows))).split(X, y, cows))      # == esquema "loco" de cmd_evaluate

def inner_splits(Xo, yo, co):
    return list(GroupKFold(len(np.unique(co))).split(Xo, yo, co))         # LOCO interno sobre las vacas de train externo

def fit_a1(X, Xa, y, idx_tr, idx_te, alpha):
    K = Xa.shape[1]
    Xtr = np.concatenate([X[idx_tr], Xa[idx_tr].reshape(-1, X.shape[1])]); ytr = np.concatenate([y[idx_tr], np.repeat(y[idx_tr], K)])
    return fit_weighted(Xtr, ytr, np.full(len(ytr), 1 / (K + 1)), X[idx_te], alpha)

def check(a, write=True):
    """Fugas: vacas de validación interna y de test externo nunca entran en ningún train ni en la selección de alpha."""
    X, y, cows, names, _ = load_canon(a); Xa = np.load(feat_file(a, POLICY_A1))["Xaug"]
    reg = json.load(open(Path(a.out_dir) / "splits.json")); rep = {"folds_externos": 0, "folds_internos": 0, "problemas": []}
    canon_loco = [set(f["test_vacas"]) for f in reg["schemes"]["loco"][0]]
    for o, (tr, te) in enumerate(outer_splits(X, y, cows)):
        rep["folds_externos"] += 1; ct, cte = set(cows[tr]), set(cows[te])
        if ct & cte: rep["problemas"].append(f"externo {o}: vaca en train y test")
        if cte != canon_loco[o]: rep["problemas"].append(f"externo {o}: distinto al LOCO canónico")
        if len(cte) != 1: rep["problemas"].append(f"externo {o}: {len(cte)} vacas de test")
        for itr, iva in inner_splits(X[tr], y[tr], cows[tr]):
            rep["folds_internos"] += 1; a_tr, a_va = tr[itr], tr[iva]
            if set(cows[a_tr]) & set(cows[a_va]): rep["problemas"].append(f"externo {o}: vaca interna en train y validación")
            if (set(cows[a_tr]) | set(cows[a_va])) & cte: rep["problemas"].append(f"externo {o}: vaca de test externo dentro del interno")
            if set(cows[a_tr]) | set(cows[a_va]) != ct: rep["problemas"].append(f"externo {o}: interno no cubre el train externo")
    # Las variantes A1 de una foto solo se usan si su foto está en el train del ajuste (fit_a1 indexa Xa con idx_tr).
    rep["variantes_A1"] = {"forma": list(Xa.shape), "nombres_alineados": bool((np.load(feat_file(a, POLICY_A1))["name"] == names).all())}
    rep["ok"] = not rep["problemas"] and rep["variantes_A1"]["nombres_alineados"]
    if write: jdump(rep, out_dir(a) / "leakage_check.json")
    print("[check]", json.dumps(rep, indent=1, ensure_ascii=False)); return rep

def out_dir(a): return Path(a.out_dir) / "nested_A0A1"

def prereg(a):
    f = out_dir(a) / "prereg.json"
    design = {"externo": "LOCO, GroupKFold(N) por vaca (== cmd_evaluate)", "interno": "LOCO sobre las N-1 vacas de train externo",
              "candidatos": ["A0", "A1 (flip p=0.5, K=3, semilla 0, peso 1/(K+1))"],
              "alpha_interno": "select_alpha canónico con filas originales del train interno (N-2 vacas)",
              "alpha_final": "select_alpha canónico con filas originales del train externo (N-1 vacas)",
              "criterio": "MAE promedio por vaca en las N-1 vacas de validación interna",
              "regla": "A1 si MAE_vaca(A1) < MAE_vaca(A0) - delta; empate o menor mejora -> A0", "deltas": DELTAS,
              "reporte_principal": "delta primario; delta=0.01 es análisis de sensibilidad",
              "comparacion": "A0 canónico (work/aug/results/A0, idéntico a work/preds_loco.csv)", "bootstrap": {"unidad": "vaca", "n": N_BOOT, "semilla": 0},
              "nota": "A1 se eligió como candidato después de ver las etapas A y B en las mismas N vacas: evaluación exploratoria, no confirmatoria."}
    if f.exists():
        old = json.load(open(f))
        if old["diseno"] != design: raise SystemExit("El diseño difiere del pre-registrado en prereg.json: no se sobrescribe.")
        return old
    p = {"fecha": time.strftime("%Y-%m-%d %H:%M:%S"), "diseno": design,
         "codigo_sha256": {k: sha256(Path(__file__).parent / k) for k in ("nested.py", "ablation.py", "augment.py", "bcs_regions.py")},
         "features_sha256": sha256(Path(a.work_dir) / "features.npz"), "a1_features_sha256": sha256(feat_file(a, POLICY_A1))}
    jdump(p, f); return p

def cmd_run(a):
    od = out_dir(a); od.mkdir(parents=True, exist_ok=True)
    if (od / "folds.json").exists() and not a.overwrite: raise SystemExit(f"Ya existen resultados en {od} (usa --overwrite).")
    pr = prereg(a); print(f"[prereg] {od / 'prereg.json'} ({pr['fecha']})")
    if not check(a)["ok"]: raise SystemExit("Verificación de fugas NO aprobada.")
    X, y, cows, names, _ = load_canon(a); Xa = np.load(feat_file(a, POLICY_A1))["Xaug"].astype(np.float64)
    reg = json.load(open(Path(a.out_dir) / "splits.json")); folds = []; t0 = time.time()
    for o, (tr, te) in enumerate(outer_splits(X, y, cows)):
        pin = {"A0": np.zeros(len(tr)), "A1": np.zeros(len(tr))}; alphas = []
        for itr, iva in inner_splits(X[tr], y[tr], cows[tr]):
            a_tr, a_va = tr[itr], tr[iva]
            al = select_alpha(X[a_tr], y[a_tr], cows[a_tr]); alphas.append(al)
            p0, al0 = fit_predict(X[a_tr], y[a_tr], cows[a_tr], X[a_va]); assert al0 == al
            pin["A0"][iva] = p0; pin["A1"][iva] = fit_a1(X, Xa, y, a_tr, a_va, al)
        m = {k: mae_vaca(v, y[tr], cows[tr]) for k, v in pin.items()}
        sel = {name: ("A1" if m["A1"] < m["A0"] - d else "A0") for name, d in DELTAS.items()}
        al = select_alpha(X[tr], y[tr], cows[tr]); assert al == reg["schemes"]["loco"][0][o]["alpha"]
        p_a0, _ = fit_predict(X[tr], y[tr], cows[tr], X[te]); p_a1 = fit_a1(X, Xa, y, tr, te, al)
        folds.append({"fold": o, "vaca": str(cows[te][0]), "test_idx": te.tolist(), "alpha_final": al,
                      "alphas_internos": sorted(set(map(float, alphas))), "mae_vaca_interno_A0": m["A0"], "mae_vaca_interno_A1": m["A1"],
                      "seleccion": sel, "pred_A0": p_a0.tolist(), "pred_A1": p_a1.tolist()})
        print(f"  [{o + 1:2d}/{len(np.unique(cows))}] vaca {cows[te][0]}: interno A0={m['A0']:.4f} A1={m['A1']:.4f} -> {sel['primario']} "
              f"(δ=0.01: {sel['sensibilidad']}) ({time.time() - t0:.0f}s)", flush=True)
    jdump(folds, od / "folds.json"); cmd_report(a)

def cmd_report(a):
    od = out_dir(a); folds = json.load(open(od / "folds.json")); X, y, cows, names, _ = load_canon(a); R = Path(a.out_dir) / "results"
    p_a0 = np.load(R / "A0" / "preds_loco_reps.npy")[0]; p_a1 = np.load(R / "A1" / "preds_loco_reps.npy")[0]
    chk_a0 = chk_a1 = 0.0; P = {k: np.zeros(len(y)) for k in DELTAS}
    for f in folds:
        te = np.array(f["test_idx"]); chk_a0 = max(chk_a0, np.abs(np.array(f["pred_A0"]) - p_a0[te]).max())
        chk_a1 = max(chk_a1, np.abs(np.array(f["pred_A1"]) - p_a1[te]).max())
        for k in DELTAS: P[k][te] = f["pred_" + f["seleccion"][k]]
    rows = []
    for label, p in [("A0 (canónico)", p_a0), ("A1 siempre", p_a1), ("Anidado δ=0 (primario)", P["primario"]), ("Anidado δ=0.01 (sensibilidad)", P["sensibilidad"])]:
        m = metrics(p, y, cows); d = paired(p, p_a0, y, cows, np.random.RandomState(0))
        rows.append({"procedimiento": label, "mae": m["mae"], "mae_lo": d["mae_ic95"][0], "mae_hi": d["mae_ic95"][1], "mae_vaca": m["mae_vaca"],
                     "mae_vaca_lo": m["mae_vaca_ic95_lo"], "mae_vaca_hi": m["mae_vaca_ic95_hi"], "rmse": m["rmse"], "r2_fotos": m["r2_fotos"],
                     "r2_vacas": m["r2_vacas"], "bias": float((p - y).mean()), "acc_05": m["acc_05"],
                     "d_mae": d["d_mae"], "d_mae_lo": d["d_mae_ic95"][0], "d_mae_hi": d["d_mae_ic95"][1],
                     "d_mae_vaca": d["d_mae_vaca"], "d_mae_vaca_lo": d["d_mae_vaca_ic95"][0], "d_mae_vaca_hi": d["d_mae_vaca_ic95"][1],
                     "vacas_mejor": d["vacas_mejor"], "vacas_peor": d["vacas_peor"]})
    cnt = {k: {"A0": sum(f["seleccion"][k] == "A0" for f in folds), "A1": sum(f["seleccion"][k] == "A1" for f in folds)} for k in DELTAS}
    with open(od / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
        w.writerows([{k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for r in rows])
    with open(od / "preds_nested_loco.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real", "pred_A0", "pred_A1", "pred_anidado_d0", "pred_anidado_d001"])
        w.writerows([[n, c, yy, round(float(a0), 4), round(float(a1), 4), round(float(q0), 4), round(float(q1), 4)]
                     for n, c, yy, a0, a1, q0, q1 in zip(names, cows, y, p_a0, p_a1, P["primario"], P["sensibilidad"])])
    per = []
    for f in sorted(folds, key=lambda f: f["vaca"]):
        te = np.array(f["test_idx"]); yt = y[te]; pa0 = np.array(f["pred_A0"]); pn = P["primario"][te]     # BCS por foto (8758 cambia por mes)
        per.append({"vaca": f["vaca"], "fotos": len(te), "bcs_real": float(yt.mean()), "bcs_rango": f"{yt.min():g}-{yt.max():g}" if yt.min() != yt.max() else f"{yt[0]:g}", "mae_int_A0": f["mae_vaca_interno_A0"], "mae_int_A1": f["mae_vaca_interno_A1"],
                    "dif_int": f["mae_vaca_interno_A1"] - f["mae_vaca_interno_A0"], "sel_d0": f["seleccion"]["primario"], "sel_d001": f["seleccion"]["sensibilidad"],
                    "pred_media_anidado": float(pn.mean()), "mae_anidado": float(np.abs(pn - yt).mean()), "mae_A0": float(np.abs(pa0 - yt).mean()),
                    "dif_vs_A0": float(np.abs(pn - yt).mean() - np.abs(pa0 - yt).mean()), "alpha_final": f["alpha_final"]})
    with open(od / "per_cow.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0])); w.writeheader()
        w.writerows([{k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for r in per])
    ci = lambda r, k: f"{r[k]:+.4f} [{r[k + '_lo']:+.4f}, {r[k + '_hi']:+.4f}]"
    md = [f"# Etapa C: selección anidada A0/A1 (LOCO externo, LOCO interno por vaca)", "",
          f"Selección δ=0 (primario): A0 en {cnt['primario']['A0']} folds, A1 en {cnt['primario']['A1']} folds de {len(folds)}.  ",
          f"Selección δ=0.01 (sensibilidad): A0 en {cnt['sensibilidad']['A0']}, A1 en {cnt['sensibilidad']['A1']}.  ",
          f"Control: predicciones finales A0 vs results/A0 máx |Δ| = {chk_a0:.2e}; A1 vs results/A1 máx |Δ| = {chk_a1:.2e}.", "",
          "| Procedimiento | MAE fotos [IC 95 %] | MAE vaca [IC 95 %] | RMSE | R² fotos | R² vacas | Bias | ≤0.5 | ΔMAE fotos vs A0 [IC 95 %] | ΔMAE vaca vs A0 [IC 95 %] | Vacas mejor/peor |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        base = r["procedimiento"] == "A0 (canónico)"
        md.append(f"| {r['procedimiento']} | {r['mae']:.4f} [{r['mae_lo']:.4f}, {r['mae_hi']:.4f}] | {r['mae_vaca']:.4f} [{r['mae_vaca_lo']:.4f}, {r['mae_vaca_hi']:.4f}] | "
                  f"{r['rmse']:.4f} | {r['r2_fotos']:.3f} | {r['r2_vacas']:.3f} | {r['bias']:+.4f} | {r['acc_05']:.1%} | "
                  f"{'—' if base else ci(r, 'd_mae')} | {'—' if base else ci(r, 'd_mae_vaca')} | {'—' if base else str(r['vacas_mejor']) + '/' + str(r['vacas_peor'])} |")
    md += ["", "| Vaca | Fotos | BCS | MAE interno A0 | MAE interno A1 | A1−A0 interno | Selección δ=0 | Selección δ=0.01 | Pred. media | MAE anidado | MAE A0 | Δ vs A0 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['vaca']} | {r['fotos']} | {r['bcs_real']:.2f} | {r['mae_int_A0']:.4f} | {r['mae_int_A1']:.4f} | {r['dif_int']:+.4f} | {r['sel_d0']} | {r['sel_d001']} | "
           f"{r['pred_media_anidado']:.3f} | {r['mae_anidado']:.4f} | {r['mae_A0']:.4f} | {r['dif_vs_A0']:+.4f} |" for r in per]
    (od / "report.md").write_text("\n".join(md) + "\n"); print("\n".join(md[:12]))
    jdump({"conteo_seleccion": cnt, "control_pred_A0_max_abs": chk_a0, "control_pred_A1_max_abs": chk_a1}, od / "summary.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Etapa C: selección anidada por vaca entre A0 y A1")
    ap.add_argument("step", choices=["check", "run", "report"])
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS_v2")
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--out_dir", default="/work/aug")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.step == "check": out_dir(a).mkdir(parents=True, exist_ok=True); check(a)
    elif a.step == "run": cmd_run(a)
    else: cmd_report(a)
