# =============================================================
# Nombre del archivo: paper_figure.py
#
# Descripción:
#   Tablas y figura de publicación del estudio de data augmentation
#   (etapas A, B y C). Solo LEE resultados ya calculados:
#     <out>/report/comparison.csv          etapa A
#     <out>/report_stageB/comparison.csv   etapa B (+ semillas de A1)
#     <out>/nested_A0A1/comparison.csv     etapa C
#   y escribe en <out>/final/ (carpeta nueva):
#     table_full.md        tabla completa A + B (ambos esquemas)
#     table_compact.{csv,md,tex}
#     figure_ablation.{pdf,png}
#
# Uso: docker compose run --rm --entrypoint python bcs paper_figure.py
# =============================================================
import argparse, csv, os
from pathlib import Path
import numpy as np

POLS = [("A1", "Horizontal flip", "Geometric"), ("A3", "Rotation ±5°", "Geometric"), ("A4", "Translation ±3 %, scale ±3 %", "Geometric"),
        ("B-BR05", "Brightness ±5 %", "Photometric"), ("B-BR10", "Brightness ±10 %", "Photometric"),
        ("B-CT05", "Contrast ±5 %", "Photometric"), ("B-CT10", "Contrast ±10 %", "Photometric"),
        ("A2", "Brightness + contrast ±10 %", "Photometric"),
        ("A5", "A1 + A2 + A3 + A4", "Combined"), ("A6", "A5 + Gaussian noise", "Combined")]

def read(path):
    with open(path) as fh: return list(csv.DictReader(fh))

def load(out: Path):
    rows = {}
    for f in (out / "report" / "comparison.csv", out / "report_stageB" / "comparison.csv"):
        for r in read(f): rows[(r["esquema"], r["politica"])] = {k: (float(v) if k not in ("esquema", "politica") else v) for k, v in r.items()}
    nested = {r["procedimiento"]: {k: (float(v) if k != "procedimiento" else v) for k, v in r.items()} for r in read(out / "nested_A0A1" / "comparison.csv")}
    return rows, nested

def verdict(r_loco, r_rep):
    lo = [r["d_mae_lo"] for r in (r_loco, r_rep)] + [r["d_mae_vaca_lo"] for r in (r_loco, r_rep)]
    hi = [r["d_mae_hi"] for r in (r_loco, r_rep)] + [r["d_mae_vaca_hi"] for r in (r_loco, r_rep)]
    # † = el IC 95 % excluye 0 en al menos 1 de 4 análisis (foto/vaca x LOCO/5x10), sin ajuste por multiplicidad
    if any(h < 0 for h in hi): return "Lower MAE†"
    if any(l > 0 for l in lo): return "Higher MAE†"
    same = np.sign(r_loco["d_mae"]) == np.sign(r_rep["d_mae"])
    return ("Lower MAE, n.s." if r_loco["d_mae"] < 0 else "Higher MAE, n.s.") if same else "No consistent change"

def tables(rows, nested, fd: Path):
    ci = lambda r, k: f"{r[k]:+.3f} [{r[k + '_lo']:+.3f}, {r[k + '_hi']:+.3f}]"
    # tabla completa
    full = ["| Scheme | Policy | MAE (photo) [95 % CI] | MAE (cow) [95 % CI] | RMSE | R² photo | R² cow | Bias | Within ±0.5 | ΔMAE photo vs A0 [95 % CI] | ΔMAE cow vs A0 [95 % CI] | Cows better/worse |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = ["A0", "A0-DUP", "A1", "A2", "A3", "A4", "A5", "A6", "B-BR05", "B-BR10", "B-CT05", "B-CT10"]
    for sch in ("loco", "rep5x10"):
        for p in order:
            r = rows[(sch, p)]; base = p == "A0"
            full.append(f"| {'LOCO' if sch == 'loco' else '5-fold × 10'} | {p} | {r['mae']:.3f} [{r['mae_lo']:.3f}, {r['mae_hi']:.3f}] | "
                        f"{r['mae_vaca']:.3f} [{r['mae_vaca_lo']:.3f}, {r['mae_vaca_hi']:.3f}] | {r['rmse']:.3f} | {r['r2_fotos']:.2f} | {r['r2_vacas']:.2f} | "
                        f"{r['bias']:+.3f} | {r['acc_05']:.0%} | {'—' if base else ci(r, 'd_mae')} | {'—' if base else ci(r, 'd_mae_vaca')} | "
                        f"{'—' if base else str(int(r['vacas_mejor'])) + '/' + str(int(r['vacas_peor']))} |")
    (fd / "table_full.md").write_text("\n".join(full) + "\n")
    # tabla compacta
    comp = []
    a0l, a0r = rows[("loco", "A0")], rows[("rep5x10", "A0")]
    comp.append({"policy": "A0", "description": "No augmentation (baseline)", "loco_mae": a0l["mae"], "loco_dmae": "", "rep_mae": a0r["mae"], "rep_dmae": "", "assessment": "Reference"})
    for p, desc, _ in POLS:
        l, r = rows[("loco", p)], rows[("rep5x10", p)]
        comp.append({"policy": p, "description": desc, "loco_mae": l["mae"], "loco_dmae": ci(l, "d_mae"), "rep_mae": r["mae"], "rep_dmae": ci(r, "d_mae"), "assessment": verdict(l, r)})
    n0 = nested["Anidado δ=0 (primario)"]
    comp.append({"policy": "Nested", "description": "A0/A1 chosen on training cows (δ = 0; A1 in 37/37 folds)", "loco_mae": n0["mae"],
                 "loco_dmae": ci(n0, "d_mae"), "rep_mae": "", "rep_dmae": "", "assessment": "Lower MAE, n.s." if n0["d_mae_hi"] >= 0 else "Lower MAE†"})
    with open(fd / "table_compact.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(comp[0])); w.writeheader()
        w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for c in comp])
    f3 = lambda v: f"{v:.3f}" if isinstance(v, float) else "—"
    md = ["| Policy | Description | LOCO MAE | LOCO ΔMAE vs A0 [95 % CI] | 5-fold×10 MAE | 5-fold×10 ΔMAE vs A0 [95 % CI] | Assessment |", "|---|---|---|---|---|---|---|"]
    md += [f"| {c['policy']} | {c['description']} | {f3(c['loco_mae'])} | {c['loco_dmae'] or '—'} | {f3(c['rep_mae'])} | {c['rep_dmae'] or '—'} | {c['assessment']} |" for c in comp]
    md += ["", "n.s.: the 95 % CI of ΔMAE includes 0 in all four analyses (photo- and cow-level MAE, LOCO and 5-fold × 10). "
           "†: the 95 % CI excludes 0 in at least one of the four analyses; not adjusted for multiple comparisons."]
    (fd / "table_compact.md").write_text("\n".join(md) + "\n")
    tex = ["\\begin{tabular}{llrlrll}", "\\hline",
           "Policy & Description & \\multicolumn{2}{c}{LOCO} & \\multicolumn{2}{c}{5-fold $\\times$ 10} & Assessment \\\\",
           " & & MAE & $\\Delta$MAE [95\\% CI] & MAE & $\\Delta$MAE [95\\% CI] & \\\\", "\\hline"]
    esc = lambda s: s.replace("%", "\\%").replace("±", "$\\pm$").replace("°", "$^\\circ$").replace("δ", "$\\delta$").replace("≥", "$\\geq$").replace("†", "$^\\dagger$")
    tex += [f"{c['policy']} & {esc(c['description'])} & {f3(c['loco_mae'])} & {c['loco_dmae'] or '--'} & {f3(c['rep_mae'])} & {c['rep_dmae'] or '--'} & {esc(c['assessment'])} \\\\" for c in comp]
    tex += ["\\hline", "\\end{tabular}", "% n.s.: 95\\% CI includes 0 in all four analyses; \\dag: CI excludes 0 in >= 1 analysis, not adjusted for multiplicity."]
    (fd / "table_compact.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))

def figure(rows, nested, fd: Path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
    ink, ink2, grid, surf, blue, gray = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#2a78d6", "#8a8984"
    labels = [desc for _, desc, _ in POLS] + ["Nested selection A0/A1 (δ = 0)"]
    groups = [g for _, _, g in POLS] + ["Selection"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True, facecolor=surf)
    for ax, sch, panel in zip(axs, ("loco", "rep5x10"), ("A  Leave-one-cow-out", "B  Cow-grouped 5-fold × 10")):
        ax.set_facecolor(surf); y = np.arange(len(labels))
        for i, (p, _, _) in enumerate(POLS):
            r = rows[(sch, p)]; ax.hlines(i, r["d_mae_lo"], r["d_mae_hi"], color=blue, lw=1.8, zorder=2)
            ax.plot(r["d_mae"], i, "o", color=blue, ms=5.5, mec=surf, mew=1, zorder=3)
            if p == "A1":                                                   # estimaciones puntuales de las otras 10 semillas
                s = [rows[(sch, f"A1@s{k}")]["d_mae"] for k in range(1, 11)]
                ax.plot(s, np.full(len(s), i + 0.28), "|", color=gray, ms=6, mew=1.2, zorder=3)
        if sch == "loco":
            n = nested["Anidado δ=0 (primario)"]; i = len(POLS)
            ax.hlines(i, n["d_mae_lo"], n["d_mae_hi"], color=blue, lw=1.8, zorder=2); ax.plot(n["d_mae"], i, "D", color=blue, ms=5, mec=surf, mew=1, zorder=3)
        else:
            ax.text(0, len(POLS), "not applicable", color=ink2, fontsize=7, ha="center", va="center")
        for j in range(1, len(groups)):
            if groups[j] != groups[j - 1]: ax.axhline(j - 0.5, color=grid, lw=0.8, zorder=0)
        ax.axvline(0, color=ink2, lw=0.9, zorder=1); ax.set_xlim(-0.035, 0.055)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02)); ax.grid(axis="x", color=grid, lw=0.6); ax.set_axisbelow(True)
        ax.set_title(panel, loc="left", fontsize=9, color=ink); ax.set_xlabel("ΔMAE vs. no augmentation (BCS units)", fontsize=8, color=ink2)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(grid); ax.tick_params(colors=ink2, labelsize=7.5, length=0)
    axs[0].set_yticks(np.arange(len(labels)), labels); axs[0].invert_yaxis()
    fig.text(0.01, 0.01, "Negative ΔMAE = lower error than A0. Points: photo-level ΔMAE; bars: 95 % CI (cow-level bootstrap, 5,000 resamples).\n"
             "Grey ticks: point estimates of A1 repeated with 10 other augmentation seeds. Diamond: nested A0/A1 selection (LOCO only).",
             fontsize=6.5, color=ink2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("pdf", "png"): fig.savefig(fd / f"figure_ablation.{ext}", dpi=300, facecolor=surf)
    print(f"-> {fd / 'figure_ablation.pdf'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out_dir", default="/work/aug"); a = ap.parse_args()
    out = Path(a.out_dir); fd = out / "final"; fd.mkdir(parents=True, exist_ok=True)
    rows, nested = load(out); tables(rows, nested, fd); figure(rows, nested, fd)
