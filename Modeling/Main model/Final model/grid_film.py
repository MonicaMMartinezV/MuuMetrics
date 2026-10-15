# =============================================================
# Nombre del archivo: grid_film.py
#
# Descripción:
#   Grid search de hiperparámetros FiLM para filmModel.py, con split
#   agrupado por vaca. Corre en el host y lanza cada entrenamiento en
#   el contenedor muumetrics-convnext (GPU).
#
#   - La selección se hace por val MAE (val = fold k+1, vacas que no
#     están en test). El test solo se reporta, nunca se usa para elegir.
#   - El baseline (--no_film) se corre con varias semillas para medir
#     el ruido entre corridas: una mejora menor a ese ruido no es real.
#   - Reanudable: las corridas con test_metrics.json se saltan.
#
# Uso (desde esta carpeta, con la imagen ya construida):
#   # Ronda 1: toda la grilla en 2 folds
#   python3 grid_film.py run --folds 0 1
#   # Ronda 2: las 5 mejores por val MAE en los 5 folds
#   python3 grid_film.py run --folds 0 1 2 3 4 --top 5
#   # Tabla de resultados
#   python3 grid_film.py summary
#
#   Flags extra para filmModel.py van después de "--":
#   python3 grid_film.py run --folds 0 -- --epochs 20
#
# Dependencias (host): solo la librería estándar de Python + Docker.
# =============================================================

import argparse, itertools, json, os, statistics as st, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

GRID = {
    "stages": ["3", "2,3", "1,2,3"],
    "freqs": [0, 1, 2, 4],
    "lr_mult": [1, 10],
    "hidden": [16, 64],
}
# Necesario para que convnext_base a 384 px quepa en 8 GB (batch efectivo 24)
DEFAULT_TRAIN_ARGS = ["--batch_size", "12", "--grad_accum_steps", "2"]

def config_name(stages, freqs, lr_mult, hidden):
    return f"s{stages.replace(',', '')}_f{freqs}_lr{lr_mult}_h{hidden}"

def all_configs():
    return [
        {"name": config_name(s, f, l, h),
         "args": ["--film_stages", s, "--film_freqs", str(f),
                  "--film_lr_mult", str(l), "--film_hidden", str(h)]}
        for s, f, l, h in itertools.product(GRID["stages"], GRID["freqs"], GRID["lr_mult"], GRID["hidden"])
    ]

def baseline_configs(seeds):
    return [{"name": f"nofilm_seed{s}", "args": ["--no_film", "--seed", str(s)]} for s in seeds]

def load_metrics(run_dir: Path):
    try:
        return (json.loads((run_dir / "val_metrics.json").read_text()),
                json.loads((run_dir / "test_metrics.json").read_text()))
    except FileNotFoundError:
        return None

def run_one(cfg, fold, args, extra):
    run_dir = Path(args.output_dir) / cfg["name"] / f"fold{fold}"
    if load_metrics(run_dir) is not None:
        print(f"[SKIP] {cfg['name']} fold{fold} (ya existe)")
        return
    cmd = [
        "docker", "run", "--rm", "--gpus", "all", "--shm-size=2g",
        "-u", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp", "-e", "TORCH_HOME=/tmp/torch", "-e", "HF_HOME=/tmp/hf",
        "--name", f"muumetrics-training-grid-{cfg['name']}-f{fold}",
        "-v", f"{args.images_dir}:/data/images:ro",
        "-v", f"{args.csv}:/data/dataset_del.csv:ro",
        "-v", f"{Path(args.output_dir).resolve()}:/outputs",
        "--entrypoint", "python", args.image, "filmModel.py",
        "--csv", "/data/dataset_del.csv", "--images_dir", "/data/images",
        "--output_dir", f"/outputs/{cfg['name']}/fold{fold}",
        "--n_folds", str(args.n_folds), "--fold", str(fold),
        *DEFAULT_TRAIN_ARGS, *cfg["args"], *extra,
    ]
    print(f"=== {cfg['name']} fold{fold} ===", flush=True)
    if args.dry_run:
        print(" ".join(cmd)); return
    # Un fallo (p.ej. OOM) no detiene la grilla: se reporta y se sigue
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[FAIL] {cfg['name']} fold{fold} (exit {res.returncode})", flush=True)

def collect(output_dir: Path):
    """{config: {fold: (val, test)}} con todas las corridas terminadas."""
    out = {}
    for cdir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        for fdir in sorted(cdir.glob("fold*")):
            m = load_metrics(fdir)
            if m is not None:
                out.setdefault(cdir.name, {})[int(fdir.name[4:])] = m
    return out

def rank_configs(results, folds):
    """Configs FiLM con todos los folds dados, ordenadas por val MAE promedio."""
    ranked = []
    for name, per_fold in results.items():
        if name.startswith("nofilm") or not all(f in per_fold for f in folds):
            continue
        ranked.append((st.mean(per_fold[f][0]["mae"] for f in folds), name))
    return [n for _, n in sorted(ranked)]

def cmd_run(args, extra):
    configs = all_configs()
    if args.top:
        results = collect(Path(args.output_dir))
        # Ranking con los folds que ya completaron todas las configs del top
        done_folds = sorted(set.intersection(*[set(v) for k, v in results.items() if not k.startswith("nofilm")]))
        if not done_folds:
            sys.exit("No hay folds comunes terminados para rankear; corre primero la ronda 1.")
        top = rank_configs(results, done_folds)[: args.top]
        print(f"Top {args.top} por val MAE en folds {done_folds}: {top}")
        configs = [c for c in configs if c["name"] in top]
    configs = baseline_configs(args.baseline_seeds) + configs
    total = len(configs) * len(args.folds)
    print(f"{len(configs)} configs x {len(args.folds)} folds = {total} corridas")
    for fold in args.folds:  # fold por fuera: resultados parciales comparables antes
        for cfg in configs:
            run_one(cfg, fold, args, extra)
    cmd_summary(args)

def fmt(xs):
    return f"{st.mean(xs):.4f}±{st.pstdev(xs):.4f}" if xs else "-"

def cmd_summary(args):
    results = collect(Path(args.output_dir))
    if not results:
        print("Sin resultados todavía."); return
    base_name = f"nofilm_seed{args.baseline_seeds[0]}"
    base = results.get(base_name, {})

    # Ruido entre semillas del baseline: rango de test MAE por fold
    seeds = [results[n] for n in results if n.startswith("nofilm_seed")]
    noise = [max(s[f][1]["mae"] for s in seeds) - min(s[f][1]["mae"] for s in seeds)
             for f in base if sum(f in s for s in seeds) > 1 and all(f in s for s in seeds)]
    print(f"\nRuido baseline ({len(seeds)} semillas): rango de test MAE por fold = {fmt(noise)}")

    rows = []
    for name, per_fold in results.items():
        folds = sorted(per_fold)
        common = [f for f in folds if f in base]
        delta = [per_fold[f][1]["mae"] - base[f][1]["mae"] for f in common]
        rows.append((st.mean(per_fold[f][0]["mae"] for f in folds), name, folds,
                     fmt([per_fold[f][0]["mae"] for f in folds]),
                     fmt([per_fold[f][1]["mae"] for f in folds]),
                     fmt([per_fold[f][1]["rmse"] for f in folds]),
                     fmt(delta) if name != base_name else "-",
                     sum(d < 0 for d in delta) if name != base_name else None, len(delta)))
    rows.sort()
    print(f"\n{'config':22s} {'folds':12s} {'val MAE':15s} {'test MAE':15s} {'test RMSE':15s} "
          f"{'ΔMAE vs base':15s} gana")
    for _, name, folds, vmae, tmae, trmse, delta, wins, n in rows:
        w = f"{wins}/{n}" if wins is not None else ""
        print(f"{name:22s} {','.join(map(str, folds)):12s} {vmae:15s} {tmae:15s} {trmse:15s} {delta:15s} {w}")
    print("\nOrdenado por val MAE (criterio de selección). ΔMAE = test FiLM − test baseline en el mismo fold "
          "(negativo = FiLM mejor); 'gana' = folds donde FiLM tiene menor test MAE.")

def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "summary"):
        s = sub.add_parser(name)
        s.add_argument("--output_dir", default=str(REPO / "Dataset" / "outputs_film_grid"))
        s.add_argument("--baseline_seeds", type=int, nargs="+", default=[42, 43, 44])
        if name == "run":
            s.add_argument("--folds", type=int, nargs="+", required=True)
            s.add_argument("--top", type=int, default=0,
                           help="Solo las N mejores configs por val MAE de las corridas existentes.")
            s.add_argument("--n_folds", type=int, default=5)
            s.add_argument("--images_dir", default=str(REPO / "Dataset" / "Images"))
            s.add_argument("--csv", default=str(REPO / "dataset_del.csv"))
            s.add_argument("--image", default="muumetrics-convnext:latest")
            s.add_argument("--dry_run", action="store_true")
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--"); argv, extra = argv[:i], argv[i + 1:]
    return p.parse_args(argv), extra

if __name__ == "__main__":
    args, extra = parse_args()
    if args.cmd == "run":
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        cmd_run(args, extra)
    else:
        cmd_summary(args)
