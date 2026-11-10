# Compara las predicciones de las vacas apartadas con su BCS real (del nombre de la carpeta).
# Uso: python3 compare_holdout.py [work_holdout/pred_holdout.csv]
import csv, collections, sys

path = sys.argv[1] if len(sys.argv) > 1 else "work_holdout/pred_holdout.csv"
g = collections.defaultdict(list)
for r in csv.DictReader(open(path)):
    g[r["foto"].split("/")[0]].append(float(r["bcs_pred"]))
errs = []
for c, v in sorted(g.items()):
    real = float(c.split("_")[1]); m = sum(v) / len(v); errs += [abs(x - real) for x in v]
    print(f"{c:11s} real={real:.2f}  pred_media={m:.2f}  error={abs(m - real):.2f}  "
          f"±0.5={sum(abs(x - real) <= 0.5 for x in v)}/{len(v)}")
print(f"MAE por foto: {sum(errs) / len(errs):.3f}")
