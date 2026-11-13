# =============================================================
# Nombre del archivo: baselines/yolo_cls.py
#
# Descripción:
#   Baseline YOLOv8-cls (Ultralytics), como los trabajos que clasifican el
#   BCS con YOLO (p. ej. Dandıl et al.). Clases = valores de BCS en pasos
#   de 0.25 presentes en el train del fold. Mismas particiones que
#   ajuste.py; Ultralytics entrena en el 80 % del train y guarda el mejor
#   modelo según la exactitud en el 20 % interno (por vaca o por foto),
#   que es su práctica estándar. Predicción continua = valor esperado
#   sobre las probabilidades de clase (también se guarda la clase más
#   probable). Las fotos se aplastan a 384 x 384 una sola vez (como el
#   modelo), así el recorte central de Ultralytics no corta la vaca.
#   Salidas: <work>/baselines/yolov8<t>_cls_<division>/
#            fold_s<s>_k<k>.npz, preds_seed<s>.csv, preds_clase_seed<s>.csv
#
# Uso: docker compose -f baselines/docker-compose.yml run --rm baselines \
#        baselines/yolo_cls.py --division vaca [--tam s] [--rapido]
# =============================================================
import argparse, os, shutil, time
from pathlib import Path
import numpy as np
from PIL import Image

from comun_b import br, cargar, particiones, interna, guardar_preds, DIVISIONES, ENTRADA

def preparar_imagenes(recs, dst: Path):
    dst.mkdir(parents=True, exist_ok=True); rutas = []
    for i, r in enumerate(recs):
        f = dst / f"{i:04d}.jpg"
        if not f.exists(): Image.open(r["path"]).convert("RGB").resize((ENTRADA, ENTRADA), Image.BICUBIC).save(f, quality=95)
        rutas.append(f)
    return rutas

def armar(carpeta: Path, rutas, y, ia, iv):
    """carpeta/{train,val}/<clase>/ con enlaces; las clases son las del train; la validación sin clase en train va a la más cercana."""
    if carpeta.exists(): shutil.rmtree(carpeta)
    clases = sorted(set(y[ia]))
    for parte, idx in (("train", ia), ("val", iv)):
        for c in clases: (carpeta / parte / f"{c:.2f}").mkdir(parents=True, exist_ok=True)
        for i in idx:
            c = min(clases, key=lambda k: abs(k - y[i]))
            (carpeta / parte / f"{c:.2f}" / rutas[i].name).symlink_to(rutas[i])
    for c in clases:                                  # Ultralytics/torchvision no aceptan carpetas de clase vacías en val
        d = carpeta / "val" / f"{c:.2f}"
        if not any(d.iterdir()): ej = next(i for i in ia if abs(y[i] - c) < 1e-9); (d / rutas[ej].name).symlink_to(rutas[ej])

def main(a):
    from ultralytics import YOLO
    work = Path(a.work_dir); recs, y, cows, names = cargar(a.data_dir)
    (work / "baselines").mkdir(parents=True, exist_ok=True)
    os.chdir(work / "baselines")                     # Ultralytics descarga sus pesos al directorio actual: que no caigan en el código
    pref = f"yolov8{a.tam}" if a.version == "8" else f"yolo11{a.tam}"
    nombre = f"{pref}_cls_{a.division}{'_rapido' if a.rapido else ''}"
    out = work / "baselines" / nombre; out.mkdir(parents=True, exist_ok=True)
    rutas = preparar_imagenes(recs, work / "baselines" / "img384"); epocas = 2 if a.rapido else a.epocas; t0 = time.time()
    print(f"[{nombre}] {len(y)} fotos, {len(np.unique(cows))} vacas | {epocas} épocas", flush=True)
    for s, splits in enumerate(particiones(y, cows, a.division, a.semillas)):
        p = np.full(len(y), np.nan); pc = np.full(len(y), np.nan)
        for k, (tr, te) in enumerate(splits):
            if a.rapido and k > 0: break
            f = out / f"fold_s{s}_k{k}.npz"
            if not f.exists():
                ia, iv = interna(tr, y, cows, a.division, s); ds = work / "baselines" / "yolo_tmp" / nombre
                armar(ds, rutas, y, ia, iv)
                m = YOLO(f"{pref}-cls.pt")
                m.train(data=str(ds), epochs=epocas, imgsz=ENTRADA, batch=16, seed=s, deterministic=False, workers=2,
                        project=str(out / "runs"), name=f"s{s}_k{k}", exist_ok=True, plots=False, verbose=False, val=True)
                best = YOLO(str(out / "runs" / f"s{s}_k{k}" / "weights" / "best.pt"))
                vals = np.array([float(best.names[j]) for j in range(len(best.names))]); E, C = [], []
                for i in range(0, len(te), 32):
                    for r in best.predict([str(rutas[j]) for j in te[i:i + 32]], imgsz=ENTRADA, verbose=False):
                        pr = r.probs.data.float().cpu().numpy(); E.append(float(pr @ vals)); C.append(float(vals[pr.argmax()]))
                np.savez(f, pred=np.array(E), clase=np.array(C)); shutil.rmtree(out / "runs" / f"s{s}_k{k}", ignore_errors=True)
            z = np.load(f); p[te], pc[te] = z["pred"], z["clase"]
            print(f"  s{s} k{k}: MAE {np.abs(z['pred'] - y[te]).mean():.3f} [{time.time() - t0:.0f}s]", flush=True)
        ok = ~np.isnan(p); print(f"[semilla {s}] MAE test {np.abs(p[ok] - y[ok]).mean():.3f} ({ok.sum()} fotos)", flush=True)
        if a.rapido: break
        guardar_preds(out / f"preds_seed{s}.csv", names, cows, y, p); guardar_preds(out / f"preds_clase_seed{s}.csv", names, cows, y, pc)
    print(f"-> {out} ({time.time() - t0:.0f}s)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--division", default="vaca", choices=DIVISIONES)
    ap.add_argument("--tam", default="s", choices=["n", "s", "m"], help="tamaño de YOLO-cls")
    ap.add_argument("--version", default="8", choices=["8", "11"], help="YOLOv8-cls o YOLO11-cls")
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--rapido", action="store_true", help="prueba de humo: 1 fold, 2 épocas")
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS_v2")
    ap.add_argument("--work_dir", default="/work")
    main(ap.parse_args())
