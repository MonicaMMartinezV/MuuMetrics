# =============================================================
# Nombre del archivo: baselines/congelados.py
#
# Descripción:
#   Baselines CONGELADOS + el mismo ridge del modelo (comun.fit_apply), con
#   las mismas particiones que ajuste.py. Responden "¿importa el modelo
#   fundacional?" y dan la referencia del propio modelo en cada división:
#     resnet50_congelado  ResNet-50 ImageNet (resnet50.a1_in1k) congelado,
#                         foto aplastada a 384 px, promedio global -> 2048
#     bcs_regions         el modelo actual (features.npz: DINOv3 bloque 21,
#                         promedio por región) con las mismas particiones;
#                         en la división "foto" es el modelo con fuga;
#                         comprueba la firma de features.npz (comun.modelo)
#   Salidas: <work>/baselines/<variante>_<division>/preds_seed<s>.csv
#
# Uso: docker compose -f baselines/docker-compose.yml run --rm baselines baselines/congelados.py
# =============================================================
import argparse, time
from pathlib import Path
import numpy as np
import torch, timm
from PIL import Image

from comun_b import br, cargar, particiones, guardar_preds, fit_apply, DIVISIONES, ENTRADA
from comun import modelo

@torch.no_grad()
def resnet50(recs, dev):
    m = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=0).to(dev).eval(); X = []
    cfg = timm.data.resolve_data_config({}, model=m); mean, std = np.array(cfg["mean"]), np.array(cfg["std"])
    for r in recs:
        x = np.asarray(Image.open(r["path"]).convert("RGB").resize((ENTRADA, ENTRADA), Image.BICUBIC)) / 255.0
        x = torch.from_numpy(((x - mean) / std).transpose(2, 0, 1)).float()[None].to(dev)
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            X.append(m(x)[0].float().cpu().numpy())
    return np.stack(X).astype(np.float64)

def main(a):
    work = Path(a.work_dir); recs, y, cows, names = cargar(a.data_dir); out = work / "baselines"; out.mkdir(parents=True, exist_ok=True)
    Xr, _, _, nr, firma, _, _ = modelo(work)       # se detiene si features.npz no es del modelo actual
    print(f"[modelo] {firma}", flush=True)
    assert list(nr) == list(names), "features.npz no coincide con los datos: corre bcs_regions.py features"
    f = out / "feat_resnet50_congelado.npz"
    if not f.exists(): np.savez_compressed(f, X=resnet50(recs, br.device()).astype(np.float32), name=names)
    V = {"bcs_regions": Xr, "resnet50_congelado": np.load(f)["X"].astype(np.float64)}
    for div in DIVISIONES:
        for s, splits in enumerate(particiones(y, cows, div, a.semillas)):
            for v, X in V.items():
                d = out / f"{v}_{div}"; d.mkdir(exist_ok=True); p = np.zeros(len(y)); t0 = time.time()
                for tr, te in splits: p[te] = fit_apply(X[tr], y[tr], cows[tr], X[te])
                guardar_preds(d / f"preds_seed{s}.csv", names, cows, y, p)
                print(f"  {v:20s} {div} semilla {s}: MAE {np.abs(p - y).mean():.3f} ({time.time() - t0:.0f}s)", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS_v2")
    ap.add_argument("--work_dir", default="/work")
    main(ap.parse_args())
