# =============================================================
# Nombre del archivo: ablaciones/recortar_zonas.py
#
# Descripción:
#   Conserva las 6 regiones del modelo base (traslapadas) pero les recorta los
#   píxeles de las zonas exclusivas que "restan" en el mapa de importancia
#   (zonas/figures/zonas_mapa.png, 37 vacas): "cadera sola" (-0.030*) y
#   "muslo solo" (-0.016). Cada región se promedia solo sobre sus píxeles
#   restantes; nada más cambia (DINOv2-L congelado, ridge + recalibración,
#   validación por vaca).
#   Las zonas rojas se eligieron antes de tener las vacas 9000-9027: la
#   diferencia en esas 27 vacas es la prueba no sesgada. El recorte del
#   "lomo solo" se eligió viendo las 64 vacas: su resultado es optimista.
#   Salidas: <work>/ablaciones/{feat_recorte_<variante>.npz, recortar_zonas.csv}
#
# Uso (GPU solo la primera vez):
#   docker compose run --rm --entrypoint python bcs ablaciones/recortar_zonas.py
# =============================================================
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bcs_regions as br
from data import load_records
from segmenter import MEAN, STD
from quitar_zonas import RULES, ZN, assign

D = 1024
VARIANTES = {"6_regiones": [], "sin_cadera_sola": ["cadera_sola"], "sin_muslo_solo": ["muslo_solo"],
             "sin_rojas": ["cadera_sola", "muslo_solo"], "sin_lomo_solo": ["lomo"],
             "sin_rojas_y_lomo_solo": ["cadera_sola", "muslo_solo", "lomo"]}

def region_weights(fuera):
    """[6,64]: qué códigos (combinación de máscaras) cuentan para cada región, sin las zonas en 'fuera'."""
    a = assign(); out = np.zeros((6, 64), np.float32)
    for r in range(6):
        for c in range(1, 64):
            if c & (1 << r) and (a[c] < 0 or ZN[a[c]] not in fuera): out[r, c] = 1
    return torch.from_numpy(out)

@torch.no_grad()
def compute(work, recs, out):
    dev = br.device(); bb = br.load_backbone(dev); G = br.SIZE // 14; bits = torch.tensor([1, 2, 4, 8, 16, 32], dtype=torch.long)
    W = {v: region_weights(f) for v, f in VARIANTES.items()}; X = {v: np.zeros((len(recs), 6 * D + 6), np.float32) for v in VARIANTES}
    for i, r in enumerate(recs):
        img = br.read_rgb(r["path"])
        x = np.asarray(Image.fromarray(img).resize((br.SIZE, br.SIZE), Image.BICUBIC)) / 255.0
        x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).float()[None].to(dev)
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            tok = bb.forward_features(x)[0, bb.num_prefix_tokens:].float().cpu()
        m = np.load(br.mask_file(work, r))["masks"].astype(np.float32) / 255.0
        mb = F.interpolate(torch.from_numpy((m > 0.5).astype(np.float32))[None], size=(br.SIZE, br.SIZE), mode="nearest")[0]
        fr = F.adaptive_avg_pool2d(F.one_hot((mb.long() * bits[:, None, None]).sum(0), 64).permute(2, 0, 1).float()[None], (G, G))[0].reshape(64, -1)
        for v in VARIANTES:
            w = W[v] @ fr; tot = w.sum(1, keepdim=True); f = (w @ tok) / tot.clamp_min(1e-6); miss = tot[:, 0] < 0.5; f[miss] = 0
            X[v][i, :6 * D] = f.reshape(-1).numpy(); X[v][i, 6 * D:] = miss.float().numpy()
        if (i + 1) % 100 == 0: print(f"  {i + 1}/{len(recs)}", flush=True)
    for v in VARIANTES: np.savez_compressed(out / f"feat_recorte_{v}.npz", X=X[v])

def main(a):
    work = Path(a.work_dir); out = work / "ablaciones"; out.mkdir(exist_ok=True)
    X6, y, cows, names = br.load_features(work); recs = load_records(a.data_dir)
    assert [r["name"] for r in recs] == list(names), "features.npz no coincide con los datos"
    if not all((out / f"feat_recorte_{v}.npz").exists() for v in VARIANTES): compute(work, recs, out)
    V = {v: np.load(out / f"feat_recorte_{v}.npz")["X"].astype(np.float64) for v in VARIANTES}
    print(f"comprobación 6_regiones vs features.npz: máx |dif| = {np.abs(V['6_regiones'] - X6).max():.4f}")
    schemes = br.splits_for(y, cows, a.seeds); keys = sorted(set(zip(cows, y)))
    nuevas = np.array([c.startswith("90") for c, _ in keys]); rows, ce = [], {}; rng = np.random.RandomState(0); t0 = time.time()
    for v, Xv in V.items():
        r = {"variante": v, "recorte": "+".join(VARIANTES[v]) or "ninguno"}
        for sch, reps in schemes.items():
            P = []
            for splits in reps:
                p = np.zeros(len(y))
                for tr, te in splits: p[te] = br.apply_model(br.fit_model(Xv[tr], y[tr], cows[tr]), Xv[te])
                P.append(p)
            P = np.mean(P, 0); m = br.metrics(P, y, cows)
            r.update({f"mae_{sch}": round(float(m["mae"]), 4), f"r2_fotos_{sch}": round(float(m["r2_fotos"]), 4), f"r2_vacas_{sch}": round(float(m["r2_vacas"]), 4)})
            if sch == "rep5x10": ce[v] = np.array([abs(P[(cows == c) & (y == b)].mean() - b) for c, b in keys])
        rows.append(r); print(f"{v}: {time.time() - t0:.0f}s", flush=True)
    for r in rows:
        d = ce[r["variante"]] - ce["6_regiones"]
        for tag, sel in (("todas", np.ones(len(d), bool)), ("nuevas", nuevas)):
            dd = d[sel]; bs = [dd[rng.randint(0, len(dd), len(dd))].mean() for _ in range(5000)]
            r[f"dif_{tag}"] = round(float(dd.mean()), 4); r[f"ic95_{tag}"] = f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]"
    with open(out / "recortar_zonas.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{'variante':24s} {'MAE loco':>8s} {'MAE 5x10':>8s} {'R2fot':>6s} {'R2vac':>6s} | dif. vs 6 regiones (5x10): todas (n={len(keys)}) | 27 vacas nuevas (n={nuevas.sum()})")
    for r in rows:
        print(f"{r['variante']:24s} {r['mae_loco']:8.3f} {r['mae_rep5x10']:8.3f} {r['r2_fotos_rep5x10']:6.3f} {r['r2_vacas_rep5x10']:6.3f} | "
              f"{r['dif_todas']:+.3f} {r['ic95_todas']} | {r['dif_nuevas']:+.3f} {r['ic95_nuevas']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS")
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--seeds", type=int, default=10)
    main(ap.parse_args())
