# =============================================================
# Nombre del archivo: transfer.py
#
# Descripción:
#   Transfer learning de BEiT-B/16 (arquitectura de Winkler et al. 2024)
#   para BCS, con la foto + las 6 máscaras de Dataset_Fr_BCS_masks.
#     0. pretrain  (opcional) SimMIM autosupervisado sobre fotos SIN
#                  etiqueta: desde cero (--init scratch) o adaptando los
#                  pesos de ImageNet (--init imagenet). Excluye las fotos
#                  y las vacas de Dataset_Fr_BCS. -> <work>/pretrain/<nombre>/
#     1. evaluate  validación por vaca, igual que BCS-Regions: LOCO y
#                  5 folds por vaca x N semillas. Dentro del train de cada
#                  fold, GroupKFold por vaca elige cuántos bloques ajustar
#                  (--grid) y la época de corte. -> <work>/metrics.csv
#     2. train     modelo final con todas las fotos -> <work>/beit_bcs.pt
#     3. predict   npz (+ su foto) -> BCS estimado (csv)
#
#   Regularización (el ajuste completo dio MAE 0.66): solo los últimos K
#   bloques, LP-FT (primero solo la cabeza), LLRD, drop-path, dropout,
#   weight decay, aumentos geométricos/color y corte temprano elegido
#   por validación interna por vaca.
#
# Uso: python transfer.py <paso> [opciones]   (ver --help)
# =============================================================
import argparse, copy, csv, json, math, random, time, hashlib
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from data import load_records, IMG_EXTS          # de ../BCS-Regions (montado en /regions)
from bcs_regions import metrics                  # mismas métricas que el modelo de referencia
from model import load_beit, BCSBeit, SimMIM, SIZE, GRID

MEAN = STD = 0.5                                 # normalización de BEiT (BeitImageProcessor)
S0 = 256                                         # resolución en caché (se recorta/escala a 224)
CLASSES = ["loin", "hips", "rump", "tailhead", "thigh_L", "thigh_R"]
FLIP9 = [0, 1, 2, 3, 4, 5, 6, 8, 7]              # al voltear: muslo izq. <-> der.

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ---------- Datos: foto + máscaras ----------
def load_dataset(a):
    """-> dict(x uint8 [N,9,S0,S0], y, vaca, name). Caché en <work>/dataset_<S0>.pt."""
    recs = load_records(a.data_dir); work = Path(a.work_dir)
    cache = work / f"dataset_{S0}.pt"; names = [r["name"] for r in recs]
    if cache.exists():
        d = torch.load(cache, weights_only=False)
        if list(d["name"]) == names: return d
    mdir = Path(a.masks_dir); x = torch.empty(len(recs), 9, S0, S0, dtype=torch.uint8)
    miss = [n for n in names if not (mdir / Path(n).with_suffix(".npz")).exists()]
    if miss: raise SystemExit(f"Faltan {len(miss)} npz en {mdir} (p.ej. {miss[0]})")
    for i, r in enumerate(recs):
        z = np.load(mdir / Path(r["name"]).with_suffix(".npz"))
        if list(z["classes"]) != CLASSES: raise SystemExit(f"Clases inesperadas en {r['name']}: {z['classes']}")
        img = np.array(Image.open(r["path"]).convert("RGB").resize((S0, S0), Image.BICUBIC))     # aplasta a cuadrado
        m = F.interpolate(torch.from_numpy(z["masks"]).float()[None], size=(S0, S0), mode="bilinear", align_corners=False)[0]
        x[i, :3] = torch.from_numpy(img).permute(2, 0, 1); x[i, 3:] = m.round().clamp(0, 255).to(torch.uint8)
    d = {"x": x, "y": np.array([r["bcs"] for r in recs]), "vaca": np.array([r["vaca"] for r in recs]), "name": np.array(names)}
    torch.save(d, cache); print(f"[datos] {len(names)} fotos + máscaras -> {cache}")
    return d

def to_input(xb: torch.Tensor) -> torch.Tensor:
    """uint8 [B,9,h,w] -> float [B,9,224,224]: RGB normalizado, máscaras en [0,1]."""
    x = xb.float() / 255.0
    if x.shape[-1] != SIZE: x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)
    return torch.cat([(x[:, :3] - MEAN) / STD, x[:, 3:]], 1)

def augment(xb: torch.Tensor) -> torch.Tensor:
    """Aumentos en GPU aplicados igual a foto y máscaras: volteo, rotación ±10°, escala, traslación, color."""
    x = xb.float() / 255.0; B, dev = x.shape[0], x.device
    flip = torch.rand(B, device=dev) < 0.5
    x[flip] = x[flip].flip(-1)[:, FLIP9]
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(10)
    sc = 1 / (0.85 + torch.rand(B, device=dev) * 0.25)                      # zoom 0.85x-1.1x
    t = (torch.rand(B, 2, device=dev) * 2 - 1) * 0.06
    th = torch.stack([torch.stack([ang.cos() * sc, -ang.sin() * sc, t[:, 0]], 1),
                      torch.stack([ang.sin() * sc, ang.cos() * sc, t[:, 1]], 1)], 1)
    grid = F.affine_grid(th, (B, 9, SIZE, SIZE), align_corners=False)
    rgb = F.grid_sample(x[:, :3], grid, mode="bilinear", padding_mode="border", align_corners=False)
    m = F.grid_sample(x[:, 3:], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    br = 0.8 + torch.rand(B, 1, 1, 1, device=dev) * 0.4; ct = 0.8 + torch.rand(B, 1, 1, 1, device=dev) * 0.4
    sat = 0.8 + torch.rand(B, 1, 1, 1, device=dev) * 0.4
    mu = rgb.mean((1, 2, 3), keepdim=True); gray = rgb.mean(1, keepdim=True)
    rgb = (((rgb * br - mu) * ct + mu) - gray) * sat + gray
    return torch.cat([(rgb.clamp(0, 1) - MEAN) / STD, m], 1)

# ---------- Entrenamiento de una configuración ----------
_BB = {}
def backbone(a):
    """Copia fresca del backbone (se lee de disco una sola vez)."""
    key = (a.backbone, a.drop_path)
    if key not in _BB: _BB[key] = load_beit(a.backbone, drop_path=a.drop_path)
    return copy.deepcopy(_BB[key])

@torch.no_grad()
def predict_idx(net, X, idx, dev, bs=32):
    net.eval(); out = []
    for i in range(0, len(idx), bs):
        x = to_input(X[idx[i:i + bs]].to(dev))
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            p = (net(x) + net(x.flip(-1)[:, FLIP9])) / 2                       # TTA: promedio con la foto volteada
        out.append(p.float().cpu())
    return torch.cat(out).numpy()

def fit(cfg, a, data, tr, dev, stop_epoch=None, evals=None, seed=0):
    """Entrena BCSBeit con cfg['k'] bloques sobre las fotos tr. evals: {nombre: idx} se predice cada
    --eval_every épocas -> (net, {nombre: {época: preds}}). El calendario de lr siempre es el de
    --epochs, así cortar en la época e equivale a lo que se vio en la validación interna."""
    seed_all(seed); X, y = data["x"], torch.tensor(data["y"], dtype=torch.float32)
    net = BCSBeit(backbone(a), cfg["k"], float(y[tr].mean()), a.head_drop).to(dev)
    opt = torch.optim.AdamW(net.param_groups(a.lr_head, a.lr_bb, a.llrd, a.wd), betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    spe = math.ceil(len(tr) / a.batch); E = a.epochs; stop = stop_epoch or E
    total, lp = E * spe, a.lp_epochs * spe; curves = {k: {} for k in (evals or {})}; step = 0
    for ep in range(1, stop + 1):
        net.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), a.batch):
            b = perm[i:i + a.batch]
            if len(b) < 2: continue
            for gi, g in enumerate(opt.param_groups):                                 # LP-FT + warmup + coseno
                if gi == 0: f = 0.5 * (1 + math.cos(math.pi * step / total))
                elif step < lp: f = 0.0
                else: f = min(1.0, (step - lp + 1) / spe) * 0.5 * (1 + math.cos(math.pi * (step - lp) / max(1, total - lp)))
                g["lr"] = g["base_lr"] * f
            x = augment(X[b].to(dev)); yt = y[b].to(dev)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                loss = F.smooth_l1_loss(net(x).float(), yt, beta=0.25)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            scaler.step(opt); scaler.update(); step += 1
        if evals and (ep % a.eval_every == 0 or ep == stop):
            for k, idx in evals.items(): curves[k][ep] = predict_idx(net, X, idx, dev)
    return net, curves

def parse_grid(s):
    return [{"k": int(k)} for k in s.split(",")]

def select(a, data, tr, dev, seed):
    """Elige (k, época) con GroupKFold por vaca dentro de tr (MAE por foto, como select_alpha)."""
    y, cows = data["y"], data["vaca"]; err = {}
    for ci, cfg in enumerate(parse_grid(a.grid)):
        for fi, (i, v) in enumerate(GroupKFold(a.inner_folds).split(tr, y[tr], cows[tr])):
            _, cur = fit(cfg, a, data, tr[i], dev, evals={"v": tr[v]}, seed=seed * 1000 + fi)
            for ep, p in cur["v"].items(): err.setdefault((ci, ep), []).append(np.abs(p - y[tr[v]]))
    (ci, ep), e = min(((k, np.concatenate(v).mean()) for k, v in err.items()), key=lambda t: t[1])
    return parse_grid(a.grid)[ci], ep, float(e)

# ---------- 1. Evaluación ----------
def run_tag(a):
    keys = ["backbone", "grid", "epochs", "lp_epochs", "lr_head", "lr_bb", "llrd", "wd", "drop_path", "head_drop", "batch", "inner_folds", "eval_every"]
    cfg = {k: getattr(a, k) for k in keys}
    return f"{Path(a.backbone).name}_{hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]}", cfg

def cmd_evaluate(a):
    t0 = time.time(); work, dev = Path(a.work_dir), device(); data = load_dataset(a)
    y, cows, names = data["y"], data["vaca"], data["name"]; uc = np.unique(cows); idx = np.arange(len(y))
    tag, cfg = run_tag(a); rdir = work / "runs" / tag; rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "config.json").write_text(json.dumps(cfg, indent=2))
    strata = np.round(y * 2).astype(int); schemes = {}
    if "loco" in a.schemes: schemes["loco"] = [list(GroupKFold(len(uc)).split(idx, y, cows))]
    if "rep5" in a.schemes:
        schemes[f"rep5x{a.seeds}"] = [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(idx, strata, cows)) for s in range(a.seeds)]
    n_total = sum(len(s) for reps in schemes.values() for s in reps); done = 0
    print(f"[evaluate] {tag}: {n_total} folds externos (cada uno: {len(parse_grid(a.grid))} config x {a.inner_folds} internos + 1 refit)")
    rows, sel_rows = [], []
    for sch, reps in schemes.items():
        P, maes = [], []
        for s, splits in enumerate(reps):
            p = np.zeros(len(y))
            for f, (tr, te) in enumerate(splits):
                fp = rdir / f"{sch}_s{s}_f{f}.json"
                if fp.exists():
                    r = json.loads(fp.read_text())
                else:
                    t1 = time.time(); best, ep, inner = select(a, data, tr, dev, seed=s)
                    net, _ = fit(best, a, data, tr, dev, stop_epoch=ep, seed=s)
                    r = {"te": te.tolist(), "pred": predict_idx(net, data["x"], te, dev).tolist(), "k": best["k"], "epoch": ep,
                         "inner_mae": inner, "test_cows": sorted(set(cows[te])), "seconds": round(time.time() - t1)}
                    fp.write_text(json.dumps(r)); del net; torch.cuda.empty_cache()
                done += 1; p[np.array(r["te"])] = r["pred"]
                sel_rows.append({"esquema": sch, "semilla": s, "fold": f, "k": r["k"], "epoca": r["epoch"], "mae_interno": round(r["inner_mae"], 4),
                                 "mae_test": round(float(np.abs(np.array(r["pred"]) - y[r["te"]]).mean()), 4), "segundos": r["seconds"]})
                el = time.time() - t0
                print(f"  [{sch} s{s} f{f}] k={r['k']} época={r['epoch']} MAE interno={r['inner_mae']:.3f} "
                      f"test={sel_rows[-1]['mae_test']:.3f} | {done}/{n_total}, {el / 60:.0f} min", flush=True)
            P.append(p); maes.append(np.abs(p - y).mean())
        P = np.mean(P, 0); m = metrics(P, y, cows)
        rows.append({"esquema": sch, **{k: round(float(v), 4) for k, v in m.items()},
                     "mae_reps_sd": round(float(np.std(maes)), 4), "fotos": len(y), "vacas": len(uc)})
        with open(work / f"preds_{sch}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real", "bcs_pred"])
            w.writerows([[n, c, yy, round(float(pp), 4)] for n, c, yy, pp in zip(names, cows, y, P)])
    for out in (work / "metrics.csv", rdir / "metrics.csv"):
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(rdir / "selections.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sel_rows[0])); w.writeheader(); w.writerows(sel_rows)
    p0 = np.zeros(len(y))
    for tr, te in GroupKFold(len(uc)).split(idx, y, cows): p0[te] = y[tr].mean()
    print(f"\n{len(y)} fotos, {len(uc)} vacas | backbone={a.backbone} | MAE prediciendo la media del train (LOCO): {np.abs(p0 - y).mean():.3f}")
    for r in rows:
        print(f"[{r['esquema']:7s}] MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  R2 fotos={r['r2_fotos']:.2f}  R2 vacas={r['r2_vacas']:.2f}  "
              f"±0.5={r['acc_05']:.0%}  MAE por vaca={r['mae_vaca']:.3f} [{r['mae_vaca_ic95_lo']:.3f}, {r['mae_vaca_ic95_hi']:.3f}]")
    print(f"-> {work / 'metrics.csv'} y {rdir} ({(time.time() - t0) / 60:.0f} min)")

# ---------- 2. Modelo final ----------
def cmd_train(a):
    work, dev = Path(a.work_dir), device(); data = load_dataset(a); tr = np.arange(len(data["y"]))
    best, ep, inner = select(a, data, tr, dev, seed=0)
    net, _ = fit(best, a, data, tr, dev, stop_epoch=ep, seed=0)
    torch.save({"state_dict": net.state_dict(), "k": best["k"], "epoch": ep, "y_mean": float(data["y"].mean()),
                "args": run_tag(a)[1]}, work / "beit_bcs.pt")
    print(f"[train] k={best['k']} época={ep} (MAE interno {inner:.3f}), {len(tr)} fotos -> {work / 'beit_bcs.pt'}")

# ---------- 3. Predicción ----------
def cmd_predict(a):
    if not a.input: raise SystemExit("predict necesita --input <npz o carpeta de npz>")
    work, dev = Path(a.work_dir), device(); ck = torch.load(a.model or work / "beit_bcs.pt", map_location="cpu", weights_only=False)
    net = BCSBeit(load_beit(ck["args"]["backbone"]), ck["k"], ck["y_mean"], ck["args"]["head_drop"])
    net.load_state_dict(ck["state_dict"]); net.to(dev).eval()
    src = Path(a.input); files = sorted(src.rglob("*.npz")) if src.is_dir() else [src]
    img_root = Path(a.images or a.data_dir); out = []
    for f in files:
        rel = f.relative_to(src) if src.is_dir() else Path(f.parent.name) / f.name
        # --input puede ser la raíz de máscaras (rel = vaca/foto.npz) o una carpeta de vaca (rel = foto.npz)
        cands = [img_root / r.with_suffix(e) for r in (rel, src.name / rel) if src.is_dir() or r is rel for e in IMG_EXTS]
        img = next((c for c in cands if c.exists()), None)
        if img is None: print(f"  sin foto para {rel}, se omite"); continue
        z = np.load(f); x = torch.empty(1, 9, S0, S0, dtype=torch.uint8)
        x[0, :3] = torch.from_numpy(np.asarray(Image.open(img).convert("RGB").resize((S0, S0), Image.BICUBIC))).permute(2, 0, 1)
        x[0, 3:] = F.interpolate(torch.from_numpy(z["masks"]).float()[None], size=(S0, S0), mode="bilinear")[0].round().clamp(0, 255).to(torch.uint8)
        out.append([str(rel), round(float(predict_idx(net, x, np.array([0]), dev)[0]), 3)])
    dst = Path(a.output or work / "predicciones.csv")
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["foto", "bcs_pred"]); w.writerows(out)
    for row in out[:10]: print(f"  {row[0]}: {row[1]}")
    print(f"[predict] {len(out)} fotos -> {dst}")

# ---------- 0. Preentrenamiento autosupervisado (SimMIM) ----------
class Corpus(torch.utils.data.Dataset):
    def __init__(self, paths):
        from torchvision import transforms as T
        self.paths = paths
        self.tf = T.Compose([T.RandomResizedCrop(SIZE, scale=(0.4, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                             T.RandomHorizontalFlip(), T.ColorJitter(0.3, 0.3, 0.2), T.ToTensor(), T.Normalize([MEAN] * 3, [STD] * 3)])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try: return self.tf(Image.open(self.paths[i]).convert("RGB"))
        except Exception: return self[random.randrange(len(self.paths))]

def simmim_mask(B, ratio=0.6, block=2):
    """Bloques de 32 px (2x2 parches) ocultos al azar -> bool [B,196]."""
    g = GRID // block; n = int(round(g * g * ratio))
    m = torch.zeros(B, g * g, dtype=torch.bool)
    for b in range(B): m[b, torch.randperm(g * g)[:n]] = True
    return m.view(B, 1, g, g).repeat_interleave(block, 2).repeat_interleave(block, 3).flatten(1)

def cmd_pretrain(a):
    work, dev = Path(a.work_dir), device(); out = work / "pretrain" / a.name; out.mkdir(parents=True, exist_ok=True)
    recs = load_records(a.data_dir, verbose=False)
    eval_cows = {r["vaca"] for r in recs}; eval_shots = {Path(r["name"]).stem.split("__")[-1] for r in recs}
    paths, skipped = [], 0
    for root in a.corpus:
        for p in sorted(Path(root).rglob("*")):
            if p.suffix.lower() not in IMG_EXTS: continue
            if any(part in eval_cows for part in p.parts) or any(s in p.stem for s in eval_shots): skipped += 1; continue
            paths.append(p)
    if not paths: raise SystemExit(f"No hay imágenes en {a.corpus}")
    print(f"[pretrain] {len(paths)} imágenes sin etiqueta ({skipped} excluidas por ser fotos/vacas de Dataset_Fr_BCS)")
    beit = load_beit(a.init, drop_path=0.1, mask_token=True); net = SimMIM(beit).to(dev)
    dl = torch.utils.data.DataLoader(Corpus(paths), batch_size=a.pt_batch, shuffle=True, drop_last=True,
                                     num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    decay = [p for n, p in net.named_parameters() if p.ndim > 1 and "token" not in n]
    no_decay = [p for n, p in net.named_parameters() if not (p.ndim > 1 and "token" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    total = a.pt_epochs * (len(dl) // a.accum); warm = a.warmup * (len(dl) // a.accum); start, step = 1, 0
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); start, step = ck["epoch"] + 1, ck["step"]
        print(f"[pretrain] continuando desde la época {ck['epoch']}")
    log = open(out / "log.csv", "a"); t0 = time.time()
    for ep in range(start, a.pt_epochs + 1):
        net.train(); tot, n = 0.0, 0
        for it, x in enumerate(dl):
            m = simmim_mask(x.shape[0]).to(dev); x = x.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                loss = net(x, m)
            scaler.scale(loss / a.accum).backward(); tot += loss.item(); n += 1
            if (it + 1) % a.accum == 0:
                lr = a.lr * (step / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
                for g in opt.param_groups: g["lr"] = lr
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
        log.write(f"{ep},{tot / max(1, n):.5f},{time.time() - t0:.0f}\n"); log.flush()
        print(f"  época {ep}/{a.pt_epochs} pérdida={tot / max(1, n):.4f} ({(time.time() - t0) / 60:.0f} min)", flush=True)
        if ep % a.save_every == 0 or ep == a.pt_epochs:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "epoch": ep, "step": step}, last)
            net.beit.save_pretrained(out)
    print(f"[pretrain] -> {out}  (úsalo con: evaluate --backbone {out})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Transfer learning BEiT-B + máscaras de región para BCS",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("step", choices=["pretrain", "evaluate", "train", "predict"])
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS", help="carpetas <vacaId>_<BCS>/ con las fotos")
    ap.add_argument("--masks_dir", default="/data/Dataset_Fr_BCS_masks", help="mismos nombres que data_dir, en .npz")
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--backbone", default="imagenet", help="imagenet | scratch | carpeta de 'pretrain' | checkpoint .ckpt/.pt (Winkler)")
    # fine-tuning
    ap.add_argument("--grid", default="2,4", help="bloques finales a ajustar; se elige dentro del train de cada fold")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lp_epochs", type=int, default=3, help="épocas solo-cabeza (LP-FT)")
    ap.add_argument("--eval_every", type=int, default=3, help="cada cuántas épocas se mide la validación interna (candidatas de corte)")
    ap.add_argument("--lr_head", type=float, default=1e-3); ap.add_argument("--lr_bb", type=float, default=3e-5)
    ap.add_argument("--llrd", type=float, default=0.75); ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--drop_path", type=float, default=0.1); ap.add_argument("--head_drop", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--inner_folds", type=int, default=4)
    ap.add_argument("--schemes", default="loco,rep5"); ap.add_argument("--seeds", type=int, default=10, help="repeticiones de 5 folds por vaca")
    # pretrain
    ap.add_argument("--init", default="scratch", help="pretrain: scratch | imagenet | carpeta")
    ap.add_argument("--name", default="beit_simmim", help="pretrain: subcarpeta en <work>/pretrain/")
    ap.add_argument("--corpus", nargs="+", default=["/corpus"], help="pretrain: carpetas con fotos sin etiqueta (recursivo)")
    ap.add_argument("--pt_epochs", type=int, default=300); ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--pt_batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=4, help="pretrain: acumulación de gradiente (batch efectivo = pt_batch x accum)")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--save_every", type=int, default=10)
    # predict
    ap.add_argument("--input"); ap.add_argument("--images", help="predict: raíz de las fotos (por defecto data_dir)")
    ap.add_argument("--model"); ap.add_argument("--output")
    a = ap.parse_args(); Path(a.work_dir).mkdir(parents=True, exist_ok=True)
    globals()[f"cmd_{a.step}"](a)
