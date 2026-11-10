# =============================================================
# Nombre del archivo: train_segmenter.py
#
# Descripción:
#   Reentrena el segmentador de regiones (misma arquitectura y receta que
#   segmenter.py / grid_v2 de BCS-RegionSeg: DINOv2-B/14 ajustado + FPN,
#   378x672, 2000 iters, bs 4, BCE+Dice, lr 3e-4, encoder x0.1) agregando
#   las fotos UNICT anotadas (prep_unict_seg.py).
#
#   Datos:
#     - robot: 40 fotos de BCS_masking.sam2 (RLE sin clase + labels_robot.json),
#       igual que grid_v2/prep.py (mismo orden, misma foto excluida).
#     - UNICT: <unict>/{train,valid,test}/*.jpg + .npz (ya girado, cola abajo),
#       puesto al centro de un lienzo 16:9 (como segmenter.segment()).
#   Pasos:
#     prep   -> <work>/data.npz
#     cv     -> 5 folds sobre el pool del robot (mismos folds que grid_v2);
#               cada fold entrena con robot del fold + UNICT train+valid y evalúa
#               en el robot fuera del fold y en UNICT test.
#     evalbase -> pesos actuales (final_all40.pt) sobre UNICT test.
#     all    -> entrena con todo (robot 40 + UNICT 40) y guarda los pesos.
# =============================================================
import argparse, json, math, os, sys, time
from pathlib import Path
import cv2, numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from segmenter import RegionSeg, REGIONS, FLIP as SWAP

ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["prep", "cv", "evalbase", "all"])
ap.add_argument("--robot", default="/robot")            # BCS_masking.sam2
ap.add_argument("--unict", default="/data/Dataset_UNICT_seg")
ap.add_argument("--work", default="/work/seg")
ap.add_argument("--weights", default="/weights/final_all40.pt")
ap.add_argument("--out", default="/work/seg/final_all40_unict40.pt")
ap.add_argument("--iters", type=int, default=2000); ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--enc_lr_mult", type=float, default=0.1)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--no_unict_test", action="store_true", help="all: dejar fuera UNICT test (para visualizarlo sin fuga)")
ap.add_argument("--no_unict", action="store_true", help="cv/all: entrenar solo con el robot (UNICT solo se evalúa)")
ap.add_argument("--aug", default="base", choices=["base", "persp", "occl", "photo"],
                help="base = receta grid_v2; persp = + perspectiva/inclinación, giro ±35°, shear y estiramiento; "
                     "occl = + rectángulos tapados (ignorados); photo = + gamma, sombra, ruido, JPEG")
ap.add_argument("--pseudo", default="", help="cv: carpeta de cuadros sin anotar (<BCS>/*.jpg) -> maestro por fold + alumno")
ap.add_argument("--pseudo_keep", type=float, default=.7, help="fracción de cuadros pseudo-etiquetados que se usan")
ap.add_argument("--student_iters", type=int, default=3000)
ap.add_argument("--tag", default="", help="cv: sufijo del json de resultados")
a = ap.parse_args(); work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
H, W = 378, 672; SH, SW = 540, 960
MEAN, STD = np.array([.485, .456, .406]), np.array([.229, .224, .225])
EXCLUDE = {"valid/2025-06-22-19-51-38"}

def rle_to_mask(seg):
    h, w = seg["size"]; s = seg["counts"]
    if isinstance(s, str):
        cnts, p = [], 0
        while p < len(s):
            x, k, more = 0, 0, True
            while more:
                c = ord(s[p]) - 48; x |= (c & 0x1f) << (5 * k); more = c & 0x20; p += 1; k += 1
                if not more and (c & 0x10): x |= -1 << (5 * k)
            if len(cnts) > 2: x += cnts[-2]
            cnts.append(x)
    else: cnts = s
    flat = np.zeros(h * w, np.uint8); pos, v = 0, 0
    for c in cnts: flat[pos:pos + c] = v; pos += c; v ^= 1
    return flat.reshape((w, h)).T

def canvas16x9(img, m):
    """Centra la foto en un lienzo 16:9 (fondo = color medio), como segmenter.segment()."""
    h, w = img.shape[:2]; wc = max(w, int(round(h * W / H))); x0 = (wc - w) // 2
    ci = np.empty((h, wc, 3), np.uint8); ci[:] = (MEAN * 255).astype(np.uint8); ci[:, x0:x0 + w] = img
    cm = np.zeros((6, h, wc), np.uint8); cm[:, :, x0:x0 + w] = m
    return ci, cm

def prep():
    lab = json.load(open(work / "labels_robot.json"))
    imgs, masks, names, splits, src = [], [], [], [], []
    for k in sorted(lab):
        if any(k.startswith(e) for e in EXCLUDE): continue
        d = json.load(open(f"{a.robot}/{k}")); M = np.zeros((6, SH, SW), np.uint8)
        for i, an in enumerate(d["annotations"]):
            M[REGIONS.index(lab[k][str(i)])] = cv2.resize(rle_to_mask(an["segmentation"]), (SW, SH), interpolation=cv2.INTER_NEAREST)
        imgs.append(cv2.resize(cv2.imread(f"{a.robot}/{k.replace('.json', '.jpg')}"), (SW, SH), interpolation=cv2.INTER_AREA)[:, :, ::-1])
        masks.append(M); names.append(os.path.basename(k)[:30]); splits.append(k.split("/")[0]); src.append("robot")
    for sp in ["train", "valid", "test"]:
        for f in sorted(Path(a.unict, sp).glob("*.npz")):
            img = cv2.imread(str(f.with_suffix(".jpg")))[:, :, ::-1]; m = (np.load(f)["masks"] > 127).astype(np.uint8)
            ci, cm = canvas16x9(img, m)
            imgs.append(cv2.resize(ci, (SW, SH), interpolation=cv2.INTER_AREA))
            masks.append(np.stack([cv2.resize(c, (SW, SH), interpolation=cv2.INTER_NEAREST) for c in cm]))
            names.append(f.stem); splits.append(sp); src.append("unict")
    np.savez_compressed(work / "data.npz", imgs=np.stack(imgs), masks=np.stack(masks), names=names, splits=splits, src=src)
    print(f"[prep] {len(names)} fotos:", {(s, p): list(zip(src, splits)).count((s, p)) for s, p in sorted(set(zip(src, splits)))})

def augment(img, m):
    h, w = img.shape[:2]
    if np.random.rand() < .5: img, m = img[:, ::-1], m[:, :, ::-1][SWAP]
    s = np.exp(np.random.uniform(-.3, .3)); ang = np.random.uniform(-20, 20)
    A = cv2.getRotationMatrix2D((w / 2 + np.random.uniform(-.15, .15) * w, h / 2 + np.random.uniform(-.15, .15) * h), ang, s)
    A[:, 2] += np.random.uniform(-.1, .1, 2) * [w, h]
    img = cv2.warpAffine(np.ascontiguousarray(img), A, (w, h), borderMode=cv2.BORDER_REFLECT)
    m = np.stack([cv2.warpAffine(np.ascontiguousarray(c), A, (w, h), flags=cv2.INTER_NEAREST) for c in m])
    img = img.astype(np.float32) * np.random.uniform(.6, 1.4) + np.random.uniform(-30, 30)
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(.6, 1.4), 0, 255); img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if np.random.rand() < .3: img = cv2.GaussianBlur(img, (0, 0), np.random.uniform(.5, 2))
    return img, m

def view_warp(w, h, rng, persp, rot, scale, shear, stretch, shift):
    """Homografía 3x3 que simula otro punto de vista: perspectiva (esquinas movidas), giro, escala, shear, estiramiento."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + rng.uniform(-persp, persp, (4, 2)) * [w, h]).astype(np.float32)
    P = cv2.getPerspectiveTransform(src, dst)
    c = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]]); ci = np.linalg.inv(c)
    t = np.deg2rad(rng.uniform(-rot, rot)); s = np.exp(rng.uniform(-scale, scale)); r = np.exp(rng.uniform(-stretch, stretch))
    sh = rng.uniform(-shear, shear)
    A = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]) @ np.diag([s * r, s / r, 1]) @ np.array([[1, sh, 0], [0, 1, 0], [0, 0, 1]])
    T = np.array([[1, 0, rng.uniform(-shift, shift) * w], [0, 1, rng.uniform(-shift, shift) * h], [0, 0, 1]])
    return T @ ci @ A @ c @ P

def apply_warp(img, m, M, border=cv2.BORDER_REFLECT):
    h, w = img.shape[:2]
    img = cv2.warpPerspective(np.ascontiguousarray(img), M, (w, h), borderMode=border)
    m = np.stack([cv2.warpPerspective(np.ascontiguousarray(c), M, (w, h), flags=cv2.INTER_NEAREST) for c in m])
    return img, m

def photometric(img):
    img = img.astype(np.float32) * np.random.uniform(.6, 1.4) + np.random.uniform(-30, 30)
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(.6, 1.4), 0, 255); img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if np.random.rand() < .3: img = cv2.GaussianBlur(img, (0, 0), np.random.uniform(.5, 2))
    return img

def augment_persp(img, m):
    """Como augment() pero con cambios de punto de vista: perspectiva ±18% por esquina (70% de las veces), giro ±35°,
    escala ±35%, shear ±0.2, estiramiento ±20%, desplazamiento ±12%."""
    if np.random.rand() < .5: img, m = img[:, ::-1], m[:, :, ::-1][SWAP]
    h, w = img.shape[:2]
    M = view_warp(w, h, np.random, .18 if np.random.rand() < .7 else 0, 35, .35, .2, .2, .12)
    img, m = apply_warp(img, m, M, border=cv2.BORDER_CONSTANT)   # sin espejo: el reflejo crearía vaca sin etiqueta
    return photometric(img), m

IGN = 2   # valor de máscara "ignorar" (píxeles tapados o pseudo-etiqueta dudosa); 0/1 = negativo/positivo

def augment_occl(img, m):
    """augment() + 1-3 rectángulos tapados (10-30% del lado) con color medio o ruido, el 50% de las veces.
    Lo tapado se marca IGN en las 6 máscaras (no se pide adivinar lo que no se ve)."""
    img, m = augment(img, m); h, w = img.shape[:2]
    if np.random.rand() < .5:
        img = img.copy(); m = m.copy()
        for _ in range(np.random.randint(1, 4)):
            rh, rw = int(h * np.random.uniform(.1, .3)), int(w * np.random.uniform(.1, .3))
            y0, x0 = np.random.randint(0, h - rh), np.random.randint(0, w - rw)
            img[y0:y0 + rh, x0:x0 + rw] = (MEAN * 255).astype(np.uint8) if np.random.rand() < .5 else \
                np.random.randint(0, 256, (rh, rw, 3), dtype=np.uint8)
            m[:, y0:y0 + rh, x0:x0 + rw] = IGN
    return img, m

def augment_photo(img, m):
    """augment() + fotométrico fuerte: ganancia por canal ±12%, gamma 0.6-1.6, sombra/gradiente de luz (40%),
    ruido gaussiano σ 0-10 (50%), compresión JPEG calidad 30-90 (50%)."""
    img, m = augment(img, m); h, w = img.shape[:2]
    x = img.astype(np.float32) / 255
    x *= np.random.uniform(.88, 1.12, 3)
    x = np.clip(x, 0, 1) ** np.random.uniform(.6, 1.6)
    if np.random.rand() < .4:      # gradiente lineal de iluminación en dirección aleatoria
        t = np.random.uniform(0, 2 * np.pi); yy, xx = np.mgrid[0:h, 0:w]
        g = (np.cos(t) * (xx / w - .5) + np.sin(t) * (yy / h - .5)); x *= (1 + np.random.uniform(.3, .7) * g)[..., None]
    x = np.clip(x * 255, 0, 255)
    if np.random.rand() < .5: x = x + np.random.normal(0, np.random.uniform(0, 10), x.shape)
    img = np.clip(x, 0, 255).astype(np.uint8)
    if np.random.rand() < .5:
        _, enc = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(np.random.randint(30, 91))])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]
    return img, m

AUG = {"base": augment, "persp": augment_persp, "occl": augment_occl, "photo": augment_photo}

def to_t(img): return torch.from_numpy(((img / 255. - MEAN) / STD).transpose(2, 0, 1)).float()
def rs(img, m): return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), np.stack([cv2.resize(c, (W, H), interpolation=cv2.INTER_NEAREST) for c in m])

def loss_fn(lo, y, wt=None):
    """BCE + Dice. wt (0/1) excluye píxeles; sin wt es exactamente la receta original."""
    if wt is None: wt = torch.ones_like(y)
    p = torch.sigmoid(lo) * wt; yw = y * wt; inter = (p * yw).sum((2, 3))
    dice = (1 - (2 * inter + 1) / (p.sum((2, 3)) + yw.sum((2, 3)) + 1)).mean()
    bce = (F.binary_cross_entropy_with_logits(lo, y, reduction="none") * wt).sum() / wt.sum().clamp(min=1)
    return bce + dice

def new_net():
    import timm
    net = RegionSeg()   # misma arquitectura; encoder desde DINOv2 preentrenado como en grid_v2
    pre = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, features_only=True,
                            dynamic_img_size=True, out_indices=(2, 5, 8, 11))
    net.e.load_state_dict(pre.state_dict()); del pre
    try: net.e.set_grad_checkpointing(True)
    except Exception: pass
    return net

def fit(IM, MK, tr, dev, extra=None, iters=None):
    """extra = lista de (img 378x672, máscara {0,1,IGN}) pseudo-etiquetadas: la mitad de cada batch sale de ahí."""
    iters = iters or a.iters
    torch.manual_seed(0); np.random.seed(0); net = new_net().to(dev).train()
    enc = list(net.e.parameters()); dec = [p for n, p in net.named_parameters() if not n.startswith("e.")]
    opt = torch.optim.AdamW([{"params": dec, "lr0": a.lr}, {"params": enc, "lr0": a.lr * a.enc_lr_mult}], lr=a.lr, weight_decay=1e-4)
    sc = torch.amp.GradScaler(); small = [rs(IM[i], MK[i]) for i in tr]; t0 = time.time()
    for it in range(iters):
        f = min(1, (it + 1) / 100) * 0.5 * (1 + math.cos(math.pi * it / iters))
        for g in opt.param_groups: g["lr"] = g["lr0"] * f
        n_ex = a.bs // 2 if extra else 0
        src = [small[np.random.randint(len(small))] for _ in range(a.bs - n_ex)] + [extra[np.random.randint(len(extra))] for _ in range(n_ex)]
        b = [AUG[a.aug](*s) for s in src]
        x = torch.stack([to_t(i) for i, _ in b]).to(dev); M = torch.from_numpy(np.stack([m for _, m in b])).to(dev)
        y = (M == 1).float(); wt = (M != IGN).float() if (M == IGN).any() else None
        with torch.autocast("cuda"): out = net(x)
        l = loss_fn(out.float(), y, wt); opt.zero_grad(); sc.scale(l).backward(); sc.step(opt); sc.update()
        if (it + 1) % 500 == 0: print(f"    iter {it + 1}/{iters} loss={l.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    return net.eval()

def ts(name):
    """'2025-05-30-16-49-34_cam4...' -> segundos (para excluir cuadros de la misma visita que una foto anotada)."""
    import datetime as dt
    return dt.datetime.strptime(name[:19], "%Y-%m-%d-%H-%M-%S").timestamp()

def load_unlabeled(annotated):
    """Cuadros completos del robot sin anotar (a.pseudo/<BCS>/*.jpg), a 378x672, sin los que están a < 10 min
    de una foto anotada (misma visita / misma vaca que una de validación)."""
    t_ann = np.array([ts(n) for n in annotated]); out, names, skip = [], [], 0
    files = sorted(f for f in Path(a.pseudo).glob("*/*.jpg") if not f.parent.name.lower().startswith("discar"))
    for f in files:
        try: t = ts(f.name)
        except ValueError: continue
        if np.abs(t_ann - t).min() < 600: skip += 1; continue
        im = cv2.imread(str(f))
        if im is None: continue
        out.append(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()); names.append(f.name)
    print(f"[pseudo] {len(out)} cuadros sin anotar ({skip} excluidos por estar a < 10 min de una foto anotada)", flush=True)
    return out, names

@torch.no_grad()
def pseudo_label(net, imgs, dev):
    """Maestro -> máscara {0,1,IGN} por imagen + puntaje (acuerdo foto vs foto volteada, mIoU).
    Píxel dudoso (0.3 < p < 0.7) = IGN. Se quedan las a.pseudo_keep mejores con las 6 regiones presentes."""
    lab, score = [], []
    for img in imgs:
        x = to_t(img)[None].to(dev)
        with torch.autocast("cuda"):
            p = torch.sigmoid(net(x)); pf = torch.sigmoid(net(x.flip(-1))).flip(-1)[:, SWAP]
        p, pf = p[0].float().cpu().numpy(), pf[0].float().cpu().numpy(); P = (p + pf) / 2
        A, B = p > .5, pf > .5
        agree = np.mean([(A[c] & B[c]).sum() / max((A[c] | B[c]).sum(), 1) for c in range(6)])
        present = all((P[c] > .5).sum() > 200 for c in range(6))
        m = (P > .5).astype(np.uint8); m[(P > .3) & (P < .7)] = IGN
        lab.append(m); score.append(agree if present else -1)
    score = np.array(score); k = int(round(len(imgs) * a.pseudo_keep))
    keep = [i for i in np.argsort(-score)[:k] if score[i] >= 0]
    print(f"[pseudo] acuerdo con volteo: mediana {np.median(score[score >= 0]):.3f}; se usan {len(keep)} "
          f"(umbral {score[keep[-1]]:.3f}); ignorado {np.mean([np.mean(lab[i] == IGN) for i in keep]):.1%} de los píxeles", flush=True)
    return [(imgs[i], lab[i]) for i in keep]

def predict(net, img, dev):
    x = to_t(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA))[None].to(dev)
    with torch.no_grad(), torch.autocast("cuda"):
        p = torch.sigmoid(net(x)); pf = torch.sigmoid(net(x.flip(-1))).flip(-1)[:, SWAP]
    P = ((p + pf) / 2)[0].float().cpu().numpy(); h, w = img.shape[:2]
    return np.stack([cv2.resize(c, (w, h)) for c in P])

def iou(P, G, th=.5):
    pr = P > th; g = G > 0
    return [float((pr[c] & g[c]).sum() / max((pr[c] | g[c]).sum(), 1)) for c in range(6)]

def summary(res):
    A = np.array(list(res.values())) if res else np.zeros((0, 6))
    return {"n": len(res), "miou": float(A.mean()) if len(A) else None, "per_class": dict(zip(REGIONS, A.mean(0).round(4).tolist())) if len(A) else {}}

def fmt(tag, s): return f"{tag}: n={s['n']} mIoU={s['miou']:.4f} | " + " ".join(f"{c}={v:.3f}" for c, v in s["per_class"].items())

def main():
    if a.step == "prep": return prep()
    dev = torch.device("cuda"); D = np.load(work / "data.npz")
    IM, MK, NM, SP, SRC = D["imgs"], D["masks"], list(D["names"]), list(D["splits"]), list(D["src"])
    robot_pool = [i for i in range(len(NM)) if SRC[i] == "robot" and SP[i] != "test"]
    robot_test = [i for i in range(len(NM)) if SRC[i] == "robot" and SP[i] == "test"]
    un_trval = [i for i in range(len(NM)) if SRC[i] == "unict" and SP[i] != "test"]
    un_test = [i for i in range(len(NM)) if SRC[i] == "unict" and SP[i] == "test"]

    if a.step == "evalbase":
        from segmenter import load_segmenter
        net = load_segmenter(a.weights, dev); res = {NM[i]: iou(predict(net, IM[i], dev), MK[i]) for i in un_test + un_trval}
        s_test = summary({k: v for k, v in res.items() if k in {NM[i] for i in un_test}}); s_all = summary(res)
        print(fmt("base -> UNICT test", s_test)); print(fmt("base -> UNICT 40", s_all))
        json.dump({"unict_test": s_test, "unict_all": s_all, "per_image": res}, open(work / "evalbase.json", "w"), indent=1)
        return

    if a.step == "all":
        tr = robot_pool + robot_test + ([] if a.no_unict else un_trval + ([] if a.no_unict_test else un_test))
        print(f"[all] entrenando con {len(tr)} fotos"); net = fit(IM, MK, tr, dev)
        if a.pseudo:   # misma receta que en cv: maestro con todas las anotadas -> pseudo-etiquetas -> alumno
            UNL, _ = load_unlabeled([NM[i] for i in range(len(NM)) if SRC[i] == "robot"])
            extra = pseudo_label(net, UNL, dev); del net; torch.cuda.empty_cache()
            print(f"[all] alumno: {len(tr)} anotadas + {len(extra)} pseudo-etiquetadas, {a.student_iters} iters")
            net = fit(IM, MK, tr, dev, extra=extra, iters=a.student_iters)
        torch.save(net.state_dict(), a.out); print(f"[all] pesos -> {a.out}"); return

    # cv: mismos folds que grid_v2 (perm sobre el pool del robot, semilla 42)
    rng = np.random.RandomState(42); perm = rng.permutation(robot_pool)
    folds = [sorted(perm[k::a.folds]) for k in range(a.folds)]
    # Vistas nuevas para medir robustez al ángulo: 3 deformaciones fijas por foto de validación (semilla = índice),
    # fondo constante (color medio) para no inventar vaca fuera del cuadro.
    def warped_views(i):
        rng = np.random.RandomState(1000 + i); h, w = IM[i].shape[:2]; out = []
        for _ in range(3):
            M = view_warp(w, h, rng, .12, 20, .2, .1, .15, .05)
            im = cv2.warpPerspective(IM[i], M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(int(v) for v in MEAN * 255))
            mk = np.stack([cv2.warpPerspective(c, M, (w, h), flags=cv2.INTER_NEAREST) for c in MK[i]]); out.append((im, mk))
        return out

    un_eval = un_trval + un_test if a.no_unict else un_test
    res_r, res_w, res_u, res_t = {}, {}, {}, {}; t0 = time.time()
    if a.pseudo: UNL, _ = load_unlabeled([NM[i] for i in range(len(NM)) if SRC[i] == "robot"])
    for k, va in enumerate(folds):
        tr = [i for i in robot_pool if i not in set(va)] + ([] if a.no_unict else un_trval)
        print(f"[cv {a.tag}] fold {k}: train {len(tr)} | aug={a.aug} | UNICT en train: {not a.no_unict} | val robot {len(va)}", flush=True)
        net = fit(IM, MK, tr, dev)
        if a.pseudo:   # el maestro solo vio el train del fold; se evalúa también (referencia pareada)
            for i in va: res_t[NM[i]] = iou(predict(net, IM[i], dev), MK[i])
            extra = pseudo_label(net, UNL, dev); del net; torch.cuda.empty_cache()
            net = fit(IM, MK, tr, dev, extra=extra, iters=a.student_iters)
            print(f"[cv {a.tag}] fold {k} maestro={summary(res_t)['miou']:.4f}", flush=True)
        for i in va:
            res_r[NM[i]] = iou(predict(net, IM[i], dev), MK[i])
            for j, (im, mk) in enumerate(warped_views(i)): res_w[f"{NM[i]}#w{j}"] = iou(predict(net, im, dev), mk)
        for i in un_eval: res_u[f"{NM[i]}#f{k}"] = iou(predict(net, IM[i], dev), MK[i])
        print(f"[cv {a.tag}] fold {k} listo: robot={summary(res_r)['miou']:.4f} robot-deformado={summary(res_w)['miou']:.4f} "
              f"UNICT={summary(res_u)['miou']:.4f} ({(time.time() - t0) / 60:.0f} min)", flush=True)
        del net; torch.cuda.empty_cache()
    out = {"robot": summary(res_r), "robot_warped": summary(res_w), "unict": summary(res_u), "unict_set": "40 (no vistas)" if a.no_unict else "test",
           "per_image_robot": res_r, "per_image_warped": res_w, "per_image_unict": res_u, "config": vars(a)}
    if res_t: out["robot_teacher"] = summary(res_t); out["per_image_teacher"] = res_t; print(fmt(f"[{a.tag}] maestro", out["robot_teacher"]))
    json.dump(out, open(work / f"cv_{a.tag or 'unict'}.json", "w"), indent=1)
    print(fmt(f"[{a.tag}] robot CV", out["robot"])); print(fmt(f"[{a.tag}] robot deformado", out["robot_warped"]))
    print(fmt(f"[{a.tag}] UNICT {out['unict_set']}", out["unict"]))

if __name__ == "__main__":
    main()
