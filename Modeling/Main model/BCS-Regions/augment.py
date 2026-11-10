# =============================================================
# Nombre del archivo: augment.py
#
# Descripción:
#   Data augmentation para el estudio de ablación (solo entrenamiento).
#   Transforma de forma coherente una foto RGB y sus 6 máscaras de
#   región (las del caché canónico <work>/masks). El descriptor se
#   calcula después con describe() de bcs_regions.py, sin cambios.
#
#   - Geométricas (imagen + máscaras con los mismos parámetros):
#       flip horizontal (+ intercambio thigh_L <-> thigh_R, índice
#       FLIP de segmenter.py), rotación, traslación y escala.
#   - Fotométricas (solo imagen): brillo/contraste, ruido gaussiano.
#
#   Los parámetros aleatorios de cada variante se sortean siempre todos,
#   con una semilla que depende solo de (BASE_SEED, foto, variante).
#   Así las políticas son anidadas: A5 usa los mismos flips que A1, los
#   mismos ángulos que A3, etc., lo que permite comparaciones pareadas.
#   La política no entra en la semilla.
#
#   Las intensidades son propuestas iniciales del equipo, no valores
#   tomados de la literatura ni optimizados.
# =============================================================
import zlib
import numpy as np
import torch, torch.nn.functional as F

from segmenter import FLIP, MEAN

BASE_SEED = 0
K_VARIANTS = 3

# Cada política activa un subconjunto de operaciones. A0 no genera filas
# aumentadas; A0-DUP genera K copias idénticas (control de ponderación).
POLICIES = {
    "A0":     None,
    "A0-DUP": {},
    "A1":     {"flip": 0.5},
    "A2":     {"bc": {"p": 0.5, "brightness": 0.10, "contrast": 0.10}},
    "A3":     {"rotate": 5.0},
    "A4":     {"translate": 0.03, "scale": (0.97, 1.03)},
    "A5":     {"flip": 0.5, "bc": {"p": 0.5, "brightness": 0.10, "contrast": 0.10}, "rotate": 5.0,
               "translate": 0.03, "scale": (0.97, 1.03)},
    "A6":     {"flip": 0.5, "bc": {"p": 0.5, "brightness": 0.10, "contrast": 0.10}, "rotate": 5.0,
               "translate": 0.03, "scale": (0.97, 1.03), "noise": 0.02},
}
FILL = (MEAN * 255).astype(np.float32)          # mismo color de relleno que el lienzo del segmentador

def variant_seed(name: str, k: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([BASE_SEED, zlib.crc32(name.encode()), k])

def sample_params(name: str, k: int) -> dict:
    """Sorteo completo e independiente de la política: u ~ U(0,1) para las compuertas y U(-1,1) para magnitudes."""
    r = np.random.default_rng(variant_seed(name, k))
    u = r.random(2); m = r.uniform(-1, 1, 6)
    return {"flip_u": float(u[0]), "bc_u": float(u[1]), "b": float(m[0]), "c": float(m[1]), "rot": float(m[2]),
            "tx": float(m[3]), "ty": float(m[4]), "s": float(m[5]), "noise_seed": int(r.integers(2**31))}

def resolve(policy: dict, p: dict) -> dict:
    """Parámetros efectivos de una variante bajo una política (0 / 1 = operación apagada)."""
    flip = "flip" in policy and p["flip_u"] < policy["flip"]
    bc = policy.get("bc"); use_bc = bc is not None and p["bc_u"] < bc["p"]
    lo, hi = policy.get("scale", (1.0, 1.0))
    return {"flip": bool(flip),
            "brightness": 1 + p["b"] * bc["brightness"] if use_bc else 1.0,
            "contrast": 1 + p["c"] * bc["contrast"] if use_bc else 1.0,
            "angle": p["rot"] * policy.get("rotate", 0.0),
            "tx": p["tx"] * policy.get("translate", 0.0), "ty": p["ty"] * policy.get("translate", 0.0),
            "scale": (lo + hi) / 2 + p["s"] * (hi - lo) / 2,
            "noise": policy.get("noise", 0.0), "noise_seed": p["noise_seed"]}

def is_identity(e: dict) -> bool:
    return (not e["flip"] and e["brightness"] == 1 and e["contrast"] == 1 and e["angle"] == 0
            and e["tx"] == 0 and e["ty"] == 0 and e["scale"] == 1 and e["noise"] == 0)

def affine_theta(h: int, w: int, angle: float, tx: float, ty: float, scale: float) -> torch.Tensor:
    """Matriz para affine_grid (salida -> entrada, coords normalizadas) de una transformación definida en píxeles:
    p_out = scale * R(angle) * p_in + (tx*w, ty*h), con centro en el centro de la imagen. Corrige el aspecto h != w."""
    a = np.deg2rad(angle); R_inv = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    D = np.diag([w / 2, h / 2]); M = np.linalg.inv(D) @ R_inv @ D / scale
    off = -M @ np.array([2 * tx, 2 * ty])
    return torch.tensor(np.concatenate([M, off[:, None]], 1), dtype=torch.float32)[None]

def warp(arr: np.ndarray, theta: torch.Tensor, fill) -> np.ndarray:
    """arr [C,h,w] float -> misma forma, muestreo bilineal; fuera del cuadro vale `fill` (por canal)."""
    f = torch.as_tensor(np.asarray(fill, np.float32)).reshape(-1, 1, 1)
    x = torch.from_numpy(arr)[None] - f
    g = F.affine_grid(theta, list(x.shape), align_corners=False)
    return (F.grid_sample(x, g, mode="bilinear", padding_mode="zeros", align_corners=False)[0] + f).numpy()

def apply(img: np.ndarray, masks: np.ndarray, e: dict):
    """img RGB uint8 [h,w,3], masks [6,h',w'] en [0,1] -> (img uint8, masks float32) transformadas con los parámetros e."""
    if is_identity(e): return img, masks
    im = img.astype(np.float32).transpose(2, 0, 1); ms = masks.astype(np.float32)
    if e["flip"]:                                                    # thigh_L/R se nombran por lado de la imagen
        im = im[:, :, ::-1].copy(); ms = ms[FLIP][:, :, ::-1].copy()
    if e["angle"] != 0 or e["tx"] != 0 or e["ty"] != 0 or e["scale"] != 1:
        th = affine_theta(im.shape[1], im.shape[2], e["angle"], e["tx"], e["ty"], e["scale"])
        im = warp(im, th, FILL); ms = warp(ms, th, np.zeros(ms.shape[0]))  # mismo theta: misma geometría normalizada
    if e["brightness"] != 1 or e["contrast"] != 1:                   # como ColorJitter: brillo y luego contraste
        im = np.clip(im * e["brightness"], 0, 255)
        gray = (0.299 * im[0] + 0.587 * im[1] + 0.114 * im[2]).mean()
        im = np.clip((im - gray) * e["contrast"] + gray, 0, 255)
    if e["noise"] > 0:
        im = im + np.random.default_rng(e["noise_seed"]).normal(0, e["noise"] * 255, im.shape).astype(np.float32)
    return np.clip(np.round(im), 0, 255).astype(np.uint8).transpose(1, 2, 0), np.clip(ms, 0, 1)
