# =============================================================
# Nombre del archivo: segmenter.py
#
# Descripción:
#   Segmentador de las 6 regiones anatómicas (multi-etiqueta, las
#   regiones se traslapan): encoder DINOv2-B/14 ajustado + decodificador
#   FPN. Entrenado con 40 imágenes anotadas en Roboflow (mIoU 0.745 en
#   test) sobre el frame completo del robot a 378x672.
#   Los recortes de Dataset_Fr_BCS se colocan al centro de un lienzo
#   16:9 para que la vaca tenga la escala con la que se entrenó
#   (IoU 0.92 contra las máscaras sacadas del frame original).
# =============================================================
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from PIL import Image

REGIONS = ["loin", "hips", "rump", "tailhead", "thigh_L", "thigh_R"]
FLIP = [0, 1, 2, 3, 5, 4]        # al voltear la imagen, muslo izq. <-> der.
H, W, DEC = 378, 672, 128
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

class RegionSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, features_only=True,
                                   dynamic_img_size=True, out_indices=(2, 5, 8, 11))
        ch = self.e.feature_info.channels()
        self.lat = nn.ModuleList([nn.Conv2d(c, DEC, 1) for c in ch])
        self.head = nn.Sequential(nn.Conv2d(DEC * len(ch), DEC, 3, padding=1), nn.BatchNorm2d(DEC), nn.ReLU(),
                                  nn.Conv2d(DEC, 6, 1))

    def forward(self, x):
        fs = self.e(x); t = fs[0].shape[-2:]; p = []; y = None
        for f, l in zip(fs[::-1], list(self.lat)[::-1]):            # FPN top-down
            z = l(f); y = z if y is None else z + F.interpolate(y, size=z.shape[-2:], mode="bilinear")
            p.append(F.interpolate(y, size=t, mode="bilinear"))
        return F.interpolate(self.head(torch.cat(p, 1)), size=x.shape[-2:], mode="bilinear")

def load_segmenter(weights: str, device) -> RegionSeg:
    net = RegionSeg(); net.load_state_dict(torch.load(weights, map_location="cpu"))
    return net.to(device).eval()

@torch.no_grad()
def segment(net: RegionSeg, img: np.ndarray, device) -> np.ndarray:
    """img RGB uint8 [h,w,3] -> probabilidades [6,h,w] float32 (promedio con la imagen volteada)."""
    h, w = img.shape[:2]
    wc = max(w, int(round(h * W / H))); x0 = (wc - w) // 2
    canvas = np.empty((h, wc, 3), np.uint8); canvas[:] = (MEAN * 255).astype(np.uint8); canvas[:, x0:x0 + w] = img
    small = np.asarray(Image.fromarray(canvas).resize((W, H), Image.BOX)) / 255.0
    x = torch.from_numpy(((small - MEAN) / STD).transpose(2, 0, 1)).float()[None].to(device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        p = (torch.sigmoid(net(x)) + torch.sigmoid(net(x.flip(-1))).flip(-1)[:, FLIP]) / 2
    p = F.interpolate(p.float(), size=(h, wc), mode="bilinear", align_corners=False)[0, :, :, x0:x0 + w]
    return p.cpu().numpy()
