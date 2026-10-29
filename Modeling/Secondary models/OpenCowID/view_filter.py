# =============================================================
# Nombre del archivo: view_filter.py
# Descripción: Clasifica el ángulo de cada recorte de CAETEC con CLIP
#   (zero-shot, sin entrenar): atrás de pie / lado / frente / echada.
#   La cámara del robot ve a la vaca desde atrás y arriba, así que para
#   cruzar con ByID conviene quedarse con las fotos "atras".
#   - Escribe vistas.csv (probabilidad por clase y clase elegida) y,
#     con --dst, copia las fotos "atras" a <dst>/<ID>/foto.jpg.
#   - Genera una hoja de muestra por clase para revisar a ojo.
#
# Uso (en la imagen muumetrics-opencowid):
#   python view_filter.py --src /data/opencowid/caetec_crops_448 \
#       --out_dir /data/opencowid/views --dst /data/opencowid/caetec_rear
# =============================================================

import argparse, csv, random, shutil
from collections import defaultdict
from pathlib import Path

import open_clip
import torch
from PIL import Image, ImageDraw

PROMPTS = {
    "atras": ["a photo of a standing cow seen from behind, showing its rump, tail and hind legs",
              "a rear view photo of a dairy cow walking away"],
    "lado": ["a side view photo of a standing dairy cow",
             "a photo of the flank of a standing cow seen from the side"],
    "frente": ["a photo of a cow's head and face seen from the front",
               "a front view photo of a dairy cow looking at the camera"],
    "echada": ["a photo of a cow lying down on the ground",
               "a photo of a dairy cow resting lying on the floor"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dst", default=None)
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="openai")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, prep = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    tok = open_clip.get_tokenizer(args.model)
    classes = list(PROMPTS)
    with torch.no_grad():
        T = []
        for c in classes:
            t = model.encode_text(tok(PROMPTS[c]).to(device)).float()
            t = t / t.norm(dim=-1, keepdim=True)
            T.append(t.mean(0) / t.mean(0).norm())
        T = torch.stack(T)

    paths = sorted(Path(args.src).glob("*/*.jpg"))
    rows = []
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            batch = paths[i:i + 32]
            x = torch.stack([prep(Image.open(p).convert("RGB")) for p in batch]).to(device)
            f = model.encode_image(x).float()
            f = f / f.norm(dim=-1, keepdim=True)
            probs = (100 * f @ T.T).softmax(-1).cpu()
            for p, pr in zip(batch, probs):
                rows.append({"id_robot": p.parent.name, "foto": p.name,
                             **{c: round(float(v), 3) for c, v in zip(classes, pr)},
                             "vista": classes[int(pr.argmax())]})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "vistas.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    by_view = defaultdict(list)
    for r in rows:
        by_view[r["vista"]].append(r)
    random.seed(0)
    W = H = 180
    for v, rs in by_view.items():
        s = random.sample(rs, min(40, len(rs)))
        sheet = Image.new("RGB", (8 * W, ((len(s) + 7) // 8) * H), "white")
        d = ImageDraw.Draw(sheet)
        for k, r in enumerate(s):
            im = Image.open(Path(args.src) / r["id_robot"] / r["foto"]).convert("RGB")
            im.thumbnail((W - 4, H - 16))
            sheet.paste(im, ((k % 8) * W + 2, (k // 8) * H + 2))
            d.text(((k % 8) * W + 2, (k // 8) * H + H - 14), f"{r['id_robot']} p={r[v]:.2f}", fill="black")
        sheet.save(out / f"muestra_{v}.jpg", quality=85)

    if args.dst:
        dst = Path(args.dst)
        if dst.exists():
            shutil.rmtree(dst)
        for r in by_view["atras"]:
            (dst / r["id_robot"]).mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(args.src) / r["id_robot"] / r["foto"], dst / r["id_robot"] / r["foto"])
    print({v: len(rs) for v, rs in by_view.items()},
          "| IDs con foto 'atras':", len({r["id_robot"] for r in by_view["atras"]}))


if __name__ == "__main__":
    main()
