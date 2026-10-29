# =============================================================
# Dockerfile: OpenCowID (re-identificación de vacas por patrón de pelaje)
#
# Parte de la imagen de entrenamiento (torch + CUDA) y agrega el repo
# upstream https://github.com/neis-lab/OpenCowID fijado a un commit.
# Parches mínimos para las versiones de la imagen:
#   - Image.ANTIALIAS ya no existe en Pillow >= 10 -> Image.LANCZOS
#   - models.resnet50(pretrained=True) -> weights=... (torchvision >= 0.13)
#   - generate_data.py: falta "import argparse"; rango de IDs y nº de
#     aumentos por argumento
#   - porespy 2.4.1 necesita scikit-image < 0.25 (usa skeletonize_3d)
#   - train.py: guarda los pesos en --save_dir
#
# Construir (desde esta carpeta):
#   docker build -t muumetrics-opencowid .
# =============================================================

FROM muumetrics-convnext:latest

ARG OPENCOWID_COMMIT=a00f085
RUN pip install porespy==2.4.1 "scikit-image<0.25" && \
    git clone https://github.com/neis-lab/OpenCowID.git /opt/OpenCowID && \
    cd /opt/OpenCowID && git checkout ${OPENCOWID_COMMIT} && \
    sed -i 's/Image.ANTIALIAS/Image.LANCZOS/' utils.py && \
    sed -i 's/models.resnet50(pretrained=True)/models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)/' models/resnet.py && \
    sed -i 's/n_jobs=64/n_jobs=-1/' infer.py && \
    sed -i "s/    parser.add_argument('--directory_path', type=str)/    parser.add_argument('--directory_path', type=str)\n    parser.add_argument('--n_ids', type=int, default=140)\n    parser.add_argument('--augments', type=int, default=500)/" generate_data.py && \
    sed -i 's/    augments_per_sample = 500/    augments_per_sample = args.augments/' generate_data.py && \
    sed -i 's/    for id in tqdm(range(1860, 2000)):/    for id in tqdm(range(args.n_ids)):/' generate_data.py && \
    sed -i "s/    parser.add_argument('--lr', type=float, default=0.00001)/    parser.add_argument('--lr', type=float, default=0.00001)\n    parser.add_argument('--save_dir', type=str, default='saved_models')/" train.py && \
    sed -i "s#save_path = f'saved_models/synthetic_loss_epoch_{epoch}.pth'#save_path = f'{SAVE_DIR}/synthetic_loss_epoch_{epoch}.pth'#" train.py && \
    sed -i "s/    args = parser.parse_args()/    args = parser.parse_args()\n    SAVE_DIR = args.save_dir\n    os.makedirs(SAVE_DIR, exist_ok=True)/" train.py && \
    sed -i '1i import os' train.py && \
    sed -i '1i import argparse' generate_data.py

# Para match_byid_caetec / view_filter: CLIP (clasificar ángulo de la foto) y
# ALIKED + LightGlue (emparejamiento de puntos entre fotos). LightGlue trae
# opencv-python, que pide librerías gráficas; se cambia por la versión headless.
RUN pip install open_clip_torch==2.24.0 \
        "git+https://github.com/cvg/LightGlue.git@eb42fee2d714" && \
    pip uninstall -y opencv-python && pip install opencv-python-headless==4.10.0.84

WORKDIR /opt/OpenCowID
ENTRYPOINT ["python"]
