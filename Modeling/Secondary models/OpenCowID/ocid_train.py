# =============================================================
# Nombre del archivo: ocid_train.py
# Descripción: Envoltorio de train.py de OpenCowID que permite
#   arrancar desde pesos ya entrenados (--init_weights), para:
#     1. entrenar con texturas sintéticas (como el repo original), y
#     2. afinar después con fotos reales etiquetadas (CAETEC).
#   Envuelve en torch.no_grad() la extracción de embeddings y la
#   evaluación KNN del original (sin eso se queda sin memoria en 8 GB).
#   Usa las mismas pérdidas y la misma selección del original: guarda
#   un checkpoint cada vez que la exactitud KNN (k=1) en validación
#   iguala o supera la mejor, y además "last.pth" al final.
#   La validación (galería/consulta) es ByID: mide si el modelo
#   reconoce vacas en vista dorsal en un día distinto.
#
# Uso (en la imagen muumetrics-opencowid, con GPU):
#   python ocid_train.py --train_data_folder /data/opencowid/synthetic \
#       --val_gallery_folder /data/opencowid/byid_val/gallery \
#       --val_data_folder /data/opencowid/byid_val/query \
#       --save_dir /data/opencowid/models/synthetic --epochs 3 --batch_size 48
# =============================================================

import argparse, sys

import torch

sys.path.insert(0, "/opt/OpenCowID")
import infer  # noqa: E402
import train as ocid_train  # noqa: E402
from utils import utils_load_data, set_model, my_utils_get_embeddings  # noqa: E402


# Versiones vectorizadas de las funciones de utils.py: mismo cálculo, sin los
# ciclos de Python sobre todos los pares de clases (con ~200 IDs, el original
# tarda ~20 min solo en los centroides y lo mismo por época).
def class_means(emb, targets):
    """Como get_mean_embeddings del original, sin imprimir las distancias entre todos los pares."""
    n = int(targets.max()) + 1
    return torch.stack([emb[targets == c].mean(0) for c in range(n)]).float().to(ocid_train.device)


def fast_optimal_centroids(class_mean, epochs=200, lr=0.1):
    C = class_mean.clone().requires_grad_()
    off_diag = ~torch.eye(len(C), dtype=torch.bool, device=C.device)
    for _ in range(epochs):
        d = torch.cdist(C, C)
        loss = -d[off_diag].sum()
        loss.backward()
        with torch.no_grad():
            C -= lr * C.grad
            C /= C.norm(dim=1, keepdim=True)
            C.grad = None
    return C.detach()


def fast_centroid_loss_2(x, labels, centroids):
    return torch.nn.functional.pairwise_distance(x, centroids[labels]).mean()


def fast_centroid_loss_repulsion(x, labels, centroids):
    inv = 1.0 / torch.cdist(x, centroids).clamp_min(1e-6)
    inv = inv.masked_fill(torch.nn.functional.one_hot(labels, len(centroids)).bool(), 0.0)
    return inv.sum() / (len(x) - 1) / len(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data_folder", required=True)
    p.add_argument("--val_gallery_folder", required=True)
    p.add_argument("--val_data_folder", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--init_weights", default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-5)
    args = p.parse_args()

    torch.manual_seed(0)
    # El original evalúa sin no_grad y guarda el grafo de cada lote (OOM en 8 GB)
    infer.infer_with_KNN = torch.no_grad()(infer.infer_with_KNN)
    ocid_train.centroid_loss_2 = fast_centroid_loss_2
    ocid_train.centroid_loss_repulsion = fast_centroid_loss_repulsion
    # fine_tune() del original guarda en la variable global SAVE_DIR de train.py
    ocid_train.SAVE_DIR = args.save_dir
    import os
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, gal_loader, val_loader = utils_load_data(
        args.train_data_folder, args.val_gallery_folder, args.val_data_folder, args.batch_size)
    net = set_model()
    if args.init_weights:
        net.load_state_dict(torch.load(args.init_weights, map_location="cpu"))
        print("Pesos iniciales:", args.init_weights)

    with torch.no_grad():
        emb, targets = my_utils_get_embeddings(net, train_loader)
    centroids = fast_optimal_centroids(class_means(emb, targets))
    ocid_train.fine_tune(net, train_loader, gal_loader, val_loader, centroids, args.epochs, args.lr)
    torch.save(net.state_dict(), f"{args.save_dir}/last.pth")
    print("Guardado:", f"{args.save_dir}/last.pth")


if __name__ == "__main__":
    main()
