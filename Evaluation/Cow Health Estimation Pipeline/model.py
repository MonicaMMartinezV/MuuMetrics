# =============================================================
# Nombre del archivo: model.py
# Autor: Ulises Orlando Carrizalez Lerín
# Fecha de creación: 11-11-2025
# Descripción: Funciones para cargar y usar modelo
# Dependencias: Funcion de discrteisacion
# =============================================================

import torch
import timm
import os
import pandas as pd
from PIL            import Image
from torchvision    import transforms
from discretizeData import discretize_value
from PIL            import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path):
    """
    Cargar modelo de pytorch para predecir BCS en imagen

    Args:
        checkpoint_path (str)  : direccion de .pth

    Returns:	
        model (model): modelo de prediccion
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    # reconstruir el modelo con el mismo nombre usado en el entrenamiento
    model = timm.create_model("convnext_base", pretrained=False, num_classes=1).to(device)
    
    # cargar los pesos
    state_dict_key = "model_state_dict" if "model_state_dict" in ckpt else "model_state"
    model.load_state_dict(ckpt[state_dict_key])

    model.eval()
    print(f"El Modelo se cargado con Exito :D device: {device}")

    return model

def get_val_transform(img_size=384):
    """
    Preparar las imágenes para que tengan el formato, tamaño y normalización que el modelo necesita

    Args:
        img_size (int)  : tamaño de imagen

    Returns:	
        _ (transforms.Compose): datos de tensor
    """
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.05)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def predict_image(model, image_path):
    """
    regresar el BCS de la imagen con modelo

    Args:
        model (model)  : modelo de prediccion
        image_path (str)  : direccion de imagen

    Returns:	
        _ (float): BCS predicho
    """
    transform = get_val_transform(384)
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img).cpu().numpy().flatten()[0]
    return discretize_value(float(pred))

def dfPredict(dir,checkpoint):
    """
    Generar df con predicciones de imagenes de un directorio.

    Args:
        dir (str)  : direccion de directorio con todas las imagenes
        checkpoint (str)  : direccion de .pth del modelo

    Returns:	
        _ (DataFrame): df con predicciones
    """
    model = load_model(checkpoint)
    results = []
    for filename in os.listdir(dir):
        if filename.lower().endswith((".jpg")):
            path = os.path.join(dir, filename)
            pred = predict_image(model,path)
            results.append({"img": filename, "BCS": pred})
    return pd.DataFrame(results)

def dfPredictTest(dir,checkpoint):
    """
    Generar df con predicciones de imagenes de un directorio con
    subdirectorios. (para recolectarlo del formato del dataset)

    Args:
        dir (str)  : direccion de directorio con todas las imagenes
        checkpoint (str)  : direccion de .pth del modelo

    Returns:	
        _ (DataFrame): df con predicciones
    """
    model = load_model(checkpoint)
    results = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.lower().endswith(('.jpg')):
                path = os.path.join(root, file)
                pred = predict_image(model,path)
                results.append({"img": file, "BCS": pred})
    return pd.DataFrame(results)