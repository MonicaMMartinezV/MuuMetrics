# =============================================================
# Nombre del archivo: data_augmentation.py
# Autor: María Soto
# Fecha de creación: 21-10-2025
# Descripción: Carga un dataset de imágenes por carpetas, aplica 
#   augmentación (rotación, flip horizontal, escala y conversión 
#   a grayscale) y guarda las imágenes originales y aumentadas 
#   en un directorio de salida separado.
# Dependencias: numpy, matplotlib, tensorflow, pathlib, os, shutil
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # type: ignore
from tensorflow.keras import layers  # type: ignore
from pathlib import Path
import os

'''
Configuración de directorios
'''

baseDir = Path(
    "./Codigos Python/Batches para correr el codigo/"
    "04. Batch Imagenes Clasificadas DataAugmentation"
)
print("Ruta absoluta:", baseDir.resolve())
print("Existe:", baseDir.exists())

# Directorio de salida fuera del dataset para no interferir con clases
outputDir = baseDir.parent / 'output'
trainOutputDir = outputDir / 'train'
valOutputDir = outputDir / 'val'
augmentedOutputDir = outputDir / 'augmented'

os.makedirs(trainOutputDir, exist_ok=True)
os.makedirs(valOutputDir, exist_ok=True)
os.makedirs(augmentedOutputDir, exist_ok=True)

'''
Carga de datasets (entrenamiento y validación)
'''
trainDs = tf.keras.utils.image_dataset_from_directory(
    baseDir,
    validation_split=0.3,
    subset="training",
    seed=42,
    image_size=(1080, 1080)
)

valDs = tf.keras.utils.image_dataset_from_directory(
    baseDir,
    validation_split=0.3,
    subset="validation",
    seed=42,
    image_size=(1080, 1080)
)

'''
Definición de augmentación
'''
dataAugmentation = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomRotation(0.4),
    layers.RandomFlip("horizontal"),
    layers.Lambda(lambda x: tf.tile(tf.image.rgb_to_grayscale(x), [1, 1, 1, 3]))
])

'''
Visualización de ejemplos
'''
for images, labels in trainDs.take(1):
    sampleImages = images[:5]
    break

augmentedImages = dataAugmentation(sampleImages)

plt.figure(figsize=(10, 6))
for i in range(5):
    # Original
    plt.subplot(2, 5, i + 1)
    plt.imshow(sampleImages[i].numpy().astype("uint8"))
    plt.title("Original")
    plt.axis("off")
    # Augmented
    plt.subplot(2, 5, i + 6)
    plt.imshow((augmentedImages[i].numpy() * 255).astype("uint8"))
    plt.title("Augmented")
    plt.axis("off")
plt.tight_layout()
plt.show()

'''
Función para guardar imágenes desde un dataset
'''

def saveImagesFromDataset(dataset, outputDirectory, maxImages=None,
                          applyAugmentation=False, augmentationLayer=None):
    count = 0
    originalLabels = sorted(trainDs.class_names)
    labelMapping = {originalLabels[i]: 2 + i * 0.25 
                    for i in range(len(originalLabels))}

    for images, labels in dataset:
        if applyAugmentation and augmentationLayer is not None:
            images = augmentationLayer(images)

        for i in range(images.shape[0]):
            if maxImages is not None and count >= maxImages:
                return

            imgData = (tf.cast(images[i] * 255, tf.uint8) 
                       if images[i].dtype != tf.uint8 else images[i])

            img = tf.keras.utils.array_to_img(imgData)
            originalLabelName = trainDs.class_names[labels[i].numpy()]
            newLabel = labelMapping[originalLabelName]

            classDir = os.path.join(outputDirectory, str(newLabel))
            os.makedirs(classDir, exist_ok=True)
            imgPath = os.path.join(classDir, f'image_{count}.png')
            img.save(imgPath)
            count += 1

        if maxImages is not None and count >= maxImages:
            break

'''
Guardar imágenes aumentadas
'''
print("Saving augmented train images...")
saveImagesFromDataset(trainDs, augmentedOutputDir, 
                      applyAugmentation=True, augmentationLayer=dataAugmentation)
print("Image saving complete.")
