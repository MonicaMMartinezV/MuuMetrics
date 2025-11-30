# =============================================================
# Nombre del archivo: modelBCS2.py
# Autores: Mónica Monserrat Martínez Vásquez,
#          Ulises Orlando Carrizalez Lerín,
#          Bárbara Paola Alcántara Vega 
#
# Descripción:
#   entrenamiento de una red convolucional separable para estimar el
#   body condition score (bcs) a partir de imágenes del dataset
#   MuMetrics. se incluyen funciones de carga, augmentación, entrenamiento,
#   callbacks y visualización del desempeño del modelo.
#
# Dependencias: os, numpy, pandas, tensorflow, matplotlib, datetime
# =============================================================

import os, math, random, pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard
import tensorflow.keras.backend as K
import datetime

# Configuración de rutas
baseDir  = r"C:\Users\Artur\OneDrive\Documents\DataSet MuMetrics"
#imagesDir = f'{baseDir}/Final Dataset/Images'
#imagesDir = f'{baseDir}/Data Transformations/Cow'
imagesDir = os.path.join(baseDir, 'DS')
outputDir = os.path.join(baseDir, 'model')
#outputDir = f'{baseDir}/4. Modeling'
checkpointDir = os.path.join(baseDir, 'Checkpoints')
#checkpointDir = f'{outputDir}/Checkpoints'
os.makedirs(checkpointDir, exist_ok=True)

# Parámetros base
imageSize    = (612, 612) #800,1000 #512,512
seed         = 42
batchSize    = 32
valTestSplit = 0.30

# Descubrimiento de clases en el directorio de imágenes
subdirs = sorted([
    d for d in os.listdir(imagesDir)
    if os.path.isdir(os.path.join(imagesDir, d))])

try:
  classNames = sorted(subdirs, key=lambda s: float(s))
except ValueError as e:
  raise ValueError(f"Valores numericos, revisar: {subdirs}") from e

bcsValues = [float(c) for c in classNames]
print("Clases detectadas (BCS):", bcsValues)

# Creación de datasets de entrenamiento y validación
trainDataset = tf.keras.utils.image_dataset_from_directory(
    imagesDir,
    labels='inferred',
    label_mode='int',
    class_names=classNames,
    color_mode='rgb',
    batch_size=batchSize,
    image_size=imageSize,
    shuffle=True,
    seed=seed,
    validation_split=valTestSplit,
    subset='training'
)

valTestDataset = tf.keras.utils.image_dataset_from_directory(
    imagesDir,
    labels='inferred',
    label_mode='int',
    class_names=classNames,
    color_mode='rgb',
    batch_size=batchSize,
    image_size=imageSize,
    shuffle=True,
    seed=seed,
    validation_split=valTestSplit,
    subset='validation'
)

# Tabla de conversión clase → valor de regresión
bcsLookupTable = tf.constant(bcsValues, dtype=tf.float32)

@tf.function
def putRegressionLabels(images, classIndices):
    """
    Convierte índices de clase a etiquetas numéricas de bcs.

    args:
        images (tf.Tensor): batch de imágenes.
        classIndices (tf.Tensor): índices enteros de clase.

    returns:
        tuple(tf.Tensor, tf.Tensor): (imágenes, etiquetas_float)
    """
    labels = tf.gather(bcsLookupTable, classIndices)
    labels = tf.expand_dims(labels , axis=-1)
    return images, labels

trainRegDataset = trainDataset.map(
  putRegressionLabels, num_parallel_calls=tf.data.AUTOTUNE)
valTestRegDataset = valTestDataset.map(
  putRegressionLabels,  num_parallel_calls=tf.data.AUTOTUNE)

def splitValidationAndTest(dataset):
    """
    Divide el dataset de validación en partes iguales para validación y prueba.

    args:
        dataset (tf.data.Dataset): dataset combinado.

    returns:
        tuple(tf.data.Dataset, tf.data.Dataset): (valDataset, testDataset)
    """
    datasetSize = dataset.cardinality().numpy()
    halfSize = datasetSize // 2
    valDataset  = dataset.take(halfSize)
    testDataset = dataset.skip(halfSize)
    return valDataset, testDataset

valRegDataset, testRegDataset = splitValidationAndTest(valTestRegDataset)

# Augmentación y normalización de imágenes
dataAugmentationLayer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.1),
], name="dataAugmentation")

rescaleLayer = tf.keras.layers.Rescaling(1./255, name="rescaleLayer")

def augmentThenScale(images, labels):
    """
    Aplica augmentación y reescalado (solo en entrenamiento).

    args:
        images (tf.Tensor): batch de imágenes.
        labels (tf.Tensor): valores bcs.

    returns:
        tuple(tf.Tensor, tf.Tensor)
    """
    images = dataAugmentationLayer(images, training=True)
    images = rescaleLayer(images)
    return images, labels

def onlyScale(images, labels):
    """Reescalado 1/255 (validación y test)."""
    images = rescaleLayer(images)
    return images, labels

autoTune = tf.data.AUTOTUNE

trainDataset = (trainRegDataset
           .shuffle(8 * batchSize, seed=seed, reshuffle_each_iteration=True)
           .map(augmentThenScale, num_parallel_calls=autoTune)
           .prefetch(autoTune))

valDataset = (valRegDataset
         .map(onlyScale, num_parallel_calls=autoTune)
         .prefetch(autoTune))

testDataset = (testRegDataset
          .map(onlyScale, num_parallel_calls=autoTune)
          .prefetch(autoTune))

def buildBcsCnnModel(inputShape):
    """
    Construye una cnn basada en separable conv2d para regresión bcs.

    args:
        inputShape (tuple): dimensiones de entrada (h, w, c).

    returns:
        tf.keras.Model
    """
    model = tf.keras.Sequential([
      layers.SeparableConv2D(32, kernel_size=3, padding='same', activation='relu',
                    kernel_regularizer=regularizers.l2(
                        1e-4), input_shape=inputShape),
      layers.SeparableConv2D(64, kernel_size=3, padding='same', activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Dropout(0.15),
      layers.SeparableConv2D(128, kernel_size=3, padding='same', activation='relu'),
      layers.SeparableConv2D(128, kernel_size=3, padding='same', activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Dropout(0.15),
      #layers.SeparableConv2D(256, kernel_size=3, padding='same', activation='relu'),
      #layers.SeparableConv2D(256, kernel_size=3, padding='same', activation='relu'),
      #layers.MaxPooling2D((2, 2)),
      #layers.Dropout(0.2),
      layers.Flatten(),
      layers.Dense(256, activation='relu'), #512 con todas las capas en kernels al doble ha sido el mejoir resultado
      layers.Dropout(0.2),
      layers.Dense(128, activation='relu'),
      layers.Dense(1, activation='linear', name='bcs_output')
      ])
    return model

inputShape = (612, 612, 3)
model = buildBcsCnnModel(inputShape)
model.summary()

# Métricas personalizadas

def mae(yTrue, yPred):
    """Error absoluto medio"""
    return tf.reduce_mean(tf.abs(yTrue - yPred))

def bias(yTrue, yPred):
    """Sesgo promedio (pred - real)"""
    return tf.reduce_mean(yPred - yTrue)

def variance(yTrue, yPred):
    """Varianza de las predicciones."""
    meanPred = tf.reduce_mean(yPred)
    return tf.reduce_mean(tf.square(yPred - meanPred))

def accuracyWithinHalfBcs(yTrue, yPred):
    """Porcentaje de aciertos ±0.5 bcs."""
    absError  = tf.abs(yTrue - yPred)
    return tf.reduce_mean(tf.cast(absError  <= 0.5, tf.float32))

# Compilación y callbacks
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Compilar el modelo
model.compile(
    optimizer=optimizer,
    loss='mae',
    metrics=[mae, bias, variance, accuracyWithinHalfBcs]
)

# Rutas de guardado
checkpointPath = os.path.join(checkpointDir, "bcsCnnWeights.weights.h5")
logsDir = os.path.join(
    outputDir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Guardar sólo los pesos
checkpointCallback = ModelCheckpoint(
    filepath=checkpointPath,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_mae',
    mode='min',
    verbose=1
)

# Detener si el modelo deja de mejorar
earlyStoppingCallback = EarlyStopping(
    monitor='val_mae',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

tensorboardCallback = TensorBoard(log_dir=logsDir)
callbacksList = [checkpointCallback, earlyStoppingCallback, tensorboardCallback]

# Entrenamiento del modelo
epochs = 150

history = model.fit(
    trainDataset,
    validation_data=valDataset,
    epochs=epochs,
    callbacks=callbacksList,
    verbose=1
)

# Guardado del modelo completo
finalModelPath = os.path.join(outputDir, "MuuMetricsBcsModel.keras")
model.save(finalModelPath)
print(f"\nModelo completo guardado en: {finalModelPath}")

# Evaluación final en conjunto de prueba
print("\nEvaluación final en conjunto de prueba:")
results = model.evaluate(testDataset, verbose=1)
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# Guardado del modelo completo
finalModelPath = os.path.join(outputDir, "MuuMetricsBcsModel.keras")
model.save(finalModelPath)
print(f"\nModelo completo guardado en: {finalModelPath}")

# Visualización del desempeño del modelo
plt.figure(figsize=(10, 5))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE over Training Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)
plt.show()