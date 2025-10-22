# Guía para ejecutar `data_augmentation.py`

## 1. Descripción

Este script realiza lo siguiente:

- Carga un dataset de imágenes organizado por carpetas (una carpeta por clase).
- Aplica augmentación de datos (rotación aleatoria, flip horizontal, normalización y conversión a grayscale).
- Visualiza ejemplos originales y aumentados.
- Guarda las imágenes aumentadas en un directorio de salida separado, fuera del dataset original, para evitar interferencias con las clases.

---

## 2. Estructura de carpetas requerida

Antes de ejecutar el script, organiza tus imágenes así:

├── 2/
│ ├── img1.jpg
│ ├── img2.jpg
├── 2.25/
│ ├── img3.jpg
│ └── ...
├── 2.5/
│ └── ...

Después de ejecutar, se creará:

output/
├── train/
├── val/
└── augmented/

## 3. Dependencias

Asegúrate de tener instaladas las siguientes librerías:

```bash
pip install numpy matplotlib tensorflow
```

4. Configuración del script

Modifica la ruta baseDir si tu dataset está en otra ubicación:

```Python
baseDir = Path("./Codigos Python/Batches para correr el codigo/"
               "04. Batch Imagenes Clasificadas DataAugmentation")
```

outputDir se genera automáticamente fuera del dataset, en la misma carpeta que 04. Batch Imagenes Clasificadas DataAugmentation.

5. Ejecución del script

Desde la terminal o Jupyter Notebook:

```bash
python data_augmentation.py
```
El script mostrará por pantalla:

La ruta absoluta de tu dataset.

Si el directorio existe.

Una visualización de 5 imágenes originales y sus versiones aumentadas.

Luego guardará las imágenes aumentadas en:

output/augmented/

6. Ajustes opcionales

Cambiar tamaño de imagen:

```Python
image_size=(1080, 1080)  # Cambia según tus necesidades
```

Cambiar porcentaje de validación:

```Python
validation_split=0.3
```
Aplicar o quitar augmentación:

```Python
saveImagesFromDataset(
    trainDs,
    augmentedOutputDir,
    applyAugmentation=True,  # False para solo guardar originales
    augmentationLayer=dataAugmentation
)
```

Limitar número de imágenes guardadas:

```Python
maxImages=100  # solo guarda 100 imágenes
```
7. Notas

Las imágenes se guardan en carpetas nombradas según un mapa de etiquetas calculado automáticamente.

La augmentación incluye:

Normalización (Rescaling(1./255))

Rotación aleatoria (RandomRotation(0.4))

Flip horizontal (RandomFlip("horizontal"))

Conversión a grayscale con 3 canales (Lambda + tf.tile)
