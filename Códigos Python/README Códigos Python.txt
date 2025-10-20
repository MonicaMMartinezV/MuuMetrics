Ejemplos de imágenes filtradas después de cada paso:

Paso 1: Quitar imágenes oscuras (quitarOscuras.py) | Quita las imágenes que tienen un porcentaje de pixeles sobre un grado de oscuridad específico (ej. 40, 0.8; 20, 0.95).

Imagen de ejemplo descartada: 2025-05-25-19-58-44_cam4_cap1

Paso 2: Quitar imágenes sin vacas (detectarVacas.py) | Quita todas las imágenes que no tienen vacas usando un algoritmo de detección con YOLOv8.

Imagen de ejemplo descartada: 2025-05-26-00-21-24_cam0_cap4

Paso 3: Quitar manualmente imágenes donde no se vean los factores necesarios para la clasificación del BCS.

Imagen de ejemplo descartada: 2025-05-27-07-45-40_cam0_cap1

Paso Brillo: Recuperar imágenes oscuras (aplicarBrillo.py) | A través de la aplicación de un filtro de brillo con un valor elegido (en este caso, 5). Después, aplicar los pasos 2 y 3 para recuperar las imágenes necesarias.

Imagen de ejemplo recuperada: 2025-05-27-22-47-30_cam0_cap1

Ejemplo de imagen utilizada: 2025-05-26-07-23-34_cam0_cap1