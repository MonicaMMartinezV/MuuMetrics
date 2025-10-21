### LimpiarVacas.py | Multi-ETL Para el Proyecto MuuMetrics



A continuación, se describirá la funcionalidad del archivo LimpiarVacas.py. Un archivo con la utilidad de limpiar los datos de imágenes de vacas proporcionados por el CAETEC. Se pueden modificar los valores, pero es preferible presionar "Enter" para elegir los valores default.



💡 Paso 1 / Función 1: Quitar imágenes oscuras | Quita las imágenes que tienen un porcentaje de pixeles sobre un grado de oscuridad específico (ej. 40, 0.8; 20, 0.95). Default = (40, 0.8)



🐄 Paso 2 / Función 2: Quitar imágenes sin vacas | Quita todas las imágenes que no tienen vacas usando un algoritmo de detección con YOLOv8.



🔍 Paso 3: Quitar manualmente imágenes donde no se vean los factores necesarios para la clasificación del BCS.



💡 Paso Brillo / Función 3: Para recuperación de imágenes oscuras filtradas por la función 1 | A través de la aplicación de un filtro de brillo con un valor elegido (en este caso, 5). Después, aplicar los pasos 2 y 3 para recuperar las imágenes necesarias.



### Instrucciones



1. Instalar las dependencias en la terminal CMD.



&nbsp;	pip install opencv-python numpy pillow tqdm ultralytics



2\. Correr el archivo en la ubicacion del directorio con el comando:



&nbsp;	python limpiarVacas.py



3\. Seguir las instrucciones en la terminal para cumplir los pasos necesarios.

