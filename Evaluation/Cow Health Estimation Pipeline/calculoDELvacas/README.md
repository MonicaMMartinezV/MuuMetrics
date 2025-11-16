# calculoDELvacas.py

## Descripción general
El script **`calculoDELvacas.py`** calcula los **Días en Leche (DEL)** más cercanos al momento en que fue tomada la imagen de una vaca.  
Combina los registros de ordeña de múltiples vacas (cada uno almacenado en un archivo `.csv` independiente) y los compara con las marcas de tiempo de las fechas más cercanas a las imágenes mostradas en otro archvo `.csv` para identificar qué vaca y qué sesión de ordeña corresponde a la imagen de una vaca.

---

## Funcionamiento del script

1. **Lectura y combinación de archivos CSV**
   - El script busca todos los archivos `.csv` dentro del directorio definido por `pathCows`.
   - Cada archivo representa a una vaca, y su nombre de archivo se utiliza como identificador (ID) de la vaca.
   - Detecta la fila correcta de encabezados, limpia los datos y combina todos los archivos en un solo *DataFrame*.

2. **Limpieza y conversión de columnas de fecha**
   - Las horas de inicio de ordeña (`Hora de inicio`) se estandarizan y se convierten al formato de fecha y hora (`datetime`).
   - Cualquier dato faltante o inválido se reporta al usuario.

3. **Procesamiento de marcas de tiempo de imágenes**
   - Se utiliza una lista de nombres de archivo simulados (por ejemplo: `2025-06-01-21-47-55_cam4_cap3`), los cuales se convierten en objetos `datetime`, útil en caso de que las imágenes se procesen en lote.
   - El script valida los nombres de las imágenes y muestra cuáles son válidos o inválidos.

4. **Lectura del conjunto de datos “Patadas”**
   - Se lee el archivo definido en `pathPatadas`.
   - Este archivo debe contener las siguientes columnas:
     - `Número del animal`
     - `DEL`
     - `Hora Inicio Ordeño`

5. **Búsqueda de la vaca más cercana en tiempo**
   - Para una marca de tiempo objetivo (de la imagen), el script identifica cuál registro de ordeña (`Hora de inicio`) está más próximo en el tiempo.
   - Después, obtiene el valor de **DEL** correspondiente a esa vaca y muestra:
     - El ID de la vaca
     - La fecha y hora más cercanas coincidentes
     - El valor de **DEL** calculado que corresponde a la fecha de la imagen

6. **Cálculo del DEL ajustado**
   - El valor de DEL se ajusta con base en la diferencia (en días) entre el registro de ordeña y la marca de tiempo de la imagen.
   - El resultado indica cuántos **días en leche** tenía la vaca al momento en que se tomó la imagen.

---

## Requisitos

El script utiliza las siguientes bibliotecas de Python:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `glob`
- `datetime`
- `os`

Puedes instalarlas con el siguiente comando:

```bash
pip install numpy pandas matplotlib seaborn glob datetime os

