# **MuuMetrics – Sistema inteligente de evaluación temprana de salud bovina**

MuuMetrics es un sistema de inteligencia artificial desarrollado para evaluar la salud de vacas Holstein mediante el análisis conjunto de su Body Condition Score (BCS) y sus Días en Leche (DEL).
El proyecto integra procesamiento de imágenes, modelos de Deep Learning y reglas de negocio para generar un semáforo BCS–DEL que permite identificar de forma temprana posibles riesgos metabólicos en el ganado del CAETEC.

Este repositorio contiene el código, pipelines técnicos y modelos.sss
La documentación completa (reportes CRISP-DM, exceles, PDFs, acuerdos, dataset, etc.) se encuentra en [Google Drive](https://drive.google.com/drive/folders/1xQ-WnKvpaFhXdCvodIKwVP_DjKes0lJr?usp=sharing) y en [Notion](), que estan enlazados en este README para fácil acceso.

---
# Documentación del proyecto por fases – CRISP DM

## Business Understanding

Comprensión del problema, valor, métricas y alcance del sistema.

Si quieres saber como es que adaptamos la metodologia a nuestro proyecto consulta [aqui]()

* [Reporte: Business Understanding]()

* Plan de Proyecto Completo (PDF)

* Objetivo de negocio:
  Determinar el estado de salud de las vacas del corral 6 del CAETEC mediante la evaluación integrada de **BCS + DEL**, apoyando decisiones técnicas en el hato.

* **Criterios de éxito (negocio):**

  * ≥ 80% coincidencia con valoración real o ±0.5 BCS considerando DEL.

---

## **Data Understanding**

Exploración inicial, calidad, estructura, fuentes y análisis preliminar.

* **Reporte: Data Understanding**

* **Dataset original desde CAETEC (Drive)**

* **Exploración inicial (notebooks/Tableau)**

---

## **Data Preparation**

Limpieza, anonimización, normalización, construcción del dataset final y generación de inputs para los modelos.

### Contenido en el repositorio

* Scripts de filtrado (eliminar imágenes sin vaca / borrosas / duplicadas).
* Métodos propios de data augmentation.
* Módulos para estandarizar proporciones, encuadres y formatos.

### Documentación externa

* **Reporte: Data Preparation**

* **Política de anonimización y manejo de datos (incluye SINIIGA)**

* **Pipeline completo ETL**

---

## **Modeling**

Aquí se encuentran TODOS tus modelos, organizados tal como lo tienes en GitHub.

### 4.1. Modelos secundarios

1. **Detección de vaca presente**
   Ubicación: 
   Objetivo: descartar imágenes sin vaca.

2. **Auto-crop para aislar la vaca**
   Ubicación:
   Objetivo: remover background y centrar la vaca.

### 4.2. Modelo principal (BCS)

Ubicación:

Incluye TODAS las versiones del modelo primario entrenado para predecir BCS.

### Documentación externa

* **Reporte: Modeling (completo)**
  
* **Comparación de modelos y experimentos**
  
* **Criterios de éxito técnicos:**
  * MAE train/test ≤ 0.4
  * Bias entre ±0.5 BCS (con DEL)
  * Varianza entre 0.05 y 0.3

---

## **Evaluation**

Integración con DEL + reglas del negocio: semáforo de salud.

### Contenido en el repositorio

* `/evaluation/semaforo_bcs_del/` → generador del semáforo
* `/evaluation/generacion_json/` → output para uso futuro

### 🔹 Documentación externa

* **Reporte: Evaluation**

* **Reglas del semáforo BCS-DEL (business rules)**

---

## **Deployment**

Guías y entregables de implementación (futuro despliegue).

* **Guía de Despliegue (PDF)**
  
* **Demo del sistema (video)**

---

# Objetivos del Proyecto

### **Objetivo de Negocio**

Determinar el estado de salud de cada vaca del corral 6 correlacionando su **BCS** con sus **Días en Leche (DEL)**.

### **Objetivo de Minería de Datos**

Categorizar a las vacas según su BCS utilizando imágenes y complementar esa clasificación con DEL para obtener un índice de riesgo metabólico (semáforo).

---

# Getting started

## Prerequisitos

* Python 3.10
* 8 GB RAM
* GPU opcional (acelera entrenamiento)

## Instalación

```bash
git clone https://github.com/MonicaMMartinezV/MuuMetrics.git
cd MuuMetrics
pip install -r requirements.txt
```

---

# Equipo

| Nombre                                | Matrícula     |
| ------------------------------------- | ------------- |
| Ulises Orlando Carrizales             | A01027715     |
| Mónica Monserrat Martínez Vásquez     | A01710965     |
| María José Soto Castro                | A01705840     |
| Tomás Pérez Vera                      | A01028008     |
| Grant Keegan                          | A01700753     |
| Bárbara Alcántara                     | A01799609     |

---

# Notion general del proyecto

[https://www.notion.so/MuuMetrics-2791c55762fc8068b8fdccd1ea386017](https://www.notion.so/MuuMetrics-2791c55762fc8068b8fdccd1ea386017)