# **MuuMetrics – Sistema inteligente de evaluación temprana de salud bovina**

MuuMetrics es un sistema de inteligencia artificial desarrollado para evaluar la salud de vacas Holstein mediante el análisis conjunto de su Body Condition Score (BCS) y sus Días en Leche (DEL).
El proyecto integra procesamiento de imágenes, modelos de Deep Learning y reglas de negocio para generar un semáforo BCS–DEL que permite identificar de forma temprana posibles riesgos metabólicos en el ganado del CAETEC.

Este repositorio contiene el código, pipelines técnicos y modelos.sss
La documentación completa (reportes CRISP-DM, exceles, PDFs, acuerdos, dataset, etc.) se encuentra en [Google Drive](https://drive.google.com/drive/folders/1xQ-WnKvpaFhXdCvodIKwVP_DjKes0lJr?usp=sharing) y en [Notion](https://gray-seaplane-cef.notion.site/MuuMetrics-2791c55762fc8068b8fdccd1ea386017), que estan enlazados en este README para fácil acceso.

Si quieres saber como es que adaptamos la metodologia a nuestro proyecto consulta [aqui](https://gray-seaplane-cef.notion.site/Entregables-CRISP-DM-2791c55762fc80319723db4e15c296ff?pvs=25#2791c55762fc80928bbbc73019c9c506)



# Objetivos del Proyecto

### **Objetivo de Negocio**

Determinar el estado de salud y condición corporal de vacas lecheras del corral seis del CAETEC.

### **Objetivo de Minería de Datos**

Categorizar las vacas con base en el Body Condition Score (BCS) mediante análisis de imágenes y  días en leche (DEL).

---
# Documentación del proyecto por fases – CRISP DM

## 1. Business Understanding

Comprensión del problema, valor, métricas y alcance del sistema.

* [Costos y beneficios de la propuesta](https://docs.google.com/document/d/1axHWBpxGsrn9QVdPUJeygye9eDbLdAcZ7ko9TrVk25Q/edit?usp=drive_link)

* [Memorandum of Understanding](https://docs.google.com/document/d/1tYcf_IZtEkEOe9UCz-_T2OchvPhVjKgmgM2hHVNGLng/edit?usp=drive_link)

* [One Page](https://docs.google.com/document/d/14SVaxU6H1D9KXOrdEGhBX7wAiB8Rpp0HaM8QdhwLCMM/edit?usp=drive_link)

* [Plan de proyecto](https://docs.google.com/document/d/1GDUuPcwiWfw8lML1qjwhfHZnJxiaok3kdK2LeJ2aGeU/edit?usp=drive_link)

* [Requisitos, supuestos y restricciones](https://docs.google.com/document/d/1Ntk5xScPihZ8uVgNhvC22ws1OZoXs8glfD6ZlE7sbc0/edit?usp=drive_link)

* [Riesgos y Contingencias](https://docs.google.com/spreadsheets/d/1EoaOeYx3wXS2SlOxsyF6TGpr9R5Bg61pAWiOXN_AG3s/edit?usp=drive_link)

* [Terminología](https://docs.google.com/document/d/1neou0geSbUkJyFNGNxVD9ZDgjk0CXehBwzSbDcNRqlo/edit?usp=drive_link)

---

## **2. Data Understanding**

Exploración inicial, calidad, estructura, fuentes y análisis preliminar.

* [Reporte inicial de descripción, calidad y exploración de datos](https://docs.google.com/document/d/1WhQmHc2qq-8_rN0itn5pkU3iVGKg_o5az70WhDmiizo/edit?usp=drive_link)

---

## **3. Data Preparation**

Limpieza, anonimización, normalización, construcción del dataset final y generación de inputs para los modelos.

* [Reporte de preparación de datos](https://docs.google.com/document/d/1qIsYqbSzOVww6H6zMj_zZK4gRgvr10D3PGJZjH1DQJU/edit?usp=drive_link)

## **4. Modeling**

Generación de herramientas de modelado.

* [Reporte de modelado](https://docs.google.com/document/d/1zcVGs0B-E2b9gopeleu4nHDKnNw2mz7b_IlUiWJcM_M/edit?usp=drive_link)

---

## **5. Evaluation**

Integración con DEL + reglas del negocio: semáforo de salud.

* [Reporte de evaluación](https://docs.google.com/document/d/1zdos6Q2AgchOKDK-zVP-26Q-k1CCMgpIdLVJDA_MHdg/edit?usp=drive_link)


## Prerequisitos

* Python 3.10
* 8 GB RAM
* GPU opcional (acelera entrenamiento)

## Instalación

```bash
git clone https://github.com/MonicaMMartinezV/MuuMetrics.git
cd MuuMetrics
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
