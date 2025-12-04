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

## 1. Entendimiento del negocio

Comprensión del problema, valor, métricas y alcance del sistema.

* [Costos y beneficios de la propuesta](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Business%20understanding/Documents/Costos%20y%20beneficios%20de%20la%20propuesta.pdf)

* [Memorandum of Understanding](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Business%20understanding/Documents/Memor%C3%A1ndum%20de%20entendimiento.pdf)

* [One Page](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Business%20understanding/Documents/One%20Page.pdf)

* [Plan de proyecto](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Business%20understanding/Documents/Plan%20de%20proyecto.pdf)

* [Requisitos, supuestos y restricciones](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Business%20understanding/Documents/Requisitos%2C%20supuestos%20y%20restricciones.pdf)

* [Riesgos y Contingencias](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Business%20understanding/Documents/Riesgos%20y%20Contingencias.xlsx)

* [Terminología](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Business%20understanding/Documents/Terminolog%C3%ADa.pdf)

---

## **2. Entendimiento de datos**

Exploración inicial, calidad, estructura, fuentes y análisis preliminar.

* [Reporte inicial de descripción, calidad y exploración de datos](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Data%20understanding/Documents/Reporte%20inicial%20de%20descripci%C3%B3n%2C%20calidad%20y%20exploraci%C3%B3n%20de%20datos.pdf)

---

## **3. Preparación de datos**

Limpieza, anonimización, normalización, construcción del dataset final y generación de inputs para los modelos.

* [Reporte de preparación de datos](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Data%20preparation/Documents/Reporte%20de%20preparaci%C3%B3n%20de%20datos.pdf)

## **4. Modelado**

Generación de herramientas de modelado.

* [Reporte de modelado](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Modeling/Documents/Reporte%20de%20modelado.pdf)

---

## **5. Evaluación**

Integración con DEL + reglas del negocio: semáforo de salud.

* [Reporte de evaluación](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Evaluation/Documents/Reporte%20de%20evaluaci%C3%B3n.pdf)

## **6. Despliegue**

Despliegue de resultados

* [Reporte final](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Deployment/Documentos/Reporte%20Final.pdf)

* [Plan de entrega](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Deployment/Documentos/Plan%20de%20entrega.pdf)

* [Plan de monitoreo y mantenimiento](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Deployment/Documentos/Plan%20de%20monitoreo%20y%20mantenimiento.pdf)

* [Documentación de la experiencia](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Deployment/Documentos/Documentaci%C3%B3n%20de%20la%20Experiencia.pdf)

* [Guía de despliegue](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Deployment/Documentos/Gu%C3%ADa%20de%20Despliegue.pdf)

* [One Page para socio formador](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Deployment/Documentos/One%20Page%20para%20los%20Socio%20Formadores.pdf)

---

## **Privacidad y seguridad de datos**

* [Log de seguridad de acceso MuuMetrics](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Data%20privacy%20and%20security/Log%20de%20Seguridad%20y%20Acceso%20MuuMetrics.pdf)

* [Políticas de privacidad y seguridad de los datos MuuMetrics](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Data%20privacy%20and%20security/PrivacidadyseguridaddelosdatosMuuMetrics.pdf)

* [Tablas de acceso](https://github.com/MonicaMMartinezV/MuuMetrics/blob/main/Data%20privacy%20and%20security/Tabla%20de%20Accesos.pdf)

## Respaldos documentación

* [Respaldo Notion](https://www.notion.so/Backups-Notion-2bf1c55762fc8092b12ef38bcdf84f92)

* [Respaldo Drive](https://drive.google.com/drive/u/5/folders/1a3VMGGWeMT5UpOKqpeJ7kKPpBm-s-3J7)


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

## Correcciones realizadas

### Módulo de Reto

Se aplicó el enfoque iterativo de la metodolgía CRISP-DM. 

Se documentó la adaptación de la metodología CRISP-DM. 

Se hizo revisión de la ortografía y redacción de los documentos. 

Se añadieron accesos directos a cada entregable por fase de la metodología en el README.md del repositorio del equipo. 

Se corrigió la introducción del reporte de modelado, haciendo más explícito el propósito de cada modelo desarrolado. 

Se añadieron accesos directos al documento de logs auditables, bitacora de cambios y tablas de acceso en el reporte grupal de privacidad de datos. 

### Módulo de Cómputo en la nube

Se añadió una sección resumiendo las observaciones del reporte. 

Se agregaron imágenes que retratan el tratamiento de los datos. 

Se agregaron referencias al contenido. 

Se agregó justificación de las herramientas a utilizar en cada fase del proyecto. 

Se añadió jsutificación al digrama de despliegue. 