from PIL import Image, ImageEnhance
import os
# Correr desde la terminal | python aplicar_brillo.py

# Preguntar rutas.
INPUT_DIR = input("Ingresa la ruta de la carpeta con las imágenes oscuras: ").strip()
OUTPUT_DIR = input("Ingresa la ruta donde quieres guardar las imágenes procesadas: ").strip()

# Factor de brillo.
while True:
    try:
        brightnessFactor = float(input("Ingresa el nivel de brillo (ej. 1.2 = leve, 1.5 = medio, 2.0 = alto... etc.): "))
        break
    except ValueError:
        print("Por favor ingresa un número válido (ej. 1.5).")

# Validar rutas.
if not os.path.isdir(INPUT_DIR):
    print("La carpeta de entrada no existe. Verifica la ruta.")
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# procesar imágenes.
count = 0
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
        imgPath = os.path.join(INPUT_DIR, filename)
        img = Image.open(imgPath)

        # Aumentar brillo
        enhancer = ImageEnhance.Brightness(img)
        brightImg = enhancer.enhance(brightnessFactor)

        # Guardar imagen procesada
        outputPath = os.path.join(OUTPUT_DIR, filename)
        brightImg.save(outputPath)
        count += 1

        print(f"✅ {filename} procesada")

print(f"\nProceso completado. {count} imágenes guardadas en: {OUTPUT_DIR}")