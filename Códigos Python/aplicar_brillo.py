from PIL import Image, ImageEnhance
import os
# Correr desde la terminal | python aplicar_brillo.py

# Preguntar rutas.
input_folder = input("Ingresa la ruta de la carpeta con las imágenes oscuras: ").strip()
output_folder = input("Ingresa la ruta donde quieres guardar las imágenes procesadas: ").strip()

# Factor de brillo.
while True:
    try:
        brightness_factor = float(input("Ingresa el nivel de brillo (ej. 1.2 = leve, 1.5 = medio, 2.0 = alto... etc.): "))
        break
    except ValueError:
        print("Por favor ingresa un número válido (ej. 1.5).")

# Validar rutas.
if not os.path.isdir(input_folder):
    print("La carpeta de entrada no existe. Verifica la ruta.")
    exit()

os.makedirs(output_folder, exist_ok=True)

# procesar imágenes.
count = 0
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # Aumentar brillo
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(brightness_factor)

        # Guardar imagen procesada
        output_path = os.path.join(output_folder, filename)
        bright_img.save(output_path)
        count += 1

        print(f"✅ {filename} procesada")

print(f"\nProceso completado. {count} imágenes guardadas en: {output_folder}")
