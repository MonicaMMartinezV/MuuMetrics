# generateGraph.py
# Script mínimo para generar una gráfica BCS vs DEL por ID de vaca
# con logs para depuración.

import sys
import os
import json
import matplotlib.pyplot as plt

def main():
    print(">>> Script generateGraph.py iniciado")

    # 1) Validar argumentos
    if len(sys.argv) != 4:
        print("Uso: python generateGraph.py <dataset.json> <ID_vaca> <output.png>")
        print(f"Argumentos recibidos: {sys.argv}")
        sys.exit(1)

    json_path = sys.argv[1]
    cow_id = sys.argv[2]       # lo tratamos como string
    output_path = sys.argv[3]

    print(f">>> json_path = {json_path}")
    print(f">>> cow_id    = {cow_id}")
    print(f">>> output    = {output_path}")

    # 2) Verificar que el JSON existe
    if not os.path.exists(json_path):
        print(f"ERROR: No se encontró el archivo JSON en {json_path}")
        sys.exit(1)

    # 3) Cargar JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR al leer el JSON: {e}")
        sys.exit(1)

    # 4) Extraer columnas esperadas
    try:
        IDs = data["ID"]
        DEL = data["DEL"]
        BCS = data["BCS"]
        Sem = data["Semaforo"]
    except KeyError as e:
        print(f"ERROR: Falta la clave en el JSON: {e}")
        sys.exit(1)

    print(">>> Claves cargadas: ID, DEL, BCS, Semaforo")

    cow_days = []
    cow_bcs = []
    cow_colors = []

    # 5) Recorrer todos los registros
    for idx_str, id_val in IDs.items():
        # id_val puede ser int, cow_id viene como string
        if str(id_val) == str(cow_id):
            day = DEL[idx_str]
            bcs = BCS[idx_str]
            sem = Sem[idx_str].lower()

            cow_days.append(day)
            cow_bcs.append(bcs)

            # colores por semáforo
            if sem == "green":
                cow_colors.append("#2ecc71")
            elif sem == "yellow":
                cow_colors.append("#f1c40f")
            else:
                cow_colors.append("#e74c3c")

    print(f">>> Registros encontrados para vaca {cow_id}: {len(cow_days)}")

    # 6) Si no hay datos, generar una gráfica vacía pero con mensaje
    if len(cow_days) == 0:
        print(f"ADVERTENCIA: No se encontraron datos para la vaca {cow_id}")
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, f"Sin datos para ID {cow_id}",
                 ha="center", va="center", fontsize=12)
        plt.axis('off')
    else:
        # Ordenar por DEL
        combined = sorted(zip(cow_days, cow_bcs, cow_colors), key=lambda x: x[0])
        cow_days, cow_bcs, cow_colors = zip(*combined)

        # 7) Graficar
        plt.figure(figsize=(8, 4))
        plt.scatter(cow_days, cow_bcs, c=cow_colors, s=70, edgecolor="black", linewidth=0.8)
        plt.plot(cow_days, cow_bcs, linestyle="--", color="#34495e")

        plt.title(f"Histórico DEL vs BCS - Vaca {cow_id}")
        plt.xlabel("Días en leche (DEL)")
        plt.ylabel("BCS")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.ylim(1.5, 4.5)

    # 8) Asegurar carpeta de salida
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 9) Guardar imagen
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=140)
        print(f">>> Gráfica guardada en: {output_path}")
    except Exception as e:
        print(f"ERROR al guardar la imagen: {e}")
        sys.exit(1)
    finally:
        plt.close()

    print(">>> Script generateGraph.py finalizado OK")

if __name__ == "__main__":
    main()