import os

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class YOLOLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Etiquetador YOLO")
        self.root.geometry("900x700")

        self.folder = ""
        self.images = []
        self.index = 0

        self.start_x = None
        self.start_y = None
        self.rect = None
        self.boxes = []  # (x1, y1, x2, y2)

        # Widgets
        self.canvas = tk.Canvas(root, width=800, height=600, bg="gray")
        self.canvas.pack(pady=10)

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack()

        self.btn_select = tk.Button(self.btn_frame, text="Seleccionar carpeta", command=self.select_folder)
        self.btn_select.grid(row=0, column=0, padx=5)

        self.btn_next = tk.Button(self.btn_frame, text="Siguiente imagen", command=self.next_image)
        self.btn_next.grid(row=0, column=1, padx=5)

        self.btn_save = tk.Button(self.btn_frame, text="Guardar anotaciones", command=self.save_annotations)
        self.btn_save.grid(row=0, column=2, padx=5)

        # Vincular eventos del mouse
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    # --- Funciones principales ---
    def select_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.folder = folder
        self.images = [f for f in os.listdir(folder)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.index = 0
        self.load_image()

    def load_image(self):
        if not self.images:
            return
        path = os.path.join(self.folder, self.images[self.index])
        img = Image.open(path)
        self.img_width, self.img_height = img.size
        img.thumbnail((800, 600))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(400, 300, image=self.tk_img)
        self.boxes = []

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)

    def on_drag(self, event):
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        end_x, end_y = event.x, event.y
        self.boxes.append((self.start_x, self.start_y, end_x, end_y))

    def save_annotations(self):
        if not self.boxes:
            return

        # Convertir a formato YOLO
        yolo_lines = []
        for (x1, y1, x2, y2) in self.boxes:
            # Normalizar coordenadas
            x_center = ((x1 + x2) / 2) / 800
            y_center = ((y1 + y2) / 2) / 600
            w = abs(x2 - x1) / 800
            h = abs(y2 - y1) / 600
            class_id = 0  # puedes cambiar esto según la clase
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # Guardar en txt con el mismo nombre
        img_name = os.path.splitext(self.images[self.index])[0]
        txt_path = os.path.join(self.folder, f"{img_name}.txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))
        print(f"Anotaciones guardadas en: {txt_path}")

    def next_image(self):
        if not self.images:
            return
        self.index = (self.index + 1) % len(self.images)
        self.load_image()

# ---- MAIN ----
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOLabeler(root)
    root.mainloop()
