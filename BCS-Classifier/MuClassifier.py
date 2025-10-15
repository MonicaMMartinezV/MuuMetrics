import os
import shutil
import tkinter  as tk
from tkinter    import filedialog, messagebox
from PIL        import Image, ImageTk

class ImageSorter:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Imágenes")
        self.root.geometry("800x600")
        self.root.iconbitmap("MUU.ico")

        self.folder_path    = ""
        self.images         = []
        self.current_index  = 0

        self.start_x = None
        self.start_y = None
        self.rect = None
        self.box = None
        
        self.set_box = False
        
        # ----- INTERFAZ -----
        self.label_info = tk.Label(root, text="MU CLASSIFIER", font=("Arial", 20))
        self.label_info.pack(pady=10)

        self.canvas = tk.Canvas(root, width=600, height=400, bg="gray")
        self.canvas.pack(pady=10)

        self.entry_label = tk.Label(root, text="Número de etiqueta:", font=("Arial", 12))
        self.entry_label.pack()

        self.entry = tk.Entry(root, font=("Arial", 12))
        self.entry.pack(pady=5)

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=10)

        self.btn_select = tk.Button(self.btn_frame, text="Seleccionar carpeta", command=self.select_folder)
        self.btn_select.grid(row=0, column=0, padx=5)

        self.btn_move = tk.Button(self.btn_frame, text="Mover imagen", command=self.move_image)
        self.btn_move.grid(row=0, column=1, padx=5)

        # Vincular eventos del mouse
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    # ----- FUNCIONES -----
    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path = folder
            self.images      = [f for f in os.listdir(folder)
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            self.current_index = 0
            if not self.images:
                messagebox.showwarning("Sin imágenes", "La carpeta no contiene imágenes.")
            else:
                self.show_image()
                self.set_box = True

    def show_image(self):
        if 0 <= self.current_index < len(self.images):
            image_path = os.path.join(self.folder_path, self.images[self.current_index])
            img        = Image.open(image_path)
            self.img_width, self.img_height = img.size
            img.thumbnail((600, 400))
            self.tk_image = ImageTk.PhotoImage(img)
            self.canvas.create_image(300, 200, image=self.tk_image)
            self.label_info.config(text=f"Imagen {self.current_index+1} de {len(self.images)}: {self.images[self.current_index]}")
            self.boxes = []
        else:
            self.canvas.delete("all")
            self.label_info.config(text="No hay más imágenes.")

    
    def on_press(self, event):
        if self.set_box:
            if self.box:
                self.canvas.delete(self.box)
                self.box = None

            self.start_x = event.x
            self.start_y = event.y
            self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)

    def on_drag(self, event):
        if self.set_box:
            if self.rect:
                self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        if self.set_box:
            end_x, end_y = event.x, event.y

            # Guardar la caja final y eliminar rectángulos temporales
            if self.rect:
                self.box = self.rect
                self.rect = None
                self.box_coords = (self.start_x, self.start_y, end_x, end_y)
            else:
                self.box_coords = None

    def save_annotation(self):
        if not hasattr(self, "box_coords") or self.box_coords is None:
            messagebox.showwarning("Aviso", "Dibuja un rectángulo antes de guardar.")
            return

        x1, y1, x2, y2 = self.box_coords

        # Convertir a formato YOLO (normalizado a tamaño del canvas)
        x_center = ((x1 + x2) / 2) / 800
        y_center = ((y1 + y2) / 2) / 600
        w = abs(x2 - x1) / 800
        h = abs(y2 - y1) / 600
        class_id = 0

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

        old_name = self.images[self.index]
        old_path = os.path.join(self.folder, old_name)
        ext = os.path.splitext(old_name)[1]
        new_filename = yolo_line + ext
        new_path = os.path.join(self.folder, new_filename)

        # Renombrar imagen
        os.rename(old_path, new_path)

        self.start_x = None
        self.start_y = None
        self.rect = None
        self.box = None
        self.box_coords is None

    def move_image(self):
        if not self.images:
            messagebox.showinfo("Aviso", "No hay imágenes cargadas.")
            return

        label = self.entry.get().strip()
        if not label:
            messagebox.showwarning("Error", "Por favor ingresa un número o etiqueta.")
            return
        elif float(label)<1 or float(label)>5:
            messagebox.showwarning("Error", "El limite de etiqueta es entre 1 y 5")
            return
        elif float(label)%0.25!=0:
            messagebox.showwarning("Error", "Recuerda que va en cuartiles")
            return

        src         = os.path.join(self.folder_path, self.images[self.current_index])
        dest_folder = os.path.join(self.folder_path, label)
        os.makedirs(dest_folder, exist_ok=True)

        dest = os.path.join(dest_folder, self.images[self.current_index])
        shutil.move(src, dest)

        self.current_index += 1
        if self.current_index < len(self.images):
            self.show_image()
        else:
            self.canvas.delete("all")
            self.label_info.config(text="Todas las imágenes han sido clasificadas.")
            self.images = []

# ----- MAIN -----
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSorter(root)
    root.mainloop()
