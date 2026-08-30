import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import rasterio

class MonitoreoSatelitalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Monitoreo Satelital")
        self.root.geometry("800x600")
        
        # Título
        self.titulo = tk.Label(root, text="Sistema de Monitoreo Satelital", font=("Helvetica", 16))
        self.titulo.pack(pady=10)

        # Botón para cargar imagen
        self.boton_cargar_imagen = tk.Button(root, text="Cargar Imagen", command=self.cargar_imagen)
        self.boton_cargar_imagen.pack(pady=10)

        # Menú de opciones
        self.menu = tk.Frame(root)
        self.menu.pack(pady=20)

        self.boton_incendios = tk.Button(self.menu, text="Detección de Incendios", command=self.detectar_incendios)
        self.boton_incendios.grid(row=0, column=0, padx=10)

        self.boton_clasificacion = tk.Button(self.menu, text="Clasificación de Terrenos", command=self.clasificar_terreno)
        self.boton_clasificacion.grid(row=0, column=1, padx=10)

        self.boton_prediccion = tk.Button(self.menu, text="Predicción de Terrenos", command=self.predecir_terreno)
        self.boton_prediccion.grid(row=0, column=2, padx=10)

        # Crear un contenedor para organizar la imagen y el slider
        self.contenedor = tk.Frame(root)
        self.contenedor.pack(fill="both", expand=True)

        # Panel para mostrar la imagen
        self.panel_resultados = tk.Label(self.contenedor)
        self.panel_resultados.grid(row=0, column=10, padx=200, pady=0)
        
        # Primero creamos el slider
        self.slider = tk.Scale(root, from_=2, to=10, orient="horizontal", label="Número de Clusters (K)",
                               command=self.clasificar_terreno)
        self.slider.set(3)  # Valor inicial
        self.slider.pack(side="bottom", fill="x", padx=10, pady=5)

        # Luego creamos el botón
        self.boton_segmentar = tk.Button(root, text="Segmentar", command=self.clasificar_terreno)
        self.boton_segmentar.pack(side="bottom", padx=10, pady=5)

        # Finalmente, ejecutamos la segmentación inicial (después de definir el slider)
        self.imagen = None
        self.imagen_tk = None
        self.clasificar_terreno()

    def cargar_imagen(self):
        """Abre un diálogo para cargar una imagen."""
        archivo_imagen = filedialog.askopenfilename(title="Seleccionar Imagen", filetypes=(("Archivos TIFF", "*.tiff"), ("Archivos JPG", "*.jpg"), ("Archivos PNG", "*.png")))
        if archivo_imagen:
            # Intentar cargar la imagen
            try:
                with rasterio.open(archivo_imagen) as src:
                    datos = src.read(1)  # lee la primera banda; ajusta según tus bandas

                # Normalizar a 0-255 para poder visualizar/procesar con OpenCV
                datos_norm = cv2.normalize(datos, None, 0, 255, cv2.NORM_MINMAX)
                self.imagen = datos_norm.astype(np.uint8)
            except Exception as e:
                self.imagen = None
                messagebox.showerror("Error", f"No se pudo leer el archivo: {e}")
                return
    
        # Verificar si la imagen fue cargada correctamente
        if self.imagen is None:
            messagebox.showerror("Error", "No se pudo cargar la imagen. Por favor, intente con otro archivo.")
            return
        
        # Convertir la imagen a RGB para Tkinter
        self.imagen = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB)
    
        # Redimensionar la imagen al tamaño del panel
        self.imagen = cv2.resize(self.imagen, (400, 300))
    
        # Convertir la imagen a formato compatible con Tkinter y mostrarla
        self.imagen_tk = ImageTk.PhotoImage(image=Image.fromarray(self.imagen))
        self.panel_resultados.config(image=self.imagen_tk)

    def clasificar_terreno(self, event=None):
        """Clasificar terrenos en la imagen."""
        if self.imagen is None:
            return

        k_clusters = self.slider.get()
        img_reshape = self.imagen.reshape((-1, 3))
        img_reshape = np.float32(img_reshape)

        criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, etiquetas, centros = cv2.kmeans(img_reshape, k_clusters, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS)

        centros = np.uint8(centros)
        imagen_segmentada = centros[etiquetas.flatten()]
        imagen_segmentada = imagen_segmentada.reshape(self.imagen.shape)

        self.imagen_tk = ImageTk.PhotoImage(image=Image.fromarray(imagen_segmentada))
        self.panel_resultados.config(image=self.imagen_tk)

    def detectar_incendios(self):
        """Detectar posibles zonas quemadas/afectadas por incendio usando el NDVI."""
        if self.imagen is None:
            return

        # self.imagen contiene el NDVI ya calculado (rango típico -1 a 1)
        ndvi = self.imagen.astype(np.float32)
        
        # Si el NDVI tiene más de 2 dimensiones (por ejemplo, 3 canales), quedarse con uno solo
        if ndvi.ndim == 3:
            ndvi = ndvi[:, :, 0]  # asume que el NDVI está en el primer canal

        # Normalizar el NDVI a 0-255 solo para poder visualizarlo con un mapa de color
        ndvi_norm = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX)
        imagen_termica = ndvi_norm.astype(np.uint8)

        # Umbral sobre el NDVI real (no sobre la versión normalizada)
        # Valores bajos/negativos de NDVI = vegetación quemada, suelo desnudo, ceniza
        umbral_fuego = -0.1  # valor ajustable entre -1 (quemado) y 1 (vegetación)
        mascara_fuego = ndvi < umbral_fuego

        # Aplicar mapa de color viridis
        imagen_color = cv2.applyColorMap(imagen_termica, cv2.COLORMAP_VIRIDIS)

        # Marcar zonas quemadas en rojo
        imagen_color[mascara_fuego] = [0, 0, 255]  # rojo en BGR

        # Convertir a RGB antes de pasar a PIL (si no, los colores salen invertidos)
        imagen_rgb = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2RGB)

        self.imagen_tk = ImageTk.PhotoImage(image=Image.fromarray(imagen_rgb))
        self.panel_resultados.config(image=self.imagen_tk)

    def predecir_terreno(self):
        """Predicción de terrenos usando un modelo de Machine Learning."""
        datos_entrenamiento = pd.DataFrame({
            'R': np.random.randint(0, 255, 100),
            'G': np.random.randint(0, 255, 100),
            'B': np.random.randint(0, 255, 100),
            'tipo_terreno': np.random.choice([0, 1, 2], 100)
        })

        X = datos_entrenamiento.drop(columns=["tipo_terreno"])
        y = datos_entrenamiento["tipo_terreno"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        predicciones = modelo.predict(X_test)
        precision = accuracy_score(y_test, predicciones) * 100

        messagebox.showinfo("Precisión del Modelo", f"La precisión del modelo es: {precision:.2f}%")

# Iniciar la aplicación
root = tk.Tk()
app = MonitoreoSatelitalApp(root)
root.mainloop()
