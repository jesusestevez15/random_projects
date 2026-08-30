import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import rasterio

# 🌍 Clase base para representar imágenes satelitales
class ImagenSatelital:
    def __init__(self, ruta_imagen):
        self.ruta_imagen = ruta_imagen
        self.imagen = cv2.imread(ruta_imagen, cv2.IMREAD_UNCHANGED)  # Carga la imagen
        if self.imagen is None:
            raise ValueError(f"No se pudo cargar la imagen en {ruta_imagen}")
    
    def tamaño_imagen(self):
        return self.imagen.shape[0], self.imagen.shape[1]
    
    def obtener_imagen(self):
        """Devuelve la imagen para mostrarla luego."""
        return cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB)  # Convertir a RGB

# 🌿 Clase derivada para analizar vegetación con NDVI
class ImagenNDVI(ImagenSatelital):
    def __init__(self, ruta_imagen, imagen_banda_roja, imagen_banda_infrarroja):
        super().__init__(ruta_imagen)
        self.imagen_banda_roja = imagen_banda_roja
        self.imagen_banda_infrarroja = imagen_banda_infrarroja

    def calcular_ndvi(self):
        """Calcula el NDVI (Índice de Vegetación de Diferencia Normalizada)."""
        # Evitar división por cero con np.where
        with rasterio.open(self.imagen_banda_roja) as src:
            banda_roja = src.read(1)  # Lee la primera banda
        with rasterio.open(self.imagen_banda_infrarroja) as src:
            banda_infrarroja = src.read(1)  # Lee la primera banda

        # Calculando el NDVI
        ndvi = (banda_infrarroja - banda_roja) / (banda_infrarroja + banda_roja)
        return ndvi

    def obtener_ndvi_visualizacion(self):
        """Devuelve la imagen NDVI para mostrarla luego."""
        ndvi = self.calcular_ndvi()
        return ndvi

# 🔍 Clase para análisis de cambios entre dos imágenes
class AnalisisCambio:
    def __init__(self, imagen1, imagen2):
        self.imagen1 = cv2.imread(imagen1, cv2.IMREAD_GRAYSCALE)
        self.imagen2 = cv2.imread(imagen2, cv2.IMREAD_GRAYSCALE)
        if self.imagen1 is None:
            raise ValueError("Error al cargar las imágenes 1 para comparación.")
        if self.imagen2 is None:
            raise ValueError("Error al cargar las imágenes 2 para comparación.")

    def diferencia(self):
        """Calcula la diferencia entre dos imágenes satelitales."""
        return cv2.absdiff(self.imagen1, self.imagen2)

    def obtener_diferencia(self):
        """Devuelve la imagen de diferencias."""
        return self.diferencia()

# 🚀 PRUEBA DEL CÓDIGO (Asume imágenes de satélite disponibles)
# NOTA: Para probar necesitas imágenes de satélite en escala de grises para NDVI

# Cargar imágenes de prueba (debes reemplazar con archivos reales)
imagen_satelite = ImagenSatelital("2020-07-01-00_00_2020-07-01-23_59_Sentinel-2_Quarterly_Mosaics_NDVI.tiff")
ancho, alto = imagen_satelite.tamaño_imagen()
imagen_rgb_1 = imagen_satelite.obtener_imagen()
imagen_satelite = ImagenSatelital("2024-07-01-00_00_2024-07-01-23_59_Sentinel-2_Quarterly_Mosaics_NDVI.tiff")
imagen_rgb_2 = imagen_satelite.obtener_imagen()

# Crear objeto de NDVI
ndvi1 = ImagenNDVI("2020-07-01-00_00_2020-07-01-23_59_Sentinel-2_Quarterly_Mosaics_NDVI.tiff", "2020-07-01-00_00_2020-07-01-23_59_Sentinel-2_Quarterly_Mosaics_B04_(Raw).tiff", "2024-07-01-00_00_2024-07-01-23_59_Sentinel-2_Quarterly_Mosaics_B08_(Raw).tiff")
ndvi_img1 = ndvi1.obtener_ndvi_visualizacion()

ndvi2 = ImagenNDVI("2024-07-01-00_00_2024-07-01-23_59_Sentinel-2_Quarterly_Mosaics_NDVI.tiff", "2024-07-01-00_00_2024-07-01-23_59_Sentinel-2_Quarterly_Mosaics_B04_(Raw).tiff", "2024-07-01-00_00_2024-07-01-23_59_Sentinel-2_Quarterly_Mosaics_B08_(Raw).tiff")
ndvi_img2 = ndvi2.obtener_ndvi_visualizacion()

# Guardar NDVI para comparación
cv2.imwrite("ndvi_1.jpg", (ndvi_img1 * 255).astype(np.uint8))
cv2.imwrite("ndvi_2.jpg", (ndvi_img2 * 255).astype(np.uint8))

# Comparación de imágenes para detección de cambios
analisis = AnalisisCambio("ndvi_1.jpg", "ndvi_2.jpg")
diferencia_img = analisis.obtener_diferencia()

# 🎨 Mostrar los cuatro plots en un solo gráfico
fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 12))

# Imagen satelital Julio 2020
axs[0,0].imshow(imagen_rgb_1)
axs[0,0].set_title("Imagen Satelital Original Julio 2020")
axs[0,0].axis("off")

# Imn satelital Julio 2024
axs[1,0].imshow(imagen_rgb_2)
axs[1,0].set_title("Imagen Satelital Original Julio 2024")
axs[1,0].axis("off")

# Diencia entre NDVIs
axs[2,0].imshow(diferencia_img, cmap="gray")
axs[2,0].set_title("Diferencia de NDVI")
axs[2,0].axis("off")

# NDVI1
axs[0,1].imshow((ndvi_img1 * 255).astype(np.uint8))
axs[0,1].set_title("NDVI1")
axs[0,1].axis("off")

# NDVI2
axs[1,1].imshow((ndvi_img2 * 255).astype(np.uint8))
axs[1,1].set_title("NDVI2")
axs[1,1].axis("off")

# Ajustar y mostrar
plt.tight_layout()
plt.show()
