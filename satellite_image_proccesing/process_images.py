import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import rasterio
from rasterio.plot import reshape_as_image

# Clase base para representar imágenes satelitales
class ImagenSatelital:
    def __init__(self, ruta_imagen):
        self.ruta_imagen = ruta_imagen
        
        with rasterio.open(ruta_imagen) as src:
            self.datos = src.read()             # (bandas, alto, ancho)
            self.perfil = src.profile           # Metadatos GeoTIFF completos
            self.crs = src.crs                  # Sistema de coordenadas (EPSG)
            self.transform = src.transform      # Geotransformación (resolución + origen)
            self.bounds   = src.bounds          # Bounding box geográfico
            self.nodata   = src.nodata          # Valor de "sin dato"
            self.n_bandas = src.count           # Número de bandas 
    
    def tamaño_imagen(self):
        """Devuelve (alto, ancho) en píxeles."""
        _, alto, ancho = self.datos.shape
        return alto, ancho
    
    def resolucion_espacial(self):
        """Devuelve la resolución en unidades del CRS (metros si es UTM)."""
        return self.transform.a, abs(self.transform.e)  # (ancho_pixel, alto_pixel)

    def obtener_imagen(self):
        """
        Devuelve imagen RGB normalizada para visualización con matplotlib.
        Si tiene más de 3 bandas, usa las 3 primeras (R, G, B).
        Si solo tiene 1 banda, la replica en los 3 canales.
        """
        if self.n_bandas >= 3:
            rgb = self.datos[:3, :, :]              # Primeras 3 bandas
        else:
            rgb = np.repeat(self.datos[:1], 3, axis=0)  # Escala de grises → RGB

        # Mover ejes: (bandas, alto, ancho) → (alto, ancho, bandas)
        rgb = reshape_as_image(rgb).astype(float)

        # Enmascarar nodata si existe
        if self.nodata is not None:
            rgb[rgb == self.nodata] = np.nan

        # Normalizar cada banda a [0, 255] ignorando NaN
        def norm(banda):
            vmin = np.nanmin(banda)
            vmax = np.nanmax(banda)
            if vmax == vmin:
                return np.zeros_like(banda, dtype=np.uint8)
            return ((banda - vmin) / (vmax - vmin) * 255).astype(np.uint8)

        return np.dstack([norm(rgb[:, :, i]) for i in range(3)])

    def obtener_banda(self, numero_banda):
        """
        Devuelve una banda específica como array 2D.
        numero_banda: 1-indexado (igual que rasterio)
        """
        if numero_banda < 1 or numero_banda > self.n_bandas:
            raise ValueError(f"Banda {numero_banda} no existe. "
                                f"Esta imagen tiene {self.n_bandas} bandas.")
        banda = self.datos[numero_banda - 1].astype(float)

        # Enmascarar nodata
        if self.nodata is not None:
            banda[banda == self.nodata] = np.nan

        return banda
    
    def info(self):
        """Imprime un resumen de los metadatos de la imagen."""
        alto, ancho = self.tamaño_imagen()
        res_x, res_y = self.resolucion_espacial()
        print(f"Archivo   : {self.ruta_imagen}")
        print(f"Tamaño    : {ancho} x {alto} px")
        print(f"Bandas    : {self.n_bandas}")
        print(f"Resolución: {res_x:.2f} x {res_y:.2f} m/px")
        print(f"CRS       : {self.crs}")
        print(f"Bounds    : {self.bounds}")
        print(f"NoData    : {self.nodata}")
        print(f"Dtype     : {self.perfil['dtype']}")

class NDVICalculo:
    def __init__(self, imagen_banda_roja, imagen_banda_infrarroja):
        self.imagen_banda_roja = imagen_banda_roja              # B04
        self.imagen_banda_infrarroja = imagen_banda_infrarroja  # B08
        return
    
    def obtener_ndvi(self):
        """Calcula el NDVI (Índice de Vegetación de Diferencia Normalizada)."""
        with rasterio.open(self.imagen_banda_roja) as src:
            banda_roja = src.read(1)  # Lee la primera banda
        with rasterio.open(self.imagen_banda_infrarroja) as src:
            banda_infrarroja = src.read(1)  # Lee la primera banda
        denominador = banda_infrarroja + banda_roja

        # Evitar división por cero
        # Calculando el NDVI
        img_ndvi = np.where(denominador == 0, 0.0,                           # Si denominador=0, NDVI=0
                       (banda_infrarroja - banda_roja) / denominador         # Si no, calcular normal
        )
        
        # Recortar valores al rango válido [-1, 1]
        img_ndvi = np.clip(img_ndvi, -1, 1)
        
        return img_ndvi
    
# Imágenes de prueba
img1 = ImagenSatelital("images/2020-07-01-00_00_2020-07-01-23_59_Sentinel-2_Quarterly_Mosaics_B04_(Raw).tiff")
img1.info()
img2 = ImagenSatelital("images/2020-07-01-00_00_2020-07-01-23_59_Sentinel-2_Quarterly_Mosaics_B08_(Raw).tiff")
img2.info()

rgb1 = img1.obtener_imagen()
rgb2 = img2.obtener_imagen()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(rgb1)
ax1.set_title("Imagen 2020 B04")
ax1.axis("off")

ax2.imshow(rgb2)
ax2.set_title("Imagen 2020 B08")
ax2.axis("off")

plt.tight_layout()
plt.show()


ndvi = NDVICalculo(img1.ruta_imagen, img2.ruta_imagen)
img_ndvi = ndvi.obtener_ndvi()
plt.imshow(img_ndvi)
plt.show()