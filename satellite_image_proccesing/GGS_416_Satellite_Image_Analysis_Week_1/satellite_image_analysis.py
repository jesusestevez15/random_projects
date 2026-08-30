import numpy as np
import matplotlib.pyplot as plt
import rasterio

image_path = "2020-07-01-00_00_2020-07-01-23_59_Sentinel-2_Quarterly_Mosaics_B04_(Raw).tiff"

dataset = rasterio.open(image_path)

print("Dataset:\n",dataset)
print("Meta:\n",dataset.meta)
print("Driver:\n",dataset.driver)
print("Transform:\n",dataset.transform,"\n")
with rasterio.open(image_path) as dataset:
    image = dataset.read()

# we can now view the image as a NumPy array
print(image.shape)
print(image)

#the square brackets mean we index at location zero to get band 1
#remember: python is zero indexed
band1 = image[0]

print(band1.min(), band1.max(), band1.mean())
crop = band1[0:900, 0:2500]

normalized = (crop - crop.min()) / (crop.max() - crop.min()) * 255

print(normalized.min(), normalized.max(), normalized.mean())

plt.imshow(normalized, cmap="gray")
plt.title("Normalized Band 1")
plt.show()