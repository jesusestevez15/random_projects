import cv2
import numpy as np
from matplotlib import pyplot as plt

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

disparity = []
a = []
b = []

# Cargo las imágenes
imgL = cv2.imread('PIA24670-Ingenuity_Helicopter_in_3D-left.png')
imgR = cv2.imread('PIA24670-Ingenuity_Helicopter_in_3D-right.png')

# Preparo las imágenes
imgL = imgL[110:1866, 110:1530]
imgR = imgR[110:1866, 110:1530]
#imgL = imgL[585:1070, 691:1398]
#imgR = imgR[585:1070, 691:1398]

dstL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
dstR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

dstL = cv2.normalize(dstL, dstL, beta=0, alpha=255,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
dstR = cv2.normalize(dstR, dstR, beta=0, alpha=255,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Creo el mapa de disparidades
print('Creando mapa de disparidades...')

stereo = cv2.StereoSGBM_create(minDisparity=-65,
                               numDisparities=15*16,
                               blockSize=3,
                               P1=1*8*11*11,
                               P2=1*32*11*11,
                               disp12MaxDiff=20,
                               preFilterCap=0,
                               uniquenessRatio=5,
                               speckleWindowSize=50,
                               speckleRange=1,
                               mode=3)

disparity = stereo.compute(dstL, dstR)  # .astype(np.float32) / 16.0

# Aplico el filtro WLS
if True:
    matcher = cv2.ximgproc.createRightMatcher(stereo)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    disparityR = matcher.compute(dstR, dstL)  # .astype(np.float32) / 16.0

    lmbda = 80000
    sigma = 3.0
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    disparity = np.int16(disparity)
    disparityR = np.int16(disparityR)

    disparityFiltered = wls_filter.filter(disparity, dstL, disparity_map_right=disparityR)
    disparityFiltered = cv2.normalize(disparityFiltered, disparityFiltered, beta=0, alpha=255,
                                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    #confidence = cv2.ximgproc_DisparityWLSFilter.getConfidenceMap(disparityFiltered)

disparity = np.int16(disparity)
disparity = cv2.normalize(disparity, disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
disparityR = cv2.normalize(disparityR, disparityR, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

factorcalibracion = 7
depth = factorcalibracion * 24.4 * 11 / disparityFiltered
depth = depth[440:1756, 175:1100]
disparity = np.uint8(disparity)
disparityR = np.uint8(disparityR)
disparityFiltered = np.uint8(disparityFiltered)

# Muestro los resultados obtenidos
#plt.subplot(121), plt.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)), plt.title('Imagen izquierda'), plt.xlabel('Píxeles'), plt.ylabel('Píxeles')
#plt.subplot(122), plt.imshow(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)), plt.title('Imagen derecha'), plt.xlabel('Píxeles'), plt.ylabel('Píxeles')
#plt.subplot(122), plt.imshow(disparity, cmap=plt.get_cmap('plasma')), plt.title('Mapa de disparidad sin filtrar'), plt.xlabel('Píxeles'), plt.ylabel('Píxeles')
#plt.subplot(121), plt.imshow(disparityFiltered, cmap=plt.get_cmap('plasma')), plt.title('Mapa de disparidad filtrado'), plt.xlabel('Píxeles'), plt.ylabel('Píxeles')
#plt.colorbar(label="Disparidad", orientation="horizontal")
plt.subplot(122), plt.imshow(depth, cmap=plt.get_cmap('jet_r')), plt.title('Mapa de profundidad'), plt.xlabel('Píxeles'), plt.ylabel('Píxeles')
plt.colorbar(label="Profundidad (m)", orientation="vertical")

plt.show()