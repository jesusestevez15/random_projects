import cv2
import numpy as np
from matplotlib import pyplot as plt


def nothing(x):
    pass


cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', (500, 100))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

newsize = (int(1530/2), int(1866/2))

disparity = []
a = []
b = []

imgL = cv2.imread('2right.png')
imgR = cv2.imread('2left.png')

width = int(imgL.shape[1] * 0.5)
height = int(imgL.shape[0] * 0.5)
dim = (width, height)
imgL = cv2.resize(imgL, dim, interpolation = cv2.INTER_AREA)
imgR = cv2.resize(imgR, dim, interpolation = cv2.INTER_AREA)


dstL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
dstR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

dstL = cv2.normalize(dstL, dstL, beta=0, alpha=255,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
dstR = cv2.normalize(dstR, dstR, beta=0, alpha=255,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

print('Creando mapa de disparidades...')

#print('¿Usar filtro en el mapa de disparidades? No: 0, Sí: 1')
#filtro = int(input())

cv2.namedWindow('disp', cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('disp', (640, 480))

cv2.createTrackbar('numDisparities', 'disp', 7, 30, nothing)
cv2.createTrackbar('blockSize', 'disp', 9, 100, nothing)
cv2.createTrackbar('preFilterCap', 'disp', 0, 100, nothing)
cv2.createTrackbar('uniquenessRatio', 'disp', 0, 100, nothing)
cv2.createTrackbar('speckleRange', 'disp', 0, 100, nothing)
cv2.createTrackbar('speckleWindowSize', 'disp', 0, 100, nothing)
cv2.createTrackbar('disp12MaxDiff', 'disp', 20, 100, nothing)
cv2.createTrackbar('minDisparity', 'disp', 0, 100, nothing)
cv2.createTrackbar('P1', 'disp', 0, 50, nothing)
cv2.createTrackbar('P2', 'disp', 0, 50, nothing)
cv2.createTrackbar('mode', 'disp', 3, 4, nothing)
cv2.createTrackbar('lambda', 'disp', 8000, 80000, nothing)
cv2.createTrackbar('sigma', 'disp', 15, 50, nothing)  # luego se divide entre 10

stereo = cv2.StereoSGBM_create()

while True:

    # Parámetros del mapa de disparidades
    numDisparities = cv2.getTrackbarPos('numDisparities',
                                        'disp') * 16  # Número máximo de disparidades (maxDis - minDis)
    blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 1  # Tamaño del bloque que matchea puntos
    preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
    minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')  #* -1 # Valor mínimo de disparidad al matchear puntos
    mode = cv2.getTrackbarPos('mode', 'disp')
    P1 = cv2.getTrackbarPos('P1', 'disp') * blockSize * blockSize  # Controla la suavidad de la disparidad
    P2 = cv2.getTrackbarPos('P2', 'disp') * blockSize * blockSize  # Controla la suavidad de la disparidad
    lmbda = cv2.getTrackbarPos('lambda', 'disp')
    sigma = cv2.getTrackbarPos('sigma', 'disp') / 10

    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)
    stereo.setMode(mode)
    stereo.setP1(P1)
    stereo.setP2(P2)

    disparity = stereo.compute(dstL, dstR)  # .astype(np.float32) / 16.0

    matcher = cv2.ximgproc.createRightMatcher(stereo)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    disparityR = matcher.compute(dstR, dstL)  # .astype(np.float32) / 16.0

    #lmbda = 80000
    #sigma = 3.0
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    disparity = np.int16(disparity)
    disparityR = np.int16(disparityR)

    disparityFiltered = wls_filter.filter(disparity, dstL, disparity_map_right=disparityR)
    #disparityFiltered = cv2.normalize(disparityFiltered, disparityFiltered, beta=0, alpha=255,
    #                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    #confidence = cv2.ximgproc_DisparityWLSFilter.getConfidenceMap(disparityFiltered)

    disparity = np.int16(disparity)
    #disparity = cv2.normalize(disparity, disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #disparityR = cv2.normalize(disparityR, disparityR, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    #disparity = np.uint8(disparity)
    #disparityR = np.uint8(disparityR)
    #disparityFiltered = np.uint8(disparityFiltered)

    cv2.imshow('disparity', disparity)
    cv2.imshow('disparity Filtered', disparityFiltered)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

plt.subplot(251), plt.imshow(dstL, 'gray'), plt.title('dstL')
plt.subplot(252), plt.imshow(dstR, 'gray'), plt.title('dstR')
plt.subplot(253), plt.imshow(disparity, cmap=plt.get_cmap('plasma')), plt.title('disparity unfiltered')

if True:
    #plt.subplot(254), plt.imshow(disparityR, cmap=plt.get_cmap('plasma')), plt.title('disparityR')
    #lt.subplot(259), plt.imshow(disparityRflip, cmap=plt.get_cmap('plasma')), plt.title('disparityR flip')
    plt.subplot(254), plt.imshow(disparityFiltered, cmap=plt.get_cmap('plasma')), plt.title('disparity filtered')

a = np.max(disparityFiltered)
print(a)
rows = disparityFiltered.shape[0]
columns = disparityFiltered.shape[1]
print(disparityFiltered.shape)
max = 0.0
for col in range(columns):
    for row in range(rows):
        if disparityFiltered[row][col] > max:
            max = disparityFiltered[row][col]
            row_max = row
            col_max = col
print(row_max, col_max)

plt.show()