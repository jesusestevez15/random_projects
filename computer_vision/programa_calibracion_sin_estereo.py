import cv2
import numpy as np
from matplotlib import pyplot as plt


# Características de las cámaras:
# Resolución: 2.0 MP, 1080P (640x480)
# Tamaño de la diagonal del sensor: 0.2362204724409449 ''
# Tamaño del píxel: 0.0075 mm/pix
# Focal: 3.6 mm
# Dimensiones (L x W x H): 91 mm x 70 mm x 35 mm


def errorcalib(objpointsL, objpointsR, rvecsL, rvecsR, tvecsL, tvecsR, mtxL, mtxR, distL, distR):
    mean_errorL = 0
    mean_errorR = 0

    for i in range(len(objpointsL)):
        imgpointsL2, _ = cv2.projectPoints(objpointsL[i], rvecsL[i], tvecsL[i], mtxL, distL)
        errorL = cv2.norm(imgpointsL[i], imgpointsL2, cv2.NORM_L2) / len(imgpointsL2)
        mean_errorL += errorL

    for i in range(len(objpointsR)):
        imgpointsR2, _ = cv2.projectPoints(objpointsR[i], rvecsR[i], tvecsR[i], mtxR, distR)
        errorR = cv2.norm(imgpointsR[i], imgpointsR2, cv2.NORM_L2) / len(imgpointsR2)
        mean_errorR += errorR

    return mean_errorL, mean_errorR


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.000001)

objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

objpointsL = []
objpointsR = []
imgpointsL = []
imgpointsR = []
disparity = []

size = (480, 640)

print('Iniciando cámaras...')

imagesL = cv2.VideoCapture(0)
imagesR = cv2.VideoCapture(2)

photosL = 1
photosR = 1

print('Buscando tablero de ajedrez...')

while True:

    trueL, imgL = imagesL.read()
    trueR, imgR = imagesR.read()

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    istrueL, cornL = cv2.findChessboardCorners(grayL, (7, 6), None)

    istrueR, cornR = cv2.findChessboardCorners(grayR, (7, 6), None)

    if istrueL:
        objpointsL.append(objp)
        cornersL = cv2.cornerSubPix(grayL, cornL, (11, 11), (-1, -1), criteria)
        imgpointsL.append(cornersL)
        cv2.drawChessboardCorners(imgL, (7, 6), cornersL, trueL)
        print('izquierda:', photosL)
        photosL = photosL + 1

    if istrueR:
        objpointsR.append(objp)
        cornersR = cv2.cornerSubPix(grayR, cornR, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)
        cv2.drawChessboardCorners(imgR, (7, 6), cornersR, trueR)
        print('derecha:', photosR)
        photosR = photosR + 1

    cv2.imshow('cornersL', imgL)
    cv2.imshow('cornersR', imgR)

    combined = cv2.addWeighted(imgL, 0.3, imgR, 0.3, 0)

    cv2.line(combined, (0, 0), (size[1], 0), (0, 255, 0), 1)
    cv2.line(combined, (0, int(size[0] / 8)), (size[1], int(size[0] / 8)), (0, 255, 0), 1)
    cv2.line(combined, (0, int(size[0] / 4)), (size[1], int(size[0] / 4)), (0, 255, 0), 1)
    cv2.line(combined, (0, int(size[0] * 3 / 8)), (size[1], int(size[0] * 3 / 8)), (0, 255, 0), 1)
    cv2.line(combined, (0, int(size[0] / 2)), (size[1], int(size[0] / 2)), (0, 255, 0), 1)
    cv2.line(combined, (0, int(size[0] * 5 / 8)), (size[1], int(size[0] * 5 / 8)), (0, 255, 0), 1)
    cv2.line(combined, (0, int(size[0] * 3 / 4)), (size[1], int(size[0] * 3 / 4)), (0, 255, 0), 1)
    cv2.line(combined, (0, int(size[0] * 7 / 8)), (size[1], int(size[0] * 7 / 8)), (0, 255, 0), 1)
    cv2.line(combined, (0, int(size[0])), (size[1], int(size[0])), (0, 255, 0), 1)

    cv2.imshow('combined', combined)

    if cv2.waitKey(1000) & 0xff == ord('q'):
        break

print('Calibrando las cámaras individualmente...')

retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpointsL, imgpointsL, grayL.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpointsR, imgpointsR, grayR.shape[::-1], None, None)

print('Parámetros internos de la cámara izquierda: ', mtxL, distL)
print('Parámetros internos de la cámara derecha: ', mtxR, distR)

print('Optimizando la matriz de ambas cámaras...')

hL, wL = imgL.shape[:2]
hR, wR = imgR.shape[:2]

newcameramtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 0, (wL, hL), True)
newcameramtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 0, (wR, hR), True)

print('Calculando el error del calibrado individual...')

mean_errorL, mean_errorR = errorcalib(objpointsL, objpointsR, rvecsL, rvecsR, tvecsL, tvecsR, mtxL, mtxR, distL, distR)

print("Error total de la calibración de la cámara izquierda: {}".format(mean_errorL / len(objpointsL)))
print("Error total de la calibración de la cámara derecha: {}".format(mean_errorR / len(objpointsR)))

print(mtxL)
print(mtxR)
print(newcameramtxL)
print(newcameramtxR)
print(distL)
print(distR)


print('Eliminando distorsión...')

imagesL = cv2.VideoCapture(0)
imagesR = cv2.VideoCapture(2)

while True:

    trueL, imgL = imagesL.read()
    trueR, imgR = imagesR.read()

    # Eliminamos la distorsión
    dstL = cv2.undistort(imgL, mtxL, distL, None, newcameramtxL)
    dstR = cv2.undistort(imgR, mtxR, distR, None, newcameramtxR)

    # Recortamos la imagen
    xL, yL, wL, hL = roiL
    xR, yR, wR, hR = roiR

    cv2.rectangle(dstL, (xL, yL), (xL + wL, yL + hL), (0, 255, 0), 3)
    cv2.rectangle(dstR, (xR, yR), (xR + wR, yR + hR), (0, 255, 0), 3)

    concatenate = np.concatenate((dstL, dstR), axis=1)
    y = concatenate.shape[0]
    x = concatenate.shape[1]
    cv2.line(concatenate, (0, 0), (wL + wR, 0), (0, 255, 0), 1)
    cv2.line(concatenate, (0, int(y / 8)), (wL + wR, int(y / 8)), (0, 255, 0), 1)
    cv2.line(concatenate, (0, int(y / 4)), (wL + wR, int(y / 4)), (0, 255, 0), 1)
    cv2.line(concatenate, (0, int(y * 3 / 8)), (wL + wR, int(y * 3 / 8)), (0, 255, 0), 1)
    cv2.line(concatenate, (0, int(y / 2)), (wL + wR, int(y / 2)), (0, 255, 0), 1)
    cv2.line(concatenate, (0, int(y * 5 / 8)), (wL + wR, int(y * 5 / 8)), (0, 255, 0), 1)
    cv2.line(concatenate, (0, int(y * 3 / 4)), (wL + wR, int(y * 3 / 4)), (0, 255, 0), 1)
    cv2.line(concatenate, (0, int(y * 7 / 8)), (wL + wR, int(y * 7 / 8)), (0, 255, 0), 1)
    cv2.line(concatenate, (0, int(y)), (wL + wR, int(y)), (0, 255, 0), 1)
    cv2.imshow('dstL / dstR', concatenate)
    combined = cv2.addWeighted(dstL, 0.3, dstR, 0.3, 0)
    cv2.imshow('combined', combined)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

L = ['\n\n Número de fotos tomadas cámara izquierda: ', str(photosL - 1),
     '\n\n Número de fotos tomadas cámara derecha: ', str(photosR - 1),
     '\n\n Matriz de cámara izquierda (mtxL): \n\n', str(mtxL),
     '\n\n Matriz de cámara derecha (mtxR): \n\n', str(mtxR),
     '\n\n Matriz óptima de cámara izquierda (newcameramtxL): \n\n', str(newcameramtxL),
     '\n\n Matriz óptima de cámara derecha (newcameramtxR): \n\n', str(newcameramtxR),
     '\n\n Vector de distorsión de cámara izquierda (dstL): \n\n', str(distL),
     '\n\n Vector de distorsión de cámara derecha (dstR): \n\n', str(distR),
     '\n\n Error de calibración cámara izquierda (mean_errorL): \n\n', str(mean_errorL / len(objpointsL)),
     '\n\n Error de calibración cámara derecha (mean_errorR): \n\n', str(mean_errorR / len(objpointsR)),
     ]

file = open('datos_calibracion_sin_estereo.txt', 'w')
file.write('Datos de la calibración:')
file.writelines(L)
file.close()

cv2.waitKey(0)

while True:
    if cv2.waitKey(0) & 0xff == ord('s'):
        np.save('objpointsL', objpointsL)
        np.save('objpointsR', objpointsR)
        np.save('imgpointsL', imgpointsL)
        np.save('imgpointsR', imgpointsR)
        np.save('mtxL', mtxL)
        np.save('mtxR', mtxR)
        np.save('distL', distL)
        np.save('distR', distR)
        np.save('newcameramtxL', newcameramtxL)
        np.save('newcameramtxR', newcameramtxR)
        np.save('roiL', roiL)
        np.save('roiR', roiR)
        break
    elif cv2.waitKey(0) & 0xff == ord('q'):
        break