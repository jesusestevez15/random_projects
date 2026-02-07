import numpy as np
import cv2 as cv
import glob
import os

# termination criteria: esto lo uso para más tarde al refinar los corners detectados con cv.cornerSubPix()
# Es el criterio que tiene en cuenta (nº iteraciones, etc) para refinar las corners
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepara los puntos objeto (object points) de la forma (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0), donde cada unidad es el
# tamaño de una casilla del tablero de ajedrez
w = 7  # w (anchura) y h (altura) son las dimensiones de los puntos objeto e imagen que quiero que calcule el programa
h = 6  # en las imágenes de los tableros de ajedrez
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)

# Vectores para guardar los puntos objeto e imagen  de todas las imagenes
objpoints0 = []  # puntos 3D en el espacio del mundo real
imgpoints0 = []  # puntos 2D en el plano imagen

i = 1

# Enciendo la cámara
frame = cv.VideoCapture(2)

# Reescalo las imágenes

width = int(frame.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(frame.get(cv.CAP_PROP_FRAME_HEIGHT))

print(width, height)

while True:
    # Leo los frames
    true, img = frame.read()

    print(len(img[0]), len(img[2]))

    #dsize = (int(width * 0.5), int(height * 0.5))
    #img = cv.resize(img, dsize)

    # Las paso a escala de grises
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)

    # Busco los corners
    true, corners = cv.findChessboardCorners(gray,
                                            (h, w),
                                            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)  # ret: booleano que dice si ha encontrado corners o no

    # Si los encuentra, añade los puntos
    if true == True:
        objpoints0.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints0.append(corners2)

        # Dibuja y muestra los corners en el tablero de ajedrez de cada imagen
        cv.drawChessboardCorners(img, (w, h), corners2, true)
        #cv.imwrite('coners_detectados_' + str(i) + ').jpg', img)
        print(i)
        i = i + 1

    cv.imshow('img', img)

    if cv.waitKey(500) & 0xff == ord('q'):
        break

#np.save('objpoints0', objpoints0)
#np.save('imgpoints0', imgpoints0)

frame.release()

cv.destroyAllWindows()

###  AHORA CALIBRAMOS  ###

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints0, imgpoints0, gray.shape[::-1], None, None)


# Antes de eliminar la distorsión usamos cv.getOptimalNewCameraMatrix(), la cual nos devuelve una nueva matriz de
# cámara mejor que la anterior conseguida. ROI sirve para posteriormente recortar la imagen no distorsionada
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

print(mtx)
print(newcameramtx)
print(dist)

# Por último, podemos calcular el error cometido al eliminar la distorsión
# Cuanto más ceracano a 0 sea, más exactos son los parámetros calculados (mtx, dist)

mean_error = 0

for i in range(len(objpoints0)):

    imgpoints02, _ = cv.projectPoints(objpoints0[i], rvecs[i], tvecs[i], mtx, dist)

    error = cv.norm(imgpoints0[i], imgpoints02, cv.NORM_L2) / len(imgpoints02)

    mean_error += error

print("total error: {}".format(mean_error / len(objpoints0)))

### GUARDO LA MATRIZ DE CÁMARA Y LOS COEFICIENTES DE DISTORSIÓN DE LA CÁMARA ###

#np.save('objpoints', objpoints0)
np.save('imgpointsL', imgpoints0)
#np.save('imgpointsR', imgpoints0)
np.save('mtxL', mtx)
#np.save('mtxR', mtx)
np.save('distL', dist)
#np.save('distR', dist)
np.save('newcameramtxL', newcameramtx)
#np.save('newcameramtxR', newcameramtx)
np.save('roiL', roi)
#np.save('roiR', roi)