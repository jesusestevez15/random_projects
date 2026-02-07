import numpy as np
import cv2 as cv
import time


def imgrect(img0, img1, mapL1, mapL2, mapR1, mapR2, roiL, roiR):

    #dst0 = cv.undistort(img0, mtxL, distL, newcameramtxL)
    #dst1 = cv.undistort(img1, mtxR, distR, newcameramtxR)
    # Elimino las distorsiones de ambas cámaras
    dst0 = cv.remap(img0, mapL1, mapL2, interpolation=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT)
    dst1 = cv.remap(img1, mapR1, mapR2, interpolation=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT)

    # Recortamos la imagen
    xL, yL, wL, hL = roiL
    xR, yR, wR, hR = roiR

    #x = int((xL + xR) / 2)
    #y = int((yL + yR) / 2)
    #w = int((wL + wR) / 2)
    #h = int((hL + hR) / 2)

    #cv.rectangle(dst0, (0, 0), (640, 455), (0, 255, 0), 3)
    #cv.rectangle(dst1, (0, 0), (640, 455), (0, 255, 0), 3)

    #dst0 = dst0[0:490, 0:680]
    #dst1 = dst1[0:490, 0:680]

    dst0 = dst0[0:455, 0:640]
    dst1 = dst1[0:455, 0:640]

    gray0 = cv.cvtColor(dst0, cv.COLOR_RGB2GRAY)
    gray1 = cv.cvtColor(dst1, cv.COLOR_RGB2GRAY)

    return dst0, dst1, gray0, gray1


# Función que dibuja los corners y los puntos de los ejes para dibujar un sistema de ejes 3D
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())

    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 3)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 3)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 3)

    return img


t1 = time.time()

# Carga la matriz de cámara y los coeficientes de distorsión de la camara calculados previamente
mtxL = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/mtxL.npy')
distL = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/distL.npy')
newcameramtxL = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/newcameramtxL.npy')
roiL = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/roiL.npy')
mtxR = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/mtxR.npy')
distR = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/distR.npy')
newcameramtxR = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/newcameramtxR.npy')
roiR = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/roiR.npy')
R = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/R.npy')
T = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/T.npy')
E = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/E.npy')
F = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/F.npy')
RL = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/RL.npy')
RR = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/RR.npy')
PL = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/PL.npy')
PR = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/PR.npy')
Q = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/Q.npy')
roiL1 = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/roiL1.npy')
roiR1 = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/roiR1.npy')
mapL1 = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/mapL1.npy')
mapL2 = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/mapL2.npy')
mapR1 = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/mapR1.npy')
mapR2 = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/mapR2.npy')
t2 = time.time()
#print('Tiempo de carga de datos y parámetros: ', t2-t1, 's')

#print(mtxL)
#print(mtxR)
#print(newcameramtxL)
#print(newcameramtxR)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 120, 0.00001)

objpL = np.zeros((6 * 7, 3), np.float32)
objpL[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

objpR = np.zeros((6 * 7, 3), np.float32)
objpR[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

t1 = time.time()

imagesL = cv.VideoCapture(1)
imagesR = cv.VideoCapture(2)

t2 = time.time()
#print('Tiempo de encendido de cámaras: ', t2-t1, 's')
photo = 1

frame_width = int(imagesL.get(3))
frame_height = int(imagesL.get(4))
size = (frame_width, frame_height)
out = cv.VideoWriter('pose.avi', cv.VideoWriter_fourcc(*'MPEG'), 20.0, (640, 480))

while True:

    t_inicio = time.time()

    ret0, imgL = imagesL.read()
    ret1, imgR = imagesR.read()

    t1 = time.time()


    # Rectificación de imágenes
    dstL, dstR, grayL, grayR = imgrect(imgL, imgR, mapL1, mapL2, mapR1, mapR2, roiL1, roiR1)
    t2 = time.time()
    #print('Tiempo de rectificación de las imágenes de ambas cámaras: ', t2 - t1, 's')

    t1 = time.time()

    retL, cornersL = cv.findChessboardCorners(grayL, (7, 6), cv.CALIB_CB_ACCURACY)  # Busca los corners de cada imagen
    retR, cornersR = cv.findChessboardCorners(grayR, (7, 6), cv.CALIB_CB_ACCURACY)

    t2 = time.time()
    #print('Tiempo en encontrar las celdas del tablero de ajedrez: ', t2 - t1, 's')

    if retL == True:  # Si los encuentra calcula los rvecs y tvecs y dibuja los ejes

        t1 = time.time()

        corners2_L = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)  # Mejora los corners encontrados

        t2 = time.time()
        #print('Tiempo en mejorar las celdas del tablero de ajedrez: ', t2 - t1, 's')

        t1 = time.time()

        # Encuentra los vectores de rotación y traslación
        retLL, rvecsL, tvecsL = cv.solvePnP(objpL, corners2_L, newcameramtxL, distL, flags=cv.SOLVEPNP_IPPE)
        tvecsLarray = float(tvecsL[2])
        distancia = 3.1 * round(tvecsLarray, 1)
        t2 = time.time()
        #print('Tiempo en determinar la POSE del tablero de ajedrez: ', t2 - t1, 's')

        t1 = time.time()

        # Proyecta los puntos 3D en el plano imagen
        imgptsL, jacL = cv.projectPoints(axis, rvecsL, tvecsL, newcameramtxL, distL)

        dstL = draw(dstL, corners2_L, imgptsL)  # Llama a la función draw() para dibujar los ejes
        cv.putText(dstL, 'Distancia: ' + str(distancia) + ' cm', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), cv.LINE_4)
        out.write(dstL)
        print(dstL.shape)
        t2 = time.time()
        #print('Tiempo en proyectar eje cartesiano en el tablero de ajedrez: ', t2 - t1, 's')

    cv.imshow('dstL', dstL)

    if retR == True:  # Si los encuentra calcula los rvecs y tvecs y dibuja los ejes

        corners2_R = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)  # Mejora los corners encontrados

        # Encuentra los vectores de rotación y traslación
        retRR, rvecsR, tvecsR = cv.solvePnP(objpR, corners2_R, newcameramtxR, distR, flags=cv.SOLVEPNP_IPPE)

        # Proyecta los puntos 3D en el plano imagen
        imgptsR, jacR = cv.projectPoints(axis, rvecsR, tvecsR, newcameramtxR, distR)

        dstR = draw(dstR, corners2_R, imgptsR)  # Llama a la función draw() para dibujar los ejes

    cv.imshow('dstR', dstR)

    concatenate = np.concatenate((dstL, dstR), axis=1)

    y = concatenate.shape[0]
    x = concatenate.shape[1]

    #cv.line(concatenate, (0,0), (1280, 0), (0, 255, 0), 1)
    #cv.line(concatenate, (0, int(y / 8)), (1280, int(y / 8)), (0, 255, 0), 1)
    #cv.line(concatenate, (0, int(y / 4)), (1280, int(y / 4)), (0, 255, 0), 1)
    #cv.line(concatenate, (0, int(y * 3 / 8)), (1280, int(y * 3 / 8)), (0, 255, 0), 1)
    #cv.line(concatenate, (0, int(y / 2)), (1280, int(y / 2)), (0, 255, 0), 1)
    #cv.line(concatenate, (0, int(y * 5 / 8)), (1280, int(y * 5 / 8)), (0, 255, 0), 1)
    #cv.line(concatenate, (0, int(y * 3 / 4)), (1280, int(y * 3 / 4)), (0, 255, 0), 1)
    #cv.line(concatenate, (0, int(y * 7 / 8)), (1280, int(y * 7 / 8)), (0, 255, 0), 1)
    #cv.line(concatenate, (0, int(y)), (1280, int(y)), (0, 255, 0), 1)
    cv.imshow('dstL / dstR', concatenate)
    #combined = cv.addWeighted(dst1, 0.3, dst0, 0.3, 0)
    #cv.imshow('combined', combined)

    t_final = time.time()
    #print('Tiempo total: ', t_final-t_inicio, 's')

    if cv.waitKey(1) & 0xff == ord('q'):
        break
    elif cv.waitKey(1) & 0xff == ord('s'):
        cv.imwrite('estimacion_pose' + str(photo) + '.png', concatenate)
        print(photo)
        print('------------------------')
        print('rvecsL: ', 180 / np.pi * rvecsL.transpose())
        print('tvecsL: ', 3.1 * tvecsL.transpose())
        print('------------------------')
        print('rvecsR: ', 180 / np.pi * rvecsR.transpose())
        print('tvecsR: ', 3.1 * tvecsR.transpose())

        photo = photo + 1

# When everything done, release the capture
imagesL.release()
imagesR.release()
out.release()

cv.destroyAllWindows()
