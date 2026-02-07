import numpy as np
import cv2 as cv

# Carga la matriz de cámara y los coeficientes de distorsión de la camara calculados previamente
mtx = np.load(
    'mtxL.npy')
dist = np.load(
    'distL.npy')
newcameramtx = np.load(
    'newcameramtxL.npy')
roi = np.load(
    'roiL.npy')

print(roi)
print(mtx)
print(dist)
print(newcameramtx)

# Función que dibuja los corners y los puntos de los ejes para dibujar un sistema de ejes 3D
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    imgpts = imgpts.astype(int)

    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 3)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 3)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 3)
    return img


w = 7  # w (anchura) y h (altura) son las dimensiones de los puntos objeto e imagen que quiero que calcule el programa
h = 6 # en las imágenes de los tableros de ajedrez

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 60, 0.001)

objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

images0 = cv.VideoCapture(0)

while True:
    ret0, img0 = images0.read()

    gray0 = cv.cvtColor(img0, cv.COLOR_RGB2GRAY)

    ret0, corners0 = cv.findChessboardCornersSB(gray0, (7, 6), None)  # Busca los corners de cada imagen

    if ret0 == True:  # Si los encuentra calcula los rvecs y tvecs y dibuja los ejes

        corners2_0 = cv.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)  # Mejora los corners encontrados

        # Encuentra los vectores de rotación y traslación
        ret0, rvecs0, tvecs0 = cv.solvePnP(objp, corners2_0, mtx, dist, cv.SOLVEPNP_ITERATIVE)

        # LAS UNIDADES EN LAS QUE ESTÁN LOS VECTORES DE ROTACIÓN Y TRASLACIÓN SON EN CASILLAS DEL TABLERO.
        # SI LAS CASILLAS MIDEN DE LADO 1 CM, LAS UNIDADES ESTARÁN EN CM.
        print('------------------------')
        print('rvecs0', 180 / np.pi * rvecs0.transpose())
        print('tvecs0', 2.35 * tvecs0.transpose())

        # Proyecta los puntos 3D en el plano imagen
        imgpts0, jac0 = cv.projectPoints(axis, rvecs0, tvecs0, mtx, dist)
        img0 = draw(img0, corners2_0, imgpts0)  # Llama a la función draw() para dibujar los ejes

    img0 = cv.undistort(img0, mtx, dist, newcameramtx)

    x, y, w, h = roi
    img0 = img0[y:y + h, x:x + w]

    cv.imshow('img0', img0)

    if cv.waitKey(10) & 0xff == ord('q'):
        break

# When everything done, release the capture
images0.release()

cv.destroyAllWindows()