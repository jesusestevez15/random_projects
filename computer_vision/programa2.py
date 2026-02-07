import cv2
import numpy as np
from matplotlib import pyplot as plt
from pixellib.instance import instance_segmentation

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

# Realiza la segmentación de imagen de las imágenes de ambas cámaras sin distorsión
def imagSegmentation(imagL, imagR):

    new_imagL = cv2.cvtColor(imagL, cv2.COLOR_RGB2BGR)
    new_imagR = cv2.cvtColor(imagR, cv2.COLOR_RGB2BGR)

    segmaskL, new_imagL = instance_video.segmentFrame(new_imagL, show_bboxes=False)
    segmaskR, new_imagR = instance_video.segmentFrame(new_imagR, show_bboxes=False)

    maskL = segmaskL['masks'] * 1.0
    maskR = segmaskR['masks'] * 1.0

    mascaraL = np.zeros((r, c, 3))
    mascaraR = np.zeros((r, c, 3))

    for i in range(maskL.shape[2]):
        mascaraL[maskL[:, :, i] == 1] = 1

    for i in range(maskR.shape[2]):
        mascaraR[maskR[:, :, i] == 1] = 1

    imgL[mascaraL == 0] = 0
    imgR[mascaraR == 0] = 0

    return imgL, imgR, mascaraL, mascaraR


def nothing(x):
    pass


# Escribe la nube de puntos en un archivo .ply
def write_ply(fn, verts, colors):

    l = len(verts)

    if len(verts) % 3 == 1:

        verts = verts[:len(verts) - 1]
        colors = colors[:len(colors) - 1]

    elif len(verts) % 3 == 2:

        verts = verts[:len(verts) - 2]
        colors = colors[:len(colors) - 2]

    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)

    verts = np.hstack([verts, colors])

    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


# Muestra las coordenadas, valor de la disparidad y distancia al clickear un píxel
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        disp = disparity[y][x]
        dispFiltered = disparityFiltered[y][x]
        distancia = points[y][x][2]
        distanciaFiltered = pointsFiltered[y][x][2]

        print(x, y, str(disp), str(dispFiltered), str(distancia), str(distanciaFiltered))


##################################
###### Parámetros iniciales ######
##################################

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.000001)

size = (480, 640)
r, c = size

objpoints = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/objpoints.npy')
imgpointsL = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/imgpointsL.npy')
imgpointsR = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/imgpointsR.npy')
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

disparity = []
a = []
b = []

print(mtxL)
print(newcameramtxL)
print(PL)
print(mtxR)
print(newcameramtxR)
print(PR)
print(distL)
print(RL)
print(distR)
print(RR)
print(Q)
print(R)

# Matriz que proyecta la imagen 2D en una 3D
# Q = np.array([[1, 0,    0,        -cx],
#               [0, 1,    0,        -cy],
#               [0, 0,    0,          f],
#               [0, 0, -1/b, (cx-cx')/b]])

# Encendido de cámaras
print('Iniciando cámaras...')

imagesL = cv2.VideoCapture(0)
imagesR = cv2.VideoCapture(2)

print('¿Usar filtro en el mapa de disparidades? No: 0, Sí: 1')
filtro = int(input())

print('¿Usar segmentación de imagen? No: 0, Sí: 1')
segment = int(input())

if bool(segment) is True:
    # Iniciamos el modelo de segmentación de imagen "mask_rcnn_coco.h5"
    instance_video = instance_segmentation(infer_speed="rapid")
    instance_video.load_model("mask_rcnn_coco.h5")

############################################
###### Creo los mapas de disparidades ######
############################################

print('Creando mapa de disparidades...')

cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', (500, 100))

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

disparity = []
disparityFiltered = []

while True:

    # Leo los frames de ambas cámaras
    trueL, imagL = imagesL.read()
    trueR, imagR = imagesR.read()
    #imagR = cv2.transform(imagR, R)

    # Elimino las distorsiones de ambas cámaras

    # imgL = cv2.undistort(imagL, mtxL, distL, newcameramtxL)
    # imgR = cv2.undistort(imagR, mtxR, distR, newcameramtxR)
    #
    # xL, yL, wL, hL = roiL
    # xR, yR, wR, hR = roiR
    # x = int((xL + xR) / 2)
    # y = int((yL + yR) / 2)
    # w = int((wL + wR) / 2)
    # h = int((hL + hR) / 2)

    imgL = cv2.remap(imagL, mapL1, mapL2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    imgR = cv2.remap(imagR, mapR1, mapR2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    # Recortamos la imagen
    xL, yL, wL, hL = roiL
    xR, yR, wR, hR = roiR

    if bool(segment) is True:
        # Segmentación de imagen
        _, _, mascaraL, mascaraR = imagSegmentation(imgL, imgR)

    # Cambio a escala de grises
    dstL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    dstR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Parámetros del mapa de disparidades
    numDisparities = cv2.getTrackbarPos('numDisparities',
                                        'disp') * 16  # Número máximo de disparidades (maxDis - minDis)
    blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 1  # Tamaño del bloque que matchea puntos
    preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
    minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')  # * -1 # Valor mínimo de disparidad al matchear puntos
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

    # Creo el mapa de disparidades sin filtrar con la cámara izquierda como referencia
    disparity = stereo.compute(dstL, dstR).astype(np.float32) / 16.0

    # Disparidad con filtro WLS
    if bool(filtro) is True:

        # Creo el matcher para la cámara derecha
        matcher = cv2.ximgproc.createRightMatcher(stereo)
        # Creo el filtro WLS (Weighted Least Squares)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        # Creo el mapa de disparidades con la cámara derecha como referencia
        disparityR = matcher.compute(dstR, dstL).astype(np.float32) / 16.0

        # Parámetros del filtro WLS
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        # Creo el mapa de disparidades filtrado
        disparityFiltered = wls_filter.filter(disparity, dstL, disparity_map_right=disparityR)

        # Normalizado el mapa de disparidades filtrado
        disparityFiltered = cv2.normalize(disparityFiltered, disparityFiltered, beta=0, alpha=255,
                                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        disparityR = cv2.normalize(disparityR, disparityR, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_32F)

        # Proyecto el mapa de disparidades en un mapa de puntos 3D
        pointsFiltered = cv2.reprojectImageTo3D(disparityFiltered, Q)

        # Los convierto a formato 8U
        disparityFiltered = np.uint8(disparityFiltered)
        disparityR = np.uint8(disparityR)

        # Muestro el mapa de disparidades filtrado
        if bool(segment) is True:
            disparityFiltered[mascaraL[:,:,1] == 0] = 0

        cv2.imshow('disp', disparityFiltered)

    # Normalizado el mapa de disparidades
    disparity = cv2.normalize(disparity, disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Proyecto el mapa de disparidades en un mapa de puntos 3D
    points = cv2.reprojectImageTo3D(disparity, Q)

    # Muestro los resultados
    if bool(segment) is True:
        disparity[mascaraL[:,:,1] == 0] = 0
        disparityR[mascaraR[:,:,1] == 0] = 0

    cv2.imshow('left', disparity)
    cv2.imshow('imgL', imgL)
    cv2.imshow('imgR', imgR)
    combined = cv2.addWeighted(dstL, 0.3, dstR, 0.3, 0)
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

    if bool(filtro) is True:
        cv2.imshow('right', disparityR)

    cv2.setMouseCallback("disp", on_EVENT_LBUTTONDOWN)
    cv2.setMouseCallback("combined", on_EVENT_LBUTTONDOWN)
    cv2.setMouseCallback("left", on_EVENT_LBUTTONDOWN)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

#########################################
###### Creo los mapas de puntos 3D ######
#########################################

colors = cv2.cvtColor(imgL, cv2.COLOR_RGB2BGR)

mask = disparity > disparity.min()
out_points = points[mask]
out_colors = colors[mask]

if bool(filtro) is True:
    maskFiltered = disparityFiltered > disparityFiltered.min()
    out_pointsFiltered = pointsFiltered[maskFiltered]
    out_colorsFiltered = colors[maskFiltered]

out = 'out.ply'
outFiltered = 'outFiltered.ply'

write_ply(out, out_points, out_colors)
if bool(filtro) is True:
    write_ply(outFiltered, out_pointsFiltered, out_colorsFiltered)

# Muestro la imagen izquierda y derecha, el mapa de disparidades sin filtrar y el filtrado
plt.subplot(221), plt.imshow(dstL, 'gray'), plt.title('dstL')
plt.subplot(222), plt.imshow(dstR, 'gray'), plt.title('dstR')
plt.subplot(223), plt.imshow(disparity, cmap=plt.get_cmap('plasma')), plt.title('disparity non filtered')
if bool(filtro) is True:
    plt.subplot(224), plt.imshow(disparityFiltered, cmap=plt.get_cmap('plasma')), plt.title('disparity filtered')
plt.show()

imagesL.release()
imagesR.release()

cv2.destroyAllWindows()