import cv2
import numpy as np
from matplotlib import pyplot as plt
from pixellib.instance import instance_segmentation
import time

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

class_names = ["BG", "person", "bicycle", "car", "motorcycle", "airplane",
                         "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                         "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                         "zebra",
                         "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                         "snowboard",
                         "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                         "tennis racket",
                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                         "orange",
                         "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                         "bed",
                         "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                         "microwave",
                         "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                         "hair dryer",
                         "toothbrush"]

def imagSegmentation(imagL, imagR):

    t1 = time.time()

    new_imgL = cv2.cvtColor(imagL, cv2.COLOR_RGB2BGR)
    new_imgR = cv2.cvtColor(imagR, cv2.COLOR_RGB2BGR)

    # Detecta los objetos
    target_classes = instance_video.select_target_classes(potted_plant=True)
    segmaskL, new_imgL = instance_video.segmentFrame(new_imgL, show_bboxes=True, segment_target_classes=target_classes)
    segmaskR, new_imgR = instance_video.segmentFrame(new_imgR, show_bboxes=True, segment_target_classes=target_classes)

    # Cargo la clase de objeto y su valor de confianza
    idsL = segmaskL['class_ids']
    scoreL = segmaskL['scores']
    idsR = segmaskR['class_ids']
    scoreR = segmaskR['scores']

    # Creo las máscaras
    maskL = segmaskL['masks'] * 1.0
    maskR = segmaskR['masks'] * 1.0

    # Genero una máscara para cada imagen que englobe todos los objetos detectados por cada cámara
    mascaraL = np.zeros((455, 640, 3))
    mascaraR = np.zeros((455, 640, 3))

    for i in range(len(scoreL)):
        # Si su valor de confianza es mayor a 0.7 muestro en pantalla la clase y su valor
        if scoreL[i] >= 0.7:
            mascaraL[maskL[:, :, i] == 1] = 1
            number = idsL[i]
            print('Imagen derecha')
            print(i, number)
            print(class_names[number])
            print(scoreL[i])
            print('---------')

    for i in range(len(scoreR)):
        # Si su valor de confianza es mayor a 0.7 muestro en pantalla la clase y su valor
        if scoreR[i] >= 0.7:
            mascaraR[maskR[:, :, i] == 1] = 1
            number = idsR[i]
            print('Imagen izquierda')
            print(i, number)
            print(class_names[number])
            print(scoreR[i])
            print('---------')

    t2 = time.time()

    print('Tiempo en detectar, clasificar y/o segmentar objetos: ' + str(t2-t1) + ' segundos')
    return imgL, imgR, new_imgL, new_imgR, mascaraL, mascaraR


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
        #a.append(x)
        #b.append(y)
        disp = disparity[y][x]
        distancia = points[y][x][2]
        if bool(filtro) is True:
            dispFiltered = disparityFiltered[y][x]
            distanciaFiltered = pointsFiltered[y][x][2]
            print(str(dispFiltered), str(distanciaFiltered))
        print(x, y, str(disp), str(distancia), str(depth[y][x]))


##################################
###### Parámetros iniciales ######
##################################

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.000001)

size = (480, 640)

objpoints = np.load('objpoints.npy')
imgpointsL = np.load('imgpointsL.npy')
imgpointsR = np.load('imgpointsR.npy')
mtxL = np.load('mtxL.npy')
mtxR = np.load('mtxR.npy')
distL = np.load('distL.npy')
distR = np.load('distR.npy')
newcameramtxL = np.load('newcameramtxL.npy')
newcameramtxR = np.load('newcameramtxR.npy')
roiL = np.load('roiL.npy')
roiR = np.load('roiR.npy')
R = np.load('R.npy')
T = np.load('T.npy')
E = np.load('E.npy')
F = np.load('F.npy')
RL = np.load('RL.npy')
RR = np.load('RR.npy')
PL = np.load('PL.npy')
PR = np.load('PR.npy')
Q = np.load('Q.npy')
roiL1 = np.load('roiL1.npy')
roiR1 = np.load('roiR1.npy')
mapL1 = np.load('mapL1.npy')
mapL2 = np.load('mapL2.npy')
mapR1 = np.load('mapR1.npy')
mapR2 = np.load('mapR2.npy')
disparity = []
a = []
b = []

#print(mtxL)
#print(newcameramtxL)
#print(PL)
#print(mtxR)
#print(newcameramtxR)
#print(PR)
#print(distL)
#print(RL)
#print(distR)
#print(RR)
#print(Q)
#print(R)

# Matriz que proyecta la imagen 2D en una 3D
cx = 319.5
cy = 239.5
f = 471.0570556
b = 8.7
Q = np.array([[1, 0,    0,        -cx],
               [0, 1,    0,        -cy],
               [0, 0,    0,          f],
               [0, 0, -1/b, (cx-cx)/b]])

# Encendido de cámaras
print('Iniciando cámaras...')

cv2.namedWindow('disp',  cv2.WINDOW_FREERATIO)
cv2.resizeWindow('disp', (10000, 100))

imagesL = cv2.VideoCapture(0)
imagesR = cv2.VideoCapture(2)

print('¿Usar filtro en el mapa de disparidades? No: 0, Sí: 1')
filtro = int(input())

print('¿Detectar y clasificar objetos? No: 0, Sí: 1')
segment = int(input())
if bool(segment) is True:
    print('¿Segmentar objetos? No: 0, Sí: 1')
    segment2 = int(input())
else:
    segment2 = 0

if bool(segment) is True:
    # Iniciamos el modelo de segmentación de imagen "mask_rcnn_coco.h5"
    instance_video = instance_segmentation(infer_speed="rapid")
    instance_video.load_model("mask_rcnn_coco.h5")

############################################
###### Creo los mapas de disparidades ######
############################################

print('Creando mapa de disparidades...')

cv2.namedWindow('disp', cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('disp', (640, 480))

cv2.createTrackbar('numDisparities', 'disp', 7, 30, nothing)
cv2.createTrackbar('blockSize', 'disp', 9, 100, nothing)
cv2.createTrackbar('disp12MaxDiff', 'disp', 20, 100, nothing)
cv2.createTrackbar('minDisparity', 'disp', 100, 200, nothing)
cv2.createTrackbar('mode', 'disp', 3, 4, nothing)
cv2.createTrackbar('lambda', 'disp', 8000, 80000, nothing)
cv2.createTrackbar('sigma', 'disp', 30, 50, nothing)  # luego se divide entre 10

stereo = cv2.StereoSGBM_create()

disparity = []
disparityFiltered = []

while True:

    # Leo los frames de ambas cámaras
    trueL, imagL = imagesL.read()
    trueR, imagR = imagesR.read()
    imgL = imagL
    imgR = imagR

    # Elimino las distorsiones de ambas cámaras
    imgL = cv2.remap(imagL, mapL1, mapL2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    imgR = cv2.remap(imagR, mapR1, mapR2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    # Recortamos la imagen
    xL, yL, wL, hL = roiL
    xR, yR, wR, hR = roiR

    imgL = imgL[0:455, 0:640]
    imgR = imgR[0:455, 0:640]

    # Cambio a escala de grises
    dstL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    dstR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Parámetros del mapa de disparidades
    numDisparities = cv2.getTrackbarPos('numDisparities',
                                        'disp') * 16  # Número máximo de disparidades (maxDis - minDis)
    blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 1  # Tamaño del bloque que matchea puntos
    #preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
    #uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
    #speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
    #speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
    minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')  * -1 # Valor mínimo de disparidad al matchear puntos
    mode = cv2.getTrackbarPos('mode', 'disp')
    P1 = cv2.getTrackbarPos('P1', 'disp') * blockSize * blockSize  # Controla la suavidad de la disparidad
    P2 = cv2.getTrackbarPos('P2', 'disp') * blockSize * blockSize  # Controla la suavidad de la disparidad
    lmbda = cv2.getTrackbarPos('lambda', 'disp')
    sigma = cv2.getTrackbarPos('sigma', 'disp') / 10

    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)
    stereo.setMode(mode)

    # Creo el mapa de disparidades sin filtrar con la cámara izquierda como referencia
    disparity = stereo.compute(dstL, dstR).astype(np.float32) / 16.0

    if bool(segment) is True:
        # Segmentación de imagen
        _, _, new_imgL, new_imgR, mascaraL, mascaraR = imagSegmentation(imgL, imgR)

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
        disparityFiltered = cv2.normalize(disparityFiltered, disparityFiltered, beta=255, alpha=1,
                                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        disparityFiltered = (256 - disparityFiltered)
        disparityR = cv2.normalize(disparityR, disparityR, beta=255, alpha=1, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_32F)
        disparityR = (256 - disparityR)

        # Proyecto el mapa de disparidades en un mapa de puntos 3D
        factorcalibracion = 2.478
        pointsFiltered = - cv2.reprojectImageTo3D(disparityFiltered, Q) * factorcalibracion
        pointsFiltered = pointsFiltered[:, 12:540]

        # Los convierto a formato 8U
        disparityFiltered = np.uint8(disparityFiltered)
        disparityR = np.uint8(disparityR)

        # Muestro el mapa de disparidades filtrado
        if bool(segment2) is True:
            disparityFiltered[mascaraL[:,:,1] == 0] = 0

        # Muestro el mapa de disparidades filtrado
        cv2.imshow('disp', disparityFiltered)

    # Normalizado el mapa de disparidades
    disparity = cv2.normalize(disparity, disparity, beta=255, alpha=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disparity = (256 - disparity)

    # Convierto los valores máximos y mínimos (producto del ruido, baja calidad de la imagen, zonas sin textura, etc.)
    # al valor de la mediana del total de la imagen
    threshold1 = 250
    threshold2 = 5
    disparitymedian = disparity[threshold1 > disparity]
    disparitymedian = disparitymedian[disparitymedian > threshold2]
    m = np.median(disparitymedian)
    disparity[disparity > threshold1] = m
    disparity[disparity < threshold2] = m

    factorcalibracion1 = 3.421504915
    factorcalibracion2 = 2.750382425
    depth = factorcalibracion1 * 8.7 * 586 / disparity
    depth = depth[:, 12:540]
    cv2.imshow('depth', depth)
    # Proyecto el mapa de disparidades en un mapa de puntos 3D
    disparity = np.uint8(disparity)
    points = - cv2.reprojectImageTo3D(disparity, Q) * factorcalibracion2
    points = points[:, 12:540]

    # Muestro los resultados
    #disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_OCEAN)

    if bool(segment2) is True:
        disparity[mascaraL[:,:,1] == 0] = 0
        if bool(filtro) is True:
            disparityR[mascaraR[:,:,1] == 0] = 0

    cv2.imshow('left', disparity)

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

    if bool(segment) is True:
        imgL = cv2.cvtColor(new_imgL, cv2.COLOR_BGR2RGB)
        imgR = cv2.cvtColor(new_imgR, cv2.COLOR_BGR2RGB)
        cv2.imshow('imgR', imgL)
        cv2.imshow('imgL', imgR)

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
colors = colors[:, 12:540]

disparity = disparity[:, 12:540]
mask = disparity > disparity.min()
out_points = points[mask]
out_colors = colors[mask]

if bool(filtro) is True:
    disparityFiltered = disparityFiltered[:, 12:540]
    maskFiltered = disparityFiltered > disparityFiltered.min()
    out_pointsFiltered = pointsFiltered[maskFiltered]
    out_colorsFiltered = colors[maskFiltered]

out = 'out.ply'
write_ply(out, out_points, out_colors)

if bool(filtro) is True:
    outFiltered = 'outFiltered.ply'
    write_ply(outFiltered, out_pointsFiltered, out_colorsFiltered)

imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

# Muestro la imagen izquierda y derecha, el mapa de disparidades sin filtrar y el filtrado
plt.subplot(221), plt.imshow(imgR), plt.title('Imagen izquierda'), plt.xlabel('Píxeles'), plt.ylabel('Píxeles')
plt.subplot(222), plt.imshow(imgL), plt.title('Imagen derecha'), plt.xlabel('Píxeles'), plt.ylabel('Píxeles')
plt.subplot(223), plt.imshow(disparity, cmap=plt.get_cmap('plasma')), plt.title('Mapa de disparidad sin filtrar'), plt.xlabel('Píxeles'), plt.ylabel('Píxeles')
if bool(filtro) is True:
    plt.subplot(224), plt.imshow(disparityFiltered, cmap=plt.get_cmap('plasma')), plt.title('Mapa de disparidad filtrado'), plt.xlabel('Píxeles'), plt.ylabel('Píxeles')
plt.colorbar(label="Disparidad", orientation="vertical")
plt.show()

imagesL.release()
imagesR.release()

cv2.destroyAllWindows()