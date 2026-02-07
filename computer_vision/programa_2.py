import cv2
import numpy as np
from matplotlib import pyplot as plt

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


def nothing(x):
    pass


def write_ply(fn, verts, colors):

    l = len(verts)

    if len(verts) % 3 == 1:

        verts = verts[:len(verts)-1]
        colors = colors[:len(colors)-1]

    elif len(verts) % 3 == 2:

        verts = verts[:len(verts)-2]
        colors = colors[:len(colors)-2]

    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)

    verts = np.hstack([verts, colors])

    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        disparidad = disparityFiltered[y][x]
        distancia = pointsFiltered[y][x][2]
        cv2.circle(disparityFiltered, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(disparityFiltered, xy, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), thickness=1)
        cv2.putText(disparityFiltered, str(disparidad), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0,),
                    thickness=1)
        cv2.putText(disparityFiltered, str(distancia), (x, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0,),
                    thickness=1)
        cv2.imshow("disp", disparityFiltered)
        print(x,y, str(disparidad), str(distancia))


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

size = (480, 640)

mtxL = np.array([[605.40549492,   0.,         298.47760266],
                 [  0.,         604.09112941, 232.96266743],
                 [  0.,           0.,           1.        ]])

newcameramtxL = np.array([[481.21969604,   0.,         295.17636696],
                          [  0.,         475.37469482, 232.4804082 ],
                          [  0.,           0.,           1.        ]])

distL = np.array([[-0.40315589,  0.14431789,  0.00100726,  0.00228183,  0.01920138]])

mtxR = np.array([[616.41177229,   0.,         323.62019162],
                 [  0.,         615.82532231, 238.50776023],
                 [  0.,           0.,           1.        ]])

newcameramtxR = np.array([[492.10787964,   0.,         323.8804575 ],
                          [  0.,         491.71749878, 236.70791575],
                          [  0.,           0.,           1.        ]])

distR = np.array([[-3.84130122e-01,  9.00404409e-02, -1.17136525e-03, -2.60319340e-04,  5.28794330e-02]])

disparity = []
a = []
b = []

Q = np.array([[1, 0,  0,       -295.17636696],
              [0, 1,  0,       -232.4804082 ],
              [0, 0,  0,        485.1049423 ],
              [0, 0, -0.1149425, -3.29932075]])

print(mtxL)
print(mtxR)
print(newcameramtxL)
print(newcameramtxR)
print(distL)
print(distR)

print('Iniciando cámaras...')

imagesL = cv2.VideoCapture(0)
imagesR = cv2.VideoCapture(2)

print('Creando mapa de disparidades...')

print('¿Usar filtro en el mapa de disparidades? No: 0, Sí: 1')
filtro = int(input())

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
cv2.createTrackbar('P1', 'disp', 0, 300, nothing)
cv2.createTrackbar('P2', 'disp', 0, 300, nothing)
cv2.createTrackbar('mode', 'disp', 3, 4, nothing)
cv2.createTrackbar('lambda', 'disp', 8000, 80000, nothing)
cv2.createTrackbar('sigma', 'disp', 15, 50, nothing)  # luego se divide entre 10

stereo = cv2.StereoSGBM_create()

disparity = []
disparityFiltered = []

while True:

    trueL, imgL = imagesL.read()
    trueR, imgR = imagesR.read()

    imgL = cv2.undistort(imgL, mtxL, distL, newcameramtxL)
    imgR = cv2.undistort(imgR, mtxR, distR, newcameramtxR)

    # xL, yL, wL, hL = roiL
    # xR, yR, wR, hR = roiR
    # x = int((xL + xR) / 2)
    # y = int((yL + yR) / 2)
    # w = int((wL + wR) / 2)
    # h = int((hL + hR) / 2)

    # imgL = imgL[y:y + h, x:x + w]
    # imgR = imgR[y:y + h, x:x + w]

    dstL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    dstR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
    blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 1
    preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
    minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')  # * -1 # probar poniendo el * -1 a ver que sale
    mode = cv2.getTrackbarPos('mode','disp')
    P1 = cv2.getTrackbarPos('P1','disp') * blockSize * blockSize
    P2 = cv2.getTrackbarPos('P2','disp') * blockSize * blockSize
    lmbda = cv2.getTrackbarPos('lambda','disp')
    sigma = cv2.getTrackbarPos('sigma','disp') / 10

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

    if bool(filtro) is True:
        matcher = cv2.ximgproc.createRightMatcher(stereo)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        disparityR = matcher.compute(dstR, dstL)  # .astype(np.float32) / 16.0

        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        disparity = np.int16(disparity)
        disparityR = np.int16(disparityR)

        disparityFiltered = wls_filter.filter(disparity, dstL, disparity_map_right=disparityR)
        disparityFiltered = cv2.normalize(disparityFiltered, disparityFiltered, beta=0, alpha=255,
                                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        #confidence = cv2.ximgproc_DisparityWLSFilter.getConfidenceMap(disparityFiltered)

        disparityR = cv2.normalize(disparityR, disparityR, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_32F)

        pointsFiltered = cv2.reprojectImageTo3D(disparityFiltered, Q) * 2.35  # Lo paso a cm

        disparityFiltered = np.uint8(disparityFiltered)
        disparityR = np.uint8(disparityR)

        cv2.imshow('disp', disparityFiltered)

    disparity = cv2.normalize(disparity, disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    points = cv2.reprojectImageTo3D(disparity, Q) * 2.35  # Lo paso a cm

    disparity = np.uint8(disparity)

    cv2.imshow('left', disparity)
    cv2.imshow('right', disparityR)
    #cv2.imshow('confidence map', confidence)
    cv2.setMouseCallback("disp", on_EVENT_LBUTTONDOWN)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

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
write_ply(outFiltered, out_pointsFiltered, out_colorsFiltered)

plt.subplot(221), plt.imshow(dstL, 'gray'), plt.title('dstL')
plt.subplot(222), plt.imshow(dstR, 'gray'), plt.title('dstR')
plt.subplot(223), plt.imshow(disparity, cmap=plt.get_cmap('plasma')), plt.title('disparity non filtered')
plt.subplot(224), plt.imshow(disparityFiltered, cmap=plt.get_cmap('plasma')), plt.title('disparity filtered')
plt.show()

imagesL.release()
imagesR.release()

cv2.destroyAllWindows()