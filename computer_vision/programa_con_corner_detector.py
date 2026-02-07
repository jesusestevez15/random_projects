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

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

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
        disparidad = disparity[y][x]
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
roiL1 = np.load('RoiL1.npy')
roiR1 = np.load('roiR1.npy')
disparity = []
a = []
b = []

print(Q)

Q = np.array([[1, 0,  0,       -331.79807233],
              [0, 1,  0,       -240.36021222],
              [0, 0,  0,       514.63076780],
              [0, 0, -0.11627907, 3.676188027]])

print(mtxL)
print(mtxR)
print(newcameramtxL)
print(newcameramtxR)
print(distL)
print(distR)

print('Iniciando cámaras...')

imagesL = cv2.VideoCapture(0)
imagesR = cv2.VideoCapture(2)

sift = cv2.SIFT_create()

while True:

    RL, RR, PL, PR, Q, roiL1, roiR1 = cv2.stereoRectify(mtxL, distL, mtxR, distR, size, R, T)

    trueL, imgL = imagesL.read()
    trueR, imgR = imagesR.read()

    imgL = cv2.undistort(imgL, mtxL, distL, newcameramtxL)
    imgR = cv2.undistort(imgR, mtxR, distR, newcameramtxR)

    xL, yL, wL, hL = roiL
    xR, yR, wR, hR = roiR
    x = int((xL + xR) / 2)
    y = int((yL + yR) / 2)
    w = int((wL + wR) / 2)
    h = int((hL + hR) / 2)

    imgL = imgL[y:y + h, x:x + w]
    imgR = imgR[y:y + h, x:x + w]

    dstL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    dstR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(dstL, None)
    kp2, des2 = sift.detectAndCompute(dstR, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    cv2.setMouseCallback("disp", on_EVENT_LBUTTONDOWN)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(dstL, dstR, lines1, pts1, pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(dstR, dstL, lines2, pts2, pts1)

    cv2.imshow('img5', img5)
    cv2.imshow('img3', img3)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

plt.subplot(221), plt.imshow(dstL, 'gray'), plt.title('img5')
plt.subplot(222), plt.imshow(dstR, 'gray'), plt.title('imgL')
plt.show()

imagesL.release()
imagesR.release()

cv2.destroyAllWindows()