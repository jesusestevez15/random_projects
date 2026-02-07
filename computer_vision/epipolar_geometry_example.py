import numpy as np
import cv2
from matplotlib import pyplot as plt


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''

    r, c, a = img1.shape
    #img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    #img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

mtxL = np.load('mtxL.npy')
mtxR = np.load('mtxR.npy')
distL = np.load('distL.npy')
distR = np.load('distR.npy')
newcameramtxL = np.load('newcameramtxL.npy')
newcameramtxR = np.load('newcameramtxR.npy')
roiL = np.load('roiL.npy')
roiR = np.load('roiR.npy')
roiL1 = np.load('RoiL1.npy')
roiR1 = np.load('roiR1.npy')

images0 = cv2.VideoCapture(0)
images1 = cv2.VideoCapture(2)

while(True):
    # Capture frame-by-frame
    ret0, img1 = images0.read()
    ret1, img2 = images1.read()

    imgL = cv2.undistort(img1, mtxL, distL, newcameramtxL)
    imgR = cv2.undistort(img2, mtxR, distR, newcameramtxR)

    xL, yL, wL, hL = roiL
    xR, yR, wR, hR = roiR
    x = int((xL + xR) / 2)
    y = int((yL + yR) / 2)
    w = int((wL + wR) / 2)
    h = int((hL + hR) / 2)

    img1 = imgL[y:y + h, x:x + w]
    img2 = imgR[y:y + h, x:x + w]

    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    print(F)
    print('---------------------------------')

    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

    cv2.imshow('Epipolar geometry 1', img5)

    cv2.imshow('Epipolar geometry 2', img3)

    #plt.subplot(121), plt.imshow(img5)
    #plt.subplot(122), plt.imshow(img3)
    #plt.show()

    if cv2.waitKey(1000) & 0xff == ord('q'):
        break

# When everything done, release the capture
images0.release()
images1.release()

cv2.destroyAllWindows()