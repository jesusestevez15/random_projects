import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2

# Cargo la máscara del objeto detectado en el programa Image_segmentation.py
# Las máscaras son matrices 3-dimensionales (x,y,z) - x: altura, y: anchura, z: objeto enmascarado (persona, botella...)
mascara = np.load('C:/Users/Jesús/Desktop/Universidad/Máster Ciencia y Tecnología Espacial/TFM/4-Segmentacion_de_imagen/mascara.npy')
mascara = mascara * 1.0
objetos = len(mascara[0,0])

mask = np.zeros((len(mascara[:]), len(mascara[0]), objetos+1))

for i in range(objetos):
    mask[:,:,i] = mascara[:,:,i]
    mask[:,:,objetos] = mask[:,:,objetos] + mascara[:,:,i]
    cv2.waitKey(0)

capture = cv2.VideoCapture(0)

while True:

    ret, frame = capture.read()

    # Proyecto la máscara al frame de la imagen
    for i in range(objetos):
        frame[mask[:,:,3] == 0] = (0, 0, 0)

    # Normalizado la imagen para que se vea bien y no saturada o quemada
    #frame = cv2.normalize(frame, frame, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Lo muestro en pantalla para verificar que se ve bien
    cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xff == ord('q'):
        break