import pixellib
import os
import cv2
import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#from utils import label_map_util
#from utils import visualization_utils as vis_util
from pixellib.semantic import semantic_segmentation
from pixellib.instance import instance_segmentation


semantic_video = semantic_segmentation()
instance_video = instance_segmentation(infer_speed="rapid")

semantic_video.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
instance_video.load_model("mask_rcnn_coco.h5")

capture = cv2.VideoCapture(0)

#instance_video.process_camera(capture, show_bboxes=True, frames_per_second=15,  output_video_name="output_video.mp4", show_frames=True, frame_name="frame")

while True:

    ret, frame = capture.read()
    new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    segmask, new_frame = instance_video.segmentFrame(new_frame, show_bboxes=True)

    mask = segmask['masks'] * 1.0
    print(mask.shape[2])
    r, c, a = mask.shape
    mascara = np.zeros((r, c, 3))
    print(mascara.shape, mask.shape)

    for i in range(mask.shape[2]):
        mascara[mask[:,:,i] == 1] = 1

    frame[mascara == 0] = 0
    cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xff == ord('q'):
        np.save('mascara', segmask['masks'])
        break