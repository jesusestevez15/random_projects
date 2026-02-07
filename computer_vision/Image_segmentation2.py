from imageai.Detection import ObjectDetection
import cv2

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("resnet50_coco_best_v2.1.0.h5")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image='ejemplo2.jpg', output_image_path='detection_ejemplo2.jpg')

cv2.imshow('detections', detections)
cv2.waitKey(0)

for eachObject in detections:
    print(eachObject["name"], ":", eachObject["percentage_probability"])