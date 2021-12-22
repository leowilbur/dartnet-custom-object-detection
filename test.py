import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from PIL import Image

net = cv2.dnn.readNetFromDarknet("cfg/yolov3-tiny.cfg", "backup/yolov3-tiny_20000.weights")
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

LABELS = open("data/yolo.names").read().strip().split("\n")
image = cv2.imread("data/images/CA1_jpg.rf.946edc5c5ab7e19b61e00806b66b44cf.jpg")
(H, W) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
       swapRB=True, crop=False)
net.setInput(blob)
layerOutputs = net.forward(ln)
# Initializing for getting box coordinates, confidences, classid 
boxes = []
confidences = []
classIDs = []
threshold = 0.15


for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > threshold:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")           
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))    
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)
idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)

mc = 0
nmc = 0
print(idxs)
if len(idxs) > 0:
   for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        print(LABELS[classIDs[i]])
        cv2.rectangle(image, (x, y), (x + w, y + h),  (0, 255, 0), 1)
        text = "{}".format(LABELS[classIDs[i]])
        cv2.putText(image, text, (x + w, y + h),                     
        cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1)
        cv2.imshow("Image", image)
        input("Press Enter to continue...") 
