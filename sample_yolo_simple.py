#!/usr/bin/python3

# Copyright (C) 2022 Infineon Technologies & pmdtechnologies ag
#
# THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
# KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
# PARTICULAR PURPOSE.

"""This sample shows how to use openCV on the depthdata we get back from either a camera or an rrf file.
The Camera's lens parameters are optionally used to remove the lens distortion and then the image is displayed using openCV windows.
Press 'd' on the keyboard to toggle the distortion while a window is selected. Press esc to exit.

Additionally this sample implements the YOLO v3 network for object detection. We convert the image to rgb and then feed this image
into the network. Then we draw bounding boxes around the found object.
"""

import sys

import numpy as np
import cv2

# insert the path to your Royale installation here:
# note that you need to use \\ or / instead of \ on Windows
ROYALE_DIR = "C:/Program Files/royale/5.4.0.2112/python"
sys.path.append(ROYALE_DIR)

import roypy
from roypy_sample_utils import CameraOpener, add_camera_opener_options
from roypy_platform_utils import PlatformHelper
from roypy_platform_utils import PlatformHelper
from roypy_convenience import RoyaleCapture

# YOLO code from: https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolo_opencv.py
CLASSES = None
with open("yoloclasses.txt", 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(CLASSES[class_id]) + " : " + "{:.2f}".format(confidence)
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def detectObjects(img):
    Width = img.shape[1]
    Height = img.shape[0]
    scale = 1/255

    blob = cv2.dnn.blobFromImage(img, scale, (416,416), (0,0,0), False, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.01
    nms_threshold = 0.5

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        
    return img

def main ():
    cap = RoyaleCapture()
    while True:
        grayImg, depthImg = cap.record_data()

        # convert the image to rgb first, because YOLO needs 3 channels, and then detect the objects
        yoloResultImageGray = detectObjects(cv2.cvtColor(grayImg, cv2.COLOR_GRAY2RGB))

        cv2.imshow("YOLO Object Detection", yoloResultImageGray)

        currentKey = cv2.waitKey(1)
        # close if escape key pressed
        if currentKey == 27: 
            break
    cap.end_recording()
    print("Done")

if (__name__ == "__main__"):
    main()