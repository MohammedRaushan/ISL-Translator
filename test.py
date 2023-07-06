import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = ['A','B','C', 'D', 'E','F','G', 'H', 'I','J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    try:
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y ,w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h/w
            if aspectRatio > 1:
                k = imgSize/h
                wCat = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCat, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((300-wCat)/2)
                imgWhite[:, wGap:wCat+wGap] = imgResize
                predictions, index = classifier.getPrediction(imgWhite, draw=False)
                print(predictions, index)
            

            else:
                k = imgSize/w
                hCat = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCat))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCat)/2)
                imgWhite[hGap:hCat + hGap, :] = imgResize
                predictions, index = classifier.getPrediction(imgWhite, draw=False)
                print(predictions, index)

            cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+150, 50), (0, 244, 83), 4, cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x,y-20),cv2.FONT_HERSHEY_COMPLEX, 2, (0, 244, 83), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (0, 244, 83), 4)
            cv2.imshow("hand", imgCrop)
            cv2.imshow("Image White", imgWhite)
    except:
        pass

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        exit()