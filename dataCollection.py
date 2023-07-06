import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "images/y"
count = 0         

while True:
    success, img = cap.read()
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

        else:
            k = imgSize/w
            hCat = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCat))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCat)/2)
            imgWhite[hGap:hCat + hGap, :] = imgResize

        cv2.imshow("hand", imgCrop)
        cv2.imshow("Image White", imgWhite)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    
    if key == ord(' '):
        count += 1               
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print(count)
    if key == ord('q'):
        exit()