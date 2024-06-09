import cvzone
import cv2
from cvzone.PoseModule import PoseDetector
import mediapipe as mp
import os

cap = cv2.VideoCapture(0)
detector = PoseDetector()

shirtPath = "Resources/Shirts"
listShirts = os.listdir(shirtPath)
print(listShirts)

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
    if lmList:
        #center = bboxInfo["center"]
        #cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
        
        lm11 = lmList[11][1:3]
        lm12 = lmList[12][1:3]
        lm13 = lmList[13][1:3]
        imgShirt = cv2.imread(os.path.join(shirtPath, listShirts[0]), cv2.IMREAD_UNCHANGED)
        #img = cvzone.overlayPNG(img, imgShirt,(100,100))
        imgShirt = cv2.resize(imgShirt,(0,0),None,0.5,0.5)
        try:
            img = cvzone.overlayPNG(img, imgShirt, lm13)
        except:
            pass
     
     
    cv2.imshow("Image", img)
    cv2.waitKey(1)