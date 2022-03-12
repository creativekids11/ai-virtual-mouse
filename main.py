import math
import cv2
import mediapipe as mp
import numpy as np
import time
from pynput.mouse import Button, Controller
import autopy

mouse=Controller()
camIndex=0
cap=cv2.VideoCapture(camIndex)
mpHands=mp.solutions.hands
Hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils
fingersCoordinate=[(8,6),(12,10),(16,14),(20,18)]
thumbCoordinate=(4,3)

##########################
wScr, hScr = autopy.screen.size()
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 7
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0



while (cap.isOpened()):
    count=[]
    lmList=[]
    success_,img=cap.read()
    cvtImg=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=Hands.process(cvtImg)

    if results.multi_hand_landmarks:
        for img_in_frame in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, img_in_frame, mpHands.HAND_CONNECTIONS)
        for id,lm in enumerate(results.multi_hand_landmarks[0].landmark):
            h,w,c=img.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            lmList.append([cx,cy])
        
        if lmList[thumbCoordinate[0]][0] > lmList[thumbCoordinate[1]][0]:
            count.append(1)
        else:
            count.append(0)

        for coordinate in fingersCoordinate:
            if lmList[coordinate[0]][1] < lmList[coordinate[1]][1]:
                count.append(1)
            else:
                count.append(0)
        if lmList[thumbCoordinate[0]][0] > lmList[thumbCoordinate[1]][0]:
            count.append(1)
        else:
            count.append(0)

        count.pop(count[-1])

    if len(lmList) != 0:
        x1, y1 = lmList[8]
        x2, y2 = lmList[12]

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),(0,0,255), 2)
        # 4. Only Index Finger : Moving Mode
        if count[0] == 1 and count[1] == 0:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
        
            # 7. Move Mouse
            mouse.position=(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (0,0,255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        if count[0] == 1 and count[1] == 1:
            # 9. Find distance between fingers
            x1m,y1m=lmList[8][0],lmList[8][1]
            x2m,y2m=lmList[12][0],lmList[12][1]
            l=math.hypot(x2m-x1m-30,y2m-y1m-30)
            cv2.line(img,(x1m,y1m),(x2m,y2m),(0,0,255),3,-1)
            # 10. Click mouse if distance short
            if not l > 63:
                mouse.click(Button.left,1)
                time.sleep(0.15)
        if count[0]==1 and count[4]==1:
            mouse.scroll(0, 1)

        if count[3]==1:
            mouse.scroll(0, -1)
    print(count)

    cv2.imshow("Hand Tracking",img)

    if cv2.waitKey(1)==113: #Q=113
        break
