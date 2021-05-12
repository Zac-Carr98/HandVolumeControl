import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20.0, None)
minVol = volRange[0]
maxVol = volRange[1]

wCam, hCam = 1280, 720


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
pLength = 0
runningCounter = 0

running = True

detector = htm.HandDetector(detectionCon=0.75)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if not success:
        print("Ignoring empty frame.")
        continue
    img = detector.find_hands(img)
    lmList = detector.find_position(img, draw=False)

    if lmList:
        # if detector.detect_fist(lmList) == "Toggle":
        #     runningCounter += 1
        # 
        #     if runningCounter == 20:
        #         running = not running
        #         runningCounter = 0
        #         if running:
        #             print("Toggled On")
        #         else:
        #             print("Toggled Off")
        if running:
            if len(lmList) != 0:
                # print(lmList[4], lmList[8])

                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                length = math.hypot(x2-x1, y2-y1)

                # print(length)

                # Hand Range 25 - 250
                # Volume Range -65 - 0

                vol = np.interp(length, [20, 160], [minVol, maxVol])
                # print(int(length), vol)
                volume.SetMasterVolumeLevel(vol, None)

                if length < 20:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

                pLength = length

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 255), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)
