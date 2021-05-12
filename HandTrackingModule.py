import cv2
import mediapipe as mp
import time
from google.protobuf.json_format import MessageToDict

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        self.results = self.hands.process(imgRGB)
        # print('Handedness: ', self.results.multi_handedness)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            for idx, hand_handedness in enumerate(self.results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                if handedness_dict['classification'][0]['label'] == "Right":
                    myHand = self.results.multi_hand_landmarks[handNo]
                    for id, lm in enumerate(myHand.landmark):
                        # print(id, lm)
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        # print(id, cx, cy)
                        lmList.append([id, cx, cy])
                        # if id == 4:
                        if draw:
                            cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            return lmList

    def detect_fist(self, lmList=None):
        indexY = 0
        indexMid = 0
        handBottomY = 0

        if lmList:
            for lms in lmList:
                if lms[0] == 7:
                    indexX, indexY = lms[1], lms[2]
                elif lms[0] == 5:
                    indexMid = lms[2]
                elif lms[0] == 0:
                    handBottomX, handBottomY = lms[1], lms[2]
            if (indexY < handBottomY) and (indexY > indexMid):
                return "FIST"



def main():
    cap = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0

    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lmList = detector.find_position(img)
        if lmList:
            if len(lmList) != 0:
                print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
