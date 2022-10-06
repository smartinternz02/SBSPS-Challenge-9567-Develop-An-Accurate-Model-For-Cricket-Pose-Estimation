import math
import cv2 as cv
import mediapipe as mp
import time
import numpy as np
from gtts import gTTS
from playsound import playsound
import os, threading

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
pTime = 0
k = 0
p = 0
cap = cv.VideoCapture('bowling_2.mp4')

def angle(lists):
    # [12 shoulder landmark, 14 elbow landmark, 16 wrist landmark]
    try:
        v1 = np.array([lists[0][0] - lists[1][0], lists[0][1] - lists[1][1]])
        v2 = np.array([lists[2][0] - lists[1][0], lists[2][1] - lists[1][1]])

        cos = v1.dot(v2) / ((np.sqrt(v1[0]**2 + v1[1]**2)) * (np.sqrt(v2[0]**2 + v2[1]**2)))

        angle = math.acos(cos) * (180/np.pi)
    except:
        angle = 180
    return angle

check = 0
skip = 0

def voice():
    if check == 1:
        language = 'en'
        my_obj = gTTS(text='not a legal delivery', lang=language, slow=False)
        my_obj.save("welcome.mp3")
        playsound('welcome.mp3')
        os.remove('welcome.mp3')

while True:
    hand = []
    _, img = cap.read() # BGR Image

    if skip < 32:
        skip += 1
        continue
    skip += 1

    img = cv.resize(img, (700, 500))
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = pose.process(imgRGB)
    height, width, channels = img.shape

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, str(int(fps)), (70, 50),
               cv.FONT_HERSHEY_PLAIN, 3,
               (255, 0, 0), 3)

    if result.pose_landmarks:
        mpDraw.draw_landmarks(img,
                              result.pose_landmarks,
                              mpPose.POSE_CONNECTIONS)

        k = 0
        for idx, lm in enumerate(result.pose_landmarks.landmark):
            cx, cy = int(lm.x * width), int(lm.y * height)
            if k == 12 or k == 14 or k == 16:
                hand.append([cx, cy])
            cv.circle(img, (cx, cy), 3, (255, 0, 0), cv.FILLED)
            k += 1
    else:
        k = k+1

    le = angle(hand)
    try:
        if hand[0][1] >= hand[1][1]:
            if le < 130:
                p = 1
                check += 1
    except:
        print('error')

    if p == 1:
        cv.putText(img, 'chucked', (500, 100),
                   cv.FONT_HERSHEY_PLAIN, 2,
                   (0, 0, 255), 2)
        cv.circle(img, (hand[0][0], hand[0][1]), 3, (0, 0, 255), cv.FILLED)
        cv.circle(img, (hand[1][0], hand[1][1]), 3, (0, 0, 255), cv.FILLED)
        cv.circle(img, (hand[2][0], hand[2][1]), 3, (0, 0, 255), cv.FILLED)

        t1 = threading.Thread(target=voice)
        t1.start()

    print('angle', le)

    cv.imshow("img", img)
    # mpDraw.plot_landmarks(result.pose_world_landmarks, mpPose.POSE_CONNECTIONS)
    cv.waitKey(1)
