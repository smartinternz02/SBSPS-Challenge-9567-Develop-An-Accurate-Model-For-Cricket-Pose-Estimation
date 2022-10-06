import cv2 as cv
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
pTime = 0
i = 0
k = 0
cap = cv.VideoCapture('umpiring/video3.mp4')
while True:
    _, img = cap.read() # BGR Image
    #i = i+1
    #img = cv.imread("D:/cricket_shots/cut/cut_"+str(i)+".jpg")
    #img = cv.resize(img, (200, 200))
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

        for idx, lm in enumerate(result.pose_landmarks.landmark):
            cx, cy = int(lm.x * width), int(lm.y * height)
            cv.circle(img, (cx, cy), 3, (255, 0, 0), cv.FILLED)
    else:
        k = k+1
        print(k)

    cv.imshow("img", img)
    # mpDraw.plot_landmarks(result.pose_world_landmarks, mpPose.POSE_CONNECTIONS)
    cv.waitKey(1)
