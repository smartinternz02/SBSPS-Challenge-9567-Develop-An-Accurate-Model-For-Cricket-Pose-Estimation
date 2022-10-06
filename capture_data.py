import cv2 as cv
import time

capture = cv.VideoCapture(0)

fourcc = cv.VideoWriter_fourcc('X', 'V', 'I', 'D')
videoWriter = cv.VideoWriter('poses/video18.mp4', fourcc, 30.0, (640, 480))

while True:
    ret, frame = capture.read()

    if ret:
        cv.imshow('video', frame)
        videoWriter.write(frame)

    if cv.waitKey(1) == 27:
        break

capture.release()
videoWriter.release()

cv.destroyAllWindows()