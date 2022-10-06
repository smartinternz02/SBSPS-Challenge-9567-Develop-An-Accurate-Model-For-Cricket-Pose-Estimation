import torch
from ThreeD_CNN import CNN3D
import cv2 as cv
import mediapipe as mp
from gtts import gTTS
from playsound import playsound
import os, threading

# Hyperparameters
path = 'umpiring/video8.mp4'
sample_size = 200
frame = 32
channels = 3
num_classes = 9
total_landmarks = 66
mode = 'umpire'

# load the 3D-CNN model
model = CNN3D(sample_size, frame, num_classes=num_classes, in_channels=channels)
saved_checkpoint = torch.load('new.pth.tar')
model.load_state_dict(saved_checkpoint['state_dict'])
model.eval()

# preprocess the data
video = cv.VideoCapture(path)
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# store the frames and landmarks
img_vector = torch.zeros((frame, sample_size, sample_size, channels), dtype=torch.float32)
landmark = torch.zeros((frame, total_landmarks), dtype=torch.float32)
fr = 0
write = " "
scores = ['', '', '', '', '', '', '', '', '', ]
voice = ''
count = 1
counter = []

i, j, k = 255, 0, 0
di, dj, dk = 255, 0, 0
si, sj, sk = 255, 0, 0
ci, cj, ck = 255, 0, 0
oi, oj, ok = 255, 0, 0
fi, fj, fk = 255, 0, 0
pi, pj, pk = 255, 0, 0
Ci, Cj, Ck = 255, 0, 0
Si, Sj, Sk = 255, 0, 0

def voice_key():
    if voice in counter:
        pass
    else:
        if mode == 'cricket':
            my_text = 'batsman played a' + voice + 'shot'
            if voice == 'stance':
                my_text = ''

            language = 'en'
            my_obj = gTTS(text=my_text, lang=language, slow=False)

            if count % 3 == 0:
                counter.clear()

            counter.append(voice)
            my_obj.save("welcome.mp3")
            playsound('welcome.mp3')
            os.remove('welcome.mp3')
        else:
            my_text = "umpire decision is " + voice
            if voice == 'None':
                my_text = ' '

            language = 'en'
            my_obj = gTTS(text=my_text, lang=language, slow=False)

            if count % 3 == 0:
                counter.clear()

            counter.append(voice)
            my_obj.save("welcome.mp3")
            playsound('welcome.mp3')
            os.remove('welcome.mp3')

# initiate with video
while True:
    count += 1
    _, img = video.read()  # BGR Image

    if fr <= frame-2:
        try:
            imgs = cv.resize(img, (sample_size, sample_size))
        except:
            break

        imgRGB = cv.cvtColor(imgs, cv.COLOR_BGR2RGB)
        result = pose.process(imgs)
        height, width, channels = img.shape

        if result.pose_landmarks:
            fr += 1

            mpDraw.draw_landmarks(img,
                                  result.pose_landmarks,
                                  mpPose.POSE_CONNECTIONS)

            land_idx = 0
            for idx, lm in enumerate(result.pose_landmarks.landmark):
                cx, cy = int(lm.x * width), int(lm.y * height)
                landmark[fr, land_idx] = cx
                landmark[fr, land_idx + 1] = cy

                land_idx += 2
        img_vector[fr, :] = torch.tensor(imgRGB)

    else:
        scores = model(img_vector.permute(3, 0, 1, 2).unsqueeze(0), landmark.unsqueeze(0))
        score = torch.argmax(scores, dim=1)
        scores = scores[0].detach().numpy()

        # for umpiring

        # for batting
        if int(score) == 0:
            voice = 'six'
            i, j, k = 0, 0, 255
            di, dj, dk = 255, 0, 0
            si, sj, sk = 255, 0, 0
            ci, cj, ck = 255, 0, 0
            oi, oj, ok = 255, 0, 0
            fi, fj, fk = 255, 0, 0
            pi, pj, pk = 255, 0, 0
            Ci, Cj, Ck = 255, 0, 0
            Si, Sj, Sk = 255, 0, 0
        elif int(score) == 1:
            di, dj, dk = 0, 0, 255
            i, j, k = 255, 0, 0
            si, sj, sk = 255, 0, 0
            ci, cj, ck = 255, 0, 0
            oi, oj, ok = 255, 0, 0
            fi, fj, fk = 255, 0, 0
            pi, pj, pk = 255, 0, 0
            Ci, Cj, Ck = 255, 0, 0
            Si, Sj, Sk = 255, 0, 0
        elif int(score) == 2:
            si, sj, sk = 0, 0, 255
            i, j, k = 255, 0, 0
            di, dj, dk = 255, 0, 0
            ci, cj, ck = 255, 0, 0
            oi, oj, ok = 255, 0, 0
            fi, fj, fk = 255, 0, 0
            pi, pj, pk = 255, 0, 0
            Ci, Cj, Ck = 255, 0, 0
            Si, Sj, Sk = 255, 0, 0
        elif int(score) == 3:
            i, j, k = 255, 0, 0
            di, dj, dk = 255, 0, 0
            si, sj, sk = 255, 0, 0
            ci, cj, ck = 255, 0, 0
            oi, oj, ok = 255, 0, 0
            fi, fj, fk = 255, 0, 0
            pi, pj, pk = 255, 0, 0
            Ci, Cj, Ck = 255, 0, 0
            Si, Sj, Sk = 255, 0, 0
            ci, cj, ck = 0, 0, 255
        elif int(score) == 4:
            i, j, k = 255, 0, 0
            di, dj, dk = 255, 0, 0
            si, sj, sk = 255, 0, 0
            ci, cj, ck = 255, 0, 0
            oi, oj, ok = 255, 0, 0
            fi, fj, fk = 255, 0, 0
            pi, pj, pk = 255, 0, 0
            Ci, Cj, Ck = 255, 0, 0
            Si, Sj, Sk = 255, 0, 0
            oi, oj, ok = 0, 0, 255
        elif int(score) == 5:
            i, j, k = 255, 0, 0
            di, dj, dk = 255, 0, 0
            si, sj, sk = 255, 0, 0
            ci, cj, ck = 255, 0, 0
            oi, oj, ok = 255, 0, 0
            fi, fj, fk = 255, 0, 0
            pi, pj, pk = 255, 0, 0
            Ci, Cj, Ck = 255, 0, 0
            Si, Sj, Sk = 255, 0, 0
            fi, fj, fk = 0, 0, 255
        elif int(score) == 6:
            i, j, k = 255, 0, 0
            di, dj, dk = 255, 0, 0
            si, sj, sk = 255, 0, 0
            ci, cj, ck = 255, 0, 0
            oi, oj, ok = 255, 0, 0
            fi, fj, fk = 255, 0, 0
            pi, pj, pk = 255, 0, 0
            Ci, Cj, Ck = 255, 0, 0
            Si, Sj, Sk = 255, 0, 0
            pi, pj, pk = 0, 0, 255
        elif int(score) == 7:
            i, j, k = 255, 0, 0
            di, dj, dk = 255, 0, 0
            si, sj, sk = 255, 0, 0
            ci, cj, ck = 255, 0, 0
            oi, oj, ok = 255, 0, 0
            fi, fj, fk = 255, 0, 0
            pi, pj, pk = 255, 0, 0
            Ci, Cj, Ck = 255, 0, 0
            Si, Sj, Sk = 255, 0, 0
            Ci, Cj, Ck = 0, 0, 255
        elif int(score) == 8:
            i, j, k = 255, 0, 0
            di, dj, dk = 255, 0, 0
            si, sj, sk = 255, 0, 0
            ci, cj, ck = 255, 0, 0
            oi, oj, ok = 255, 0, 0
            fi, fj, fk = 255, 0, 0
            pi, pj, pk = 255, 0, 0
            Ci, Cj, Ck = 255, 0, 0
            Si, Sj, Sk = 255, 0, 0
            Si, Sj, Sk = 0, 0, 255

        if mode == 'cricket':
            if score == 0:
                voice = 'stance'
            elif score == 1:
                voice = 'defence'
            elif score == 2:
                voice = 'straight drive'
            elif score == 3:
                voice = 'cover drive'
            elif score == 4:
                voice = 'on drive'
            elif score == 5:
                voice = 'Flick'
            elif score == 6:
                voice = 'pull'
            elif score == 7:
                voice = 'cut'
            elif score == 8:
                voice = 'sweep'
        elif mode == 'umpire':
            if score == 0:
                voice = 'None'
            elif score == 1:
                voice = 'out'
            elif score == 2:
                voice = 'six'
            elif score == 3:
                voice = 'four'
            elif score == 4:
                voice = 'wide'
            elif score == 5:
                voice = 'no ball'
            elif score == 6:
                voice = 'free hit'
            elif score == 7:
                voice = 'bye'
            elif score == 8:
                voice = 'leg bye'

        fr = 0
        img_vector = torch.zeros((frame, sample_size, sample_size, channels))
        landmark = torch.zeros((frame, total_landmarks))

        t1 = threading.Thread(target=voice_key)
        t1.start()

    if mode == 'cricket':
        cv.putText(img, 'stance ', (60, 170), cv.FONT_HERSHEY_PLAIN, 1.5, (i, j, k), 2)
        cv.putText(img, 'defence ', (60, 130), cv.FONT_HERSHEY_PLAIN, 1.5, (di, dj, dk), 2)
        cv.putText(img, 'straight drive ', (60, 190), cv.FONT_HERSHEY_PLAIN, 1.5, (si, sj, sk), 2)
        cv.putText(img, 'cover drive ', (60, 110), cv.FONT_HERSHEY_PLAIN, 1.5, (ci, cj, ck), 2)
        cv.putText(img, 'on drive ', (60, 70), cv.FONT_HERSHEY_PLAIN, 1.5, (oi, oj, ok), 2)
        cv.putText(img, 'flick ', (60, 150), cv.FONT_HERSHEY_PLAIN, 1.5, (fi, fj, fk), 2)
        cv.putText(img, 'pull ', (60, 50), cv.FONT_HERSHEY_PLAIN, 1.5, (pi, pj, pk), 2)
        cv.putText(img, 'cut ', (60, 210), cv.FONT_HERSHEY_PLAIN, 1.5, (Ci, Cj, Ck), 2)
        cv.putText(img, 'sweep ', (60, 90), cv.FONT_HERSHEY_PLAIN, 1.5, (Si, Sj, Sk), 2)
    elif mode == 'umpire':
        cv.putText(img, 'None ', (60, 130), cv.FONT_HERSHEY_PLAIN, 1.5, (i, j, k), 2)
        cv.putText(img, 'Out ', (60, 110), cv.FONT_HERSHEY_PLAIN, 1.5, (di, dj, dk), 2)
        cv.putText(img, 'Six ', (60, 170), cv.FONT_HERSHEY_PLAIN, 1.5, (si, sj, sk), 2)
        cv.putText(img, 'Four ', (60, 190), cv.FONT_HERSHEY_PLAIN, 1.5, (ci, cj, ck), 2)
        cv.putText(img, 'Wide ', (60, 50), cv.FONT_HERSHEY_PLAIN, 1.5, (oi, oj, ok), 2)
        cv.putText(img, 'No Ball ', (60, 210), cv.FONT_HERSHEY_PLAIN, 1.5, (fi, fj, fk), 2)
        cv.putText(img, 'Free Hit ', (60, 90), cv.FONT_HERSHEY_PLAIN, 1.5, (pi, pj, pk), 2)
        cv.putText(img, 'Bye ', (60, 70), cv.FONT_HERSHEY_PLAIN, 1.5, (Ci, Cj, Ck), 2)
        cv.putText(img, 'Leg Bye ', (60, 150), cv.FONT_HERSHEY_PLAIN, 1.5, (Si, Sj, Sk), 2)

    cv.imshow('img', img)
    cv.waitKey(1)
