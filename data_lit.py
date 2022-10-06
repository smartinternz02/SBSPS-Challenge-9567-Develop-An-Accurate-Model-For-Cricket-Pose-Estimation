import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2 as cv
import mediapipe as mp

class CricketShot(Dataset):
    def __init__(self, path, frame, channels, width, height):
        super(CricketShot, self).__init__()
        self.data = pd.read_csv(path)
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.frame = frame
        self.channels = channels
        self.width = width
        self.height = height
        self.total_landmarks = 66

    def video_get(self, img_vector, landmark, path):
        """
        :param img_vector: of shape (frame, height, width, channel)
        :param landmark: of shape (frame, vector)
        :param path: img path
        :return:
        """

        cap = cv.VideoCapture(path)
        frame = 0
        while frame <= 30:
            _, img = cap.read() # BGR Image

            img = cv.resize(img, (self.width, self.height))

            imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            result = self.pose.process(imgRGB)
            height, width, channels = img.shape

            if result.pose_landmarks:
                frame += 1

                land_idx = 0
                for idx, lm in enumerate(result.pose_landmarks.landmark):
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    landmark[frame, land_idx] = cx
                    landmark[frame, land_idx+1] = cy

                    land_idx += 2

            img_vector[frame, :] = torch.tensor(imgRGB)

        return img_vector, landmark

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path = self.data.iloc[item, 0]
        label = self.data.iloc[item, 1]

        frames, landmarks = self.video_get(torch.zeros((self.frame, self.width, self.height, self.channels)),
                                           torch.zeros((self.frame, self.total_landmarks)),
                                           'bat_frames'
                                           '/'+path)

        return frames.permute(3, 0, 1, 2), landmarks, label

if __name__ == '__main__':
    dataset = CricketShot("data2.csv", 32, 3, 200, 200)
    loader = DataLoader(dataset, batch_size=1)

    for f, l, L in loader:
        print(f.shape, l.shape, L)


