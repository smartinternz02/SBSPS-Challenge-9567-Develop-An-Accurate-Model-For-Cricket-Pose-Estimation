import torch.nn as nn
import math
import torch
from OneD_CNN import CNN1D


class CNN3D(nn.Module):
    def __init__(self, size, frames, num_classes=6, in_channels=3):
        super(CNN3D, self).__init__()
        self.layer_1D = CNN1D(frames)
        self.group1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)))

        last_duration = int(math.floor(frames / 16))
        last_size = int(math.ceil(size / 32))
        self.fc1 = nn.Sequential(
            nn.Linear((512 * last_duration * last_size * last_size), 120),
            nn.ReLU(),
            nn.Dropout(0.1))
        self.fc2 = nn.Sequential(
            nn.Linear(120, 120),
            nn.ReLU(),
            nn.Dropout(0.1))
        self.fc3 = nn.Sequential(
            nn.Linear(240, 120),
            nn.ReLU(),
            nn.Dropout(0.1))
        self.fc = nn.Sequential(
            nn.Linear(120, num_classes))

    def forward(self, x, landmarks):
        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        out = self.group5(out)

        # flatten the layer
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        # output of 1D CNN network
        out1D = self.layer_1D(landmarks)

        # concatenate output layers
        out = torch.concat((out, out1D), dim=1)
        out = self.fc3(out)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    net = CNN3D(200, 32, 5, 3).to("cuda")
    inputs = torch.randn(1, 3, 32, 200, 200).to("cuda")
    landmark = torch.randn(1, 32, 66).to('cuda')
    print(net(inputs, landmark).shape)