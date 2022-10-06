import torch.nn as nn
import torch


# define the model using pytorch
class CNN1D(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(10))
        self.layer2 = nn.Flatten()
        self.layer3 = nn.Sequential(
            nn.Linear(64*6, 120),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(120, 120),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

if __name__ == '__main__':
    inp = torch.zeros(1, 32, 66)
    model = CNN1D(32)
    print(model(inp).shape)