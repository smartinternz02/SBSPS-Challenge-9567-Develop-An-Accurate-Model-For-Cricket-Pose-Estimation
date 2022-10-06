import torch
import torch.nn as nn
from ThreeD_CNN import CNN3D
from data_lit import CricketShot
from torch.utils.data import DataLoader
import torch.optim as optim

# Hyperparameters
sample_size = 200
frame = 32
channels = 3
num_classes = 9
path = "batting.csv"
epochs = 50
learning_rate = 3e-4
device = "cuda"
load_model = False
save_model = True
checkpoint_path = 'batting_2.pth.tar' # 'check_point.pth.tar'
BATCH_SIZE = 1

# save the model
def save_checkpoint(state, path=checkpoint_path):
    print('The checkpoint is saved')
    torch.save(state, path)

# load the checkpoint
def load_checkpoint(saved_checkpoint):
    print('The checkpoint is loaded')
    model.load_state_dict(saved_checkpoint['state_dict'])
    optimizer.load_state_dict(saved_checkpoint['optimizer'])


# data loading
dataset = CricketShot(path, frame, channels, sample_size, sample_size)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Model
model = CNN3D(sample_size, frame, num_classes=num_classes, in_channels=channels).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()

# load the model
if load_model:
    load_checkpoint(torch.load(checkpoint_path))

# train the model
for epoch in range(epochs):
    print(f'epochs: {epoch}/{epochs}')

    if save_model and epoch % 5 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    '''
    count = 0
    accuracy = []
    model.eval()
    for idx, (train, landmark, label) in enumerate(loader):
        train = train.to(device)
        landmark = landmark.to(device)
        label = label

        score = model(train, landmark)
        ans = torch.argmax(score, dim=1)
        print(label, ans)

        if count >= 10:
            break
        count += 1
    '''

    losses = []
    for idx, (train, landmark, label) in enumerate(loader):
        train = train.to(device)
        landmark = landmark.to(device)
        label = label.to(device)

        score = model(train, landmark)
        lossVal = loss(score, label)
        losses.append(lossVal.item())

        optimizer.zero_grad()
        lossVal.backward()
        optimizer.step()

    print("losses: ", sum(losses)/len(losses))
