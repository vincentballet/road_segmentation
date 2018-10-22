"""
[DEPRECATED] CNN model.
"""
import torch.nn as nn
import torch.nn.functional as F
NUM_LABELS = 2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=NUM_CHANNELS, out_channels=6, kernel_size=5, padding=10)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=5, padding=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 11 * 11, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_LABELS)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,8 * 11 * 11)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
   