import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Adjusted layers for 50x100 input size
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)  # Output: 64 x 25 x 50
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)      # Output: 64 x 13 x 25
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2) # Output: 128 x 13 x 25
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)       # Output: 128 x 7 x 13
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) # Output: 256 x 7 x 13
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)        # Output: 256 x 4 x 7
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1) # Output: 512 x 4 x 7
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)        # Output: 512 x 2 x 4

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4096, 1024)  # Adjusted input size for fully connected layer
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 8)  # 8 output classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
