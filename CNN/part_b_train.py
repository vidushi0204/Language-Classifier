import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from trainloader import CustomImageDataset  # Adjust the import path as necessary
from modelcnn_b import cnn  # Import the cnn model class
import numpy as np
import torchvision.transforms as transforms
import sys

transform = transforms.Compose([
    transforms.Resize((50, 100)),
    transforms.ToTensor(),
])

# Set manual seed for reproducibility
torch.manual_seed(0)

def train_model(train_dataset_root, save_weights_path):
    # Create dataset and dataloader
    train_dataset = CustomImageDataset(root_dir=train_dataset_root, csv=os.path.join(train_dataset_root, "public_train.csv"),transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)


    # Initialize model, loss, and optimizer
    model = cnn().float()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(8):
        # print(epoch)
        model.train()
        
        for images, labels in train_loader:
            images, labels = images.float(), labels.long()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), os.path.join(save_weights_path, "part_b_multi_model.pth"))

if __name__ == '__main__':
    train_dataset_root = sys.argv[1]
    save_weights_path = sys.argv[2]
    train_model(train_dataset_root, save_weights_path)
