import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from trainloader import CustomImageDataset
from modelcnn_a import cnn
import torchvision.transforms as transforms
import sys

transform = transforms.Compose([
    transforms.Resize((50, 100)),
    transforms.ToTensor(),
])

def train_model(train_dataset_root, save_weights_path):
    torch.manual_seed(0) 

    train_dataset = CustomImageDataset(root_dir=train_dataset_root, csv=os.path.join(train_dataset_root, "public_train.csv"), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

    model = cnn().float()
    cel = nn.BCEWithLogitsLoss()
    adam_optim = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for _ in range(8):
        model.train()
        for images, labels in train_loader:
            images, labels = images.float(), labels.float()  

            adam_optim.zero_grad()
            outputs = model(images)
            loss = cel(outputs.squeeze(), labels)
            loss.backward()
            adam_optim.step()

        torch.save(model.state_dict(), os.path.join(save_weights_path, f"part_a_binary_model.pth"))

if __name__ == "__main__":

    train_dataset_root = sys.argv[1]
    save_weights_path = sys.argv[2]

    train_model(train_dataset_root, save_weights_path)
