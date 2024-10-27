import os
import time  # Import the time module
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from trainloader import CustomImageDataset  # Adjust the import path as necessary
from modelcnn_c import CNN  # Import the cnn model class
import torchvision.transforms as transforms
import sys
# Set manual seed for reproducibility
torch.manual_seed(0)

def train_model(train_dataset_root, save_weights_path):
    transform = transforms.Compose([
        transforms.Resize((50, 100)),
        transforms.ToTensor(),
    ])

    # Create dataset and dataloader
    train_dataset = CustomImageDataset(
        root_dir=train_dataset_root,
        csv=os.path.join(train_dataset_root, "public_train.csv"),
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

    # Initialize model, loss, and optimizer
    print("MODEL")
    model = CNN().float()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Start total timer
    total_start_time = time.time()

    # Training loop
    num_epoch = 80
    for epoch in range(1,num_epoch+1):
        print("Epoch: ", epoch, end=" ")
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.float(), labels.long()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total

        print(f' Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%', end=", ")

        # Calculate and print the total time since the start
        total_elapsed_time = time.time() - total_start_time
        print(f'Time since start: {total_elapsed_time:.2f} seconds')

        # Save model weights every 3 epochs starting from epoch 9
        if (epoch>=num_epoch or total_elapsed_time>1700):
            torch.save(model.state_dict(), os.path.join(save_weights_path, "part_c_multi_model.pth"))
        if(total_elapsed_time>1750):
            break

if __name__ == '__main__':
    train_dataset_root = sys.argv[1]
    save_weights_path = sys.argv[2]
    train_model(train_dataset_root, save_weights_path)
