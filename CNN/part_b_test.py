import os
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader
from testloader import CustomImageDataset  # Adjust the import path as necessary
from modelcnn_b import cnn  # Import the cnn model class
import torchvision.transforms as transforms
import sys

transform = transforms.Compose([
    transforms.Resize((50, 100)),
    transforms.ToTensor(),
])
def test_model(test_dataset_root, load_weights_path, save_predictions_path):
    # Create dataset and dataloader
    test_dataset = CustomImageDataset(root_dir=test_dataset_root, csv=os.path.join(test_dataset_root, "public_test.csv"),transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize model
    model = cnn().float()
    model.load_state_dict(torch.load(os.path.join(load_weights_path,"part_b_multi_model.pth")))
    model.eval()

    predictions = []
    
    with torch.no_grad():
        for images in test_loader:
            images = images.float()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.numpy())

    # Save predictions
    with open(os.path.join(save_predictions_path,"predictions.pkl"), 'wb') as f:
        pickle.dump(np.array(predictions), f)

if __name__ == '__main__':
    test_dataset_root = sys.argv[1]
    load_weights_path = sys.argv[2]
    save_predictions_path = sys.argv[3]

    test_model(test_dataset_root, load_weights_path, save_predictions_path)
