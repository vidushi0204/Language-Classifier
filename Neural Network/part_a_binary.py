import pandas as pd 
import numpy as np
import os 
import sys
import pickle
from preprocessor import CustomImageDataset, DataLoader, numpy_transform



# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of sigmoid function
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Binary Cross-Entropy Loss function
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-9  # to prevent log(0)
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

# Xavier Initialization for weights
def xavier_init(size_in, size_out):
    return np.random.randn(size_in, size_out) * np.sqrt(2.0 / size_in)

# Initialize biases to zeros
def init_bias(size_out):
    return np.zeros((size_out,))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        np.random.seed(0)
        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2 / input_size) # 625, 512
        self.b1 = np.zeros((1, hidden_size1)) # 1, 512
        
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2 / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2)) 
        
        self.W3 = np.random.randn(hidden_size2, hidden_size3) * np.sqrt(2 / hidden_size2)
        self.b3 = np.zeros((1, hidden_size3))
        
        self.W4 = np.random.randn(hidden_size3, output_size) * np.sqrt(2 / hidden_size3) # 128, 1
        self.b4 = np.zeros((1, output_size))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1 # 25 , 625 * 625, 512 = 25, 512
        self.a1 = sigmoid(self.z1) 
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2 # 25, 512 * 512, 256 = 25, 256
        self.a2 = sigmoid(self.z2)
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3 # 25, 256 * 256, 128 = 25, 128
        self.a3 = sigmoid(self.z3)
        
        self.z4 = np.dot(self.a3, self.W4) + self.b4 # 25, 128 * 128, 1 = 25, 1
        y_pred = sigmoid(self.z4)
        
        return y_pred
    
    def backward(self, X, y, y_pred, learning_rate):
        m = X.shape[0]
        # print(y_pred.shape)
        y = y.reshape(y_pred.shape)
        # print(y.shape)
        dz4 = y_pred - y
        # print(dz4.shape)
        dW4 = (1 / m) * np.dot(self.a3.T, dz4)  # 128 ,25 * 25, 1 = 128, 1
        db4 = (1 / m) * np.sum(dz4, axis=0, keepdims=True)
        
        dz3 = np.dot(dz4, self.W4.T) * sigmoid_derivative(self.z3) # 25, 1 * 1, 128 = 25, 128
        dW3 = (1 / m) * np.dot(self.a2.T, dz3)  # 256, 25 * 25, 128 = 256, 128
        db3 = (1 / m) * np.sum(dz3, axis=0, keepdims=True) 
        
        dz2 = np.dot(dz3, self.W3.T) * sigmoid_derivative(self.z2)
        dW2 = (1 / m) * np.dot(self.a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)
        
        dz1 = np.dot(dz2, self.W2.T) * sigmoid_derivative(self.z1)
        dW1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)
        
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W4 -= learning_rate * dW4 # 
        self.b4 -= learning_rate * db4


    def train_step(self, X_batch, y_batch, learning_rate):
        y_pred = self.forward(X_batch)
        loss = binary_cross_entropy(y_batch, y_pred)
        self.backward(X_batch, y_batch, y_pred, learning_rate)
        return loss

    def save_weights(self, filename='weights.pkl'):
        # reshape the biases to 1D array
        self.save_b1 = self.b1.reshape(-1)
        self.save_b2 = self.b2.reshape(-1)
        self.save_b3 = self.b3.reshape(-1)
        self.save_b4 = self.b4.reshape(-1)
        weights = {
            'fc1': self.W1,
            'fc2': self.W2,
            'fc3': self.W3,
            'fc4': self.W4
        }
        biases = {
            'fc1': self.save_b1,
            'fc2': self.save_b2,
            'fc3': self.save_b3,
            'fc4': self.save_b4
        }
        with open(filename, 'wb') as f:
            pickle.dump({'weights': weights, 'bias': biases}, f)

if __name__ == '__main__':
    # Define the root directory and CSV file for the dataset
    root_dir = sys.argv[1]
    weights_path = sys.argv[2]
    csv_path = os.path.join(root_dir, "train.csv")  # or "val.csv" depending on mode

    # Create the custom dataset
    dataset = CustomImageDataset(root_dir=root_dir, csv=csv_path, transform=numpy_transform)

    # Create the DataLoader
    batch_size = 256
    dataloader = DataLoader(dataset, batch_size=batch_size)
    network = NeuralNetwork(input_size=625, hidden_size1=512, hidden_size2=256, hidden_size3=128, output_size=1)


    epochs = 15
    learning_rate = 0.001

    # network.save_weights(weights_path+'/initial_weights.pkl')
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            loss = network.train_step(X_batch, y_batch, learning_rate)
            epoch_loss += loss
        # network.save_weights(weights_path+ f'/weights_{epoch+1}.pkl')
        
        # avg_loss = epoch_loss / len(dataloader)
        # print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    network.save_weights(weights_path + "/weights.pkl")

