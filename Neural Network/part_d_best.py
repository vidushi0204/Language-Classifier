import numpy as np
import pickle
import time 
import sys

def sigmoid(x):
    # cap the value of x to prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss2(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]
def cross_entropy_loss(y_true, y_pred, weights, reg_lambda):
    """
    Cross-entropy loss with L2 regularization.
    
    y_true: true labels (one-hot encoded).
    y_pred: predicted labels (softmax output).
    weights: list of weight matrices for regularization.
    reg_lambda: regularization strength (lambda).
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    cross_entropy = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    # L2 regularization term: sum of squares of all weights
    l2_regularization = (reg_lambda / 2) * sum(np.sum(W ** 2) for W in weights)
    
    # Total loss = cross-entropy loss + regularization term
    return cross_entropy + l2_regularization

# In your notebook
import os 
# Import the required classes and functions from preprocess.py
from preprocessor import CustomImageDataset, DataLoader, numpy_transform

class NeuralNetworkAdam:
    def __init__(self, layer_sizes, learning_rate=0.001, reg_lambda=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.num_layers = len(layer_sizes) - 1  # Input + hidden + output
        self.weights = []
        self.biases = []
        self.reg_lambda = reg_lambda  # Regularization strength
        self.ln = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = []  # First moment estimate for weights
        self.v_w = []  # Second moment estimate for weights
        self.m_b = []  # First moment estimate for biases
        self.v_b = []  # Second moment estimate for biases
        self.t = 0  # Time step (for bias correction)

        # Initialize weights, biases, and moment estimates for each layer
        for i in range(self.num_layers):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)
            self.m_w.append(np.zeros_like(W))
            self.v_w.append(np.zeros_like(W))
            self.m_b.append(np.zeros_like(b))
            self.v_b.append(np.zeros_like(b))

    def forward(self, X):
        self.activations = [X]
        self.zs = []

        for i in range(self.num_layers - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = sigmoid(z)
            self.zs.append(z)
            self.activations.append(a)

        # Final layer (softmax)
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        y_pred = softmax(z)
        self.zs.append(z)
        self.activations.append(y_pred)

        return y_pred
    
    def backward(self, X, y, y_pred):
        m = X.shape[0]
        dz = y_pred - y

        # Update weights and biases for the last layer
        dW = (1 / m) * np.dot(self.activations[-2].T, dz) + (self.reg_lambda / m) * self.weights[-1]
        db = (1 / m) * np.sum(dz, axis=0, keepdims=True)

        # Perform Adam update for the last layer
        self._adam_update(dW, db, -1)

        # Backpropagation for the hidden layers
        for i in range(self.num_layers - 2, -1, -1):
            dz = np.dot(dz, self.weights[i+1].T) * sigmoid_derivative(self.zs[i])
            dW = (1 / m) * np.dot(self.activations[i].T, dz) + (self.reg_lambda / m) * self.weights[i]
            db = (1 / m) * np.sum(dz, axis=0, keepdims=True)

            # Perform Adam update for hidden layers
            self._adam_update(dW, db, i)

    def _adam_update(self, dW, db, layer_index):
        # Increment time step
        self.t += 1
        
        # Update biased first moment estimate
        self.m_w[layer_index] = self.beta1 * self.m_w[layer_index] + (1 - self.beta1) * dW
        self.m_b[layer_index] = self.beta1 * self.m_b[layer_index] + (1 - self.beta1) * db

        # Update biased second moment estimate
        self.v_w[layer_index] = self.beta2 * self.v_w[layer_index] + (1 - self.beta2) * (dW ** 2)
        self.v_b[layer_index] = self.beta2 * self.v_b[layer_index] + (1 - self.beta2) * (db ** 2)

        # Bias correction
        m_w_hat = self.m_w[layer_index] / (1 - self.beta1 ** self.t)
        m_b_hat = self.m_b[layer_index] / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w[layer_index] / (1 - self.beta2 ** self.t)
        v_b_hat = self.v_b[layer_index] / (1 - self.beta2 ** self.t)

        # Update weights and biases using Adam's update rule
        self.weights[layer_index] -= self.ln * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        self.biases[layer_index] -= self.ln * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def train_step(self, X_batch, y_batch):
        num_classes = self.weights[-1].shape[1]
        y_batch_one_hot = one_hot_encode(y_batch, num_classes)
        y_pred = self.forward(X_batch)
        loss = cross_entropy_loss(y_batch_one_hot, y_pred, self.weights, self.reg_lambda)
        self.backward(X_batch, y_batch_one_hot, y_pred)
        return loss

    def save_weights(self, filename='weights_adam.pkl'):
        weights_and_biases = {
            'weights': self.weights,
            'biases': [b.reshape(-1) for b in self.biases]  # Flatten biases
        }
        with open(filename, 'wb') as f:
            pickle.dump(weights_and_biases, f)

def run(hidden_layers, reg_lambda, learning_rate, train_loader, test_loader, save_weights_path, save_predictions_path):
    input_size = 625
    output_size = 8
    layer_sizes = [input_size] + hidden_layers + [output_size]
    network = NeuralNetworkAdam(layer_sizes, learning_rate, reg_lambda)
    epochs = 100

    start_time = time.time()
    batch_size = 256

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            loss = network.train_step(X_batch, y_batch)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        elapsed_time = time.time() - start_time
        if elapsed_time >= 60 * 14:  # Stop after 15 minutes
            print("Training stopped after 15 minutes.")
            break

    # Save the trained model weights
    network.save_weights(save_weights_path + "/weights.pkl")

    # Test the model
    all_predictions = []
    for X in test_loader:
        predictions = network.forward(X)
        class_predictions = np.argmax(predictions, axis=1)
        all_predictions.extend(class_predictions)

    with open(save_predictions_path + "/predictions.pkl", 'wb') as f:
        pickle.dump(all_predictions , f)

from part_d_testloader import TestImageDataset, TestDataLoader, numpy_transform
from part_d_trainloader import TrainImageDataset, TrainDataLoader

if __name__ == "__main__":
    # Argument parser
    
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    save_weights = sys.argv[3]
    save_predictions = sys.argv[4]

    train_csv_path = train_path+ "/train.csv"
    test_csv_path = test_path + "/val.csv"

    train_dataset = TrainImageDataset(root_dir=train_path, csv=train_csv_path, transform=numpy_transform)
    test_dataset = TestImageDataset(root_dir=test_path, csv=test_csv_path, transform=numpy_transform)

    train_loader = TrainDataLoader(train_dataset, batch_size=256)
    test_loader = TestDataLoader(test_dataset, batch_size=256)

    # Hyperparameters
    hidden_layerss = [[400, 200, 100, 50]]
    reg_lambdas = [0.001]
    learning_rates = [0.0012]

    # Run training and testing
    for hidden_layers in hidden_layerss:
        for reg_lambda in reg_lambdas:
            for learning_rate in learning_rates:
                run(hidden_layers, reg_lambda, learning_rate, train_loader, test_loader, save_weights, save_predictions)


    