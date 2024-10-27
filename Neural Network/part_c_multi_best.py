import time
import os 
from preprocessor import CustomImageDataset, DataLoader, numpy_transform
import numpy as np
import pickle
import sys


def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size,  learning_rate=0.001,b1=0.9,b2=0.99, k =0.5 ,epsilon=1e-8):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size1))
        
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2 / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))
        
        self.W3 = np.random.randn(hidden_size2, hidden_size3) * np.sqrt(2 / hidden_size2)
        self.b3 = np.zeros((1, hidden_size3))
        
        self.W4 = np.random.randn(hidden_size3, hidden_size4) * np.sqrt(2 / hidden_size3)
        self.b4 = np.zeros((1, hidden_size4))
        
        self.W5 = np.random.randn(hidden_size4, output_size) * np.sqrt(2 / hidden_size4)
        self.b5 = np.zeros((1, output_size))
        self.t = 1
        self.mW1, self.mb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.mW2, self.mb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        self.mW3, self.mb3 = np.zeros_like(self.W3), np.zeros_like(self.b3)
        self.mW4, self.mb4 = np.zeros_like(self.W4), np.zeros_like(self.b4)
        self.mW5, self.mb5 = np.zeros_like(self.W5), np.zeros_like(self.b5)
        self.vW1, self.vb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.vW2, self.vb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        self.vW3, self.vb3 = np.zeros_like(self.W3), np.zeros_like(self.b3)
        self.vW4, self.vb4 = np.zeros_like(self.W4), np.zeros_like(self.b4)
        self.vW5, self.vb5 = np.zeros_like(self.W5), np.zeros_like(self.b5)
        self.beta1 = b1
        self.beta2 = b2
        self.k = k
        self.epsilon = epsilon
        self.optimizer = 'adam'
        self.learning_rate = learning_rate
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1 
        self.a1 = sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = sigmoid(self.z3)
        
        self.z4 = np.dot(self.a3, self.W4) + self.b4
        self.a4 = sigmoid(self.z4)
        
        self.z5 = np.dot(self.a4, self.W5) + self.b5
        y_pred = softmax(self.z5)
        
        return y_pred
    
    def backward(self, X, y, y_pred):
        m = X.shape[0]
        y = y.reshape(y_pred.shape)  # Ensure y is one-hot encoded
        
        dz5 = y_pred - y
        dW5 = (1 / m) * np.dot(self.a4.T, dz5)
        db5 = (1 / m) * np.sum(dz5, axis=0, keepdims=True)
        
        dz4 = np.dot(dz5, self.W5.T) * sigmoid_derivative(self.z4)
        dW4 = (1 / m) * np.dot(self.a3.T, dz4)
        db4 = (1 / m) * np.sum(dz4, axis=0, keepdims=True)
        
        dz3 = np.dot(dz4, self.W4.T) * sigmoid_derivative(self.z3)
        dW3 = (1 / m) * np.dot(self.a2.T, dz3)
        db3 = (1 / m) * np.sum(dz3, axis=0, keepdims=True)
        
        dz2 = np.dot(dz3, self.W3.T) * sigmoid_derivative(self.z2)
        dW2 = (1 / m) * np.dot(self.a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)
        
        dz1 = np.dot(dz2, self.W2.T) * sigmoid_derivative(self.z1)
        dW1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)
        
    
    
        self.t += self.k  # Increment time step

        # Adam implementation
        self.mW1 = self.beta1 * self.mW1 + (1 - self.beta1) * dW1
        self.mb1 = self.beta1 * self.mb1 + (1 - self.beta1) * db1
        self.mW2 = self.beta1 * self.mW2 + (1 - self.beta1) * dW2
        self.mb2 = self.beta1 * self.mb2 + (1 - self.beta1) * db2
        self.mW3 = self.beta1 * self.mW3 + (1 - self.beta1) * dW3
        self.mb3 = self.beta1 * self.mb3 + (1 - self.beta1) * db3
        self.mW4 = self.beta1 * self.mW4 + (1 - self.beta1) * dW4
        self.mb4 = self.beta1 * self.mb4 + (1 - self.beta1) * db4
        self.mW5 = self.beta1 * self.mW5 + (1 - self.beta1) * dW5
        self.mb5 = self.beta1 * self.mb5 + (1 - self.beta1) * db5

        self.vW1 = self.beta2 * self.vW1 + (1 - self.beta2) * (dW1 ** 2)
        self.vb1 = self.beta2 * self.vb1 + (1 - self.beta2) * (db1 ** 2)
        self.vW2 = self.beta2 * self.vW2 + (1 - self.beta2) * (dW2 ** 2)
        self.vb2 = self.beta2 * self.vb2 + (1 - self.beta2) * (db2 ** 2)
        self.vW3 = self.beta2 * self.vW3 + (1 - self.beta2) * (dW3 ** 2)
        self.vb3 = self.beta2 * self.vb3 + (1 - self.beta2) * (db3 ** 2)
        self.vW4 = self.beta2 * self.vW4 + (1 - self.beta2) * (dW4 ** 2)
        self.vb4 = self.beta2 * self.vb4 + (1 - self.beta2) * (db4 ** 2)
        self.vW5 = self.beta2 * self.vW5 + (1 - self.beta2) * (dW5 ** 2)
        self.vb5 = self.beta2 * self.vb5 + (1 - self.beta2) * (db5 ** 2)

        mW1_corr = self.mW1 / (1 - self.beta1 ** self.t)
        mb1_corr = self.mb1 / (1 - self.beta1 ** self.t)
        vW1_corr = self.vW1 / (1 - self.beta2 ** self.t)
        vb1_corr = self.vb1 / (1 - self.beta2 ** self.t)

        mW2_corr = self.mW2 / (1 - self.beta1 ** self.t)
        mb2_corr = self.mb2 / (1 - self.beta1 ** self.t)
        vW2_corr = self.vW2 / (1 - self.beta2 ** self.t)
        vb2_corr = self.vb2 / (1 - self.beta2 ** self.t)

        mW3_corr = self.mW3 / (1 - self.beta1 ** self.t)
        mb3_corr = self.mb3 / (1 - self.beta1 ** self.t)
        vW3_corr = self.vW3 / (1 - self.beta2 ** self.t)
        vb3_corr = self.vb3 / (1 - self.beta2 ** self.t)

        mW4_corr = self.mW4 / (1 - self.beta1 ** self.t)
        mb4_corr = self.mb4 / (1 - self.beta1 ** self.t)
        vW4_corr = self.vW4 / (1 - self.beta2 ** self.t)
        vb4_corr = self.vb4 / (1 - self.beta2 ** self.t)

        mW5_corr = self.mW5 / (1 - self.beta1 ** self.t)
        mb5_corr = self.mb5 / (1 - self.beta1 ** self.t)
        vW5_corr = self.vW5 / (1 - self.beta2 ** self.t)
        vb5_corr = self.vb5 / (1 - self.beta2 ** self.t)

        self.W1 -= self.learning_rate * mW1_corr / (np.sqrt(vW1_corr) + self.epsilon)
        self.b1 -= self.learning_rate * mb1_corr / (np.sqrt(vb1_corr) + self.epsilon)
        self.W2 -= self.learning_rate * mW2_corr / (np.sqrt(vW2_corr) + self.epsilon)
        self.b2 -= self.learning_rate * mb2_corr / (np.sqrt(vb2_corr) + self.epsilon)
        self.W3 -= self.learning_rate * mW3_corr / (np.sqrt(vW3_corr) + self.epsilon)
        self.b3 -= self.learning_rate * mb3_corr / (np.sqrt(vb3_corr) + self.epsilon)
        self.W4 -= self.learning_rate * mW4_corr / (np.sqrt(vW4_corr) + self.epsilon)
        self.b4 -= self.learning_rate * mb4_corr / (np.sqrt(vb4_corr) + self.epsilon)
        self.W5 -= self.learning_rate * mW5_corr / (np.sqrt(vW5_corr) + self.epsilon)
        self.b5 -= self.learning_rate * mb5_corr / (np.sqrt(vb5_corr) + self.epsilon)

    def train_step(self, X_batch, y_batch):
        num_classes = 8
        y_batch_one_hot = one_hot_encode(y_batch, num_classes)
        y_pred = self.forward(X_batch)
        loss = cross_entropy_loss(y_batch_one_hot, y_pred)
        self.backward(X_batch, y_batch_one_hot, y_pred)
        return loss


    def save_weights(self, filename):
        weights = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'W3': self.W3,
            'b3': self.b3,
            'W4': self.W4,
            'b4': self.b4,
            'W5': self.W5,
            'b5': self.b5
        }
        with open(filename, 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self, filename):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
       
if __name__ == '__main__':

    root_dir = sys.argv[1]
    weights_path = sys.argv[2]
    csv_path = os.path.join(root_dir, "train.csv")  # or "val.csv" depending on mode

    # Create the custom dataset
    dataset = CustomImageDataset(root_dir=root_dir, csv=csv_path, transform=numpy_transform)

    network = NeuralNetwork(
        input_size=625,
        hidden_size1=512,
        hidden_size2=256,
        hidden_size3=128,
        hidden_size4=32,
        output_size=8,
        learning_rate=0.0018,
        b1 = 0.9,
        b2 = 0.999,
        k = 0.5,
    )

    epochs = 100
    start_time = time.time()
    batch_size = 256
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for epoch in range(epochs):
        epoch_loss = 0.0
        i=0
        for X_batch, Y_batch in dataloader:
            i+=1
            loss = network.train_step(X_batch, Y_batch)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(dataloader)

        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        elapsed_time = time.time() - start_time
        if elapsed_time >=  60*14:
            print("Training stopped after 15 minutes.")
            break



    # final_data = DataLoader(dataset,len(dataset))
    # desc(dataset)
    # for X, y in final_data:
    #     predictions = network.forward(X)
    # print(predictions)
    # class_predictions = np.argmax(predictions, axis=1)
    # one hot encode y 
    # y = one_hot_encode(y, 8)
    # print("Final Loss", cross_entropy_loss(y, predictions))
    # print("y is",y)
    # predictions = one_hot_encode(class_predictions, 8)
    # print("predictions are",predictions)
    # print("Final Accuracy", np.mean(np.argmax(y, axis=1) == np.argmax(predictions, axis=1)))
    # print("Final Loss 2", cross_entropy_loss(y, predictions))
    # Final save of weights
    network.save_weights(weights_path + '/weights_c.pkl')
    # with open('class_predictions.pkl', 'wb') as f:
    #     pickle.dump(class_predictions, f)
