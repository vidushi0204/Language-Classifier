# Handwritten Script Classification into 8 Indian Languages

## Objective

The objective of this project is to develop and optimize a **basic Neural Network** and a **Convolutional Neural Network (CNN)** to classify handwritten scripts from **8 Indian languages**. The NNs are trained on labeled image data and tested for their ability to accurately predict the script category for unseen images. 

## CNN Model Structure and Hyperparameters

The CNN architecture consists of multiple convolutional layers for feature extraction, followed by fully connected layers for classification. Below is a table describing the structure and key hyperparameters used in the model:

| **Layer**           | **Type**               | **Details**                                                     |
|---------------------|------------------------|-----------------------------------------------------------------|
| Input               | Grayscale Image         | Size: (1, 50, 100)                                              |
| **Conv1**           | 2D Convolution          | Filters: 32, Kernel Size: 3x3, Stride: 1, Padding: 1            |
| **Activation1**     | ReLU                    | Activation Function                                             |
| **Pool1**           | Max Pooling             | Kernel Size: 2x2, Stride: 2                                     |
| **Conv2**           | 2D Convolution          | Filters: 64, Kernel Size: 3x3, Stride: 1, Padding: 1            |
| **Activation2**     | ReLU                    | Activation Function                                             |
| **Pool2**           | Max Pooling             | Kernel Size: 2x2, Stride: 2                                     |
| **Flatten**         | Reshape Layer           | Reshapes feature maps into a single vector for fully connected layers |
| **FC1**             | Fully Connected Layer   | Output: 512 neurons                                             |
| **Activation3**     | ReLU                    | Activation Function                                             |
| **FC2**             | Fully Connected Layer   | Output: 8 classes (for multi-class classification)              |

### Hyperparameter Values

The table below summarizes the hyperparameters used in training the CNN model:

| **Hyperparameter**   | **Value**                                                   |
|---------------------|-------------------------------------------------------------|
| **Learning Rate**    | 0.001                                                       |
| **Optimizer**        | Adam                                                        |
| **Batch Size**       | 128                                                         |
| **Number of Epochs** | 8                                                           |
| **Loss Function**    | Cross-Entropy Loss                                          |
| **Activation**       | ReLU (Rectified Linear Unit)                                |
| **Pooling**          | Max Pooling (Kernel Size: 2x2, Stride: 2)                   |
| **Weight Initialization** | PyTorch default initialization (He/Kaiming for ReLU)    |

## Training and Testing

1. **Training**:
    - The CNN is trained using the Adam optimizer and cross-entropy loss. The dataset is split into mini-batches of size 128, and the model is trained over 8 epochs.
    - Model weights are saved after each epoch to allow evaluation on the test set.
    
2. **Testing**:
    - Accuracy for Neural Network: 62%
    - Accuracy for Convolutional Neural Network: 86%
    
---
