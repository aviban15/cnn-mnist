# CNN MNIST Classification with PyTorch

A Convolutional Neural Network implementation for MNIST digit classification using PyTorch. This project demonstrates a lightweight CNN architecture that achieves over 99% accuracy on the MNIST test dataset.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Data Transformations](#data-transformations)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Usage](#usage)

## 🎯 Overview

This project implements a custom CNN architecture for handwritten digit recognition using the MNIST dataset. The model is designed to be efficient with **less than 25,000 (24,488) parameters** while achieving excellent classification performance.

## 📊 Dataset

- **Dataset**: MNIST Handwritten Digits
- **Training samples**: 60,000 images
- **Test samples**: 10,000 images
- **Image size**: 28×28 pixels (grayscale)
- **Classes**: 10 digits (0-9)

## 🏗️ Model Architecture

The CNN model consists of the following layers:

```
Net(
  (conv1): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1))
  (conv3): Conv2d(16, 24, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=384, out_features=50, bias=True)
  (fc2): Linear(in_features=50, out_features=10, bias=True)
)
```

### Architecture Details:

| Layer | Type | Output Shape | Parameters |
|-------|------|-------------|------------|
| Conv2d-1 | Convolution | [-1, 8, 26, 26] | 80 |
| Conv2d-2 | Convolution | [-1, 16, 11, 11] | 1,168 |
| Conv2d-3 | Convolution | [-1, 24, 9, 9] | 3,480 |
| Linear-4 | Fully Connected | [-1, 50] | 19,250 |
| Linear-5 | Output | [-1, 10] | 510 |

**Total Parameters**: 24,488 (all trainable)
**Model Size**: ~0.09 MB

### Forward Pass:
1. **Input**: 28×28×1 grayscale image
2. **Conv1**: 3×3 convolution → ReLU → 8×26×26
3. **MaxPool1**: 2×2 pooling → 8×13×13
4. **Conv2**: 3×3 convolution → ReLU → 16×11×11
5. **Conv3**: 3×3 convolution → ReLU → 24×9×9
6. **MaxPool2**: 2×2 pooling → 24×4×4
7. **Flatten**: → 384 features
8. **FC1**: Fully connected → ReLU → 50 features
9. **FC2**: Output layer → Log Softmax → 10 classes

## 🔄 Data Transformations

### Training Transformations:
- **Random Center Crop**: 22×22 pixels (10% probability)
- **Resize**: Back to 28×28
- **Random Rotation**: ±15 degrees
- **Normalization**: Mean=0.1307, Std=0.3081
- **Tensor Conversion**

### Test Transformations:
- **Normalization**: Mean=0.1407, Std=0.4081
- **Tensor Conversion**

## ⚙️ Training Configuration

- **Optimizer**: Adam (lr=0.01)
- **Scheduler**: StepLR (step_size=5, gamma=0.1)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 256
- **Epochs**: 10
- **Device**: GPU/CPU (based on CUDA availability)

## 📈 Results

### Training Progress:

| Epoch | Train Accuracy | Test Accuracy | Test Loss |
|-------|---------------|---------------|-----------|
| 1 | 91.21% | 98.01% | 0.0003 |
| 2 | 97.13% | 98.15% | 0.0002 |
| 3 | 97.74% | 98.57% | 0.0002 |
| 4 | 98.03% | 98.77% | 0.0002 |
| 5 | 98.09% | 98.77% | 0.0001 |
| 6 | 98.76% | 99.08% | 0.0001 |
| 7 | 98.92% | 99.14% | 0.0001 |
| 8 | 98.97% | 99.10% | 0.0001 |
| 9 | 99.02% | 99.26% | 0.0001 |
| 10 | 99.07% | 99.13% | 0.0001 |

### Final Performance:
- **Best Test Accuracy**: 99.26% (Epoch 9)
- **Final Test Accuracy**: 99.13%
- **Final Train Accuracy**: 99.07%

### Key Achievements:
- ✅ Achieved >99% test accuracy, starting >95% in the first epoch
- ✅ Efficient model with <25K parameters
- ✅ Good generalization (minimal overfitting)
- ✅ Stable training progression

## 📁 Project Structure

```
cnn-mnist/
├── notebooks/
│   └── CNN_MNIST_PyTorch.ipynb    # Main notebook with implementation
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## 📦 Dependencies

- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities
- `torchsummary` - Model summary visualization
- `matplotlib` - Plotting and visualization
- `tqdm` - Progress bars
- `numpy` - Numerical computations

## 🚀 Usage

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/CNN_MNIST_PyTorch.ipynb
   ```

2. Run all cells sequentially to:
   - Load and visualize the MNIST dataset
   - Define the CNN architecture
   - Train the model for 10 epochs
   - Evaluate performance and plot results

3. The notebook includes:
   - Data loading and preprocessing
   - Model architecture definition
   - Training and testing loops
   - Performance visualization
   - Sample predictions display

## 📊 Model Performance Analysis

The model demonstrates excellent performance characteristics:

- **Quick Convergence**: Reaches 98%+ test accuracy by epoch 1
- **Stable Learning**: Consistent improvement without significant fluctuations
- **Good Generalization**: Test accuracy closely follows training accuracy
- **Efficient Architecture**: High performance with minimal parameters

The learning rate scheduler effectively reduces the learning rate after 5 epochs, leading to fine-tuned performance in the later stages of training.

---

*This project was developed to demonstrate practical CNN implementation for image classification tasks.*