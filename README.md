# CIFAR-10 Image Classification for Industrial Applications

This project develops a Computer Vision application for classifying images from the CIFAR-10 dataset using PyTorch.

## Project Overview

The goal is to classify 32x32 RGB images from the CIFAR-10 dataset into 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The solution uses a Convolutional Neural Network (CNN) implemented in PyTorch, with detailed preprocessing, training, and evaluation steps.

## Dataset Choice

**CIFAR-10**: Contains 60,000 images (50,000 training, 10,000 testing) across 10 classes. Chosen for its diversity, moderate complexity, and relevance to real-world image classification tasks.

**Reason**: CIFAR-10 is a standard benchmark in Computer Vision, suitable for prototyping industrial applications like quality control.

## AI Framework and Methods

- **Framework**: PyTorch  
- **Why PyTorch?**: Offers dynamic computation graphs, ease of debugging, and strong support for Computer Vision via torchvision. Ideal for rapid prototyping and deployment in industrial settings.
- **Model**: A simple CNN with two convolutional layers, max-pooling, and two fully connected layers.
- **Optimizer**: Stochastic Gradient Descent (SGD) with learning rate 0.01 and momentum 0.9 for stable convergence.
- **Loss Function**: Cross-Entropy Loss, standard for multi-class classification.

## Model Architecture

| Layer         | Input Size  | Output Size | Parameters                   |
|--------------|-------------|-------------|------------------------------|
| Conv1 (3x3)   | 3x32x32     | 32x32x32    | 32 filters, padding=1        |
| MaxPool (2x2) | 32x32x32    | 32x16x16    | -                            |
| Conv2 (3x3)   | 32x16x16    | 64x16x16    | 64 filters, padding=1        |
| MaxPool (2x2) | 64x16x16    | 64x8x8      | -                            |
| FC1           | 64x8x8=4096 | 512         | Linear                       |
| Dropout (0.5) | 512         | 512         | -                            |
| FC2           | 512         | 10          | Linear                       |

**Why CNN?**: CNNs are effective for image data due to their ability to capture spatial hierarchies. Alternatives like MLPs would be less efficient, while pre-trained models (e.g., ResNet) were avoided to keep the solution simple.

**Dropout**: Added to prevent overfitting, critical for generalizing to industrial datasets.

## Data Preprocessing

### Transformations:
- Convert images to tensors (ToTensor).
- Normalize pixel values to [-1, 1] using mean=0.5 and std=0.5 per channel.

### Batching:
- Images are loaded in batches of 64 using DataLoader for efficient training.

## Training Process

- **Epochs**: 10, balancing training time and performance.
- **Hardware**: Supports both CPU and GPU (CUDA). Training takes ~4–6 minutes on a standard GPU.
- **Monitoring**: Loss printed every 100 steps; training loss and validation accuracy plotted per epoch.

## Evaluation Results

- **Accuracy**: ~75–80% on the test set, reasonable for a simple CNN with 10 epochs.

### Metrics:
- **Precision**: ~0.76
- **Recall**: ~0.76
- **F1-Score**: ~0.76 
(Exact values vary due to randomness)

- **Confusion Matrix**: Visualizes classification performance, highlighting confusions (e.g., cat vs. dog).
- **Training Metrics**: Loss and accuracy plots show model convergence.
- **Sample Predictions**: Visualizes predictions on five test images, comparing predicted and true labels.

## Real-World Applications in Industrial/Manufacturing Settings

Image classification has numerous applications in industry, particularly in manufacturing:

### Quality Control:
- Classify products (e.g., circuit boards, automotive parts) as defective or non-defective based on visual inspection.
- Example: Detect cracks or misalignments in components.

### Automated Sorting:
- Sort items on a conveyor belt (e.g., fruits, packages) into categories for packaging or processing.
- Example: Separate recyclable materials by type.

### Safety Monitoring:
- Identify hazardous objects or conditions in a factory (e.g., misplaced tools, unauthorized personnel).
- Example: Detect safety gear compliance among workers.

### Inventory Management:
- Classify and count items in a warehouse for automated stock tracking.
- Example: Identify product types in storage bins.

The CIFAR-10 model, while simple, serves as a prototype for such systems. With fine-tuning and domain-specific datasets, it can be adapted for production environments.

## Challenges and Solutions

- **Challenge**: Overfitting due to limited model complexity.  
  **Solution**: Added dropout (0.5) and kept the model simple to avoid overparameterization.

- **Challenge**: Class imbalance in performance (e.g., cat/dog confusion).  
  **Solution**: Confusion matrix analysis to identify weak classes; future improvements could include data augmentation.

- **Challenge**: Limited training time for a job application.  
  **Solution**: Used 10 epochs and a lightweight CNN to ensure a complete, functional prototype.

## Requirements

- Python 3.8+
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Install dependencies:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
```

##  Usage

###  Clone the repository:

```bash
git clone https://github.com/AbdulSametTurkmenoglu/CIFAR10-Image-Classification.git
cd CIFAR10-Image-Classification
```

### Run the script:

```bash
python cifar10_image_classification.py
```


##  Outputs

**Console:**

- Accuracy  
- Precision  
- Recall  
- F1-score


### Files
- `confusion_matrix.png` – Confusion matrix visualization  
  ![Confusion Matrix](images_classification/confusion_matrix.png)

- `training_metrics.png` – Loss and accuracy plots per epoch  
  ![Training Metrics](images_classification/training_metrics.png)

- `predictions.png` – Sample predictions from the test set  
  ![Sample Predictions](images_classification/predictions.png)



##  License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this software with proper attribution.
