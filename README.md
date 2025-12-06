# Deepfake Cat Detector 🐱🔍

A machine learning project that uses neural networks to detect deepfake cat images, distinguishing between real and artificially generated cat photos.

## What is this?

This project implements a **Multi-Layer Perceptron (MLP)** neural network using PyTorch to classify cat images as either real or fake (deepfake). The model analyzes 32x32 pixel RGB cat images and outputs a binary classification with high accuracy.

The project includes:
- A custom PyTorch neural network implementation
- K-fold cross-validation for robust model evaluation
- Hyperparameter tuning and optimization
- Comparison with traditional machine learning baselines
- Early stopping to prevent overfitting

## Key Features

### 🧠 Neural Network Architecture
- **2-layer MLP** with 128 and 64 hidden units
- ReLU activation functions for non-linearity
- Dropout and weight decay for regularization
- Binary classification output (real/fake)

### 📊 Robust Evaluation
- **5-fold cross-validation** for reliable performance metrics
- **Early stopping** with patience parameter to prevent overfitting
- Standard error reporting alongside accuracy metrics
- Comparison with sklearn logistic regression baseline

### ⚙️ Hyperparameter Optimization
- Learning rate tuning (`0.001`, `0.01`, `0.1`)
- Weight decay regularization (`0.0`, `0.01`)
- Batch size optimization
- Comprehensive grid search evaluation

### 🔍 Data Handling
- **2,000 cat images** (1,000 real, 1,000 fake)
- Automatic data preprocessing and normalization
- Efficient PyTorch DataLoader implementation
- Image visualization capabilities

## Why?

### The Problem
With the rapid advancement of generative AI and deepfake technology, distinguishing between real and artificially generated content has become increasingly challenging. This is particularly relevant for:

- **Content verification** on social media platforms
- **Digital forensics** and authenticity verification
- **Research** into AI-generated content detection
- **Educational purposes** to understand neural network classification

### The Solution
This project demonstrates how traditional machine learning techniques can be effective for binary classification tasks, even in the era of sophisticated generative models. By focusing on cat images, we create a controlled environment to:

- Study the effectiveness of MLPs vs. traditional classifiers
- Understand the impact of various hyperparameters
- Implement best practices for model evaluation
- Explore the balance between model complexity and performance

## Getting Started

### Prerequisites
- Python 3.7+
- PyTorch
- scikit-learn
- matplotlib
- pandas
- Jupyter Notebook

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Deepfake-Cat-Detector.git
cd Deepfake-Cat-Detector
```

2. **Install dependencies:**
```bash
pip install torch torchvision scikit-learn matplotlib pandas jupyter
```

3. **Download the dataset:**
   - Download `hw2_data.pt` from the provided source
   - Place it in the project root directory

### Usage

1. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

2. **Open and run `hw2.ipynb`:**
   - The notebook contains all code and explanations
   - Run cells sequentially to reproduce results
   - Modify hyperparameters as needed for experimentation

3. **Key sections to explore:**
   - **Data Loading & Visualization:** See sample real vs fake cat images
   - **Baseline Model:** Logistic regression performance benchmark
   - **Neural Network:** Custom MLP implementation and training
   - **Hyperparameter Tuning:** Grid search results and analysis
   - **Results:** Comprehensive accuracy metrics and comparisons

### Expected Results
- **Baseline (Logistic Regression):** ~63.4% validation accuracy
- **Neural Network:** Improved performance with proper hyperparameter tuning
- **Cross-validation:** Robust evaluation across multiple data splits

### Customization

You can modify the neural network architecture by adjusting:
- Number of hidden layers
- Hidden unit sizes
- Activation functions
- Learning rates and optimization parameters
- Early stopping patience

---

