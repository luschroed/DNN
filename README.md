# PyTorch Neural Network – Hyperparameter Optimization

A small end-to-end deep learning project using **PyTorch** and **Optuna** to train, optimize, and evaluate a neural network on the Fashion-MNIST dataset.

The main goal of this project is to demonstrate the complete machine learning workflow:

- training a baseline neural network
- hyperparameter optimization with Optuna
- training a final model with the optimized parameters
- evaluating the model on an unseen test set
- analyzing classification errors using a confusion matrix

---

## Project Overview

The project uses a fully connected neural network to classify images from the Fashion-MNIST dataset into 10 clothing categories.

The workflow consists of three main stages:

```text
Fashion-MNIST
      │
      ▼
Baseline Model
      │
      ▼
Hyperparameter Optimization
      │
      ▼
Final Optimized Model
      │
      ▼
Test Evaluation
      │
      ▼
Error Analysis
````

---

## Dataset

The project uses the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

Fashion-MNIST contains grayscale images of clothing items with a resolution of **28 × 28 pixels**.

The dataset contains 10 classes:

| Label | Class       |
| ----: | ----------- |
|     0 | T-shirt/top |
|     1 | Trouser     |
|     2 | Pullover    |
|     3 | Dress       |
|     4 | Coat        |
|     5 | Sandal      |
|     6 | Shirt       |
|     7 | Sneaker     |
|     8 | Bag         |
|     9 | Ankle boot  |

The dataset is automatically downloaded using `torchvision`.

---

## Model Architecture

The model is a fully connected neural network.

The 28 × 28 input image is flattened into a vector of 784 features.

The architecture contains:

```text
Input
784 features
     │
     ▼
Fully Connected Layer
256 hidden units
     │
     ▼
ReLU
     │
     ▼
Dropout
     │
     ▼
Output Layer
10 classes
```

The number of hidden units and dropout rate are configurable hyperparameters.

---

## Baseline Model

Before performing hyperparameter optimization, a baseline model was trained to establish a reference performance.

The baseline achieved:

**Validation Accuracy: 88.38%**

This baseline serves as the reference point for evaluating the effect of hyperparameter optimization.

---

## Hyperparameter Optimization

Hyperparameter optimization was performed using **Optuna**.

A total of **20 trials** were evaluated.

The following hyperparameters were optimized:

| Hyperparameter | Search         |
| -------------- | -------------- |
| Learning rate  | Continuous     |
| Hidden units   | 64 / 128 / 256 |
| Dropout        | Continuous     |
| Batch size     | 32 / 64 / 128  |

The objective of the optimization was to maximize validation accuracy.

### Best Optuna Trial

The best trial achieved:

**Validation Accuracy: 87.71%**

The corresponding hyperparameters were:

```text
Learning rate: 0.0008626881131442
Hidden units:  256
Dropout:       0.0878650272774308
Batch size:    128
```

The Optuna trials are stored in:

```text
results/optuna_trials.csv
```

---

## Final Model

The best hyperparameter configuration found by Optuna was used to train the final model for 10 epochs.

The final training configuration was:

| Parameter     |        Value |
| ------------- | -----------: |
| Learning rate | 0.0008626881 |
| Hidden units  |          256 |
| Dropout       |     0.087865 |
| Batch size    |          128 |
| Epochs        |           10 |

The best validation performance was reached at **epoch 9**.

```text
Best Validation Accuracy: 88.97%
Best Epoch: 9
```

The model was automatically saved based on the best validation accuracy.

---

## Results

### Baseline vs. Optimized Model

| Model     | Validation Accuracy | Test Accuracy |
| --------- | ------------------: | ------------: |
| Baseline  |              88.38% |             – |
| Optimized |              88.97% |    **88.15%** |

Hyperparameter optimization improved validation accuracy by:

**+0.59 percentage points**

The final optimized model achieved:

**88.15% test accuracy**

on the unseen Fashion-MNIST test set.

The test set was not used during hyperparameter optimization.

---

## Training Behavior

The final model achieved its best validation accuracy at epoch 9:

```text
Epoch 8   → 88.79%
Epoch 9   → 88.97%  ← Best
Epoch 10  → 88.49%
```

At the same time, validation loss increased slightly after epoch 9.

This indicates that the model began to show signs of overfitting toward the end of training.

---

## Error Analysis

A confusion matrix was generated to analyze the model's classification behavior.

### Per-Class Accuracy

| Class       | Accuracy |
| ----------- | -------: |
| T-shirt/top |    82.0% |
| Trouser     |    96.8% |
| Pullover    |    81.9% |
| Dress       |    93.0% |
| Coat        |    82.3% |
| Sandal      |    96.2% |
| Shirt       |    64.1% |
| Sneaker     |    93.1% |
| Bag         |    96.8% |
| Ankle boot  |    95.3% |

The model performs particularly well on visually distinctive classes such as trousers, bags, sandals, and ankle boots.

The most challenging class is **Shirt**, with an accuracy of 64.1%.

### Most Common Misclassifications

The most frequent classification errors were:

```text
Shirt          → T-shirt/top : 123
T-shirt/top    → Shirt       : 111
Pullover       → Coat        : 91
Shirt          → Coat        : 87
Shirt          → Pullover    : 85
```

These results indicate that the model has difficulty distinguishing visually similar upper-body clothing categories.

In particular, the confusion between **Shirt** and **T-shirt/top** occurs in both directions, suggesting that these classes are intrinsically difficult to separate based on the available image information.

---

## Visualizations

The project generates several visualizations:

### Optuna Optimization History

Shows the validation accuracy achieved across the 20 optimization trials.

```text
results/plots/optimization_history.png
```

### Hyperparameter Analysis

The relationship between individual hyperparameters and validation accuracy can be found in:

```text
results/plots/
├── learning_rate_vs_accuracy.png
├── hidden_units_vs_accuracy.png
├── dropout_vs_accuracy.png
└── batch_size_vs_accuracy.png
```

### Baseline vs. Optimized

```text
results/plots/baseline_vs_optimized.png
```

### Confusion Matrix

```text
results/plots/confusion_matrix.png
```

---

## Project Structure

```text
DNN/
│
├── README.md
├── requirements.txt
│
├── models/
│   ├── __init__.py
│   └── model.py
│
├── train.py
├── train_final.py
├── optimize.py
├── evaluate.py
├── plot_optuna.py
├── compare_models.py
│
└── results/
    ├── optuna_trials.csv
    │
    └── plots/
        ├── optimization_history.png
        ├── learning_rate_vs_accuracy.png
        ├── hidden_units_vs_accuracy.png
        ├── dropout_vs_accuracy.png
        ├── batch_size_vs_accuracy.png
        ├── baseline_vs_optimized.png
        └── confusion_matrix.png
```

The dataset and trained model files are excluded from version control.

---

## Installation

Clone the repository and create a Python environment.

For example, using Conda:

```bash
conda create -n pytorch python=3.11
conda activate pytorch
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If a CUDA-enabled PyTorch installation is available, the model can use an NVIDIA GPU automatically.

Otherwise, training falls back to CPU.

---

## Usage

### 1. Train the baseline model

```bash
python train.py
```

### 2. Run hyperparameter optimization

```bash
python optimize.py
```

The results are saved to:

```text
results/optuna_trials.csv
```

### 3. Train the final model

After optimization:

```bash
python train_final.py
```

The best model is saved as:

```text
results/best_model.pt
```

### 4. Evaluate the final model

```bash
python evaluate.py
```

This generates the confusion matrix and per-class evaluation.

### 5. Generate Optuna visualizations

```bash
python plot_optuna.py
```

### 6. Compare baseline and optimized models

```bash
python compare_models.py
```

---


## Key Takeaways

This project demonstrates a complete neural network development workflow using PyTorch.

The main results are:

```text
Baseline Validation Accuracy
88.38%

        ↓

Optuna Hyperparameter Optimization
20 trials

        ↓

Final Validation Accuracy
88.97%

        ↓

Final Test Accuracy
88.15%
```

The optimization produced a modest improvement over the baseline.

More importantly, the project demonstrates how hyperparameter optimization, model selection, final evaluation, and error analysis can be combined into a reproducible machine learning workflow.

The confusion matrix shows that the remaining errors are concentrated primarily among visually similar clothing categories, particularly shirts, T-shirts/tops, coats, and pullovers.

```
```
