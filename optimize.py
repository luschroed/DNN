import os

import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from models.model import NeuralNetwork
from train import train_model

from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)

import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

EPOCHS = 5
N_TRIALS = 20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# ============================================================
# Dataset
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# ============================================================
# Train / Validation Split
# ============================================================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)


# ============================================================
# Objective Function
# ============================================================

def objective(trial):

    # --------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-4,
        1e-2,
        log=True
    )

    hidden_units = trial.suggest_categorical(
        "hidden_units",
        [64, 128, 256]
    )

    dropout = trial.suggest_float(
        "dropout",
        0.0,
        0.5
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64, 128]
    )


    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = NeuralNetwork(
        hidden_units=hidden_units,
        dropout=dropout
    ).to(DEVICE)


    # --------------------------------------------------------
    # Loss & Optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS
    )


    # --------------------------------------------------------
    # Return best validation accuracy
    # --------------------------------------------------------

    best_accuracy = max(
        history["val_accuracy"]
    )

    return best_accuracy


# ============================================================
# Optuna Study
# ============================================================

study = optuna.create_study(
    direction="maximize",
    study_name="fashion_mnist_hpo"
)


study.optimize(
    objective,
    n_trials=N_TRIALS
)


# ============================================================
# Results
# ============================================================

print()
print("=" * 60)
print("Hyperparameter Optimization Complete")
print("=" * 60)

print(f"Number of trials: {len(study.trials)}")

print()
print("Best Trial:")
print(f"Validation Accuracy: {study.best_value:.2f}%")

print()
print("Best Hyperparameters:")

for parameter, value in study.best_params.items():
    print(f"{parameter}: {value}")


# ============================================================
# Save Results
# ============================================================

os.makedirs("results", exist_ok=True)

df = study.trials_dataframe()

df.to_csv(
    "results/optuna_trials.csv",
    index=False
)

print()
print("Trial results saved to:")
print("results/optuna_trials.csv")



# ============================================================
# Optuna Visualizations
# ============================================================

os.makedirs("results/plots", exist_ok=True)


# Optimization history
fig = plot_optimization_history(study)

fig.figure.savefig(
    "results/plots/optimization_history.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig.figure)


# Hyperparameter importance
fig = plot_param_importances(study)

fig.figure.savefig(
    "results/plots/parameter_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig.figure)


# Parallel coordinate plot
fig = plot_parallel_coordinate(study)

fig.figure.savefig(
    "results/plots/parallel_coordinates.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig.figure)