import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from models.model import NeuralNetwork
from train import train_model


# ============================================================
# Configuration
# ============================================================

# Best hyperparameters found by Optuna
LEARNING_RATE = 0.0008626881131442573
HIDDEN_UNITS = 256
DROPOUT = 0.08786502727743081
BATCH_SIZE = 128

EPOCHS = 10

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

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
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


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Model
# ============================================================

model = NeuralNetwork(
    hidden_units=HIDDEN_UNITS,
    dropout=DROPOUT
).to(DEVICE)


print()
print("Final Model Configuration")
print("-" * 40)
print(f"Learning rate: {LEARNING_RATE}")
print(f"Hidden units:  {HIDDEN_UNITS}")
print(f"Dropout:       {DROPOUT}")
print(f"Batch size:    {BATCH_SIZE}")
print(f"Epochs:        {EPOCHS}")
print("-" * 40)


# ============================================================
# Loss & Optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=DEVICE,
    epochs=EPOCHS,
    save_best_model=True,
    model_path="results/best_model.pt"
)


# ============================================================
# Best Validation Accuracy
# ============================================================

best_val_accuracy = history["best_val_accuracy"]
best_epoch = history["best_epoch"]

print()
print("=" * 50)
print("Training Complete")
print("=" * 50)
print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)
print(
    f"Best Epoch: " 
    f"{best_epoch}"
)


# ============================================================
# Test Evaluation
# ============================================================

model.eval()

correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        test_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


test_loss /= len(test_loader)

test_accuracy = (
    100 * correct / total
)


# ============================================================
# Final Results
# ============================================================

print()
print("=" * 50)
print("Final Test Results")
print("=" * 50)
print(f"Test Loss:       {test_loss:.4f}")
print(f"Test Accuracy:   {test_accuracy:.2f}%")
print("=" * 50)


# ============================================================
# Save Training Metadata
# ============================================================

os.makedirs("results", exist_ok=True)

metadata = {
    "hyperparameters": {
        "learning_rate": LEARNING_RATE,
        "hidden_units": HIDDEN_UNITS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    },
    "best_epoch": best_epoch,
    "best_validation_accuracy": best_val_accuracy,
    "test_accuracy": test_accuracy,
}


torch.save(
    metadata,
    "results/final_results.pt"
)


print()
print("Best model saved to:")
print("results/best_model.pt")

print()
print("Training metadata saved to:")
print("results/final_results.pt")

