import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models.model import NeuralNetwork


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 128

HIDDEN_UNITS = 256
DROPOUT = 0.08786502727743081

MODEL_PATH = "results/best_model.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Dataset
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Class Names
# ============================================================

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# ============================================================
# Model
# ============================================================

model = NeuralNetwork(
    hidden_units=HIDDEN_UNITS,
    dropout=DROPOUT
).to(DEVICE)


# ============================================================
# Load Best Model
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# train.py saves a state_dict directly.
model.load_state_dict(checkpoint)

model.eval()


print()
print("Loaded model:")
print(MODEL_PATH)


# ============================================================
# Evaluation
# ============================================================

criterion = nn.CrossEntropyLoss()

all_predictions = []
all_labels = []

test_loss = 0.0
correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        test_loss += loss.item()

        _, predictions = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


test_loss /= len(test_loader)

test_accuracy = (
    100 * correct / total
)


# ============================================================
# Print Test Results
# ============================================================

print()
print("=" * 50)
print("Test Set Evaluation")
print("=" * 50)

print(
    f"Test Loss:     {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.2f}%"
)

print("=" * 50)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


os.makedirs(
    "results/plots",
    exist_ok=True
)


fig, ax = plt.subplots(
    figsize=(10, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d"
)

plt.title(
    "Confusion Matrix - Optimized Model"
)

plt.tight_layout()

plt.savefig(
    "results/plots/confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# Per-Class Accuracy
# ============================================================

print()
print("=" * 50)
print("Per-Class Accuracy")
print("=" * 50)


for i, class_name in enumerate(class_names):

    total_class = cm[i].sum()

    correct_class = cm[i, i]

    accuracy = (
        100 * correct_class / total_class
    )

    print(
        f"{class_name:15s}: "
        f"{accuracy:6.2f}%"
    )


# ============================================================
# Most Common Confusions
# ============================================================

print()
print("=" * 50)
print("Most Common Misclassifications")
print("=" * 50)


# Copy matrix so we can ignore diagonal
cm_errors = cm.copy()

for i in range(len(class_names)):
    cm_errors[i, i] = 0


# Find the five largest off-diagonal values
flat_indices = cm_errors.argsort(
    axis=None
)[::-1]


reported = 0

for flat_index in flat_indices:

    actual, predicted = (
        divmod(
            flat_index,
            len(class_names)
        )
    )

    count = cm_errors[
        actual,
        predicted
    ]

    if count == 0:
        break

    print(
        f"{class_names[actual]:15s}"
        f" -> "
        f"{class_names[predicted]:15s}"
        f": {count}"
    )

    reported += 1

    if reported >= 5:
        break


print()
print(
    "Confusion matrix saved to:"
    " results/plots/confusion_matrix.png"
)
