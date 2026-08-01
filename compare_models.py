import os

import matplotlib.pyplot as plt


# ============================================================
# Results
# ============================================================

baseline_accuracy = 88.38
optimized_validation_accuracy = 88.97
optimized_test_accuracy = 88.15


# ============================================================
# Create Results Directory
# ============================================================

os.makedirs("results/plots", exist_ok=True)


# ============================================================
# Validation Accuracy Comparison
# ============================================================

models = [
    "Baseline",
    "Optimized"
]

validation_accuracy = [
    baseline_accuracy,
    optimized_validation_accuracy
]


plt.figure(figsize=(8, 5))

bars = plt.bar(
    models,
    validation_accuracy
)

plt.ylabel("Validation Accuracy (%)")
plt.title("Baseline vs Optimized Model")

plt.ylim(80, 95)

plt.grid(
    axis="y",
    alpha=0.3
)


# Add values above bars

for bar, value in zip(
    bars,
    validation_accuracy
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.2,
        f"{value:.2f}%",
        ha="center"
    )


plt.tight_layout()

plt.savefig(
    "results/plots/baseline_vs_optimized.png",
    dpi=300
)

plt.close()


# ============================================================
# Print Results
# ============================================================

improvement = (
    optimized_validation_accuracy
    - baseline_accuracy
)

print()
print("=" * 50)
print("Model Comparison")
print("=" * 50)

print(
    f"Baseline Validation Accuracy: "
    f"{baseline_accuracy:.2f}%"
)

print(
    f"Optimized Validation Accuracy: "
    f"{optimized_validation_accuracy:.2f}%"
)

print(
    f"Improvement: "
    f"+{improvement:.2f} percentage points"
)

print(
    f"Optimized Test Accuracy: "
    f"{optimized_test_accuracy:.2f}%"
)

print("=" * 50)

print()
print(
    "Plot saved to:"
    " results/plots/baseline_vs_optimized.png"
)

