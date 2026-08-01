import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

RESULTS_FILE = "results/optuna_trials.csv"
PLOT_DIR = "results/plots"

os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# Load Optuna Results
# ============================================================

df = pd.read_csv(RESULTS_FILE)

print("Loaded Optuna results:")
print(df.head())

print()
print("Available columns:")
print(df.columns.tolist())


# ============================================================
# Optimization History
# ============================================================

df["best_so_far"] = df["value"].cummax()


plt.figure(figsize=(8, 5))

plt.plot(
    df["number"],
    df["value"],
    marker="o",
    label="Trial Accuracy"
)

plt.plot(
    df["number"],
    df["best_so_far"],
    label="Best Accuracy So Far"
)

plt.xlabel("Trial")
plt.ylabel("Validation Accuracy (%)")
plt.title("Optuna Optimization History")

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "optimization_history.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# Hyperparameter vs Accuracy
# ============================================================

parameters = {
    "params_learning_rate": "Learning Rate",
    "params_hidden_units": "Hidden Units",
    "params_dropout": "Dropout",
    "params_batch_size": "Batch Size"
}


for parameter, label in parameters.items():

    if parameter not in df.columns:
        continue

    plt.figure(figsize=(8, 5))

    plt.scatter(
        df[parameter],
        df["value"]
    )

    plt.xlabel(label)
    plt.ylabel("Validation Accuracy (%)")

    plt.title(
        f"{label} vs Validation Accuracy"
    )

    plt.grid()

    plt.tight_layout()

    filename = (
        parameter.replace("params_", "")
        + "_vs_accuracy.png"
    )

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            filename
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# Best Trial
# ============================================================

best_trial = df.loc[
    df["value"].idxmax()
]


print()
print("=" * 60)
print("Best Trial")
print("=" * 60)

print(
    f"Trial:               "
    f"{int(best_trial['number'])}"
)

print(
    f"Validation Accuracy: "
    f"{best_trial['value']:.2f}%"
)

print(
    f"Learning Rate:       "
    f"{best_trial['params_learning_rate']}"
)

print(
    f"Hidden Units:        "
    f"{int(best_trial['params_hidden_units'])}"
)

print(
    f"Dropout:             "
    f"{best_trial['params_dropout']}"
)

print(
    f"Batch Size:          "
    f"{int(best_trial['params_batch_size'])}"
)

print()
print("Plots saved to:")
print(PLOT_DIR)
