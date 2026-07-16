import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# FILE PATHS
# =====================================================

RESULTS_DIR = Path("experiment_results")
RESULTS_FILE = RESULTS_DIR / "metrics.csv"

# =====================================================
# LOAD METRICS
# =====================================================

df = pd.read_csv(RESULTS_FILE)

# =====================================================
# AVERAGE RESULTS PER ROUND
# =====================================================

summary = (
    df.groupby("Round")
      .agg({
          "Accuracy": "mean",
          "Loss": "mean"
      })
      .reset_index()
)

print(summary)

# =====================================================
# ACCURACY GRAPH
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(
    summary["Round"],
    summary["Accuracy"],
    marker="o",
    linewidth=2
)

plt.title("Global Average Accuracy per Federated Round")
plt.xlabel("Federated Round")
plt.ylabel("Accuracy")
plt.grid(True)

accuracy_path = RESULTS_DIR / "accuracy_plot.png"

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# LOSS GRAPH
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(
    summary["Round"],
    summary["Loss"],
    marker="o",
    linewidth=2
)

plt.title("Global Average Loss per Federated Round")
plt.xlabel("Federated Round")
plt.ylabel("Loss")
plt.grid(True)

loss_path = RESULTS_DIR / "loss_plot.png"

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nGraphs generated successfully!")

print(f"\nSaved:")
print(accuracy_path)
print(loss_path)