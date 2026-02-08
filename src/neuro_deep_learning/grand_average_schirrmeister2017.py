import sys
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Logging configuration
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# -------------------- CONFIG --------------------
DATASET_NAME = "Schirrmeister2017"
RESULTS_DIR = Path("results")
SAVE_DIR = RESULTS_DIR / "grand_average"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- FUNCTIONS --------------------
def load_results():
    accuracies = [
        92.2, 87.2, 96.2, 98.1, 97.5,
        83.1, 88.8, 83.1, 88.8, 80.9,
        84.5, 90.6, 81.0, 66.9,
    ]
    subjects = [f"S{i}" for i in range(1, 15)]
    return subjects, accuracies

def compute_grand_average(accuracies):
    return float(np.mean(accuracies))

def get_bar_colors(accuracies):
    colors = []
    for acc in accuracies:
        if acc >= 90:
            colors.append("#2E7D32")
        elif acc >= 80:
            colors.append("#1565C0")
        else:
            colors.append("#D84315")
    return colors

def plot_results(subjects, accuracies, grand_avg):
    plt.figure(figsize=(12, 6))
    colors = get_bar_colors(accuracies)
    bars = plt.bar(subjects, accuracies, color=colors)
    plt.axhline(
        y=grand_avg,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Grand Average ({grand_avg:.2f}%)",
    )
    plt.axhline(
        y=25,
        color="gray",
        linestyle=":",
        label="Chance (25%)",
    )
    plt.ylabel("Accuracy (%)")
    plt.title(f"{DATASET_NAME} (High Gamma): Accuracy per Subject")
    plt.ylim(0, 105)
    plt.legend(loc="lower right")
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    save_path = SAVE_DIR / f"grand_average_{DATASET_NAME}.png"
    plt.savefig(save_path, dpi=300)
    logger.info(f"Plot saved to: {save_path}")

def main():
    logger.info(f"Generating Grand Average Plot for {DATASET_NAME}")
    subjects, accuracies = load_results()
    logger.info(f"Data loaded for {len(subjects)} subjects.")
    grand_avg = compute_grand_average(accuracies)
    logger.info(f"{DATASET_NAME} GRAND AVERAGE: {grand_avg:.2f}%")
    plot_results(subjects, accuracies, grand_avg)

if __name__ == "__main__":
    main()

