import sys
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Logging configuration
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# -------------------- CONFIG --------------------
DATASET_NAME = "BNCI2014_001"

RESULTS_DIR = Path("results")
SAVE_DIR = RESULTS_DIR / "grand_average"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- FUNCTIONS --------------------
def load_results():
    accuracies = [
        81.6, 36.1, 85.1, 53.5, 36.5, 47.9, 84.4, 78.1, 71.2
    ]
    subjects = [f"S{i}" for i in range(1, 10)]
    return subjects, accuracies

def compute_grand_average(accuracies):
    return float(np.mean(accuracies))

def get_bar_colors(accuracies):
    colors = []
    for acc in accuracies:
        if acc >= 70:
            colors.append("#2E7D32")  # Dark Green (Good)
        elif acc >= 50:
            colors.append("#1565C0")  # Blue (Okay)
        else:
            colors.append("#D84315")  # Orange (Needs Improvement)
    return colors

def plot_results(subjects, accuracies, grand_avg):
    plt.figure(figsize=(12, 6))

    colors = get_bar_colors(accuracies)
    bars = plt.bar(subjects, accuracies, color=colors)

    # Grand Average Line
    plt.axhline(
        y=grand_avg,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Grand Average ({grand_avg:.2f}%)",
    )
    # Chance Level Line
    plt.axhline(
        y=25,
        color="gray",
        linestyle=":",
        label="Chance (25%)",
    )

    plt.ylabel("Accuracy (%)")
    plt.title(f"{DATASET_NAME}: Accuracy per Subject (38Hz Low-Pass)")
    plt.ylim(0, 100)
    plt.legend(loc="upper right")

    # Add numbers on top of bars
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
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