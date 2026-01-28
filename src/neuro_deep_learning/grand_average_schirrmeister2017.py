import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# CONFIGURATION
dataset_name = "Schirrmeister2017"

# Setup Directories
RESULTS_DIR = Path("results")
SAVE_DIR = RESULTS_DIR / "grand_average"
SAVE_DIR.mkdir(parents=True, exist_ok=True)  # Creates folder if missing

print(f"--- Generating Grand Average Plot for {dataset_name} ---")

# 1. HARDCODED RESULTS (Derived from the manual run)
# S1 to S14
accuracies = [
    92.2, 87.2, 96.2, 98.1, 97.5,   # S1-S5
    83.1, 88.8, 83.1, 88.8, 80.9,   # S6-S10
    84.5, 90.6, 81.0, 66.9          # S11-S14
]

subjects = [f'S{i}' for i in range(1, 15)]

# 2. Calculate Statistics
grand_avg = np.mean(accuracies)
print(f"Data loaded for {len(subjects)} subjects.")
print(f" {dataset_name} GRAND AVERAGE: {grand_avg:.2f}%")

# 3. PLOTTING SECTION
plt.figure(figsize=(12, 6))

# Color logic: Green if > 90%, Blue if > 80%, Orange if lower
colors = []
for acc in accuracies:
    if acc >= 90: colors.append('#2E7D32') # Dark Green
    elif acc >= 80: colors.append('#1565C0') # Blue
    else: colors.append('#D84315') # Orange

bars = plt.bar(subjects, accuracies, color=colors)

# Grand Average Line
plt.axhline(y=grand_avg, color='red', linestyle='--', linewidth=2, label=f'Grand Average ({grand_avg:.2f}%)')
plt.axhline(y=25, color='gray', linestyle=':', label='Chance (25%)')

# Labels and Titles
plt.ylabel('Accuracy (%)')
plt.title(f'{dataset_name} (High Gamma): Accuracy per Subject')
plt.ylim(0, 105)
plt.legend(loc='lower right')

# Add numbers on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()

# 4. SAVE THE FIGURE
save_path = SAVE_DIR / f"grand_average_{dataset_name}.png"
plt.savefig(save_path, dpi=300)
print(f"Plot saved to: {save_path}")

# plt.show() # Optional: Uncomment to pop up the window

