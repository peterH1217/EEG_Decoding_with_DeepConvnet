"""Project configuration."""
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Paths:
    """Standard project paths derived dynamically."""
    project_root: Path
    data: Path
    logs: Path
    results: Path
    models: Path
    figures: Path

    @staticmethod
    def from_here() -> "Paths":
        # Resolves to: project_root/src/neuro_deep_learning/config.py -> parents[2] is project_root
        root = Path(__file__).resolve().parents[2]
        return Paths(
            project_root=root,
            data=root / "data",
            logs=root / "logs",
            results=root / "results",
            models=root / "results" / "models",
            figures=root / "results" / "figures"  # Or "grand_average" based on your preference
        )

# Initialize paths
PATHS = Paths.from_here()

# --- Data Parameters ---
SAMPLING_RATE = 250  # Hz (Matches Paper)

# --- Filtering Strategy ---
# The paper compares "0-f_end" and "4-f_end". We use 4.0 Hz to remove eye artifacts (EOG).
LOW_CUTOFF = 4.0     # Hz 

# The paper analyzes up to 125 Hz for the High Gamma Dataset.
# We set this to None so we don't kill the Gamma band features.
HIGH_CUTOFF = 38.0 #Hz
