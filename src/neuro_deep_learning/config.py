# src/config.py
import os

# Paths
LOG_DIR = "logs"
RESULTS_DIR = "results"

# Data Parameters
SAMPLING_RATE = 250  # Hz (Matches Paper)

# Filtering strategy:
# The paper compares "0-f_end" and "4-f_end". We use 4.0 Hz to remove eye artifacts (EOG).
LOW_CUTOFF = 4.0     # Hz 

# The paper analyzes up to 125 Hz for the High Gamma Dataset.
# We set this to None (or 120) so we don't kill the Gamma band features.
HIGH_CUTOFF = None   # None means "Do not filter high freqs" (or use 120.0)

