import logging
import sys
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

# --- IMPORTS ---
from neuro_deep_learning import fetch, dataset, ui

# ---------------- Logging configuration ----------------
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# ---------------- Project paths ----------------
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def process_dataset(dataset_name: str) -> None:
    """
    Runs the full processing pipeline for a single dataset.
    Keeps main() clean and under 50 lines.
    """
    logger.info(f"--- PROCESSING DATASET: {dataset_name} ---")

    # 1. Fetch
    raw_train, raw_test = fetch.get_dataset(subject_id=1, dataset_name=dataset_name)

    # 2. Preprocess (Filter & Z-score)
    raw_train = dataset.preprocess_data(raw_train)
    raw_test = dataset.preprocess_data(raw_test)

    # 3. Visualize
    ui.plot_psd(raw_train, save_path=RESULTS_DIR / f"psd_plot_{dataset_name}.png")
    ui.plot_raw_trace(raw_train, save_path=RESULTS_DIR / f"raw_eeg_trace_{dataset_name}.png")

    # 4. Epochs (Smart)
    X_train, y_train = dataset.make_epochs(raw_train, tmin=0.0, tmax=4.0)
    # (Optional: You can process X_test here if needed for evaluation later)
    
    logger.debug(f"{dataset_name} epochs train: {X_train.shape}, y: {y_train.shape}")

    # 5. Artifact Removal
    X_train, y_train = dataset.remove_artifact_trials(X_train, y_train, threshold_std=20)
    
    logger.debug(f"{dataset_name} after artifact removal: {X_train.shape[0]} trials")

    if X_train.shape[0] == 0:
        logger.error(f"{dataset_name}: 0 trials left after artifact removal; skipping.")
        return

    # 6. Crops Dataset (Memory Safe)
    trial_len = X_train.shape[-1]
    crop_size = min(500, trial_len)
    train_ds = dataset.CropsDataset(X_train, y_train, crop_size=crop_size, stride=1)

    # 7. Loader Test
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    xb, yb = next(iter(train_loader))

    # Log results (No prints!)
    logger.info(f"Finished {dataset_name}. Final Shapes -> Crops: {xb.shape}, Labels: {yb.shape}")

def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(description="Train Neuro Deep Learning Model")
    parser.add_argument("--dataset", type=str, default="BNCI2014_001", help="Dataset to process")
    args = parser.parse_args()

    try:
        process_dataset(args.dataset)
    except Exception as e:
        logger.exception(f"Failed to process {args.dataset}: {e}")

    logger.info(">>> Pipeline complete.")

if __name__ == "__main__":
    main()