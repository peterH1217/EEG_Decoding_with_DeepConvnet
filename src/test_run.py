# src/test_run.py
import logging
import sys
from pathlib import Path

from torch.utils.data import DataLoader
import data_loader, visualization  # IMPORTANT: package import

# ---------------- Logging configuration ----------------
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# ---------------- Project paths ----------------
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def main():
    logger.info(">>> STEP 1: Starting Multi-Dataset Pipeline...")

    datasets_to_run = ["BNCI2014_001", "Schirrmeister2017"]

    for dataset_name in datasets_to_run:
        logger.info(f"\n--- PROCESSING DATASET: {dataset_name} ---")

        try:
            # 1) Load + preprocess (MOABB auto-downloads & caches)
            raw_train, raw_test = data_loader.load_and_process_subject(
                subject_id=1,
                dataset_name=dataset_name,
            )

            # 2) Save visualizations (absolute paths)
            visualization.plot_psd(raw_train, save_path=RESULTS_DIR / f"psd_plot_{dataset_name}.png")
            visualization.plot_raw_trace(raw_train, save_path=RESULTS_DIR / f"raw_eeg_trace_{dataset_name}.png")

            # 3) Epoching (dataset-specific)
            if dataset_name == "BNCI2014_001":
                X_train, y_train = data_loader.make_epochs_bci(raw_train, tmin=0.0, tmax=4.0)
                X_test,  y_test  = data_loader.make_epochs_bci(raw_test,  tmin=0.0, tmax=4.0)
            else:
                X_train, y_train = data_loader.make_epochs_hgd(raw_train, tmin=0.0, tmax=4.0)
                X_test,  y_test  = data_loader.make_epochs_hgd(raw_test,  tmin=0.0, tmax=4.0)

            print(f"[DEBUG] {dataset_name} epochs train: {X_train.shape}, y: {y_train.shape}")

            # 4) Minimal artifact removal (±800 µV) BEFORE z-scoring
            X_train, y_train, keep_tr = data_loader.remove_artifact_trials_uV(X_train, y_train, threshold_uV=800.0)
            X_test,  y_test,  keep_te = data_loader.remove_artifact_trials_uV(X_test,  y_test,  threshold_uV=800.0)

            print(f"[DEBUG] {dataset_name} after artifact train trials: {X_train.shape[0]}")

            if X_train.shape[0] == 0:
                logger.error(f"{dataset_name}: 0 trials left after artifact removal; skipping.")
                continue

            # 5) Normalize for CNN (z-score per trial per channel)
            X_train = data_loader.zscore_per_channel(X_train)
            X_test  = data_loader.zscore_per_channel(X_test)

            # 6) Memory-safe cropped training dataset (on-the-fly)
            trial_len = X_train.shape[-1]
            crop_size = min(500, trial_len)  # never exceed trial length
            stride = 1

            train_ds = data_loader.make_crops_dataset_for_cnn(X_train, y_train, crop_size=crop_size, stride=stride)

            print(f"[DEBUG] {dataset_name} crop_size={crop_size}, total_crops={len(train_ds)}")

            if len(train_ds) == 0:
                logger.error(f"{dataset_name}: 0 crops produced; skipping.")
                continue

            # Windows tip: start with num_workers=0, then increase if stable
            train_loader = DataLoader(
                train_ds,
                batch_size=64,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )

            xb, yb = next(iter(train_loader))
            print(f"\n--- SHAPES for {dataset_name} ---")
            print("Epochs shape:", X_train.shape)
            print("One batch crops (CNN input):", xb.shape)
            print("Batch labels:", yb.shape)
            print("Total number of crops:", len(train_ds))
            print("-----------------------------------\n")

            logger.info(f"Finished {dataset_name} successfully.")

        except Exception as e:
            logger.exception(f"Failed to process {dataset_name}: {e}")

    logger.info(">>> Pipeline complete.")

if __name__ == "__main__":
    main()
