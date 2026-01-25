import logging
import sys
import argparse
import numpy as np
import torch
import json
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from neuro_deep_learning import fetch, dataset, visualization, cnn

# ---------------- Logging configuration ----------------
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# ---------------- Project paths ----------------
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------- Hyperparameters ----------------
TRAIN_SIZE = 0.8
N_EPOCHS = 300      # You can reduce this to 150-200 for faster Schirrmeister runs if needed
STRIDE = 100        # Keep this small for high accuracy

PATIENCE = 50
CROP_SIZE = 500
BATCH_SIZE = 64


def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    """
    Crop-level training loop.
    Handles both (xb, yb) and (xb, yb, tb) inputs automatically.
    """
    if is_train:
        model.train()
    else:
        model.eval()

    correct, total, running_loss = 0, 0, 0.0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in loader:
            # 1. Unpack batch (Handle cases with or without Trial Index)
            if len(batch) == 3:
                xb, yb, _ = batch
            else:
                xb, yb = batch

            # 2. Add extra dimension for Conv2d (Batch, 1, Channels, Time)
            xb = xb.unsqueeze(1).to(device) 
            yb = yb.to(device)

            if is_train:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
            else:
                out = model(xb)
                loss = criterion(out, yb)

            _, pred = torch.max(out.data, 1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()
            running_loss += loss.item()

    acc = 100 * correct / max(1, total)
    avg_loss = running_loss / max(1, len(loader))
    return acc, avg_loss


def predict_trials_by_mean_logits(model, loader, device):
    """
    Paper-style evaluation:
    aggregate all crop logits per trial -> mean -> 1 prediction per trial.
    loader must yield: (xb, yb, trial_idx)
    """
    model.eval()
    trial_logits = {}  # trial_idx -> list[logits]
    trial_label  = {}  # trial_idx -> label

    with torch.no_grad():
        for xb, yb, tb in loader:
            xb = xb.unsqueeze(1).to(device)
            logits = model(xb).detach().cpu()  # (B, n_classes)

            for logit, y, t in zip(logits, yb, tb):
                t = int(t.item())
                y = int(y.item())
                trial_logits.setdefault(t, []).append(logit)
                trial_label[t] = y

    trial_ids = sorted(trial_logits.keys())
    y_true, y_pred = [], []

    for t in trial_ids:
        mean_logit = torch.stack(trial_logits[t]).mean(dim=0)
        pred = int(torch.argmax(mean_logit).item())
        y_true.append(trial_label[t])
        y_pred.append(pred)

    return np.array(y_true), np.array(y_pred)


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_acc is None:
            self.best_acc = val_acc
        elif val_acc < self.best_acc + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_acc = val_acc
            self.counter = 0


def process_dataset(dataset_name: str) -> None:
    print(f"--- PROCESSING DATASET: {dataset_name} ---")

    # 1. DETERMINE SUBJECTS
    # If BNCI2014_001, we know there are 9 subjects.
    if dataset_name == 'BNCI2014_001':
         subject_ids = list(range(1, 10)) # [1, 2, ..., 9]
    else:
         # For Schirrmeister or Physionet, fetch list dynamically
         subject_ids = fetch.get_participants(dataset_name)

    # --- DEBUG MODE: Uncomment below to run ONLY Subject 1 for testing ---
    # subject_ids = [1] 
    
    grand_accuracies = []

    # 2. MAIN LOOP: ONE SUBJECT AT A TIME (Replication Standard)
    for subject_id in subject_ids:
        print(f"\n" + "="*50)
        print(f"  TRAINING SUBJECT {subject_id} of {len(subject_ids)}")
        print(f"="*50)

        # A. LOAD DATA (Specific to this subject)
        raw_train, raw_test = fetch.get_dataset(subject_id=subject_id, dataset_name=dataset_name)
        raw_train = dataset.preprocess_data(raw_train)
        raw_test  = dataset.preprocess_data(raw_test)

        X_train, y_train = dataset.make_epochs(raw_train, tmin=-0.5, tmax=4.0)
        X_test, y_test   = dataset.make_epochs(raw_test,  tmin=-0.5, tmax=4.0)

        X_train, y_train = dataset.remove_artifact_trials(X_train, y_train, threshold_std=20)
        X_test, y_test   = dataset.remove_artifact_trials(X_test, y_test, threshold_std=20)
        
        print(f"Subject {subject_id}: {len(X_train)} train trials | {len(X_test)} test trials")

        # B. SPLIT TRAIN/VAL
        split_idx = int(len(X_train) * TRAIN_SIZE)

        train_ds = dataset.CropsDataset(X_train[:split_idx], y_train[:split_idx], crop_size=CROP_SIZE, stride=STRIDE)
        val_ds   = dataset.CropsDataset(X_train[split_idx:], y_train[split_idx:], crop_size=CROP_SIZE, stride=STRIDE)
        test_ds  = dataset.CropsDataset(X_test, y_test, crop_size=CROP_SIZE, stride=STRIDE)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

        # C. FRESH MODEL & OPTIMIZER
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cnn.DeepConvNet(
            n_channels=X_train.shape[1],
            n_classes=4,
            input_window_samples=CROP_SIZE
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        early_stopper = EarlyStopping(patience=PATIENCE)

        # D. TRAINING LOOP
        best_val_acc = -1.0
        
        # We don't need to store full history for every subject unless you want 14 graphs.
        # Let's just track the best model.
        
        for epoch in range(N_EPOCHS):
            # Train (Crop-level)
            run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)

            # Validation (Trial-level)
            y_true_val, y_pred_val = predict_trials_by_mean_logits(model, val_loader, device)
            val_acc = (y_true_val == y_pred_val).mean() * 100

            # Save Best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(),
                    RESULTS_DIR / f"best_model_{dataset_name}_S{subject_id}.pth"
                )
            
            # Early Stopping
            early_stopper(val_acc)
            if early_stopper.early_stop:
                print(f"  Early stopping at epoch {epoch} (Best Val: {best_val_acc:.1f}%)")
                break
            
            # Print status every 10 epochs
            if epoch % 10 == 0:
                print(f"  Ep {epoch}: Val Acc {val_acc:.1f}%")

        # E. FINAL TEST EVALUATION
        # Load the best weights for THIS subject
        model.load_state_dict(torch.load(RESULTS_DIR / f"best_model_{dataset_name}_S{subject_id}.pth", map_location=device))
        
        y_true_test, y_pred_test = predict_trials_by_mean_logits(model, test_loader, device)
        final_test_acc = (y_true_test == y_pred_test).mean() * 100
        
        print(f"✅ FINAL RESULT - Subject {subject_id}: {final_test_acc:.2f}%")
        grand_accuracies.append(final_test_acc)
        
        # Optional: Save a confusion matrix for this subject
        cm_path = RESULTS_DIR / f"confusion_matrix_{dataset_name}_S{subject_id}.png"
        class_mapping = fetch.get_dataset_class_mapping(dataset_name)
        visualization.plot_test_confusion_matrix(
            y_true_test, y_pred_test,
            class_mapping=class_mapping,
            title=f"S{subject_id} Confusion Matrix (Acc: {final_test_acc:.1f}%)",
            save_path=cm_path
        )

    # 3. GRAND AVERAGE REPORT
    avg_acc = np.mean(grand_accuracies)
    print("\n" + "="*50)
    print(f"REPLICATION COMPLETE: {dataset_name}")
    print(f"Grand Average Accuracy: {avg_acc:.2f}%")
    print(f"Individual Scores: {grand_accuracies}")
    print("="*50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Neuro Deep Learning Model")
    parser.add_argument("--dataset", type=str, default="BNCI2014_001", help="Dataset to process")
    args, unknown = parser.parse_known_args()

    try:
        process_dataset(args.dataset)
    except Exception as e:
        logger.exception(f"Failed to process {args.dataset}: {e}")

    logger.info(">>> Pipeline complete.")


if __name__ == "__main__":
    main()