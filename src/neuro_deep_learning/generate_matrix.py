import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from neuro_deep_learning import fetch, dataset, cnn
from pathlib import Path

# --- CONFIGURATION ---
DATASET_NAME = "BNCI2014_001"
STRIDE = 100    # Matches the filename in your screenshot
CROP_SIZE = 500
BATCH_SIZE = 64

# Automatically find the saved model in the results folder
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
model_files = list(RESULTS_DIR.glob(f"best_model_{DATASET_NAME}*stride_{STRIDE}.pth"))
if not model_files:
    raise FileNotFoundError("Could not find the .pth model file!")
MODEL_PATH = model_files[0] # Pick the first one found
print(f"Loading model from: {MODEL_PATH}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data (Session 2 Only for Test)
    print("Loading Test Data...")
    subject_ids = fetch.get_participants(DATASET_NAME)
    # Just load Subject 1 for consistency with your pilot
    X_test_all, y_test_all = [], []
    
    # NOTE: If you trained on subject 1 only, set this to [1]
    subject_ids = [1] 

    for subject_id in subject_ids:
        _, raw_test = fetch.get_dataset(subject_id, DATASET_NAME)
        raw_test = dataset.preprocess_data(raw_test)
        X_s2, y_s2 = dataset.make_epochs(raw_test, tmin=-0.5, tmax=4.0)
        X_s2, y_s2 = dataset.remove_artifact_trials(X_s2, y_s2)
        X_test_all.append(X_s2); y_test_all.append(y_s2)

    X_test = np.concatenate(X_test_all, axis=0)
    y_test = np.concatenate(y_test_all, axis=0)

    # 2. Prepare Data Loader
    test_ds = dataset.CropsDataset(X_test, y_test, crop_size=CROP_SIZE, stride=STRIDE)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Load Model Architecture
    model = cnn.DeepConvNet(n_channels=22, n_classes=4, input_window_samples=CROP_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 4. Run Predictions (Trial-wise voting)
    print("Running Predictions...")
    trial_logits = {}
    trial_labels = {}
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3: xb, yb, tb = batch
            else: continue # Skip if no trial index
            
            xb = xb.unsqueeze(1).to(device)
            logits = model(xb).detach().cpu()
            
            for l, y, t in zip(logits, yb, tb):
                t_idx = int(t.item())
                trial_logits.setdefault(t_idx, []).append(l)
                trial_labels[t_idx] = int(y.item())

    # Aggregate votes
    y_true, y_pred = [], []
    for t in sorted(trial_logits.keys()):
        mean_logit = torch.stack(trial_logits[t]).mean(dim=0)
        pred = torch.argmax(mean_logit).item()
        y_true.append(trial_labels[t])
        y_pred.append(pred)

    # 5. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = ["Right Hand", "Left Hand", "Feet", "Tongue"] # Standard 2a classes
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f"Confusion Matrix (Acc: {100*np.mean(np.array(y_true)==np.array(y_pred)):.1f}%)")
    
    save_path = RESULTS_DIR / "final_confusion_matrix.png"
    plt.savefig(save_path)
    print(f"✅ Success! Saved matrix to: {save_path}")

if __name__ == "__main__":
    main()