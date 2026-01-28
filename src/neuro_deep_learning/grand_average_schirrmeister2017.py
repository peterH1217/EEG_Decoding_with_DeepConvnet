import sys
import os
import matplotlib.pyplot as plt  # <--- Added for plotting

# Fix imports so it finds your 'src' folder
sys.path.append(os.path.join(os.getcwd(), 'src'))

import torch
import numpy as np
from neuro_deep_learning import fetch, dataset, cnn, train
from pathlib import Path

# --- CONFIGURATION FOR HIGH GAMMA ---
dataset_name = "Schirrmeister2017"
subject_ids = list(range(1, 15))   # Subjects 1 to 14
STRIDE = 500                       # Testing with non-overlapping crops (Standard)
CROP_SIZE = 500                    

# Setup Directories
RESULTS_DIR = Path("results")
SAVE_DIR = RESULTS_DIR / "grand_average"
SAVE_DIR.mkdir(parents=True, exist_ok=True)  # <--- Creates folder if missing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accuracies = []
found_subjects = []  # <--- To track labels (S1, S2...)

print(f"--- Calculating Grand Average for {dataset_name} ---")

for subject_id in subject_ids:
    # Try to find the model file (handling different naming conventions)
    potential_files = list((RESULTS_DIR / "models").glob(f"best_model_{dataset_name}_S{subject_id}*.pth"))
    
    if not potential_files:
        print(f"Warning: Model for S{subject_id} not found. Skipping.")
        continue
    
    model_path = potential_files[0] # Pick the first match

    try:
        # 1. Load Data
        _, raw_test = fetch.get_dataset(subject_id, dataset_name)
        raw_test = dataset.preprocess_data(raw_test)
        X_test, y_test = dataset.make_epochs(raw_test, tmin=-0.5, tmax=4.0)
        X_test, y_test = dataset.remove_artifact_trials(X_test, y_test)
        
        # 2. Prepare Loader
        test_ds = dataset.CropsDataset(X_test, y_test, crop_size=CROP_SIZE, stride=STRIDE)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

        # 3. Load Model (Auto-detect 128 channels)
        n_chans = X_test.shape[1] 
        model = cnn.DeepConvNet(n_channels=n_chans, n_classes=4, input_window_samples=CROP_SIZE).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # 4. Predict
        y_true, y_pred = train.predict_trials_by_mean_logits(model, test_loader, device)
        acc = (y_true == y_pred).mean() * 100
        
        # Store for plotting
        accuracies.append(acc)
        found_subjects.append(f"S{subject_id}")
        print(f"Subject {subject_id}: {acc:.2f}%")
        
    except Exception as e:
        print(f"Error processing S{subject_id}: {e}")



if len(accuracies) > 0:
    grand_avg = np.mean(accuracies)
    print(f"🏆 {dataset_name} GRAND AVERAGE: {grand_avg:.2f}%")
    
    # --- PLOTTING SECTION ---
    plt.figure(figsize=(12, 6))
    
    # Use real calculated data
    bars = plt.bar(found_subjects, accuracies, color=['#4CAF50' if x > 80 else '#2196F3' for x in accuracies])

    # Grand Average Line
    plt.axhline(y=grand_avg, color='red', linestyle='--', linewidth=2, label=f'Grand Average ({grand_avg:.1f}%)')
    plt.axhline(y=25, color='gray', linestyle=':', label='Chance (25%)')

    plt.ylabel('Accuracy (%)')
    plt.title('Schirrmeister 2017 (High Gamma): Accuracy per Subject')
    plt.ylim(0, 100)
    plt.legend()

    # Add numbers on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    
    # --- SAVE THE FIGURE ---
    save_path = SAVE_DIR / f"grand_average_{dataset_name}.png"
    plt.savefig(save_path)
    print(f"✅ Plot saved to: {save_path}")
    
    # plt.show() # Optional

else:
    print("No models found! Check your folder path.")

