"""Build datasets from raw data."""
import logging
import numpy as np
import mne
from mne import Epochs
import torch
from torch.utils.data import Dataset
from neuro_deep_learning.config import SAMPLING_RATE, LOW_CUTOFF, HIGH_CUTOFF
from neuro_deep_learning.logger import logger

def preprocess_data(raw):
    """
    Apply filtering and Z-score normalization to raw EEG data.
    Keeps stim channels during filtering to ensure we don't lose event triggers.
    """
    logger.info(f"Preprocessing: Filtering ({LOW_CUTOFF}-{HIGH_CUTOFF} Hz)...")
    
    # 1. Pick EEG types but keep STIM for now (needed for events)
    raw = raw.copy()
    if raw.info["sfreq"] != SAMPLING_RATE:
        logger.info(f"Resampling from {raw.info['sfreq']} to {SAMPLING_RATE}Hz")
        raw.resample(SAMPLING_RATE)
        
    eeg_picks = mne.pick_types(raw.info, eeg=True, stim=False)
    # Filter only EEG, skip stim
    raw.filter(LOW_CUTOFF, HIGH_CUTOFF, picks=eeg_picks, fir_design='firwin', skip_by_annotation='edge', verbose=False)
    
    # 2. Z-score Normalization (Channel-wise on continuous data)
    logger.info("Preprocessing: Applying Z-score normalization...")
    data = raw.get_data(picks='eeg')
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    
    # Apply normalization only to EEG channels
    raw._data[eeg_picks] = (data - mean) / std
    
    return raw

def make_epochs(raw, tmin=0.0, tmax=4.0):
    """
    Robust Epoching that handles both BCI (Annotations) and HGD (Stim Channels).
    Now accepts tmin/tmax to match train.py requirements.
    """
    logger.info(f"Segmenting raw data into epochs (tmin={tmin}, tmax={tmax})...")
    
    # Strategy A: Annotations (Works for BCI 2a)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    # Strategy B: Stim Channels (Fallback for High Gamma Dataset)
    if len(events) == 0:
        logger.warning("No annotations found! Checking STIM channels...")
        stim_channels = [ch for ch in raw.ch_names if "STI" in ch.upper() or "TRIG" in ch.upper()]
        if stim_channels:
            events = mne.find_events(raw, stim_channel=stim_channels[0], verbose=False)
            # Create dummy event_id for HGD top 4 classes if needed
            codes = events[:, 2]
            unique_codes, counts = np.unique(codes, return_counts=True)
            top_codes = unique_codes[np.argsort(-counts)][:4] 
            event_id = {str(c): c for c in top_codes}
        else:
            raise RuntimeError("Could not find Events (Annotations or Stim Channel)!")

    # Epoching
    epochs = Epochs(
        raw, 
        events, 
        event_id=event_id,
        tmin=tmin, 
        tmax=tmax, 
        baseline=None, 
        preload=True, 
        verbose=False
    )
    
    epochs.pick_types(eeg=True)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    
    # Remap y to 0, 1, 2, 3 so it works with PyTorch CrossEntropy
    unique_y = np.sort(np.unique(y))
    y_map = {val: i for i, val in enumerate(unique_y)}
    y_mapped = np.array([y_map[val] for val in y], dtype=np.int64)
    
    return X, y_mapped

def remove_artifact_trials(X, y, threshold_std=20):
    """
    Remove trials with extreme values (artifacts).
    """
    # Check max value in each trial
    max_vals = np.max(np.abs(X), axis=(1, 2))
    keep_mask = max_vals <= threshold_std
    
    logger.info(f"Artifact Removal: Keeping {np.sum(keep_mask)}/{len(keep_mask)} trials")
    return X[keep_mask], y[keep_mask]

# --- The "Memory-Safe" Class (Crucial for HGD) ---
class CropsDataset(Dataset):
    """
    PyTorch Dataset that generates crops on-the-fly.
    """
    def __init__(self, X, y, crop_size=500, stride=10):
        # Keep data as float32 to save space
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.crop_size = crop_size
        self.stride = stride
        
        n_trials, n_channels, n_time = self.X.shape
        
        # Calculate how many crops fit per trial
        self.crops_per_trial = (n_time - crop_size) // stride + 1
        self.total_crops = n_trials * self.crops_per_trial

    def __len__(self):
        return self.total_crops

    def __getitem__(self, idx):
        # Map global index -> (trial_index, crop_index)
        trial_idx = idx // self.crops_per_trial
        crop_idx = idx % self.crops_per_trial
        
        start = crop_idx * self.stride
        end = start + self.crop_size
        
        # Extract crop (Channels, Time)
        x_crop = self.X[trial_idx, :, start:end]
        y_label = self.y[trial_idx]
        
        return torch.from_numpy(x_crop), torch.tensor(y_label)