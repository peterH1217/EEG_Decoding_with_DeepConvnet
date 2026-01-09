# src/data_loader.py
import logging
import ssl
import mne
from moabb.datasets import BNCI2014_001
import config
import numpy as np


logger = logging.getLogger(__name__)

# --- MAC FIX: IGNORE SSL ERRORS ---
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

def get_dataset(subject_id: int):
    """
    Downloads and loads data for a specific subject.
    Dynamically finds session and run names to avoid KeyErrors.
    """
    logger.info(f"Step 1: Loading data for Subject {subject_id} via MOABB...")
    
    ds = BNCI2014_001()
    ds.subject_list = [subject_id]
    
    # Download data
    data = ds.get_data()
    
    # --- DYNAMIC LOOKUP (The Fix) ---
    # We don't guess '0train' or 'run_0'. We look at what is actually there.
    subject_data = data[subject_id]
    
    # 1. Find the Training Session name (usually '0train' or 'session_T')
    train_session_name = list(subject_data.keys())[0] 
    # 2. Find the Test Session name (usually '1test' or 'session_E')
    test_session_name = list(subject_data.keys())[1]
    
    logger.info(f"DEBUG: Found sessions: {train_session_name}, {test_session_name}")

    # 3. Get the run data (The 'KeyError' Fix)
    # We grab the first run available in the session, whatever it is named.
    runs_train = subject_data[train_session_name]
    run_train_name = list(runs_train.keys())[0] # e.g. 'run_0'
    
    runs_test = subject_data[test_session_name]
    run_test_name = list(runs_test.keys())[0]
    
    logger.info(f"DEBUG: Found runs: {run_train_name}, {run_test_name}")
    
    raw_train = runs_train[run_train_name]
    raw_test = runs_test[run_test_name]
    
    return raw_train, raw_test

def preprocess_data(raw: mne.io.Raw):
    """
    Applies filtering and channel selection.
    """
    logger.info("Step 2-5: Preprocessing...")
    
    # 1. Pick EEG channels
    raw.pick_types(eeg=True)
    
    # 2. Resample if needed
    if raw.info['sfreq'] != config.SAMPLING_RATE:
        raw.resample(config.SAMPLING_RATE)
        
    # 3. Filter
    raw.filter(l_freq=config.LOW_CUTOFF, h_freq=config.HIGH_CUTOFF, method='iir', verbose=False)
    
    # 4. Normalization
    # We will apply "Z-score normalization" (subtract mean, divide by std) to each channel independently.
    logger.info("Step 4: Applying Channel-wise Normalization (Z-score)...")
    raw.apply_function(lambda x: (x - x.mean()) / x.std())
    
    return raw

def load_and_process_subject(subject_id: int):
    """
    Master function.
    """
    raw_train, raw_test = get_dataset(subject_id)
    raw_train = preprocess_data(raw_train)
    raw_test = preprocess_data(raw_test)
    
    logger.info(f"Successfully processed Subject {subject_id}")
    return raw_train, raw_test


def make_epochs(raw: mne.io.Raw):
    """
    Create trials (epochs) following Schirrmeister et al. (2017).
    Epochs cover [-0.5, +4.0] seconds relative to trial start.
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    picks = mne.pick_types(raw.info, eeg=True)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=-0.5,
        tmax=4.0,
        picks=picks,
        baseline=None,
        preload=True,
        verbose=False,
    )

    X = epochs.get_data()          # (trials, channels, time)
    y = epochs.events[:, 2]        # event codes
    return X, y


def crop_trials_schirrmeister(X, y):
    """
    Cropped training exactly following Schirrmeister et al. (2017):
    - sampling rate: config.SAMPLING_RATE (250 Hz)
    - crop length: 2 s (500 samples)
    - step: 1 sample (0.004 s)
    - first crop starts at -0.5 s
    - last crop ends at +4.0 s
    - 625 crops per trial
    """
    sfreq = config.SAMPLING_RATE
    crop_size = int(2.0 * sfreq)   # 500 samples
    X_crops = []
    y_crops = []

    for trial_idx in range(len(X)):
        trial = X[trial_idx]       # (channels, time)
        label = y[trial_idx]

        # trial length should be ~1125 samples at 250 Hz
        T = trial.shape[-1]

        # enforce 625 crops (paper)
        last_start = 624            # 0..624 -> 625 crops

        for start in range(0, last_start + 1):
            crop = trial[:, start:start + crop_size]
            X_crops.append(crop)
            y_crops.append(label)

    return np.asarray(X_crops), np.asarray(y_crops)
