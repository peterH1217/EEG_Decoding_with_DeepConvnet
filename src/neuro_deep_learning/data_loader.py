# src/data_loader.py
import logging
import ssl
import numpy as np
import mne
import torch
from torch.utils.data import Dataset
from moabb.datasets import BNCI2014_001, Schirrmeister2017

import config  # IMPORTANT: keep package import

logger = logging.getLogger(__name__)

# Ignore SSL errors (harmless on Windows, useful on some Macs)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass


# ---------------- Dataset loading (MOABB auto-cache) ----------------
def get_dataset(subject_id: int, dataset_name: str):
    logger.info(f"Step 1: Loading Subject {subject_id} from {dataset_name}...")

    if dataset_name == "BNCI2014_001":
        ds = BNCI2014_001()
    elif dataset_name == "Schirrmeister2017":
        ds = Schirrmeister2017()
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    ds.subject_list = [subject_id]
    data = ds.get_data()

    subject_data = data[subject_id]
    session_names = list(subject_data.keys())

    train_session_name = session_names[0]
    test_session_name = session_names[1] if len(session_names) > 1 else session_names[0]
    logger.info(f"DEBUG: Found sessions: {train_session_name}, {test_session_name}")

    runs_train = subject_data[train_session_name]
    run_train_name = list(runs_train.keys())[0]

    runs_test = subject_data[test_session_name]
    run_test_name = list(runs_test.keys())[0]

    return runs_train[run_train_name], runs_test[run_test_name]


# ---------------- Preprocessing (keep STIM!) ----------------
def preprocess_data(raw: mne.io.Raw):
    """
    Preprocess EEG while KEEPING stim/trigger channels (needed for HGD events).
    We only filter/resample EEG channels; stim channels are preserved untouched.
    """
    raw = raw.copy()

    # Keep EEG + stim channels so triggers remain available
    raw.pick_types(eeg=True, stim=True)

    # Resample (applies to all kept channels; ok)
    if raw.info["sfreq"] != config.SAMPLING_RATE:
        raw.resample(config.SAMPLING_RATE)

    # Filter ONLY EEG channels (never filter stim)
    eeg_picks = mne.pick_types(raw.info, eeg=True, stim=False)
    raw.filter(
        l_freq=config.LOW_CUTOFF,
        h_freq=config.HIGH_CUTOFF,
        picks=eeg_picks,
        method="iir",
        verbose=False,
    )

    return raw


def load_and_process_subject(subject_id: int, dataset_name: str):
    raw_train, raw_test = get_dataset(subject_id, dataset_name)
    raw_train = preprocess_data(raw_train)
    raw_test = preprocess_data(raw_test)
    logger.info(f"Successfully processed Subject {subject_id} from {dataset_name}")
    return raw_train, raw_test


# ---------------- Epoching helpers ----------------
def _choose_top_n_events(events, event_id, n_classes=4):
    codes = events[:, 2]
    uniq, cnt = np.unique(codes, return_counts=True)
    top = uniq[np.argsort(-cnt)][: int(n_classes)]
    code_to_name = {v: k for k, v in event_id.items()}
    chosen = {code_to_name[c]: int(c) for c in top if c in code_to_name}
    if len(chosen) < 2:
        raise RuntimeError(f"Too few classes selected: {chosen}")
    return chosen


def make_epochs_bci(raw, tmin=0.0, tmax=4.0, baseline=None):
    """
    BNCI2014_001: usually has annotations with class names.
    """
    raw = raw.copy()
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    if not event_id:
        raise RuntimeError("BCI: No annotations found; cannot epoch.")

    # Prefer common MI labels if present, else fallback to top-4
    prefer = [k for k in ["left_hand", "right_hand", "feet", "tongue", "left", "right"] if k in event_id]
    chosen = {k: event_id[k] for k in prefer} if prefer else _choose_top_n_events(events, event_id, n_classes=4)

    # Epoch on EEG only
    raw.pick_types(eeg=True)
    epochs = mne.Epochs(
        raw, events, event_id=chosen,
        tmin=tmin, tmax=tmax,
        baseline=baseline,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    # Keep in volts for artifact removal; convert to µV later if needed
    X_volts = epochs.get_data().astype(np.float32)
    y_codes = epochs.events[:, 2]
    code_list = sorted(set(y_codes))
    code_to_label = {c: i for i, c in enumerate(code_list)}
    y = np.array([code_to_label[c] for c in y_codes], dtype=np.int64)
    return X_volts, y


def make_epochs_hgd(raw, tmin=0.0, tmax=4.0, baseline=None):
    """
    Schirrmeister2017 (HGD): may rely on stim/trigger channel.
    We find events BEFORE dropping stim; then epoch on EEG only.
    """
    raw = raw.copy()

    # 1) Find events (needs stim sometimes)
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    if not event_id:
        stim_candidates = [
            ch for ch in raw.ch_names
            if ("STI" in ch.upper()) or ("TRIG" in ch.upper()) or ("STIM" in ch.upper())
        ]
        if not stim_candidates:
            raise RuntimeError(f"HGD: No annotations and no stim channel. First channels: {raw.ch_names[:30]}")
        events = mne.find_events(raw, stim_channel=stim_candidates[0], shortest_event=1, verbose=False)

        # Create a dummy mapping from the most frequent codes
        codes = events[:, 2]
        uniq, cnt = np.unique(codes, return_counts=True)
        top = uniq[np.argsort(-cnt)][:4]
        event_id = {f"class_{i}": int(c) for i, c in enumerate(top)}

    chosen = _choose_top_n_events(events, event_id, n_classes=4)

    # 2) Epoch on EEG only
    raw.pick_types(eeg=True)
    epochs = mne.Epochs(
        raw, events, event_id=chosen,
        tmin=tmin, tmax=tmax,
        baseline=baseline,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    X_volts = epochs.get_data().astype(np.float32)
    y_codes = epochs.events[:, 2]
    code_list = sorted(set(y_codes))
    code_to_label = {c: i for i, c in enumerate(code_list)}
    y = np.array([code_to_label[c] for c in y_codes], dtype=np.int64)
    return X_volts, y


# ---------------- Artifact removal (±800 µV) ----------------
def remove_artifact_trials_uV(X_volts, y, threshold_uV=800.0):
    """
    Remove trials where ANY channel exceeds ±threshold_uV.
    Note: X_volts is in volts -> convert to µV internally.
    """
    X_uV = X_volts * 1e6
    keep = np.max(np.abs(X_uV), axis=(1, 2)) <= float(threshold_uV)
    return X_volts[keep], y[keep], keep


# ---------------- Normalization for CNN (optional but recommended) ----------------
def zscore_per_channel(X_volts, eps=1e-8):
    """
    Z-score each trial independently per channel: (x-mean)/std over time.
    Keeps shape the same and is safe for CNN training.
    """
    mean = X_volts.mean(axis=2, keepdims=True)
    std = X_volts.std(axis=2, keepdims=True)
    return (X_volts - mean) / (std + eps)


# ---------------- Cropped training (memory-safe, on-the-fly) ----------------
class CropsDataset(Dataset):
    """
    On-the-fly crops for CNN training: returns (crop[ch, crop_size], label).
    X should be float32 (we convert), shape (trials, ch, time).
    """
    def __init__(self, X, y, crop_size=500, stride=1, max_crops_per_trial=None):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.crop_size = int(crop_size)
        self.stride = int(stride)

        n_trials, _, n_time = self.X.shape
        if n_trials == 0:
            self.crops_per_trial = 0
            self.total = 0
            return

        cpt = (n_time - self.crop_size) // self.stride + 1
        if cpt < 1:
            # no valid crop
            self.crops_per_trial = 0
            self.total = 0
            return

        if max_crops_per_trial is not None:
            cpt = min(cpt, int(max_crops_per_trial))

        self.crops_per_trial = cpt
        self.total = n_trials * cpt

    def __len__(self):
        return int(self.total)

    def __getitem__(self, idx):
        t = idx // self.crops_per_trial
        c = idx % self.crops_per_trial
        start = c * self.stride
        end = start + self.crop_size
        x = self.X[t, :, start:end]
        return torch.from_numpy(x), torch.tensor(self.y[t])


def make_crops_dataset_for_cnn(X, y, crop_size=500, stride=1, max_crops_per_trial=None):
    return CropsDataset(X, y, crop_size=crop_size, stride=stride, max_crops_per_trial=max_crops_per_trial)
