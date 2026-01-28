"""Fetch data from external sources."""
import logging
import ssl
from moabb.datasets import BNCI2014_001, Schirrmeister2017, PhysionetMI
from neuro_deep_learning.logger import logger
from mne import concatenate_raws

# --- SSL Hack ---
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass


def get_participants(dataset_name: str) -> list[int]:
    if dataset_name == "BNCI2014_001":
        ds = BNCI2014_001()
        return ds.subject_list
    elif dataset_name == "Schirrmeister2017":
        ds = Schirrmeister2017()
        return ds.subject_list
    elif dataset_name == "PhysionetMI":
        ds = PhysionetMI()
        return ds.subject_list
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataset_class_mapping(dataset_name: str) -> dict[int, str]:
    """
    Returns a dictionary mapping class number -> class name for the given dataset.
    """
    if dataset_name == "BNCI2014_001":
        ds = BNCI2014_001()
    elif dataset_name == "Schirrmeister2017":
        ds = Schirrmeister2017()
    elif dataset_name == "PhysionetMI":
        ds = PhysionetMI()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Use event_id attribute, which exists in all three datasets
    # return ds.event_id
    mapping = {v: k for k, v in ds.event_id.items()}
    return mapping


def get_dataset(subject_id: int, dataset_name: str):
    logger.info(f"Step 1: Loading Subject {subject_id} from {dataset_name}...")
    
    # 1. Select Dataset
    if dataset_name == 'BNCI2014_001':
        ds = BNCI2014_001()
    elif dataset_name == 'Schirrmeister2017':
        ds = Schirrmeister2017()
    elif dataset_name == 'PhysionetMI':
        ds = PhysionetMI()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 2. Extract Data
    ds.subject_list = [subject_id]
    data = ds.get_data()
    subject_data = data[subject_id]
    session_name = list(subject_data.keys())[0] # Usually 'session_0'
    runs = subject_data[session_name]

    # --- FIX FOR LEAKAGE ---
    if dataset_name == 'Schirrmeister2017':
        # MOABB usually labels them as 'train' and 'test' or '0' and '1'
        # Let's verify keys to be safe, but typically:
        if 'train' in runs and 'test' in runs:
            raw_train = runs['train']
            raw_test = runs['test']
        else:
            # Fallback: If keys are just numbers (e.g., '0', '1'), assume 0=Train, 1=Test
            # (Schirrmeister MOABB typically has 2 run entries: Train and Test)
            run_keys = sorted(list(runs.keys()))
            logger.info(f"Splitting runs: {run_keys}")
            
            # Simple split: First part Train, last part Test
            # Note: Verify if your specific download has 2 runs or 13. 
            # If 2 runs: 0 is train, 1 is test.
            raw_train = runs[run_keys[0]]
            raw_test = runs[run_keys[1]]
            
    else:
        # Keep your old logic for BNCI (which has 2 sessions)
        session_names = list(subject_data.keys())
        if len(session_names) > 1:
            # True 2-session dataset (like BNCI)
            runs_train = subject_data[session_names[0]]
            runs_test  = subject_data[session_names[1]]
            
            raw_train = concatenate_raws([runs_train[r] for r in runs_train])
            raw_test  = concatenate_raws([runs_test[r] for r in runs_test])
        else:
            # Fallback for other 1-session datasets (like Physionet)
            # You should split by run index manually here too
            runs_all = [runs[r] for r in runs]
            raw_train = concatenate_raws(runs_all[:-1]) # Use all except last run
            raw_test = runs_all[-1] # Use last run for test

    return raw_train, raw_test