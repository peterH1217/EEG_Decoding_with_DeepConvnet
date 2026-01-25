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
    """
    Fetch data for a specific subject. 
    Matches original data_loader.py signature: (subject_id, dataset_name)
    """
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

    # 2. Extract specific subject data
    ds.subject_list = [subject_id]
    data = ds.get_data()
    
    # Handle MOABB dictionary structure
    subject_data = data[subject_id]
    session_names = list(subject_data.keys())
    train_session_name = session_names[0]
    
    # Use test session if available, else duplicate train (for demo)
    test_session_name = session_names[1] if len(session_names) > 1 else session_names[0]
    
    # Get all Train Runs
    runs_train = subject_data[train_session_name]
    # Collect all 'Raw' objects from the dictionary values
    raws_train_list = [runs_train[run_name] for run_name in runs_train.keys()]
    # Stitch them together
    raw_train = concatenate_raws(raws_train_list)
    
    # 2. Get all Test Runs
    runs_test = subject_data[test_session_name]
    raws_test_list = [runs_test[run_name] for run_name in runs_test.keys()]
    raw_test = concatenate_raws(raws_test_list)

    return raw_train, raw_test