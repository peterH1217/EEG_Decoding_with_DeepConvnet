"""Fetch data from external sources."""
import logging
import ssl
from moabb.datasets import BNCI2014_001, Schirrmeister2017, PhysionetMI
from neuro_deep_learning.logger import logger

# --- SSL Hack ---
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

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
    
    # Extract Runs
    runs_train = subject_data[train_session_name]
    run_train_name = list(runs_train.keys())[0]
    raw_train = runs_train[run_train_name]
    
    runs_test = subject_data[test_session_name]
    run_test_name = list(runs_test.keys())[0]
    raw_test = runs_test[run_test_name]

    return raw_train, raw_test