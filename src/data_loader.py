# src/data_loader.py
import logging
import ssl
import mne
from moabb.datasets import BNCI2014_001, Schirrmeister2017
from src import config

logger = logging.getLogger(__name__)

# Mac Fix: Ignoring SSL Errors
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

def get_dataset(subject_id: int, dataset_name):
    """
    Downloads and loads data for a specific subject from the specified dataset.
    Supported datasets: 'BNCI2014_001' (BCI IV 2a) and 'Schirrmeister2017' (High Gamma).
    """
    logger.info(f"Step 1: Loading Subject {subject_id} from {dataset_name}...")

    # Dataset Switch
    if dataset_name == 'BNCI2014_001':
        ds = BNCI2014_001()
    elif dataset_name == 'Schirrmeister2017':
        ds = Schirrmeister2017()
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    ds.subject_list = [subject_id]
    
    # Downloading data (Note to self: MOABB handles the caching automatically)
    data = ds.get_data()
    
    # DYNAMIC LOOKUP: works for both datasets because we look at the keys dynamically
    subject_data = data[subject_id]
    session_names = list(subject_data.keys())
    
    # The first session is Training, second is Test (or Evaluation)
    train_session_name = session_names[0]
    # Handling rare cases where a dataset might only have one session listed
    test_session_name = session_names[1] if len(session_names) > 1 else session_names[0]

    logger.info(f"DEBUG: Found sessions: {train_session_name}, {test_session_name}")

    # Extract runs (Training)
    runs_train = subject_data[train_session_name]
    run_train_name = list(runs_train.keys())[0]
    
    # Extract runs (Test)
    runs_test = subject_data[test_session_name]
    run_test_name = list(runs_test.keys())[0]

    return runs_train[run_train_name], runs_test[run_test_name]

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

def load_and_process_subject(subject_id: int, dataset_name):
    """
    Master function to load and process data.
    """
    # Pass the dataset_name to the get_dataset function
    raw_train, raw_test = get_dataset(subject_id, dataset_name)
    
    # Preprocessing (Filtering/Normalization) is the same for both!
    raw_train = preprocess_data(raw_train)
    raw_test = preprocess_data(raw_test)
    
    logger.info(f"Successfully processed Subject {subject_id} from {dataset_name}")
    return raw_train, raw_test


