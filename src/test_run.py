import logging
from src.data_loader import load_and_process_subject
from src.visualization import plot_psd, plot_raw_trace
import sys
import data_loader
import visualization
import numpy as np

#from src import data_loader, visualization

# Configuring logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info(">>> STEP 1: Starting Multi-Dataset Pipeline...")
    
    # 1. Defining the list of datasets we want to process
    # 'BNCI2014_001' = BCI Competition IV 2a
    # 'Schirrmeister2017' = High Gamma Dataset (HGD)
    datasets_to_run = ['BNCI2014_001', 'Schirrmeister2017']
    
    for dataset_name in datasets_to_run:
        logger.info(f"\n--- PROCESSING DATASET: {dataset_name} ---")
        
        try:
            # 2. Loading Data (Dynamic Switch)
            raw_train, raw_test = load_and_process_subject(1, dataset_name=dataset_name)

            # 3. Generating Visualizations (with unique filenames!)
            logger.info(f"Generating visualizations for {dataset_name}...")
            
            # We add the dataset_name to the filename so they don't overwrite each other
            plot_psd(
                raw_train, 
                save_path=f"results/psd_plot_{dataset_name}.png"
            )
            
            plot_raw_trace(
                raw_train, 
                save_path=f"results/raw_eeg_trace_{dataset_name}.png"
            )
            
            logger.info(f"Successfully finished {dataset_name}. Check results folder!")

        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}")

    logger.info("\n>>> Pipeline complete: we have results for both datasets.")

if __name__ == '__main__':
    main()
except Exception as e:
    logger.exception("Something went wrong!")





# Configure Logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger("test_script")

def main():
    logger.info(">>> STEP 1: Loading Data...")
    raw_train, raw_test = data_loader.load_and_process_subject(subject_id=1)

    logger.info(">>> STEP 2: Generating Visualizations...")
    visualization.plot_power_spectrum(raw_train)
    visualization.plot_raw_segment(raw_train)

    logger.info(">>> STEP 3: Making epochs (trials)...")
    X_train, y_train = data_loader.make_epochs(raw_train)
    X_test,  y_test  = data_loader.make_epochs(raw_test)

    logger.info(">>> STEP 4: Cropped training...")
    X_train_crops, y_train_crops = data_loader.crop_trials_schirrmeister(X_train, y_train)

    print("X_train.shape        :", X_train.shape)         # (n_trials, n_channels, n_times)
    print("y_train.shape        :", y_train.shape)         # (n_trials,)

    print("X_train_crops.shape  :", X_train_crops.shape)   # (n_trials*625, n_channels, 500)
    print("y_train_crops.shape  :", y_train_crops.shape)   # (n_trials*625,)

if __name__ == "__main__":
    try:
        main()
        print("\n>>> SUCCESS! Check the 'results' folder for your images!")
    except Exception:
        logger.exception("Something went wrong!")
