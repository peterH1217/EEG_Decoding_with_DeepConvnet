import logging
import sys
import data_loader
import visualization
import numpy as np

#from src import data_loader, visualization

# Configure Logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger("test_script")

try:
    logger.info(">>> STEP 1: Loading Data...")
    raw_train, raw_test = data_loader.load_and_process_subject(1)
    
    logger.info(">>> STEP 2: Generating Visualizations for Presentation...")
    
    # Plot 1: Did the filter work? (Should see a drop off after 38Hz)
    visualization.plot_power_spectrum(raw_train)
    
    # Plot 2: What does the signal look like?
    visualization.plot_raw_segment(raw_train)
    
    print("\n>>> SUCCESS! Check the 'results' folder for your images!")

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
