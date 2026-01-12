# src/test_run.py
import logging
import sys
from src import data_loader, visualization  # Clean imports

# Configuring logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

def main():
    logger.info(">>> STEP 1: Starting Multi-Dataset Pipeline...")
    
    # 1. Defining Datasets
    # 'BNCI2014_001' = BCI Competition IV 2a
    # 'Schirrmeister2017' = High Gamma Dataset (HGD)
    datasets_to_run = ['BNCI2014_001', 'Schirrmeister2017']
    
    for dataset_name in datasets_to_run:
        logger.info(f"\n--- PROCESSING DATASET: {dataset_name} ---")
        
        try:
            # Helen's Part (Preprocessing & Visualization):
            
            # 2. Load Data (Dynamic Switch)
            # We pass the dataset_name to tell the loader which one to grab
            raw_train, raw_test = data_loader.load_and_process_subject(1, dataset_name=dataset_name)

            # 3. Generate Visualizations
            logger.info(f"Generating visualizations for {dataset_name}...")
            
            visualization.plot_psd(
                raw_train, 
                save_path=f"results/psd_plot_{dataset_name}.png"
            )
            
            visualization.plot_raw_trace(
                raw_train, 
                save_path=f"results/raw_eeg_trace_{dataset_name}.png"
            )

            # Peter's part (Epoching & Cropping) -- adding this to the logic

            logger.info(f"Step 3: Making epochs (trials) for {dataset_name}...")
            X_train, y_train = data_loader.make_epochs(raw_train)
            X_test, y_test   = data_loader.make_epochs(raw_test)


            logger.info(f"Step 4: Cropped training (Schirrmeister logic)...")
            X_train_crops, y_train_crops = data_loader.crop_trials_schirrmeister(X_train, y_train)

            # Verifying Shapes
            print(f"\n--- SHAPES for {dataset_name} ---")
            print(f"Original Trials : {X_train.shape}")       # (n_trials, channels, time)
            print(f"Cropped Trials  : {X_train_crops.shape}") # (n_trials * 625, channels, crop_size)
            print(f"-----------------------------------\n")

            logger.info(f"Successfully finished {dataset_name}. Check results folder!")

        except Exception as e:
            logger.exception(f"Failed to process {dataset_name}: {e}")

    logger.info("\n>>> Pipeline complete: processed and cropped both datasets.")

if __name__ == '__main__':
    main()