import matplotlib.pyplot as plt
import seaborn as sns
import mne
import logging
# import neuro_deep_learning.config as config  <-- Removed (unused)

# Configure logging
logger = logging.getLogger(__name__)

def plot_psd(raw, save_path="results/psd_plot.png"):
    """
    Plots the Power Spectral Density (PSD) of the raw data.
    """
    logger.info("Plotting power spectral density (dB=True).")
    
    # Critical for servers/scripts: Use non-interactive backend
    current_backend = plt.get_backend()
    plt.switch_backend('Agg') 
    
    fig = raw.compute_psd(fmax=50).plot(show=False)
    
    # Save the figure
    fig.savefig(save_path)
    logger.info(f"Saved PSD plot to {save_path}")
    plt.close(fig)
    
    # Restore backend (optional)
    plt.switch_backend(current_backend)

def plot_raw_trace(raw, save_path="results/raw_eeg_trace.png", duration=5, start_time=0):
    """
    Plots a few seconds of raw EEG traces for Motor channels (C3, Cz, C4).
    """
    logger.info(f"Plotting first {duration} seconds of raw EEG...")
    
    plt.switch_backend('Agg') # Ensure no window pops up
    
    # Select specific motor cortex channels to make the plot readable
    possible_channels = ['C3', 'Cz', 'C4']
    channels_to_plot = [ch for ch in possible_channels if ch in raw.ch_names]
    
    if not channels_to_plot:
        logger.warning("Could not find C3, Cz, or C4. Plotting first 3 channels instead.")
        channels_to_plot = raw.ch_names[:3]

    # Extract data
    data, times = raw.get_data(picks=channels_to_plot, start=int(start_time * raw.info['sfreq']), 
                               stop=int((start_time + duration) * raw.info['sfreq']), return_times=True)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    for i, channel_name in enumerate(channels_to_plot):
        plt.plot(times, data[i], label=channel_name)
        
    plt.title(f"First {duration} seconds of Motor Cortex EEG")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (Normalized)")
    plt.legend(loc="upper right")
    
    # Save
    plt.savefig(save_path)
    logger.info(f"Saved Raw EEG plot to {save_path}")
    plt.close()