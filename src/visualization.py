import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import config

def plot_power_spectrum(raw: mne.io.Raw, title="Power Spectrum Density (PSD)"):
    """
    Generates a PSD plot to check if filtering (4-38Hz) worked.
    """
    # MNE plot_psd automatically picks good channels
    fig = raw.compute_psd(fmax=60).plot(show=False)
    
    save_path = os.path.join(config.RESULTS_DIR, "psd_plot.png")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved PSD plot to {save_path}")
    plt.close()

def plot_raw_segment(raw: mne.io.Raw, duration=5):
    """
    Plots the first 5 seconds of EEG data.
    """
    # CORRECTED CHANNEL NAMES (No 'EEG-' prefix)
    picks = ['C3', 'Cz', 'C4']
    
    # Check if these channels actually exist, otherwise pick the first 3
    available_channels = raw.ch_names
    final_picks = [ch for ch in picks if ch in available_channels]
    
    if not final_picks:
        print(f"Warning: Preferred channels {picks} not found. Using first 3 channels.")
        final_picks = available_channels[:3]
    
    data, times = raw.get_data(picks=final_picks, start=0, stop=int(duration * config.SAMPLING_RATE), return_times=True)
    
    plt.figure(figsize=(10, 6))
    for i, channel_data in enumerate(data):
        # Offset each channel so they don't overlap
        plt.plot(times, channel_data + (i * 5e-5), label=final_picks[i]) # 5e-5 is a scaling factor for visibility
        
    plt.title(f"First {duration} seconds of Motor Cortex EEG")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (Volts)")
    plt.legend()
    
    save_path = os.path.join(config.RESULTS_DIR, "raw_eeg_trace.png")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved Raw EEG plot to {save_path}")
    plt.close()