import matplotlib.pyplot as plt
import seaborn as sns
import mne
import numpy as np
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


def plot_training_history(history: dict, dataset_name: str, results_path = None) -> None:
    # plt.style.use()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    ax.plot(history['test_acc'], label='Testing Accuracy', linewidth=2)

    ax.set_title(f"Performance: {dataset_name}", fontsize=14)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if results_path is not None:
      plt.savefig(results_path)
      plt.show()
      plt.close()


def plot_test_confusion_matrix(y_true, y_pred, class_mapping: dict[int, str], title="Confusion Matrix", save_path=None):
    """
    Plots a clean confusion matrix with counts and percentages.

    Parameters:
        y_true: np.array or list of true labels
        y_pred: np.array or list of predicted labels
        class_mapping: dict {class_num: class_name}
        title: Title of the plot
        save_path: Path to save the figure. If None, it will just show it.
    """
    # Sort classes by numeric code and format names
    sorted_classes = sorted(class_mapping.keys())
    class_names = [class_mapping[c].replace('_', ' ').capitalize() for c in sorted_classes]
    n_classes = len(class_names)

    # Compute confusion matrix (rows=predictions, columns=targets)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[p, t] += 1

    total_samples = np.sum(cm)
    cm_normalized = cm / total_samples

    # Plot
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2, style="white")
    ax = sns.heatmap(cm_normalized, annot=False, cmap="Reds", cbar=True,
                     xticklabels=class_names, yticklabels=class_names,
                     linewidths=1, linecolor='white', square=True)

    # Add counts + percentages
    for i in range(n_classes):
        for j in range(n_classes):
            count = cm[i, j]
            pct = cm[i, j] / total_samples * 100
            text = f"{count}\n{pct:.1f}%"
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                    color='black' if i == j else 'gray', fontsize=12, weight='bold' if i==j else 'normal')

    # Labels and title
    ax.set_xlabel('Targets', fontsize=14)
    ax.set_ylabel('Predictions', fontsize=14)
    ax.set_title(title, fontsize=16)

    # Rotate y-axis labels to vertical
    ax.set_yticklabels(class_names, rotation=0, va='center')
    ax.set_xticklabels(class_names, rotation=0, ha='right')

    ax.set_facecolor('white')
    plt.tight_layout()

    if save_path is not None:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to: {save_path}")
        except Exception as e:
            logger.info(f"Failed to save figure: {e}")

    plt.show()
    plt.close()

