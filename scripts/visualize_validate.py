import os
from nibabel import load as nib_load
import mne
import numpy as np
import matplotlib.pyplot as plt

# Constant file paths for dataset files
T1_FILE = '/Users/jsnunki/Code/neuro-viz/datasets/Simultaneous_eeg-fmri/bids_mri_fmri/sub-001/ses-001/anat/sub-001_ses-001_acq-highres_T1w.nii.gz'
FMRI_FILES = [
    '/Users/jsnunki/Code/neuro-viz/datasets/Simultaneous_eeg-fmri/bids_mri_fmri/sub-001/ses-001/func/sub-001_ses-001_task-eoec_bold.nii.gz',
    '/Users/jsnunki/Code/neuro-viz/datasets/Simultaneous_eeg-fmri/bids_mri_fmri/sub-001/ses-001/func/sub-001_ses-001_task-rest_bold.nii.gz'
]
EEG_FILES = [
    '/Users/jsnunki/Code/neuro-viz/datasets/Simultaneous_eeg-fmri/bids_eeg/derivatives/gac_with_events_edf/01_EEG_fMRI_OCOA_GAC_with_events.edf',
    '/Users/jsnunki/Code/neuro-viz/datasets/Simultaneous_eeg-fmri/bids_eeg/derivatives/gac_with_events_edf/01_EEG_fMRI_resting_EC_GAC_with_events.edf',
    '/Users/jsnunki/Code/neuro-viz/datasets/Simultaneous_eeg-fmri/bids_eeg/derivatives/gac_with_events_edf/01_EEG_fMRI_resting_EO_GAC_with_events.edf'
]


# Function to validate whether a file exists
def validate_file(filepath):
    """
    Validates the existence of the file at the specified path.

    Args:
        filepath (str): The path to the file.

    Returns:
        bool: True if the file exists; False otherwise.
    """
    if not os.path.isfile(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return False
    return True


# Function to visualize T1-weighted anatomical imaging
def visualize_t1(filepath):
    """
    Visualizes T1-weighted anatomical images in sagittal, coronal, and axial views.

    Args:
        filepath (str): The path to the T1-weighted image file.
    """
    if not validate_file(filepath):
        return

    t1_img = nib_load(filepath)
    t1_data = t1_img.get_fdata()

    # Display sagittal, coronal, and axial views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(t1_data[t1_data.shape[0] // 2, :, :], cmap='gray')
    axes[0].set_title("Sagittal")
    axes[1].imshow(t1_data[:, t1_data.shape[1] // 2, :], cmap='gray')
    axes[1].set_title("Coronal")
    axes[2].imshow(t1_data[:, :, t1_data.shape[2] // 2], cmap='gray')
    axes[2].set_title("Axial")
    plt.suptitle("T1-Weighted Image Visualization")
    plt.show()


# Function to visualize fMRI data
def visualize_fmri(filepath):
    """
    Visualizes fMRI data, including an axial slice of the first volume and the time series of a voxel.

    Args:
        filepath (str): The path to the fMRI data file.
    """
    if not validate_file(filepath):
        return

    fmri_img = nib_load(filepath)
    fmri_data = fmri_img.get_fdata()

    # Display the first fMRI axial slice
    plt.imshow(fmri_data[:, :, fmri_data.shape[2] // 2, 0], cmap='hot')
    plt.title("fMRI Initial Volume - Axial Slice")
    plt.colorbar()
    plt.show()

    # Plot the time series from the center voxel
    voxel_time_series = fmri_data[fmri_data.shape[0] // 2,
                        fmri_data.shape[1] // 2,
                        fmri_data.shape[2] // 2, :]
    plt.plot(voxel_time_series)
    plt.title("Voxel Time Series")
    plt.xlabel("Time (TRs)")
    plt.ylabel("Signal Intensity")
    plt.show()


# Function to visualize EEG signals and their annotations
def visualize_eeg(filepath):
    """
    Visualizes EEG signals and annotations using the MNE library.

    Args:
        filepath (str): The path to the EEG data file in EDF format.
    """
    if not validate_file(filepath):
        return

    # Load EEG data in EDF format
    raw = mne.io.read_raw_edf(filepath, preload=True)

    # Apply band-pass filtering (1–50 Hz)
    raw_filtered = raw.copy().filter(l_freq=0.5, h_freq=40.0)

    # Plot EEG signals (filtered, 10-second duration)
    raw_filtered.plot(duration=10, n_channels=10, scalings='auto', title="EEG Signals (Filtered)")

    # Compute and plot Power Spectral Density (PSD)
    psd = raw_filtered.compute_psd(method='welch', fmax=50)
    fig = psd.plot(average=True, amplitude=False)  # Generate PSD plot
    fig.suptitle("Power Spectral Density (Filtered)")  # Add title to PSD plot

    # If event annotations are available, display them
    if raw_filtered.annotations:
        print("Annotations (Events):")
        print(raw_filtered.annotations)
        raw_filtered.plot_annotations()


# Main function for executing the visualizations
def main():
    """
    Main function to execute the visualizations for T1, fMRI, and EEG data.
    """
    # Visualize T1-weighted anatomical images
    print("\n[INFO] Visualizing T1-weighted image...")
    visualize_t1(T1_FILE)

    # Visualize fMRI data
    for fmri_file in FMRI_FILES:
        print(f"\n[INFO] Visualizing fMRI data: {fmri_file}")
        visualize_fmri(fmri_file)

    # Visualize EEG data
    for eeg_file in EEG_FILES:
        print(f"\n[INFO] Visualizing EEG data: {eeg_file}")
        visualize_eeg(eeg_file)


# Entry point of the script
if __name__ == "__main__":
    main()
