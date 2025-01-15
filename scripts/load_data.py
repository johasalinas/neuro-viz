import nibabel as nib
import mne
import numpy as np
import matplotlib.pyplot as plt
import os

# Define paths to data directories
T1_DIR = '/Users/jsnunki/Code/neuro-viz/datasets/Simultaneous_eeg-fmri/bids_mri_fmri/sub-001/ses-001/anat'
FMRI_DIR = '/Users/jsnunki/Code/neuro-viz/datasets/Simultaneous_eeg-fmri/bids_mri_fmri/sub-001/ses-001/func'
EEG_DIR = '/Users/jsnunki/Code/neuro-viz/datasets/Simultaneous_eeg-fmri/bids_eeg/derivatives/gac_with_events_edf'

# Construct file paths dynamically
T1_FILES = [
    os.path.join(T1_DIR, 'sub-001_ses-001_acq-highres_T1w.nii.gz')
]

FMRI_FILES = [
    os.path.join(FMRI_DIR, 'sub-001_ses-001_task-eoec_bold.nii.gz'),
    os.path.join(FMRI_DIR, 'sub-001_ses-001_task-rest_bold.nii.gz')
]

EEG_FILES = [
    os.path.join(EEG_DIR, '01_EEG_fMRI_OCOA_GAC_with_events.edf'),
    os.path.join(EEG_DIR, '01_EEG_fMRI_resting_EC_GAC_with_events.edf'),
    os.path.join(EEG_DIR, '01_EEG_fMRI_resting_EO_GAC_with_events.edf')
]


# 1. Load anatomical data (T1) using NiBabel
def load_t1(filepath):
    """
    Load T1-weighted anatomical data using NiBabel and visualize it.
    :param filepath: Path to the T1 image file.
    :return: Loaded T1 image object.
    """
    t1_img = nib.load(filepath)
    t1_data = t1_img.get_fdata()
    print("\n[INFO] T1 Image Loaded:")
    print(f"Dimensions: {t1_data.shape}")
    print(f"Affine Matrix: \n{t1_img.affine}")

    # Visualize the middle axial slice
    plt.imshow(t1_data[:, :, t1_data.shape[2] // 2], cmap='gray')
    plt.title('Axial Slice T1 - Anatomy')
    plt.show()
    return t1_img


# 2. Load functional MRI (fMRI) data
def load_fmri(filepath):
    """
    Load fMRI data using NiBabel and perform basic validation.
    :param filepath: Path to the fMRI file.
    :return: Loaded fMRI image object.
    """
    fmri_img = nib.load(filepath)
    fmri_data = fmri_img.get_fdata()

    # Ensure the fMRI data has a time dimension (len(shape) == 4)
    if len(fmri_data.shape) < 4:
        raise ValueError(f"[ERROR] fMRI data {filepath} does not have a time dimension.")

    # Extract the repetition time (TR) if available
    tr = fmri_img.header.get_zooms()[3] if len(fmri_img.header.get_zooms()) > 3 else None
    if tr is None:
        print("[WARNING] TR is missing in the fMRI header.")

    print("\n[INFO] fMRI Image Loaded:")
    print(f"Dimensions: {fmri_data.shape}")
    print(f"TR (Repetition Time): {tr} sec")

    # Visualize the middle axial slice of the first volume
    plt.imshow(fmri_data[:, :, fmri_data.shape[2] // 2, 0], cmap='hot')
    plt.title('fMRI Volume 1 - Axial Slice')
    plt.show()
    return fmri_img


# 3. Load EEG data (EDF files) using MNE-Python
def load_eeg(filepath):
    """
    Load EEG data from an EDF file and display basic information.
    :param filepath: Path to the EEG file.
    :return: MNE Raw object containing EEG data.
    """
    raw = mne.io.read_raw_edf(filepath, preload=True)
    print("\n[INFO] EEG Data Loaded:")
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Sampling Frequency: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]} seconds")

    # Visualize the first 20 seconds of the EEG signals
    raw.plot(duration=100, n_channels=10, title=f'EEG Data Preview: {os.path.basename(filepath)}')
    return raw


# 4. Main execution block
if __name__ == "__main__":
    """
    Main workflow for loading and validating T1, fMRI, and EEG data.
    """
    # Ensure all required files exist
    missing_files = [f for f in T1_FILES + FMRI_FILES + EEG_FILES if not os.path.exists(f)]
    if missing_files:
        print(f"[ERROR] The following files were not found:\n{missing_files}")
    else:
        # Load anatomical (T1) data
        t1_img = load_t1(T1_FILES[0])

        # Load functional MRI data
        fmri_imgs = [load_fmri(f) for f in FMRI_FILES]

        # Load EEG data
        eeg_raws = [load_eeg(f) for f in EEG_FILES]

        print("\n[INFO] Data loading completed.")
