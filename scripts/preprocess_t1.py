import SimpleITK as sitk
import os


def preprocess_t1(file_path, output_path, domain_sigma=0.1, range_sigma=20.0, normalize=True):
    """
    Preprocess T1-weighted MRI data for visualization and analysis.
    Includes bias field correction, intensity normalization, and bilateral smoothing.

    Args:
        file_path (str): Path to the input T1-weighted MRI file.
        output_path (str): Path to save the preprocessed T1 file.
        domain_sigma (float): Spatial sigma for the bilateral filter.
        range_sigma (float): Intensity sigma for the bilateral filter.
        normalize (bool): Whether to normalize intensities to the range [0, 255].
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] File not found: {file_path}")

    print("[INFO] Loading T1 MRI image...")
    image = sitk.ReadImage(file_path)

    # Step 1: Convert to sitkFloat32 if necessary (required for N4BiasFieldCorrection)
    if image.GetPixelID() != sitk.sitkFloat32:
        print("[INFO] Converting image to sitkFloat32 for processing...")
        image = sitk.Cast(image, sitk.sitkFloat32)

    # Step 2: Apply Bias Field Correction
    print("[INFO] Applying Bias Field Correction...")
    corrected_image = sitk.N4BiasFieldCorrection(image)

    # Step 3: Normalize intensities to [0, 255] (if enabled)
    if normalize:
        print("[INFO] Normalizing intensities to range [0, 255]...")
        intensity_filter = sitk.RescaleIntensityImageFilter()
        intensity_filter.SetOutputMinimum(0)
        intensity_filter.SetOutputMaximum(255)
        corrected_image = intensity_filter.Execute(corrected_image)

    # Step 3: Apply Adaptive Histogram Equalization
    print("[INFO] Enhancing contrast with Adaptive Histogram Equalization...")
    adaptive_histogram_filter = sitk.AdaptiveHistogramEqualizationImageFilter()
    contrast_enhanced_image = adaptive_histogram_filter.Execute(corrected_image)

    # Step 4: Apply Bilateral Smoothing
    print(f"[INFO] Applying Bilateral Smoothing (domain_sigma={domain_sigma}, range_sigma={range_sigma})...")
    smoothed_image = sitk.Bilateral(corrected_image, domainSigma=domain_sigma, rangeSigma=range_sigma)

    # Step 5: Save the preprocessed image
    print(f"[INFO] Saving preprocessed T1 MRI to {output_path}...")
    sitk.WriteImage(smoothed_image, output_path)
    print("[INFO] Preprocessing completed successfully.")




def save_intermediate_images(image, step_name, output_dir):
    """
    Save intermediate images during preprocessing for validation purposes.

    Args:
        image (sitk.Image): Image to save.
        step_name (str): Name of the preprocessing step.
        output_dir (str): Directory to save intermediate images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{step_name}.nii.gz")
    sitk.WriteImage(image, output_path)
    print(f"[INFO] Saved intermediate image: {step_name} at {output_path}")


def preprocess_t1_with_validation(file_path, output_path, domain_sigma=0.1, range_sigma=20.0, normalize=True):
    """
    Preprocess T1-weighted MRI data with intermediate validation outputs.

    Args:
        file_path (str): Path to the input T1-weighted MRI file.
        output_path (str): Path to save the final preprocessed T1 file.
        domain_sigma (float): Spatial sigma for the bilateral filter.
        range_sigma (float): Intensity sigma for the bilateral filter.
        normalize (bool): Whether to normalize intensities to the range [0, 255].
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] File not found: {file_path}")

    print("[INFO] Loading T1 MRI image...")
    image = sitk.ReadImage(file_path)

    # Step 1: Convert to sitkFloat32 if necessary
    if image.GetPixelID() != sitk.sitkFloat32:
        print("[INFO] Converting image to sitkFloat32 for processing...")
        image = sitk.Cast(image, sitk.sitkFloat32)

    # Step 2: Apply Bias Field Correction
    print("[INFO] Applying Bias Field Correction...")
    corrected_image = sitk.N4BiasFieldCorrection(image)


    # Step 4: Apply Bilateral Smoothing
    print(f"[INFO] Applying Bilateral Smoothing (domain_sigma={domain_sigma}, range_sigma={range_sigma})...")
    smoothed_image = sitk.Bilateral(corrected_image, domainSigma=domain_sigma, rangeSigma=range_sigma)

    # Step 5: Save the preprocessed image
    print(f"[INFO] Saving final preprocessed T1 MRI to {output_path}...")
    sitk.WriteImage(smoothed_image, output_path)
    print("[INFO] Preprocessing with validation completed successfully.")


if __name__ == "__main__":
    # Input and output directories
    input_dir = "/Users/jsnunki/Code/neuro-viz/datasets/Simultaneous_eeg-fmri/bids_mri_fmri/sub-001/ses-001/anat/"
    output_dir = "/Users/jsnunki/Code/neuro-viz/datasets/Simultaneous_eeg-fmri/bids_mri_fmri/sub-001/ses-001/anat/preprocessed_anat"
    os.makedirs(output_dir, exist_ok=True)

    # Example T1 MRI file
    t1_file = os.path.join(input_dir, "sub-001_ses-001_acq-highres_T1w.nii.gz")
    output_file = os.path.join(output_dir, "preprocessed_t1.nii.gz")

    try:
        preprocess_t1_with_validation(t1_file, output_file)
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")

