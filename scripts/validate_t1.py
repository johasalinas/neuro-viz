import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def check_and_normalize_intensity(image, lower_bound=0, upper_bound=255):
    """
    Verifies and normalizes the intensity of the image to the given range if needed.

    Parameters:
        image (sitk.Image): Image loaded with SimpleITK.
        lower_bound (int): Lower bound of the range (default: 0).
        upper_bound (int): Upper bound of the range (default: 255).

    Returns:
        sitk.Image: Normalized image if required.
    """
    img_array = sitk.GetArrayFromImage(image)
    min_int, max_int = img_array.min(), img_array.max()
    print(f"[INFO] Current intensity range: [{min_int}, {max_int}]")

    # Normalize the image if its intensity values are outside the desired range
    if not (lower_bound <= min_int and max_int <= upper_bound):
        print(f"[INFO] Normalizing intensities to the range [{lower_bound}, {upper_bound}]...")
        normalized_array = (img_array - min_int) / (max_int - min_int) * (upper_bound - lower_bound) + lower_bound
        return sitk.GetImageFromArray(normalized_array)

    print("[INFO] Normalization is not required.")
    return image


def check_orientation(image, view="axial"):
    """
    Checks whether the orientation of the image is correct according to the specified view.

    Parameters:
        image (sitk.Image): Image loaded with SimpleITK.
        view (str): Expected view: "axial", "sagittal", or "coronal".

    Returns:
        bool: True if correctly oriented, False otherwise.
    """
    direction = image.GetDirection()
    print(f"[INFO] Current direction matrix: {direction}")

    # Check orientation based on the specified view
    if view == "axial":
        # Z-axis should point upwards
        is_correct = direction[8] > 0  # Last element of the direction matrix
    elif view == "sagittal":
        # X-axis should be aligned correctly
        is_correct = direction[0] > 0
    elif view == "coronal":
        # Y-axis should be aligned correctly
        is_correct = direction[4] > 0
    else:
        raise ValueError("Unsupported view. Use: 'axial', 'sagittal', or 'coronal'")

    print(f"[INFO] Orientation is {'correct' if is_correct else 'incorrect'} for the {view} view.")
    return is_correct


def check_edge_clarity(image, threshold=0.1):
    """
    Checks the clarity of the edges based on intensity gradients.

    Parameters:
        image (sitk.Image): Image loaded with SimpleITK.
        threshold (float): Minimum threshold to distinguish clear edges (default: 0.1).

    Returns:
        bool: True if edges are clear, False otherwise.
    """
    img_array = sitk.GetArrayFromImage(image)

    # Select a middle slice of the volume
    if img_array.ndim == 3:
        middle_slice = img_array[img_array.shape[0] // 2, :, :]  # Axial slice by default
    else:
        middle_slice = img_array  # For 2D images

    # Compute gradient in both axes
    gradient_y, gradient_x = np.gradient(middle_slice)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)  # Gradient magnitude

    # Normalize the gradient
    gradient_normalized = gradient_magnitude / gradient_magnitude.max()

    # Visualization (optional, disable if unnecessary)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Middle Slice of the Image")
    plt.imshow(middle_slice, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Gradient Magnitude (Edges)")
    plt.imshow(gradient_normalized, cmap="hot")
    plt.axis("off")
    plt.show()

    # Check edge clarity based on the threshold
    edge_clarity = np.mean(gradient_normalized > threshold)
    print(f"[INFO] Percentage of pixels with clear edges: {edge_clarity:.2%}")
    return edge_clarity > 0.05  # Use 5% criterion for clear edges


if __name__ == "__main__":
    # Path to the T1 MRI file
    t1_file_path = "/Users/jsnunki/Code/neuro-viz/datasets/Simultaneous_eeg-fmri/bids_mri_fmri/sub-001/ses-001/anat/preprocessed_anat/preprocessed_t1.nii.gz"

    # Load the image using SimpleITK
    print("[INFO] Loading T1 MRI image...")
    image = sitk.ReadImage(t1_file_path)

    # 1. Check and normalize intensities
    print("[INFO] Checking intensity range...")
    normalized_image = check_and_normalize_intensity(image)

    # 2. Check orientation (axial view)
    print("[INFO] Checking orientation...")
    is_oriented_correctly = check_orientation(normalized_image, view="axial")
    if not is_oriented_correctly:
        print("[WARN] Image orientation is not correct. It should be reoriented.")

    # 3. Check edge clarity
    print("[INFO] Checking edge clarity...")
    are_edges_clear = check_edge_clarity(normalized_image)
    if not are_edges_clear:
        print("[WARN] Edge clarity is not sufficient. Consider reviewing the processing or quality of the file.")

    # Final verification
    print("[INFO] Verification complete.")
    if is_oriented_correctly and are_edges_clear:
        print("[SUCCESS] The image meets the expected criteria.")
    else:
        print("[FAILURE] The image does not meet some of the criteria.")
