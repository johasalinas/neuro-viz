import vtk
import os


def load_t1_image(file_path):
    """
    Load a T1 image using vtkNIFTIImageReader.

    Parameters:
    file_path (str): The path to the T1 image file.

    Returns:
    vtk.vtkNIFTIImageReader: Reader object containing the loaded T1 image.
    """
    # Check if the specified file path exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] File not found: {file_path}")

    # Initialize the VTK reader for NIFTI format
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(file_path)  # Set the file path to be loaded by the reader
    reader.Update()  # Load the T1 image
    print("[INFO] T1 Image loaded successfully.")
    return reader


def segment_t1_image(t1_reader, lower_threshold, upper_threshold):
    """
    Segment the T1 image using a thresholding method.

    Parameters:
    t1_reader (vtk.vtkNIFTIImageReader): Reader object containing the loaded T1 image.
    lower_threshold (float): Lower threshold value for segmentation.
    upper_threshold (float): Upper threshold value for segmentation.

    Returns:
    vtk.vtkImageThreshold: Threshold object with the segmented image.
    """
    # Initialize the vtkImageThreshold object for thresholding
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputConnection(t1_reader.GetOutputPort())  # Connect the T1 image data
    threshold.ThresholdBetween(lower_threshold, upper_threshold)  # Set threshold range
    threshold.SetInValue(1)  # Pixel values inside the range are set to 1
    threshold.SetOutValue(0)  # Pixel values outside the range are set to 0
    threshold.Update()  # Apply the thresholding
    print(f"[INFO] Image segmented with thresholds: {lower_threshold}-{upper_threshold}")
    return threshold


def generate_surface(threshold, smooth_iterations=100):
    """
    Generate a 3D surface mesh from the segmented image using the Marching Cubes algorithm.

    Parameters:
    threshold (vtk.vtkImageThreshold): Threshold object containing the segmented image.
    smooth_iterations (int): Number of iterations to smooth the generated surface.

    Returns:
    vtk.vtkPolyData: Smoothed polygonal surface data.
    """
    # Initialize Marching Cubes for surface extraction
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputConnection(threshold.GetOutputPort())  # Connect the segmented image as input
    marching_cubes.SetValue(0, 0.5)  # Set the isovalue for surface extraction
    marching_cubes.Update()  # Generate the initial polygonal surface

    print("[INFO] Surface generated using Marching Cubes.")

    # Apply a smoothing filter for surface refinement
    smooth_filter = vtk.vtkSmoothPolyDataFilter()
    smooth_filter.SetInputConnection(marching_cubes.GetOutputPort())  # Use the surface as input
    smooth_filter.SetNumberOfIterations(smooth_iterations)  # Set the number of smoothing iterations
    smooth_filter.Update()  # Smooth the surface
    print(f"[INFO] Surface smoothed with {smooth_iterations} iterations.")
    return smooth_filter.GetOutput()


def overlay_t1_and_surface(t1_reader, surface_polydata):
    """
    Overlay the T1 image with the reconstructed surface for visualization.

    Parameters:
    t1_reader (vtk.vtkNIFTIImageReader): Reader object containing the loaded T1 image.
    surface_polydata (vtk.vtkPolyData): Reconstructed 3D surface mesh data.
    """
    # Configure the mapper and actor for the T1 image
    t1_mapper = vtk.vtkImageSliceMapper()
    t1_mapper.SetInputConnection(t1_reader.GetOutputPort())  # Connect T1 image as input
    t1_actor = vtk.vtkImageSlice()
    t1_actor.SetMapper(t1_mapper)

    # Configure the mapper and actor for the reconstructed surface
    surface_mapper = vtk.vtkPolyDataMapper()
    surface_mapper.SetInputData(surface_polydata)  # Use the surface data as input
    surface_actor = vtk.vtkActor()
    surface_actor.SetMapper(surface_mapper)

    # Adjust the surface actor properties to enhance visualization
    surface_actor.GetProperty().SetOpacity(0.1)  # Set surface transparency
    surface_actor.GetProperty().SetDiffuse(0.8)  # Enable diffuse lighting
    surface_actor.GetProperty().SetSpecular(0.3)  # Adjust specular highlights
    surface_actor.GetProperty().SetSpecularPower(30)  # Adjust intensity of specular highlights

    # Create the renderer and window for visualization
    renderer = vtk.vtkRenderer()
    renderer.AddViewProp(t1_actor)  # Add the T1 image actor
    renderer.AddActor(surface_actor)  # Add the reconstructed surface actor
    renderer.SetBackground(0.1, 0.1, 0.1)  # Set the renderer background color (dark gray)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)  # Attach the renderer
    render_window.SetSize(800, 800)  # Set the window size

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)  # Attach the render window to the interactor

    print("[INFO] Rendering T1 and surface overlay. Close the window to proceed.")
    render_window.Render()  # Render the visualization
    interactor.Start()  # Start interactive visualization


def save_surface_to_file(surface_polydata, output_file):
    """
    Save the reconstructed surface mesh to a .vtk file.

    Parameters:
    surface_polydata (vtk.vtkPolyData): Reconstructed surface data to be saved.
    output_file (str): File path for saving the surface.
    """
    # Initialize the VTK writer for polydata
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_file)  # Set the file path
    writer.SetInputData(surface_polydata)  # Set the surface data as input
    writer.Write()  # Write the file
    print(f"[INFO] Surface saved to {output_file}")


if __name__ == "__main__":
    # Define input and output file paths
    t1_file = "/Users/jsnunki/Code/neuro-viz/datasets/Simultaneous_eeg-fmri/bids_mri_fmri/sub-001/ses-001/anat/preprocessed_anat/segment_t1_fsl/preprocessed_t1_brain.nii.gz"
    surface_path = "/Users/jsnunki/Code/neuro-viz/results/reconstructed_brain_surface.vtk"

    try:
        # Step 1: Load the T1 image
        t1_reader = load_t1_image(t1_file)

        # Step 2: Segment the T1 image using thresholding
        threshold = segment_t1_image(t1_reader, lower_threshold=20, upper_threshold=80)

        # Step 3: Generate a 3D surface using the Marching Cubes algorithm
        surface_polydata = generate_surface(threshold, smooth_iterations=300)
        print(
            f"[INFO] Surface generated with {surface_polydata.GetNumberOfPoints()} points and {surface_polydata.GetNumberOfCells()} cells."
        )

        # Step 4: Overlay the T1 image with the reconstructed surface for validation/visualization
        overlay_t1_and_surface(t1_reader, surface_polydata)

        # Step 5: Save the reconstructed surface to a file
        save_surface_to_file(surface_polydata, surface_path)
    except Exception as e:
        print(f"[ERROR] {e}")
