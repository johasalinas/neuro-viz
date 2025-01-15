import vtk
import os


def load_fmri_image(file_path):
    """
    Load an fMRI image using vtkNIFTIImageReader.

    Parameters:
    file_path (str): Path to the fMRI image file.

    Returns:
    vtk.vtkNIFTIImageReader: Reader object containing the fMRI data.

    Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If the fMRI image does not contain scalar data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] fMRI file not found: {file_path}")

    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(file_path)
    reader.Update()

    if reader.GetOutput().GetPointData().GetScalars() is None:
        raise ValueError("[ERROR] fMRI image does not contain scalar data.")

    print("[INFO] fMRI Image loaded successfully.")
    return reader


def apply_smoothing(surface_polydata, iterations=50, pass_band=0.1):
    """
    Apply smoothing to a brain surface to remove artifacts.

    Parameters:
    surface_polydata (vtk.vtkPolyData): Input surface polydata.
    iterations (int): Number of smoothing iterations.
    pass_band (float): Passband value for the smoother.

    Returns:
    vtk.vtkPolyData: Smoothed surface polydata.
    """
    print("[INFO] Applying smoothing filter...")
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(surface_polydata)
    smoother.SetNumberOfIterations(iterations)
    smoother.SetPassBand(pass_band)
    smoother.Update()

    smoothed_polydata = smoother.GetOutput()
    print("[INFO] Surface smoothing completed.")
    return smoothed_polydata


def map_fmri_to_surface(fmri_reader, surface_polydata):
    """
    Map fMRI activations onto the reconstructed brain surface.

    Parameters:
    fmri_reader (vtk.vtkNIFTIImageReader): Reader object containing the fMRI data.
    surface_polydata (vtk.vtkPolyData): Polydata of the reconstructed brain surface.

    Returns:
    vtk.vtkPolyData: Surface polydata with mapped fMRI activations as scalars.
    """
    interpolator = vtk.vtkPointLocator()
    interpolator.SetDataSet(fmri_reader.GetOutput())
    interpolator.BuildLocator()

    scalars = vtk.vtkFloatArray()
    scalars.SetName("fMRI Activations")
    fmri_data = fmri_reader.GetOutput().GetPointData().GetScalars()

    # Map scalar data from fMRI to each point on the brain surface
    for i in range(surface_polydata.GetNumberOfPoints()):
        surface_point = surface_polydata.GetPoint(i)
        try:
            closest_point_idx = interpolator.FindClosestPoint(surface_point)
            interpolated_value = fmri_data.GetTuple1(closest_point_idx)
        except RuntimeError:
            interpolated_value = 0.0  # Default to 0 for unmapped points

        scalars.InsertNextValue(interpolated_value)

    surface_polydata.GetPointData().SetScalars(scalars)
    print("[INFO] fMRI data mapped onto the brain surface.")
    return surface_polydata


def enhanced_color_mapping(surface_polydata):
    """
    Apply enhanced color mapping for fMRI activations to improve visual perception.

    Parameters:
    surface_polydata (vtk.vtkPolyData): Surface with fMRI activation data.

    Returns:
    vtk.vtkPolyDataMapper: Pre-configured mapper with color mapping applied.
    """
    scalar_range = surface_polydata.GetPointData().GetScalars().GetRange()
    min_value, max_value = scalar_range
    adjusted_max = max_value * 0.9  # Adjust for better saturation

    # Create a custom color transfer function
    color_transfer_function = vtk.vtkColorTransferFunction()
    color_transfer_function.AddRGBPoint(min_value, 0.0, 0.0, 1.0)  # Blue
    color_transfer_function.AddRGBPoint((min_value + adjusted_max) * 0.5, 0.0, 1.0, 0.0)  # Green
    color_transfer_function.AddRGBPoint(adjusted_max, 1.0, 0.0, 0.0)  # Red

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(surface_polydata)
    mapper.SetLookupTable(color_transfer_function)
    mapper.SetScalarRange(scalar_range[0], adjusted_max)
    mapper.SetScalarModeToUsePointData()

    return mapper


def visualize_fmri_on_surface(surface_polydata, mapper=None):
    """
    Visualize the brain surface with fMRI activations.

    Parameters:
    surface_polydata (vtk.vtkPolyData): Surface polydata with fMRI activations.
    mapper (vtk.vtkPolyDataMapper): Pre-configured mapper for optimized rendering. If None, a default mapper is used.
    """
    if mapper is None:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(surface_polydata)
        scalars = surface_polydata.GetPointData().GetScalars()
        if scalars is not None:
            scalar_range = scalars.GetRange()
            mapper.SetScalarRange(scalar_range)
            print(f"[INFO] Scalar range for visualization: {scalar_range}")

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetDiffuse(1.0)
    actor.GetProperty().SetSpecular(0.5)
    actor.GetProperty().SetSpecularPower(50)

    # Setup rendering environment
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    print("[INFO] Visualizing fMRI data on the brain surface.")
    render_window.Render()
    interactor.Start()


def save_mapped_surface(surface_polydata, output_path):
    """
    Save the brain surface with fMRI activations to a file.

    Parameters:
    surface_polydata (vtk.vtkPolyData): Surface polydata with activations.
    output_path (str): File path to save the surface polydata.
    """
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(surface_polydata)
    writer.Write()
    print(f"[INFO] Mapped surface saved to {output_path}")


if __name__ == "__main__":
    # Input and output file paths
    fmri_path = (
        "/Users/jsnunki/Code/neuro-viz/datasets/Simultaneous_eeg-fmri/bids_mri_fmri/sub-001/ses-001/func/preprocessed_fmri/preprocess_eoec_fmri.feat/filtered_eoec_func_data.nii.gz"
    )
    surface_path = "/Users/jsnunki/Code/neuro-viz/results/reconstructed_brain_surface.vtk"
    output_path = "/Users/jsnunki/Code/neuro-viz/results/mapped_fmri_to_surface.vtk"

    try:
        # Load fMRI data
        fmri_reader = load_fmri_image(fmri_path)

        # Load brain surface
        surface_reader = vtk.vtkPolyDataReader()
        surface_reader.SetFileName(surface_path)
        surface_reader.Update()
        surface_polydata = surface_reader.GetOutput()

        # Map fMRI data to the surface
        mapped_surface = map_fmri_to_surface(fmri_reader, surface_polydata)

        # Apply smoothing filter
        smoothed_surface = apply_smoothing(mapped_surface, iterations=100, pass_band=0.08)

        # Apply enhanced color mapping
        optimized_color_mapper = enhanced_color_mapping(smoothed_surface)

        # Visualize the surface with fMRI activations
        visualize_fmri_on_surface(smoothed_surface, optimized_color_mapper)

        # Save the mapped surface to a file
        save_mapped_surface(smoothed_surface, output_path)
        print(f"[INFO] Successfully exported the file to {output_path}")

    except Exception as e:
        print(f"[ERROR] {e}")