import vtk
from vtk.util.colors import light_grey
import pyvista as pv
import numpy as np


def load_nifti(file_path):
    """
    Load NIfTI data using VTK.

    Args:
        file_path (str): Path to the NIfTI file (.nii or .nii.gz).

    Returns:
        vtk.vtkNIFTIImageReader: The reader object with loaded data.
    """
    print("[INFO] Loading NIfTI data...")
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader


def segment_data(reader, lower_threshold=30, upper_threshold=60):
    """
    Segment the data using intensity thresholds.

    Args:
        reader (vtk.vtkNIFTIImageReader): The NIfTI reader object containing the data.
        lower_threshold (int): Minimum threshold for segmentation.
        upper_threshold (int): Maximum threshold for segmentation.

    Returns:
        vtk.vtkImageThreshold: Segment containing the binary mask of the data.
    """
    print(f"[INFO] Segmenting data with thresholds: {lower_threshold}-{upper_threshold}")
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputConnection(reader.GetOutputPort())
    threshold.ThresholdBetween(lower_threshold, upper_threshold)
    threshold.ReplaceInOn()
    threshold.SetInValue(1)  # Keep values within the threshold range
    threshold.ReplaceOutOn()
    threshold.SetOutValue(0)  # Set values outside the range to zero
    threshold.Update()
    return threshold


def calculate_dynamic_isovalue(threshold):
    """
    Calculate a dynamic isovalue based on the intensity range of segmented data.

    Args:
        threshold (vtk.vtkImageThreshold): Binary segmented data.

    Returns:
        float: The dynamic isovalue calculated as 60% of the intensity range.
    """
    print("[INFO] Calculating dynamic isovalue...")
    scalars = threshold.GetOutput().GetPointData().GetScalars()
    if not scalars:
        raise ValueError("[ERROR] Segmented data contains no scalars!")

    scalar_range = scalars.GetRange()
    print(f"[DEBUG] Scalar range for isosurface: {scalar_range}")
    isovalue = (scalar_range[0] + scalar_range[1]) * 0.6  # Using 60% of the range
    print(f"[DEBUG] Using dynamic isovalue: {isovalue}")
    return isovalue


def generate_surface(threshold, isovalue):
    """
    Generate a 3D surface using the Marching Cubes algorithm.

    Args:
        threshold (vtk.vtkImageThreshold): Binary segmented data.
        isovalue (float): Isovalue defining the surface boundaries.

    Returns:
        vtk.vtkPolyData: Polydata object representing the 3D surface.
    """
    print(f"[INFO] Generating surface with isovalue: {isovalue}")
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputConnection(threshold.GetOutputPort())
    marching_cubes.SetValue(0, isovalue)
    marching_cubes.Update()

    polydata = marching_cubes.GetOutput()
    if polydata.GetNumberOfPoints() == 0:
        raise ValueError("[ERROR] No surface was generated! Check thresholds or isovalue.")
    return polydata


def apply_laplacian_smoothing(polydata, iterations=100, relaxation_factor=0.1):
    """
    Apply Laplacian smoothing to reduce noise and improve surface quality.

    Args:
        polydata (vtk.vtkPolyData): Input polygonal mesh surface.
        iterations (int): Number of smoothing repetitions.
        relaxation_factor (float): Controls the strength of smoothing per iteration.

    Returns:
        vtk.vtkPolyData: Smoothed surface mesh.
    """
    print("[INFO] Applying Laplacian smoothing...")
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputData(polydata)
    smoother.SetNumberOfIterations(iterations)
    smoother.SetRelaxationFactor(relaxation_factor)
    smoother.Update()
    return smoother.GetOutput()


def fill_holes(polydata, hole_size=50.0):
    """
    Fill small holes in the surface mesh.

    Args:
        polydata (vtk.vtkPolyData): Input polygonal mesh surface.
        hole_size (float): The maximum area of the holes to be filled.

    Returns:
        vtk.vtkPolyData: Surface mesh with filled holes.
    """
    print("[INFO] Filling holes...")
    fill_holes_filter = vtk.vtkFillHolesFilter()
    fill_holes_filter.SetInputData(polydata)
    fill_holes_filter.SetHoleSize(hole_size)
    fill_holes_filter.Update()
    return fill_holes_filter.GetOutput()


def remove_small_components(polydata):
    """
    Remove small disconnected components from the surface mesh.

    Args:
        polydata (vtk.vtkPolyData): Input polygonal mesh surface.

    Returns:
        vtk.vtkPolyData: Mesh containing only the largest connected component.
    """
    print("[INFO] Removing small disconnected components...")
    connectivity_filter = vtk.vtkConnectivityFilter()
    connectivity_filter.SetInputData(polydata)
    connectivity_filter.SetExtractionModeToLargestRegion()
    connectivity_filter.Update()
    return connectivity_filter.GetOutput()


def smooth_surface(polydata, iterations=600, pass_band=0.2):
    """
    Apply physical smoothing to the surface using a Windowed Sinc filter.

    Args:
        polydata (vtk.vtkPolyData): Input polygonal mesh surface.
        iterations (int): Number of smoothing iterations.
        pass_band (float): Smoothing frequency threshold.

    Returns:
        vtk.vtkPolyData: Smoothed surface representation.
    """
    print("[INFO] Smoothing the surface mesh...")
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(polydata)
    smoother.SetNumberOfIterations(iterations)
    smoother.SetPassBand(pass_band)
    smoother.Update()
    return smoother.GetOutput()

def apply_color_mapping(polydata):
    """
    Apply a color mapping to the surface mesh using scalar values.

    Args:
        polydata (vtk.vtkPolyData): Input polygonal mesh with scalar data.

    Returns:
        vtk.vtkPolyDataMapper: The mapper object with applied colors.
    """
    print("[INFO] Applying color mapping...")
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    color_transfer_function = vtk.vtkColorTransferFunction()
    color_transfer_function.AddRGBPoint(30, 0.0, 0.0, 1.0)  # Blue for low values
    color_transfer_function.AddRGBPoint(70, 1.0, 0.0, 0.0)  # Red for high values
    mapper.SetLookupTable(color_transfer_function)
    return mapper


def setup_lighting(renderer):
    """
    Configure multiple light sources for better visualization.

    Args:
        renderer (vtk.vtkRenderer): The renderer responsible for the 3D scene.
    """
    print("[INFO] Setting up lighting...")
    light1 = vtk.vtkLight()
    light1.SetPosition(1, 1, 1)  # Place the first light source
    renderer.AddLight(light1)

    light2 = vtk.vtkLight()
    light2.SetPosition(-1, -1, 1)  # Place the second light source
    renderer.AddLight(light2)


def visualize_with_vtk(polydata):
    """
    Visualize the 3D surface using VTK with proper lighting and shading.

    Args:
        polydata (vtk.vtkPolyData): Input 3D surface data.
    """
    print("[INFO] Visualizing the surface using VTK...")
    mapper = apply_color_mapping(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetInterpolationToPhong()  # Use Phong shading for realism

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.3)  # Background color

    # Configure lighting
    setup_lighting(renderer)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    render_window.Render()  # Render the visualization
    interactor.Start()  # Initialize the interactive viewer

def export_surface(polydata, file_path, file_format="vtk"):
    """
    Export the reconstructed surface to a file.

    Args:
        polydata (vtk.vtkPolyData): Surface to be exported.
        file_path (str): Path to save the exported file.
        file_format (str): File format for the export ('vtk', 'stl', etc.).
    """
    print(f"[INFO] Exporting surface to {file_path} in {file_format.upper()} format...")

    if file_format.lower() == "vtk":
        writer = vtk.vtkPolyDataWriter()
    elif file_format.lower() == "stl":
        writer = vtk.vtkSTLWriter()
    else:
        raise ValueError("[ERROR] Unsupported file format. Use 'vtk' or 'stl'.")

    writer.SetInputData(polydata)
    writer.SetFileName(file_path)
    writer.Write()
    print("[INFO] Export successfully completed.")


if __name__ == "__main__":
    file_path = "/Users/jsnunki/Code/neuro-viz/datasets/Simultaneous_eeg-fmri/bids_mri_fmri/sub-001/ses-001/anat/preprocessed_anat/segment_t1_fsl/preprocessed_t1_brain.nii.gz"

    try:
        # Pipeline: Load, process, and visualize the surface
        reader = load_nifti(file_path)
        segmented = segment_data(reader)
        dynamic_isovalue = calculate_dynamic_isovalue(segmented)
        brain_surface = generate_surface(segmented, dynamic_isovalue)
        laplacian_surface = apply_laplacian_smoothing(brain_surface)
        filled_surface = fill_holes(laplacian_surface)
        cleaned_surface = remove_small_components(filled_surface)
        final_surface = smooth_surface(cleaned_surface)
        visualize_with_vtk(final_surface)


        # Export the surface to a .vtk file
        export_surface(final_surface, "/Users/jsnunki/Code/neuro-viz/results/reconstructed_brain_surface.vtk", file_format="vtk")

        # Optional: Visualize the surface
        visualize_with_vtk(final_surface)

    except Exception as e:
        print(f"[ERROR] {e}")
