import vtk
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


class BrainVisualizerApp(QtWidgets.QMainWindow):
    """
    Main application for brain visualization using PyQt5 and VTK.
    This application loads and visualizes surface data (*.vtk).
    """

    def __init__(self, path):
        """
        Initializes the BrainVisualizerApp with the path to the surface file.

        Args:
            path (str): Path to the surface (VTK) file.
        """
        super().__init__()
        self.path = path
        self.init_ui()
        self.setup_vtk()

    def init_ui(self):
        """
        Sets up the PyQt5 user interface, including controls and layout.
        """
        self.setWindowTitle("Brain Surface Visualization")
        self.setGeometry(100, 100, 1200, 800)

        # Main Widget
        self.frame = QtWidgets.QFrame()
        self.vtk_layout = QtWidgets.QVBoxLayout()
        self.frame.setLayout(self.vtk_layout)

        # Side Panel for Controls
        self.side_panel = QtWidgets.QVBoxLayout()
        self.opacity_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setToolTip("Adjust Surface Opacity")
        self.opacity_slider.valueChanged.connect(self.update_opacity)

        # Adding widgets to the side panel layout
        self.side_panel.addWidget(QtWidgets.QLabel("Surface Opacity"))
        self.side_panel.addWidget(self.opacity_slider)

        # Main Layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.frame, stretch=5)  # VTK Widget
        layout.addLayout(self.side_panel, stretch=1)  # Side panel for controls

        # Set Central Widget
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def setup_vtk(self):
        """
        Sets up the VTK components, including the renderer, render window, and interactor.
        Initializes the surface data for visualization.
        """
        # Renderer and RenderWindow
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.1)  # Dark background

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        # Embed VTK widget in the PyQt layout
        self.vtk_widget = QtWidgets.QWidget(self.frame)
        self.vtk_layout.addWidget(self.vtk_widget)

        # Load Surface Data
        self.setup_surface()

        # Adjust Camera for initial view
        self.renderer.ResetCamera()

    def setup_surface(self):
        """
        Loads and sets up the surface data using the given VTK file path.
        """
        # Reader for VTK surface data
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.path)
        reader.Update()

        # Surface Mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(reader.GetOutput())
        scalar_range = reader.GetOutput().GetPointData().GetScalars().GetRange()
        mapper.SetScalarRange(scalar_range)

        # Surface Actor
        self.surface_actor = vtk.vtkActor()
        self.surface_actor.SetMapper(mapper)
        self.surface_actor.GetProperty().SetOpacity(0.5)  # Default opacity
        self.renderer.AddActor(self.surface_actor)

    def update_opacity(self, value):
        """
        Updates the opacity of the surface based on slider input.

        Args:
            value (int): Slider value (0-100), representing opacity percentage.
        """
        opacity = value / 100.0  # Convert percentage to fraction
        self.surface_actor.GetProperty().SetOpacity(opacity)
        self.render_window.Render()

    def start(self):
        """
        Starts the VTK interactor for rendering the visualization.
        """
        self.render_window.Render()
        self.interactor.Initialize()
        self.interactor.Start()


if __name__ == "__main__":
    import sys

    # Path to surface data (*.vtk)
    mapped_fmri_to_surface = "/Users/jsnunki/Code/neuro-viz/results/mapped_fmri_to_surface.vtk"

    # Create and start the BrainVisualizerApp
    app = QtWidgets.QApplication(sys.argv)
    visualizer = BrainVisualizerApp(mapped_fmri_to_surface)
    visualizer.show()
    sys.exit(app.exec_())