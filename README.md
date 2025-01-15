# 🌟 Brain Visualization Repository

Welcome to the **Brain Visualization for Depression Diagnosis** project repository! This repository demonstrates the integration of anatomical T1-weighted MRI data and functional MRI (fMRI) data into an interactive 3D visualization tool. 🚀

---

## ✨ Project Highlights

- **🧠 Multimodal Data Integration**: Combine T1 anatomical data and fMRI functional data seamlessly.
- **🎛️ Intuitive GUI**: Explore brain data interactively with real-time controls.
- **📊 High-Quality Volume Rendering**: Smooth rendering of volumetric data and functional overlays.
- **🎨 Surface Reconstruction**: Generate stunning 3D cortical surfaces with mapped activity.

For this demo, we used the **Simultaneous EEG-fMRI dataset (DOI: [10.17632/crhybxpdy6.2](https://doi.org/10.17632/crhybxpdy6.2))**, focusing on **one subject's eyes-open condition**.

---

## 🔧 Features

- **🖱️ User-Friendly Controls**: Built with PyQt5, allowing you to adjust opacity, toggle modalities, and interact with brain data.
- **📡 Functional Mapping**: Overlay fMRI data on anatomical surfaces for precise visualization.
- **🔍 Modular Scripts**: Easy-to-follow pipeline for data preprocessing, reconstruction, and visualization.

---

## 🛠️ Technologies and Libraries Used

### Core Libraries:
1. **VTK (Visualization Toolkit)** 🖥️
   - Volume and surface rendering.
   - Key Functions: `vtkNIFTIImageReader`, `vtkMarchingCubes`, `vtkGPUVolumeRayCastMapper`.
2. **PyQt5** 🖼️
   - Interactive GUI for real-time visualization and controls.
3. **NiBabel** 📦
   - Handling NIfTI neuroimaging formats.


### Prerequisites
- Python 3.8 or above 🐍
- Libraries:
  - Matplotlib
  - NumPy
  - PyVista
---

## 📂 Pipeline and Scripts

### 1️⃣ Load Data
- **Script**: `load_data.py`
- **📜 Description**: Loads raw T1 and fMRI data into memory.
- **Outcome**: Data is validated and ready for preprocessing.

### 2️⃣ Validate and Visualize
- **Script**: `visualize_validate.py`
- **📜 Description**: Ensures raw data integrity and correctness through visualization.
- **Outcome**: Confirms data readiness for further processing.

### 3️⃣ Preprocess T1
- **Script**: `preprocess_t1.py`
- **📜 Description**: Cleans and segments T1 data using FSL and VTK.
- **Outcome**: Segmented T1 image ready for functional overlay.

### 4️⃣ Align T1 to Surface
- **Script**: `align_brain_t1_surface.py`
- **📜 Description**: Aligns T1 anatomical data with the surface for accurate integration.
- **Outcome**: Properly aligned T1 surface.

### 5️⃣ Surface Reconstruction
- **Script**: `surface_reconstruction.py`
- **📜 Description**: Reconstructs a smooth cortical surface from T1 data.
- **Outcome**: High-quality 3D surface for functional mapping.

### 6️⃣ Map fMRI to Surface
- **Script**: `map_fmri_and_brain.py`
- **📜 Description**: Maps fMRI activations onto the reconstructed cortical surface.
- **Outcome**: Functional data visualized on the anatomical surface.

### 7️⃣ GUI and Volume Rendering
- **Script**: `gui_visualization.py`
- **📜 Description**: Provides interactive tools for exploring the dataset.
- **Outcome**: A user-friendly 3D visualization interface.

---

## 🚀 Getting Started

### Prerequisites:
- Python 3.8+
- Required Libraries:
  - VTK
  - PyQt5
  - FSL
  - NiBabel
  - NumPy
  - Matplotlib

### Installation:
1. Clone this repository:
   ```bash
   git clone https://github.com/johasalinas/neuro-viz.git
   cd neuro-viz

---
## 🛠️ Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Running the Demo

1. Preprocess the data using provided scripts.
2. Launch the interactive GUI:
   ```bash
   python gui_visualization.py
---

## 📚 References
* Gallego-Rudolf, J., et al. (2023). Simultaneous EEG-fMRI dataset. Mendeley Data. DOI: 10.17632/crhybxpdy6.2
* Schroeder, W., et al. (2006). The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics. Kitware Inc.
* Smith, S. M., et al. (2004). Advances in functional and structural MR image analysis and implementation as FSL. NeuroImage.
* Engel, K., et al. (2014). Real-Time Volume Graphics. CRC Press.
* PyQt5 and VTK Documentation
