"""
PyQt5 version of the RF Coil Random Search Optimizer with Field Visualization
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
import random
import logging
import signal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QMessageBox, QCheckBox, QComboBox, QScrollArea,
                             QGroupBox, QRadioButton, QFileDialog, QGridLayout)
from PyQt5.QtCore import Qt
from pyvistaqt import QtInteractor

# ---------------------------
# Helper Functions
# ---------------------------

def generate_base_coil(params, num_points=200):
    """
    Generate the base coil centerline using the given parameters.
    The coil centerline is defined by a modified sigmoid and trigonometric functions.
    The r₀ value in params is used to adjust the radius in the y and z equations.
    """
    t = np.linspace(0, 1, num_points)
    f = (1/(1+np.exp(-params['alpha']*(t-0.5))) - 1/(1+np.exp(params['alpha']/2))) / \
        (1/(1+np.exp(-params['alpha']/2)) - 1/(1+np.exp(params['alpha']/2)))
    
    x_local = params['length'] * f
    y_local = (params['radius_y'] + params['r0']) * np.cos(2*np.pi*params['turns']*t)
    z_local = (params['radius_y'] + params['r0']) * np.sin(2*np.pi*params['turns']*t)
    
    coil_points_local = np.vstack((x_local, y_local, z_local)).T
    
    theta = params.get('theta', 54.7356 * np.pi/180)
    psi = np.deg2rad(90) - theta
    R_y = np.array([[np.cos(psi), 0, -np.sin(psi)],
                    [0,           1,  0],
                    [np.sin(psi), 0,  np.cos(psi)]])

    coil_points = coil_points_local.dot(R_y.T)
    return coil_points

def center_coil(coil_points):
    """
    Center the coil by subtracting its geometric centroid.
    Returns the shifted coil points and the computed centroid.
    """
    centroid = np.mean(coil_points, axis=0)
    offset = -centroid
    return coil_points + offset, centroid

def compute_frames(curve):
    """
    Compute a smooth, continuously transported frame (tangent, normal, binormal)
    along a curve using a simple parallel transport algorithm.
    """
    n_pts = curve.shape[0]
    tangents = np.zeros((n_pts, 3))
    normals = np.zeros((n_pts, 3))
    binormals = np.zeros((n_pts, 3))
    
    tangent0 = curve[1] - curve[0]
    tangent0 = tangent0 / np.linalg.norm(tangent0)
    tangents[0] = tangent0
    ref = np.array([0, 0, 1])
    if abs(np.dot(tangent0, ref)) > 0.9:
        ref = np.array([1, 0, 0])
    normal0 = np.cross(tangent0, ref)
    normal0 = normal0 / np.linalg.norm(normal0)
    binormal0 = np.cross(tangent0, normal0)
    binormal0 = binormal0 / np.linalg.norm(binormal0)
    normals[0] = normal0
    binormals[0] = binormal0
    
    for i in range(1, n_pts):
        t_new = curve[i] - curve[i-1]
        t_new = t_new / np.linalg.norm(t_new)
        tangents[i] = t_new
        v = normals[i-1] - np.dot(normals[i-1], t_new) * t_new
        if np.linalg.norm(v) < 1e-6:
            v = normals[i-1]
        n_new = v / np.linalg.norm(v)
        normals[i] = n_new
        b_new = np.cross(t_new, n_new)
        b_new = b_new / np.linalg.norm(b_new)
        binormals[i] = b_new
        
    return tangents, normals, binormals

def generate_surface_curves(coil_points, cross_params, k=10):
    """
    Generate k surface curves around the coil centerline.
    """
    num_points = coil_points.shape[0]
    tangents, normals, binormals = compute_frames(coil_points)
    curves = [np.zeros((num_points, 3)) for _ in range(k)]
    
    for i in range(num_points):
        n_vec = normals[i]
        b_vec = binormals[i]
        for j in range(k):
            theta = 2 * np.pi * j / k
            r_val = cross_params['r0']
            offset = r_val * (np.cos(theta) * n_vec + np.sin(theta) * b_vec)
            curves[j][i, :] = coil_points[i] + offset
    return curves

def randomize_params(params, bounds):
    """
    Randomly generate a new set of parameters by sampling uniformly within the given bounds.
    """
    new_params = {}
    for key in params.keys():
        lower = bounds[key]['min']
        upper = bounds[key]['max']
        new_params[key] = random.uniform(lower, upper)
    return new_params

def biot_savart_Bx(coil_points, eval_point):
    """
    Compute a simplified Bx at eval_point from a coil defined by coil_points.
    """
    mu0_4pi = 1e-7
    B = np.array([0.0, 0.0, 0.0])
    for i in range(len(coil_points) - 1):
        p1 = coil_points[i]
        p2 = coil_points[i + 1]
        dl = p2 - p1
        r_vec = eval_point - (p1 + p2) / 2.0
        r_norm = np.linalg.norm(r_vec)
        if r_norm < 1e-6:
            continue
        dB = mu0_4pi * np.cross(dl, r_vec) / (r_norm**3)
        B += dB
    return B[0]

def evaluate_coil(coil_points, sample_points):
    """
    Evaluate a coil by computing Bx at each sample point.
    """
    Bx_vals = [biot_savart_Bx(coil_points, pt) for pt in sample_points]
    Bx_vals = np.array(Bx_vals)
    avg_Bx = np.mean(np.abs(Bx_vals))
    var_Bx = np.var(Bx_vals)
    return avg_Bx, var_Bx

def get_volume_sample_points(volume):
    """
    Generate sample points within a cylindrical volume.
    """
    points = []
    origin = np.array([0, 0, 0])
    radius = volume['radius']
    length = volume['length']
    
    axis = np.array([volume['axis_x'], volume['axis_y'], volume['axis_z']])
    axis = axis / np.linalg.norm(axis)
    if abs(axis[0]) < 0.9:
        ref = np.array([1, 0, 0])
    else:
        ref = np.array([0, 1, 0])
    n_vec = np.cross(axis, ref)
    n_vec = n_vec / np.linalg.norm(n_vec)
    b_vec = np.cross(axis, n_vec)
    b_vec = b_vec / np.linalg.norm(b_vec)
    
    n_length = 3
    n_radial = 3
    n_angular = 4
    
    for i in range(n_length):
        s_offset = -length / 2 + i * (length / (n_length - 1))
        for j in range(n_radial):
            r = radius * j / (n_radial - 1)
            if r == 0:
                points.append(origin + s_offset * axis)
            else:
                for theta in np.linspace(0, 2 * np.pi, n_angular, endpoint=False):
                    radial_offset = r * (np.cos(theta) * n_vec + np.sin(theta) * b_vec)
                    points.append(origin + s_offset * axis + radial_offset)
    return points

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Returns the rotation matrix that aligns vec1 to vec2.
    """
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-6:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return R

# ---------------------------
# Default Parameters and Bounds
# ---------------------------
base_coil_params = {
    'radius_y': 2.6,
    'turns': 7,
    'length': 15.0,
    'alpha': 2.0,
    'r0': 0.5
}

default_coil_bounds = {
    'radius_y': {'min': 2.6, 'max': 2.6},
    'turns':    {'min': 7,   'max': 7},
    'length':   {'min': 14.5, 'max': 17.5},
    'alpha':    {'min': 1.0, 'max': 5.0},
    'r0':       {'min': 0.5, 'max': 0.6}
}

base_cross_params = {'r0': 0.5}
default_cross_bounds = {'r0': {'min': 0.5, 'max': 0.6}}

default_volume = {
    'radius': 1.7,
    'length': 10.0,
    'spacing': 0.1,
    'axis_x': 1.41421356,
    'axis_y': 0.0,
    'axis_z': 1.0,
}

# Constant for near-zero threshold
EPSILON = 1e-20

# Global unit conversion
MM_TO_M  = 1e-3
MU0_SI   = 4 * np.pi * 1e-7
MU0_MM   = MU0_SI / MM_TO_M

# Setup logging
logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def compute_biot_savart_field(observation_points: np.ndarray,
                              coil_points: np.ndarray,
                              current: float = 1.0,
                              mu0: float = MU0_MM) -> np.ndarray:        
    """
    Compute the magnetic field B (in Tesla) at the given observation_points
    due to a piecewise-linear coil defined by coil_points.
    """
    n_seg = coil_points.shape[0] - 1
    if n_seg < 1:
        return np.zeros((observation_points.shape[0], 3), dtype=np.float64)

    B = np.zeros((observation_points.shape[0], 3), dtype=np.float64)
    for i in range(n_seg):
        r1 = coil_points[i]
        r2 = coil_points[i+1]
        dl = r2 - r1
        mid = 0.5 * (r1 + r2)
        r_vec = observation_points - mid
        cross_vals = np.cross(dl, r_vec)
        r_norm = np.linalg.norm(r_vec, axis=1)**3
        valid = r_norm > EPSILON
        B[valid] += cross_vals[valid] / r_norm[valid, np.newaxis]

    B *= (mu0 * current) / (4.0 * np.pi)
    return B

def compute_vector_potential(obs_pts: np.ndarray,
                            coil_pts: np.ndarray,
                            current: float,
                            mu0: float = MU0_MM) -> np.ndarray:
    """
    Vector potential A(r) for a poly-line coil (all in mm).
    Returns A in Tesla·mm.
    """
    n_seg = coil_pts.shape[0] - 1
    A = np.zeros((len(obs_pts), 3), dtype=np.float64)
    for i in range(n_seg):
        r1, r2 = coil_pts[i], coil_pts[i+1]
        dl = r2 - r1
        mid = 0.5 * (r1 + r2)
        R = obs_pts - mid
        Rn = np.linalg.norm(R, axis=1)
        mask = Rn > 1e-12
        A[mask] += dl / Rn[mask, None]
    A *= mu0 * current / (4.0 * np.pi)
    return A

# ---------------------------
# Magnetic Field Visualizer Class
# ---------------------------
class MagneticFieldVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Magnetic Field Visualizer")
        self.setGeometry(100, 100, 1200, 700)
        
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        
        self.surf_poly = None
        self.centerline = None
        self.surface_curves = None
        self.last_region_points = None
        self.last_B = None
        self.last_E = None
        self.last_computed_region_type = None
        self.region_counts = {"Volume": 0, "Plane": 0, "Line": 0}
        self.last_region_points_by_type = {"Volume": None, "Plane": None, "Line": None}
        
        self.main_layout = QHBoxLayout(self)
        self.setLayout(self.main_layout)
        
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.controls_widget)
        self.main_layout.addWidget(scroll_area, stretch=0)
        
        self.vtk_widget = QtInteractor(self)
        self.vtk_widget.setMinimumWidth(600)
        self.main_layout.addWidget(self.vtk_widget, stretch=1)
        self.plotter = self.vtk_widget
        
        self.create_region_type_selector()
        self.create_region_inputs()
        self.create_exclusion_controls()
        self.create_visualization_options()
        self.create_field_visualization_options()
        self.create_source_controls()
        self.create_compute_export_controls()
        
        self.field_type_dropdown.currentIndexChanged.connect(self.plot_magnetic_field)
        self.vector_scale_input.editingFinished.connect(self.plot_magnetic_field)
        
        # Connect all visibility checkboxes to refresh the current view
        self.hide_centerline_checkbox.toggled.connect(self.refresh_current_view)
        self.hide_surface_curves_checkbox.toggled.connect(self.refresh_current_view)
        self.hide_coil_geometry_checkbox.toggled.connect(self.refresh_current_view)
        self.hide_region_checkbox.toggled.connect(self.refresh_current_view)
        self.hide_grid_checkbox.toggled.connect(self.refresh_current_view)
        self.hide_axes_checkbox.toggled.connect(self.refresh_current_view)
        self.exclude_interior_checkbox.toggled.connect(self.refresh_current_view)
        self.exclusion_distance_input.editingFinished.connect(self.refresh_current_view)
        
        # Connect grid settings to refresh view
        self.grid_spacing_input.editingFinished.connect(self.refresh_current_view)
        self.grid_side_length_input.editingFinished.connect(self.refresh_current_view)
        
        # Connect region dropdown to refresh when changed (but only if there's something to refresh)
        self.region_dropdown.currentIndexChanged.connect(self.on_region_type_changed)
        
        self.update_inputs()

    def set_coil_data(self, centerline: np.ndarray, surface_curves: list, surface_mesh=None):
        """Set the coil data for visualization and computation"""
        if centerline is not None and centerline.shape[0] >= 2:
            self.centerline = pv.PolyData(centerline.astype(np.float32))
            self.centerline.lines = np.hstack([[centerline.shape[0]], np.arange(centerline.shape[0])])
        else:
            self.centerline = None
            
        self.surface_curves = surface_curves
        self.surf_poly = surface_mesh
        
        # Automatically visualize region after data is set
        if centerline is not None and surface_curves is not None:
            self.visualize_region()

    def ensure_plotter(self) -> None:
        if getattr(self.plotter, "_closed", False):
            logger.debug("Plotter was closed; reinitializing.")
            self.vtk_widget = QtInteractor(self)
            self.main_layout.addWidget(self.vtk_widget, stretch=1)
            self.plotter = self.vtk_widget

    def create_region_type_selector(self) -> None:
        self.controls_layout.addWidget(QLabel("Select Computation Region Type:"))
        self.region_dropdown = QComboBox()
        self.region_dropdown.addItems(["Volume", "Plane", "Line"])
        self.controls_layout.addWidget(self.region_dropdown)

    def create_region_inputs(self) -> None:
        self.volume_group = QGroupBox("Volume Region Limits")
        vol_layout = QVBoxLayout()
        self.xmin_input = self.create_labeled_input("Xmin:", "-5", vol_layout)
        self.xmax_input = self.create_labeled_input("Xmax:", "17", vol_layout)
        self.ymin_input = self.create_labeled_input("Ymin:", "0", vol_layout)
        self.ymax_input = self.create_labeled_input("Ymax:", "0", vol_layout)
        self.zmin_input = self.create_labeled_input("Zmin:", "-5", vol_layout)
        self.zmax_input = self.create_labeled_input("Zmax:", "17", vol_layout)
        self.points_spacing_input = self.create_labeled_input("Spacing:", "0.5", vol_layout)
        self.volume_group.setLayout(vol_layout)
        self.controls_layout.addWidget(self.volume_group)

        self.plane_group = QGroupBox("Plane Region Limits")
        plane_layout = QVBoxLayout()
        self.plane_origin_input = self.create_labeled_input("Origin (x,y,z):", "0,0,0", plane_layout)
        self.plane_normal_input = self.create_labeled_input("Normal (x,y,z):", "0,0,1", plane_layout)
        self.plane_length_input = self.create_labeled_input("Length:", "10", plane_layout)
        self.plane_width_input = self.create_labeled_input("Width:", "10", plane_layout)
        self.plane_points_spacing_input = self.create_labeled_input("Spacing:", "0.5", plane_layout)
        self.plane_group.setLayout(plane_layout)
        self.controls_layout.addWidget(self.plane_group)

        self.line_group = QGroupBox("Line Region Limits")
        line_layout = QVBoxLayout()
        self.line_start_input = self.create_labeled_input("Start (x,y,z):", "0,0,0", line_layout)
        self.line_end_input = self.create_labeled_input("End (x,y,z):", "1,1,1", line_layout)
        self.line_points_input = self.create_labeled_input("Number of Points:", "10", line_layout)
        self.line_group.setLayout(line_layout)
        self.controls_layout.addWidget(self.line_group)

    def create_exclusion_controls(self) -> None:
        self.exclude_interior_checkbox = QCheckBox("Exclude points inside surface mesh")
        self.exclude_interior_checkbox.setChecked(True)
        self.controls_layout.addWidget(self.exclude_interior_checkbox)
        self.exclusion_distance_input = self.create_labeled_input("Exclusion distance:", "0.25", self.controls_layout, inline=True)

    def create_visualization_options(self) -> None:
        self.controls_layout.addWidget(QLabel("Visualization Options:"))
        hide_group1 = QHBoxLayout()
        self.hide_centerline_checkbox = QCheckBox("Hide centerline")
        self.hide_surface_curves_checkbox = QCheckBox("Hide surface curves")
        self.hide_coil_geometry_checkbox = QCheckBox("Hide coil geometry")
        hide_group1.addWidget(self.hide_centerline_checkbox)
        hide_group1.addWidget(self.hide_surface_curves_checkbox)
        hide_group1.addWidget(self.hide_coil_geometry_checkbox)
        self.controls_layout.addLayout(hide_group1)
        hide_group2 = QHBoxLayout()
        self.hide_region_checkbox = QCheckBox("Hide region points")
        self.hide_grid_checkbox = QCheckBox("Hide grid points")
        self.hide_axes_checkbox = QCheckBox("Hide coordinate axes")
        hide_group2.addWidget(self.hide_region_checkbox)
        hide_group2.addWidget(self.hide_grid_checkbox)
        hide_group2.addWidget(self.hide_axes_checkbox)
        self.controls_layout.addLayout(hide_group2)
        self.grid_spacing_input = self.create_labeled_input("Grid Spacing:", "1", self.controls_layout, inline=True)
        self.grid_side_length_input = self.create_labeled_input("Grid Cube Side Length:", "20", self.controls_layout, inline=True)
        self.visualize_region_button = QPushButton("Visualize Region")
        self.visualize_region_button.clicked.connect(self.visualize_region)
        self.controls_layout.addWidget(self.visualize_region_button)

    def create_field_visualization_options(self) -> None:
        self.controls_layout.addWidget(QLabel("Magnetic Field Visualization Options:"))
        self.field_type_dropdown = QComboBox()
        self.field_type_dropdown.addItems([
            "Vector- Magnitude (Color), Direction(Arrow)",
            "Vector, Magnitude & Direction",
            "Magnitude",
            "Bx (Color Only)",
            "Bx (Arrow)",
            "By (Color Only)",
            "By (Arrow)",
            "Bz (Color Only)",
            "Bz (Arrow)"
        ])
        self.controls_layout.addWidget(self.field_type_dropdown)
        self.vector_scale_input = self.create_labeled_input("Vector Scale:", "1.0", self.controls_layout, inline=True)

    def create_source_controls(self) -> None:
        self.controls_layout.addWidget(QLabel("Select path for current:"))
        source_layout = QHBoxLayout()
        self.surface_curves_radio = QRadioButton("Surface Curves")
        self.centerline_radio = QRadioButton("Centerline")
        self.surface_curves_radio.setChecked(True)
        source_layout.addWidget(self.surface_curves_radio)
        source_layout.addWidget(self.centerline_radio)
        self.controls_layout.addLayout(source_layout)
        self.current_input = self.create_labeled_input("Coil Current (A):", "1.0", self.controls_layout, inline=True)

    def create_compute_export_controls(self) -> None:
        self.field_type_lbl = QLabel("Select field type to compute:")
        self.field_compute_dropdown = QComboBox()
        self.field_compute_dropdown.addItems(["B-field (magnetic)", "E-field (electric)"])
        self.controls_layout.addWidget(self.field_type_lbl)
        self.controls_layout.addWidget(self.field_compute_dropdown)
        
        freq_box = QHBoxLayout()
        freq_box.addWidget(QLabel("Frequency (MHz):"))
        self.freq_input = QLineEdit("400")
        self.freq_input.setFixedWidth(60)
        self.freq_input.setEnabled(False)  # Greyed out by default
        freq_box.addWidget(self.freq_input)
        self.controls_layout.addLayout(freq_box)
        self.compute_field_button = QPushButton("Compute Field")
        self.compute_field_button.clicked.connect(self.compute_field)
        self.controls_layout.addWidget(self.compute_field_button)
        self.inductance_button = QPushButton("Compute inductance from calculated B-Field")
        self.inductance_button.clicked.connect(self.compute_inductance)
        self.controls_layout.addWidget(self.inductance_button)
        note_lbl = QLabel("<i>Ensure the specified volume fully encloses the coil</i>")
        self.controls_layout.addWidget(note_lbl)
        ind_layout = QHBoxLayout()
        ind_layout.addWidget(QLabel("Calculated static inductance:"))
        self.inductance_value_lbl = QLabel("")
        ind_layout.addWidget(self.inductance_value_lbl, stretch=1)
        self.controls_layout.addLayout(ind_layout)
        self.export_basename_input = self.create_labeled_input("Export CSV Base Name:", "field_export", self.controls_layout, inline=True)
        self.export_button = QPushButton("Export Data")
        self.export_button.clicked.connect(self.export_data)
        self.controls_layout.addWidget(self.export_button)

    def create_labeled_input(self, label_text: str, default_value: str, parent_layout, inline: bool = False) -> QLineEdit:
        if inline:
            layout = parent_layout if isinstance(parent_layout, QHBoxLayout) else QHBoxLayout()
        else:
            layout = QHBoxLayout()
        label = QLabel(label_text)
        input_field = QLineEdit()
        input_field.setText(default_value)
        layout.addWidget(label)
        layout.addWidget(input_field)
        if not inline:
            parent_layout.addLayout(layout)
        else:
            if not isinstance(parent_layout, QHBoxLayout):
                parent_layout.addLayout(layout)
        return input_field

    def update_inputs(self) -> None:
        region_type = self.region_dropdown.currentText()
        self.volume_group.setVisible(region_type == "Volume")
        self.plane_group.setVisible(region_type == "Plane")
        self.line_group.setVisible(region_type == "Line")

    def on_region_type_changed(self) -> None:
        """Handle region type dropdown change"""
        self.update_inputs()  # Update input visibility
        # Only refresh if there's something displayed that would be affected
        if self.last_region_points is not None:
            self.refresh_current_view()

    def refresh_current_view(self) -> None:
        """Refresh the current visualization based on what was last displayed"""
        # Determine what should be refreshed based on current state
        if self.last_B is not None and self.last_region_points is not None:
            # If magnetic field was computed and displayed, refresh that
            self.plot_magnetic_field()
        elif self.last_region_points is not None:
            # If only region visualization was shown, refresh that
            self.visualize_region()
        else:
            # If nothing specific was shown, refresh basic coil visualization
            self.refresh_basic_visualization()

    def refresh_basic_visualization(self) -> None:
        """Refresh basic coil visualization (mesh, centerline, surface curves, axes)"""
        self.ensure_plotter()
        self.plotter.clear()
        
        # Add surface mesh if available and not hidden
        if self.surf_poly is not None and not self.hide_coil_geometry_checkbox.isChecked():
            self.plotter.add_mesh(self.surf_poly, color="lightblue", opacity=0.5, label="Surface Mesh")
        
        # Add centerline if available and not hidden
        if self.centerline is not None and not self.hide_centerline_checkbox.isChecked():
            self.plotter.add_mesh(self.centerline, color="magenta", line_width=3, label="Centerline")
        
        # Add surface curves if available and not hidden
        if self.surface_curves is not None and len(self.surface_curves) > 0 and not self.hide_surface_curves_checkbox.isChecked():
            for idx, curve in enumerate(self.surface_curves):
                if len(curve) > 1:
                    curve_poly = pv.PolyData(curve.astype(np.float32))
                    curve_poly.lines = np.hstack([[len(curve)], np.arange(len(curve))])
                    self.plotter.add_mesh(curve_poly, color="cyan", line_width=3, 
                                        label=f"Surface Curve {idx}" if idx == 0 else None)
        
        # Add coordinate axes if not hidden
        if not self.hide_axes_checkbox.isChecked():
            self.add_coordinate_system()
        
        # Add grid points if not hidden
        self.add_grey_points()

    def get_volume_region_points(self) -> np.ndarray:
        try:
            xmin = float(self.xmin_input.text())
            xmax = float(self.xmax_input.text())
            ymin = float(self.ymin_input.text())
            ymax = float(self.ymax_input.text())
            zmin = float(self.zmin_input.text())
            zmax = float(self.zmax_input.text())
            spacing = float(self.points_spacing_input.text())
        except ValueError as e:
            logger.error("Invalid volume region input: %s", e)
            return np.empty((0, 3), dtype=np.float32)
        x_vals = np.arange(xmin, xmax + spacing, spacing)
        y_vals = np.arange(ymin, ymax + spacing, spacing)
        z_vals = np.arange(zmin, zmax + spacing, spacing)
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T.astype(np.float32)
        if self.exclude_interior_checkbox.isChecked():
            points = self.filter_interior_points(points)
        return points

    def get_plane_region_points(self) -> np.ndarray:
        try:
            origin = np.array([float(x) for x in self.plane_origin_input.text().split(",")], dtype=np.float32)
            normal = np.array([float(x) for x in self.plane_normal_input.text().split(",")], dtype=np.float32)
            length = float(self.plane_length_input.text())
            width = float(self.plane_width_input.text())
            spacing = float(self.plane_points_spacing_input.text())
        except ValueError as e:
            logger.error("Invalid plane region input: %s", e)
            return np.empty((0, 3), dtype=np.float32)
        norm_len = np.linalg.norm(normal)
        if norm_len < EPSILON:
            logger.error("Normal vector cannot be zero.")
            return np.empty((0, 3), dtype=np.float32)
        normal = normal / norm_len
        u = np.cross(normal, [1, 0, 0])
        if np.linalg.norm(u) < EPSILON:
            u = np.cross(normal, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        xs = np.arange(-length/2, length/2 + spacing, spacing)
        ys = np.arange(-width/2, width/2 + spacing, spacing)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        plane_points = origin + np.outer(X.ravel(), u) + np.outer(Y.ravel(), v)
        plane_points = plane_points.astype(np.float32)
        if self.exclude_interior_checkbox.isChecked():
            plane_points = self.filter_interior_points(plane_points)
        return plane_points

    def get_line_region_points(self) -> np.ndarray:
        try:
            start_point = np.array([float(x) for x in self.line_start_input.text().split(",")], dtype=np.float32)
            end_point = np.array([float(x) for x in self.line_end_input.text().split(",")], dtype=np.float32)
            num_points = int(self.line_points_input.text())
        except ValueError as e:
            logger.error("Invalid line region input: %s", e)
            return np.empty((0, 3), dtype=np.float32)
        line_points = np.linspace(start_point, end_point, num_points).astype(np.float32)
        if self.exclude_interior_checkbox.isChecked():
            line_points = self.filter_interior_points(line_points)
        return line_points

    def get_region_points(self, region_type: str) -> np.ndarray:
        if region_type == "Volume":
            return self.get_volume_region_points()
        elif region_type == "Plane":
            return self.get_plane_region_points()
        elif region_type == "Line":
            return self.get_line_region_points()
        return np.empty((0, 3), dtype=np.float32)

    def filter_interior_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            logger.debug("No points provided for filtering.")
            return points
        if self.surf_poly is None:
            logger.debug("Surface mesh not loaded. Skipping interior point filtering.")
            return points
        try:
            exclusion_distance = float(self.exclusion_distance_input.text())
        except ValueError:
            logger.error("Invalid exclusion distance; using 0.")
            exclusion_distance = 0.0

        test_poly = pv.PolyData(points)
        enclosed_result = test_poly.select_enclosed_points(self.surf_poly, check_surface=True)
        inside_mask = enclosed_result["SelectedPoints"]
        keep_mask = inside_mask == 0

        if exclusion_distance > 0:
            try:
                out_poly = test_poly.compute_implicit_distance(self.surf_poly, inplace=False)
                distances = np.abs(out_poly["implicit_distance"])
                if isinstance(distances, np.ndarray):
                    keep_mask &= (distances > exclusion_distance)
                else:
                    logger.error("'implicit_distance' array not found.")
                    keep_mask &= False
            except Exception as e:
                logger.error("compute_implicit_distance failed: %s", e)
                keep_mask &= False

        filtered_points = test_poly.points[keep_mask]
        n_excluded = np.count_nonzero(~keep_mask)
        logger.info("Excluded %d points based on criteria. %d remain.", n_excluded, filtered_points.shape[0])
        if filtered_points.shape[0] == 0:
            logger.warning("All points were excluded based on the exclusion distance.")
        return filtered_points

    def add_coordinate_system(self) -> None:
        origin = np.array([0, 0, 0])
        axes = {
            'X': ([1, 0, 0], 'red'),
            'Y': ([0, 1, 0], 'green'),
            'Z': ([0, 0, 1], 'blue')
        }
        for label, (direction, color) in axes.items():
            arrow = pv.Arrow(start=origin, direction=direction, scale=1.0)
            self.plotter.add_mesh(arrow, color=color, label=label)
            tip = origin + np.array(direction) * 1.1
            self.plotter.add_point_labels(tip.reshape(1, -1), [label],
                                          font_size=10, text_color=color)

    def add_grey_points(self) -> None:
        if self.hide_grid_checkbox.isChecked():
            return
        try:
            spacing = float(self.grid_spacing_input.text())
            side_length = float(self.grid_side_length_input.text())
        except ValueError:
            logger.error("Invalid grid spacing or side length. Using defaults (1, 20).")
            spacing = 1.0
            side_length = 20.0
        half_side = side_length / 2
        x_vals = np.arange(-half_side, half_side + spacing, spacing)
        y_vals = np.arange(-half_side, half_side + spacing, spacing)
        z_vals = np.arange(-half_side, half_side + spacing, spacing)
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T.astype(np.float32)
        grid_poly = pv.PolyData(grid_points)
        self.plotter.add_mesh(grid_poly, color="grey", point_size=2,
                              render_points_as_spheres=True, label="Coordinate Grid")

    def visualize_region(self) -> None:
        logger.debug("Entering visualize_region()")
        self.ensure_plotter()
        region_type = self.region_dropdown.currentText()
        self.plotter.clear()
        logger.debug("Plotter cleared for region visualization")
        if self.surf_poly is not None and not self.hide_coil_geometry_checkbox.isChecked():
            self.plotter.add_mesh(self.surf_poly, color="lightblue", opacity=0.5, label="Surface Mesh")
        if self.centerline is not None and not self.hide_centerline_checkbox.isChecked():
            self.plotter.add_mesh(self.centerline, color="magenta", line_width=3, label="Centerline")
        if not self.hide_axes_checkbox.isChecked():
            self.add_coordinate_system()
        self.add_grey_points()

        if not self.hide_region_checkbox.isChecked():
            if region_type == "Volume":
                pts = self.get_volume_region_points()
                volume_poly = pv.PolyData(pts)
                self.plotter.add_mesh(volume_poly, color="orange", point_size=4,
                                      render_points_as_spheres=True, label="Volume Points")
                try:
                    xmin = float(self.xmin_input.text())
                    xmax = float(self.xmax_input.text())
                    ymin = float(self.ymin_input.text())
                    ymax = float(self.ymax_input.text())
                    zmin = float(self.zmin_input.text())
                    zmax = float(self.zmax_input.text())
                    corners = np.array([
                        [xmin, ymin, zmin],
                        [xmax, ymin, zmin],
                        [xmax, ymax, zmin],
                        [xmin, ymax, zmin],
                        [xmin, ymin, zmax],
                        [xmax, ymin, zmax],
                        [xmax, ymax, zmax],
                        [xmin, ymax, zmax]
                    ], dtype=np.float32)
                    corner_poly = pv.PolyData(corners)
                    self.plotter.add_mesh(corner_poly, color="red", point_size=8,
                                          render_points_as_spheres=True, label="Bounding Box")
                except ValueError:
                    logger.error("Invalid volume bounding box parameters.")
            elif region_type == "Plane":
                pts = self.get_plane_region_points()
                plane_poly = pv.PolyData(pts)
                self.plotter.add_mesh(plane_poly, color="orange", point_size=4,
                                      render_points_as_spheres=True, label="Plane Points")
                try:
                    origin = np.array([float(x) for x in self.plane_origin_input.text().split(",")], dtype=np.float32)
                    normal = np.array([float(x) for x in self.plane_normal_input.text().split(",")], dtype=np.float32)
                    length = float(self.plane_length_input.text())
                    width = float(self.plane_width_input.text())
                    norm_len = np.linalg.norm(normal)
                    if norm_len < EPSILON:
                        raise ValueError("Normal vector is zero")
                    normal = normal / norm_len
                    u = np.cross(normal, [1, 0, 0])
                    if np.linalg.norm(u) < EPSILON:
                        u = np.cross(normal, [0, 1, 0])
                    u = u / np.linalg.norm(u)
                    v = np.cross(normal, u)
                    v = v / np.linalg.norm(v)
                    corners = np.array([
                        origin - (length/2)*u - (width/2)*v,
                        origin + (length/2)*u - (width/2)*v,
                        origin + (length/2)*u + (width/2)*v,
                        origin - (length/2)*u + (width/2)*v,
                    ], dtype=np.float32)
                    corner_poly = pv.PolyData(corners)
                    self.plotter.add_mesh(corner_poly, color="red", point_size=8,
                                          render_points_as_spheres=True, label="Plane Corners")
                    normal_arrow = pv.Arrow(start=origin, direction=normal, scale=1.0)
                    self.plotter.add_mesh(normal_arrow, color="red", label="Normal Vector")
                except Exception as e:
                    logger.error("Error visualizing plane details: %s", e)
            elif region_type == "Line":
                pts = self.get_line_region_points()
                if pts.shape[0] > 1:
                    line_poly = pv.PolyData(pts)
                    connectivity = np.hstack([[pts.shape[0]], np.arange(pts.shape[0])])
                    line_poly.lines = connectivity
                    self.plotter.add_mesh(line_poly, color="orange", line_width=4, label="Line Region")
                    endpoints = np.array([pts[0], pts[-1]], dtype=np.float32)
                    epoly = pv.PolyData(endpoints)
                    self.plotter.add_mesh(epoly, color="red", point_size=8,
                                          render_points_as_spheres=True, label="Line Endpoints")
                elif pts.shape[0] == 1:
                    # Single point - render without spheres to avoid errors
                    line_poly = pv.PolyData(pts)
                    self.plotter.add_mesh(line_poly, color="orange", point_size=10, label="Line Point")
                # If 0 points, don't try to visualize
        if self.surface_curves is not None and len(self.surface_curves) > 0 and not self.hide_surface_curves_checkbox.isChecked():
            for curve in self.surface_curves:
                if len(curve) < 2:
                    continue
                curve = np.array(curve, dtype=np.float32)
                curve_poly = pv.PolyData(curve)
                lines = np.hstack([[curve.shape[0]], np.arange(curve.shape[0])])
                curve_poly.lines = lines
                curve_poly.verts = np.empty((0,), dtype=np.int64)
                self.plotter.add_mesh(curve_poly, color="cyan", line_width=2, label="Surface Curves")
        self.vtk_widget.interactor.GetRenderWindow().SetWindowName(f"Region Visualization: {region_type}")
        self.plotter.show()

    def compute_field(self) -> None:
        region_type = self.region_dropdown.currentText()
        logger.info("Computing field for region type: %s", region_type)

        try:
            total_current = float(self.current_input.text())
        except ValueError:
            logger.error("Invalid current value; reverting to 1.0 A")
            total_current = 1.0

        region_points = self.get_region_points(region_type)
        if region_points.size == 0:
            logger.warning("No region points found. Cannot compute field.")
            return

        use_surface = self.surface_curves_radio.isChecked()
        if use_surface:
            if self.surface_curves is None or len(self.surface_curves) == 0:
                logger.error("No surface curves loaded")
                return
            valid_curves = [np.asarray(c, dtype=np.float32) for c in self.surface_curves
                            if len(c) >= 2]
            if not valid_curves:
                logger.error("No valid surface curves (need ≥2 points each)")
                return
            current_per_curve = total_current / len(valid_curves)
        else:
            if self.centerline is None or self.centerline.n_points < 2:
                logger.error("Invalid or missing centreline")
                return
            coil_points = self.centerline.points

        computing_B = (self.field_compute_dropdown.currentText() == "B-field (magnetic)")

        if computing_B:
            B_total = np.zeros((region_points.shape[0], 3), dtype=np.float64)
            if use_surface:
                for curve in valid_curves:
                    B_total += compute_biot_savart_field(region_points, curve,
                                                        current=current_per_curve,
                                                        mu0=MU0_MM)
            else:
                B_total = compute_biot_savart_field(region_points, coil_points,
                                                    current=total_current,
                                                    mu0=MU0_MM)
            scalars = np.linalg.norm(B_total, axis=1)
            bar_title = "|B| (T)"
            self.last_B = B_total
            self.last_E = None
        else:
            try:
                freq_MHz = float(self.freq_input.text())
            except ValueError:
                QMessageBox.critical(self, "Field Error",
                                    "Frequency must be a number (MHz).")
                return
            omega = 2 * np.pi * freq_MHz * 1e6
            A_total = np.zeros((region_points.shape[0], 3), dtype=np.float64)
            if use_surface:
                for curve in valid_curves:
                    A_total += compute_vector_potential(region_points, curve,
                                                        current=current_per_curve,
                                                        mu0=MU0_MM)
            else:
                A_total = compute_vector_potential(region_points, coil_points,
                                                current=total_current,
                                                mu0=MU0_MM)
            E_vec = omega * A_total * MM_TO_M
            E_mag = np.linalg.norm(E_vec, axis=1)
            scalars = E_mag
            bar_title = "|E| (V/m)"
            self.last_E = E_vec
            self.last_B = None

        self.last_region_points = region_points
        self.last_computed_region_type = region_type
        if self.points_changed(self.last_region_points_by_type.get(region_type),
                            region_points):
            self.region_counts[region_type] += 1
            self.last_region_points_by_type[region_type] = region_points.copy()

        logger.info("Computed %s for %d points.", bar_title, region_points.shape[0])
        if computing_B:
            self.plot_magnetic_field()
        else:
            self.plot_electric_field()

    def plot_magnetic_field(self) -> None:
        if self.last_B is None or self.last_region_points is None:
            logger.info("No computed B-field to plot yet.")
            return

        self.ensure_plotter()
        B = self.last_B
        region_points = self.last_region_points
        self.plotter.clear()
        if self.surf_poly is not None and not self.hide_coil_geometry_checkbox.isChecked():
            self.plotter.add_mesh(self.surf_poly, color="lightblue", opacity=0.5, label="Surface Mesh")
        if self.centerline is not None and not self.hide_centerline_checkbox.isChecked():
            self.plotter.add_mesh(self.centerline, color="magenta", line_width=2, label="Centerline")
        if not self.hide_axes_checkbox.isChecked():
            self.add_coordinate_system()
        if not self.hide_grid_checkbox.isChecked():
            self.add_grey_points()

        if not self.hide_region_checkbox.isChecked():
            self.visualize_region_by_type(region_points, self.last_computed_region_type)

        if self.surface_curves is not None and len(self.surface_curves) > 0 and not self.hide_surface_curves_checkbox.isChecked():
            for curve in self.surface_curves:
                if len(curve) < 2:
                    continue
                curve = np.array(curve, dtype=np.float32)
                curve_poly = pv.PolyData(curve)
                lines = np.hstack([[curve.shape[0]], np.arange(curve.shape[0])])
                curve_poly.lines = lines
                curve_poly.verts = np.empty((0,), dtype=np.int64)
                self.plotter.add_mesh(curve_poly, color="cyan", line_width=2, label="Surface Curves")

        field_poly = pv.PolyData(region_points)
        Bmag = np.linalg.norm(B, axis=1)
        try:
            vector_scale = float(self.vector_scale_input.text())
        except ValueError:
            vector_scale = 1.0
        field_type = self.field_type_dropdown.currentText()

        if field_type == "Vector- Magnitude (Color), Direction(Arrow)":
            field_poly["B"] = B
            field_poly["Bmag"] = Bmag
            arrows = field_poly.glyph(orient="B", scale=False, factor=vector_scale)
            # Safer calculation for scalar assignment to handle edge cases
            n_arrows = arrows.n_points
            n_region = region_points.shape[0]
            if n_region > 0 and n_arrows > 0:
                repeat_factor = n_arrows // n_region if n_arrows >= n_region else 1
                arrows["Bmag"] = np.repeat(Bmag, repeat_factor)[:n_arrows]
            self.plotter.add_mesh(arrows, scalars="Bmag", cmap="jet",
                                  scalar_bar_args={"title": "|B| (T)"})
        elif field_type == "Vector, Magnitude & Direction":
            max_mag = Bmag.max() if Bmag.size else 1.0
            if max_mag < EPSILON:
                max_mag = 1.0
            scale_factor = vector_scale / max_mag
            field_poly["B"] = B
            field_poly["Bmag"] = Bmag
            arrows = field_poly.glyph(orient="B", scale="Bmag", factor=scale_factor)
            # Safer calculation for scalar assignment
            n_arrows = arrows.n_points
            n_region = region_points.shape[0]
            if n_region > 0 and n_arrows > 0:
                repeat_factor = n_arrows // n_region if n_arrows >= n_region else 1
                arrows["Bmag"] = np.repeat(Bmag, repeat_factor)[:n_arrows]
            self.plotter.add_mesh(arrows, scalars="Bmag", cmap="jet",
                                  scalar_bar_args={"title": "|B| (T)","fmt": "%.2e"})
        elif field_type == "Magnitude":
            field_poly["Bmag"] = Bmag
            self.plotter.add_mesh(field_poly, scalars="Bmag", cmap="jet",
                                  point_size=5, render_points_as_spheres=True,
                                  scalar_bar_args={"title": "|B| (T)","fmt": "%.2e"})
        elif field_type == "Bx (Color Only)":
            bx_abs = np.abs(B[:, 0])
            field_poly["BxAbs"] = bx_abs
            self.plotter.add_mesh(field_poly, scalars="BxAbs", cmap="jet",
                                  point_size=5, render_points_as_spheres=True,
                                  scalar_bar_args={"title": "|Bx| (T)","fmt": "%.2e"})
        elif field_type == "Bx (Arrow)":
            bx = B[:, 0]
            bx_abs = np.abs(bx)
            max_bx = bx_abs.max() if bx_abs.size else 1.0
            if max_bx < EPSILON:
                max_bx = 1.0
            scale_factor = vector_scale / max_bx
            bx_dir = np.zeros_like(B)
            bx_dir[:, 0] = np.sign(bx)
            field_poly["Bx_dir"] = bx_dir
            field_poly["Bx_len"] = bx_abs
            arrows = field_poly.glyph(orient="Bx_dir", scale="Bx_len", factor=scale_factor)
            # Safer calculation for scalar assignment
            n_arrows = arrows.n_points
            n_region = region_points.shape[0]
            if n_region > 0 and n_arrows > 0:
                repeat_factor = n_arrows // n_region if n_arrows >= n_region else 1
                arrows["BxAbs"] = np.repeat(bx_abs, repeat_factor)[:n_arrows]
            self.plotter.add_mesh(arrows, scalars="BxAbs", cmap="jet",
                                  scalar_bar_args={"title": "|Bx| (T)","fmt": "%.2e"})
        elif field_type == "By (Color Only)":
            by_abs = np.abs(B[:, 1])
            field_poly["ByAbs"] = by_abs
            self.plotter.add_mesh(field_poly, scalars="ByAbs", cmap="jet",
                                  point_size=5, render_points_as_spheres=True,
                                  scalar_bar_args={"title": "|By| (T)","fmt": "%.2e"})
        elif field_type == "By (Arrow)":
            by = B[:, 1]
            by_abs = np.abs(by)
            max_by = by_abs.max() if by_abs.size else 1.0
            if max_by < EPSILON:
                max_by = 1.0
            scale_factor = vector_scale / max_by
            by_dir = np.zeros_like(B)
            by_dir[:, 1] = np.sign(by)
            field_poly["By_dir"] = by_dir
            field_poly["By_len"] = by_abs
            arrows = field_poly.glyph(orient="By_dir", scale="By_len", factor=scale_factor)
            # Safer calculation for scalar assignment
            n_arrows = arrows.n_points
            n_region = region_points.shape[0]
            if n_region > 0 and n_arrows > 0:
                repeat_factor = n_arrows // n_region if n_arrows >= n_region else 1
                arrows["ByAbs"] = np.repeat(by_abs, repeat_factor)[:n_arrows]
            self.plotter.add_mesh(arrows, scalars="ByAbs", cmap="jet",
                                  scalar_bar_args={"title": "|By| (T)","fmt": "%.2e"})
        elif field_type == "Bz (Color Only)":
            bz_abs = np.abs(B[:, 2])
            field_poly["BzAbs"] = bz_abs
            self.plotter.add_mesh(field_poly, scalars="BzAbs", cmap="jet",
                                  point_size=5, render_points_as_spheres=True,
                                  scalar_bar_args={"title": "|Bz| (T)","fmt": "%.2e"})
        elif field_type == "Bz (Arrow)":
            bz = B[:, 2]
            bz_abs = np.abs(bz)
            max_bz = bz_abs.max() if bz_abs.size else 1.0
            if max_bz < EPSILON:
                max_bz = 1.0
            scale_factor = vector_scale / max_bz
            bz_dir = np.zeros_like(B)
            bz_dir[:, 2] = np.sign(bz)
            field_poly["Bz_dir"] = bz_dir
            field_poly["Bz_len"] = bz_abs
            arrows = field_poly.glyph(orient="Bz_dir", scale="Bz_len", factor=scale_factor)
            # Safer calculation for scalar assignment
            n_arrows = arrows.n_points
            n_region = region_points.shape[0]
            if n_region > 0 and n_arrows > 0:
                repeat_factor = n_arrows // n_region if n_arrows >= n_region else 1
                arrows["BzAbs"] = np.repeat(bz_abs, repeat_factor)[:n_arrows]
            self.plotter.add_mesh(arrows, scalars="BzAbs", cmap="jet",
                                  scalar_bar_args={"title": "|Bz| (T)"})
        self.vtk_widget.interactor.GetRenderWindow().SetWindowName(f"B-field Visualization: {field_type}")
        self.plotter.show()

    def plot_electric_field(self) -> None:
        if self.last_E is None or self.last_region_points is None:
            logger.info("No E-field to plot yet.")
            return

        self.ensure_plotter()
        E_vec = self.last_E
        Emag = np.linalg.norm(E_vec, axis=1)
        pts = self.last_region_points
        self.plotter.clear()
        if self.surf_poly is not None and not self.hide_coil_geometry_checkbox.isChecked():
            self.plotter.add_mesh(self.surf_poly, color="lightblue", opacity=0.5)

        if self.centerline is not None and not self.hide_centerline_checkbox.isChecked():
            self.plotter.add_mesh(self.centerline, color="magenta", line_width=2)
        if not self.hide_axes_checkbox.isChecked():
            self.add_coordinate_system()
        if not self.hide_grid_checkbox.isChecked():
            self.add_grey_points()
        if not self.hide_region_checkbox.isChecked():
            self.visualize_region_by_type(pts, self.last_computed_region_type)

        field_poly = pv.PolyData(pts)
        field_poly["E"] = E_vec
        field_poly["Emag"] = Emag
        try:
            vscale = float(self.vector_scale_input.text())
        except ValueError:
            vscale = 1.0
        max_e = Emag.max() if Emag.size else 1.0
        if max_e < EPSILON:
            max_e = 1.0
        arrows = field_poly.glyph(orient="E", scale="Emag", factor=vscale / max_e)
        # Safer calculation for scalar assignment
        n_arrows = arrows.n_points
        n_region = pts.shape[0]
        if n_region > 0 and n_arrows > 0:
            repeat_factor = n_arrows // n_region if n_arrows >= n_region else 1
            arrows["Emag"] = np.repeat(Emag, repeat_factor)[:n_arrows]
        self.plotter.add_mesh(
            arrows,
            scalars="Emag",
            cmap="jet",
            scalar_bar_args={"title": "|E| (V/m)", "fmt": "%.2e"},
        )

        self.vtk_widget.interactor.GetRenderWindow().SetWindowName(
            "E-field Visualization: Vector, magnitude & direction"
        )
        self.plotter.show()

    def visualize_region_by_type(self, region_points: np.ndarray, region_type: str) -> None:
        if region_type == "Volume":
            volume_poly = pv.PolyData(region_points)
            self.plotter.add_mesh(volume_poly, color="orange", point_size=4,
                                  render_points_as_spheres=True, label="Region Points")
        elif region_type == "Plane":
            plane_poly = pv.PolyData(region_points)
            self.plotter.add_mesh(plane_poly, color="orange", point_size=4,
                                  render_points_as_spheres=True, label="Region Points")
        elif region_type == "Line":
            if region_points.shape[0] > 1:
                line_poly = pv.PolyData(region_points)
                connectivity = np.hstack([[region_points.shape[0]], np.arange(region_points.shape[0])])
                line_poly.lines = connectivity
                self.plotter.add_mesh(line_poly, color="orange", line_width=4, label="Region Points")
                endpoints = np.array([region_points[0], region_points[-1]], dtype=np.float32)
                epoly = pv.PolyData(endpoints)
                self.plotter.add_mesh(epoly, color="red", point_size=8,
                                      render_points_as_spheres=True, label="Line Endpoints")
            elif region_points.shape[0] == 1:
                # Single point - render without spheres to avoid errors
                line_poly = pv.PolyData(region_points)
                self.plotter.add_mesh(line_poly, color="orange", point_size=10, label="Region Point")
            # If 0 points, don't try to visualize

    def compute_inductance(self) -> None:
        if self.last_B is None or self.last_region_points is None:
            QMessageBox.warning(self, "Inductance",
                               "Please compute a magnetic field first.")
            return
        if self.last_computed_region_type != "Volume":
            QMessageBox.warning(self, "Inductance",
                               "Inductance requires a Volume region grid.")
            return

        try:
            spacing = float(self.points_spacing_input.text())
            current = float(self.current_input.text())
        except ValueError:
            QMessageBox.critical(self, "Inductance",
                                "Invalid spacing or current value.")
            return

        spacing_mm = float(self.points_spacing_input.text())
        dV_m3 = (spacing_mm * MM_TO_M) ** 3
        Bmag2 = np.sum(self.last_B**2, axis=1)
        energy = 0.5 / MU0_SI * np.sum(Bmag2) * dV_m3
        L = 2 * energy / current**2
        self.inductance_value_lbl.setText(f"{L:.2e} H")
        logger.info("Computed inductance ≈ %.6e H", L)

    @staticmethod
    def points_changed(old_points: np.ndarray, new_points: np.ndarray) -> bool:
        if old_points is None:
            return True
        if old_points.shape[0] != new_points.shape[0]:
            return True
        return not np.allclose(old_points, new_points, rtol=1e-7, atol=1e-12)

    def export_data(self) -> None:
        if self.last_region_points is None or self.last_B is None or self.last_computed_region_type is None:
            QMessageBox.warning(self, "Export Data", "No computed field data to export. Please compute first.")
            return

        region_type = self.last_computed_region_type
        count = self.region_counts.get(region_type, 0)
        base_name = self.export_basename_input.text().strip() or "field_export"
        file_name = f"{base_name}-{region_type.lower()}{count}.csv"

        data = np.column_stack((self.last_region_points, self.last_B))
        header = "x,y,z,Bx,By,Bz"
        fmt = ["%.4f", "%.4f", "%.4f"] + ["%.20e"] * 3

        try:
            np.savetxt(
                file_name,
                data,
                delimiter=",",
                header=header,
                comments="",
                fmt=fmt,
            )
            logger.info("Exported field data to '%s'.", file_name)
        except Exception as e:
            logger.error("Failed to export data: %s", e)
            QMessageBox.critical(self, "Export Error", f"Failed to export data: {e}")

    def exit_application(self) -> None:
        logger.debug("exit_application() called")
        try:
            self.vtk_widget.close()
            logger.debug("VTK widget closed successfully")
        except Exception as e:
            logger.error("Error closing VTK widget: %s", e)
        QApplication.instance().quit()

    def closeEvent(self, event) -> None:
        logger.debug("closeEvent() triggered")
        try:
            self.vtk_widget.close()
            logger.debug("VTK widget closed in closeEvent()")
        except Exception as e:
            logger.error("Error closing VTK widget in closeEvent: %s", e)
        QApplication.instance().quit()
        event.accept()

# ---------------------------
# New Tab Classes
# ---------------------------

class MeshProcessorTab(QWidget):

    """Mesh processing tab with STL/STEP import, meshing, centerline, surface curve extraction, and visualization, matching convert.py logic and UI, with Export to Visualizer."""
    def __init__(self, parent=None):
        super().__init__(parent)
        import helpers
        import pyvista as pv
        from pyvistaqt import QtInteractor
        self.helpers = helpers
        self.pv = pv
        self.init_ui()

    def init_ui(self):
        from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QRadioButton, QButtonGroup, QComboBox, QDoubleSpinBox, QSpinBox, QFileDialog, QMessageBox
        main_layout = QHBoxLayout(self)
        control_layout = QVBoxLayout()

        # File type radio buttons
        self.stl_radio = QRadioButton("Accept STL Files")
        self.stl_radio.setChecked(True)
        self.stl_radio.toggled.connect(self.set_accept_stl)
        self.stp_radio = QRadioButton("Accept STP Files")
        self.stp_radio.toggled.connect(self.set_accept_stp)
        self.msh_radio = QRadioButton("Accept MSH Files")
        self.msh_radio.toggled.connect(self.set_accept_msh)
        filetype_group = QButtonGroup()
        filetype_group.addButton(self.stl_radio)
        filetype_group.addButton(self.stp_radio)
        filetype_group.addButton(self.msh_radio)
        control_layout.addWidget(QLabel("File Type Filter:"))
        control_layout.addWidget(self.stl_radio)
        control_layout.addWidget(self.stp_radio)
        control_layout.addWidget(self.msh_radio)

        self.btn_load_file = QPushButton("Load File")
        self.btn_load_file.clicked.connect(self.load_file)
        self.btn_load_file.setToolTip("Load a .stp or .stl file")
        control_layout.addWidget(self.btn_load_file)
        self.loaded_file = QLabel("No file loaded")
        self.loaded_file.setStyleSheet("color: green;")
        control_layout.addWidget(self.loaded_file)

        self.btn_process = QPushButton("Generate Centerline")
        self.btn_process.clicked.connect(self.generate_centerline)
        self.btn_process.setEnabled(False)
        self.btn_process.setToolTip("Generate a centerline for loaded coil")
        control_layout.addWidget(self.btn_process)

        self.btn_second_process = QPushButton("Generate Surface Curves")
        self.btn_second_process.clicked.connect(self.generate_surface_curves)
        self.btn_second_process.setEnabled(False)
        self.btn_second_process.setToolTip("Generate surface curves for centerline")
        control_layout.addWidget(self.btn_second_process)

        self.btn_clear = QPushButton("Clear Plot")
        self.btn_clear.clicked.connect(self.clear_plot)
        self.btn_clear.setToolTip("Clears the current plot")
        control_layout.addWidget(self.btn_clear)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: red;")
        control_layout.addWidget(self.status_label)

        # Sizing mode dropdown for STP
        self.stp_sizing_dropdown = QComboBox()
        self.stp_sizing_dropdown.addItems(["curvature", "uniform"])
        self.stp_sizing_dropdown.setCurrentText("uniform")
        self.stp_sizing_dropdown.currentTextChanged.connect(lambda val: setattr(self, 'sizing_mode', val))
        self.stp_sizing_dropdown.setToolTip("Choose between Uniform and Curvature Modes for STP Files")
        self.stp_sizing_dropdown.setEnabled(False)
        control_layout.addWidget(QLabel("Sizing Mode:"))
        control_layout.addWidget(self.stp_sizing_dropdown)

        # Element size input
        self.element_size_input = QDoubleSpinBox()
        self.element_size_input.setRange(0.01, 10.0)
        self.element_size_input.setSingleStep(0.01)
        self.element_size_input.setValue(0.15)
        self.element_size_input.setEnabled(False)
        control_layout.addWidget(QLabel("Element Size"))
        control_layout.addWidget(self.element_size_input)

        self.max_element_size_factor_input = QDoubleSpinBox()
        self.max_element_size_factor_input.setRange(1.0, 5.0)
        self.max_element_size_factor_input.setSingleStep(0.01)
        self.max_element_size_factor_input.setValue(2.0)
        self.max_element_size_factor_input.setEnabled(False)
        control_layout.addWidget(QLabel("Max Element Size Factor"))
        control_layout.addWidget(self.max_element_size_factor_input)

        self.feature_angle_input = QSpinBox()
        self.feature_angle_input.setRange(0, 180)
        self.feature_angle_input.setValue(65)
        self.feature_angle_input.setEnabled(False)
        control_layout.addWidget(QLabel("Feature Angle"))
        control_layout.addWidget(self.feature_angle_input)

        self.trim_points_input = QSpinBox()
        self.trim_points_input.setRange(0, 1000)
        self.trim_points_input.setValue(0)
        self.trim_points_input.setEnabled(False)
        control_layout.addWidget(QLabel("Trim Points"))
        control_layout.addWidget(self.trim_points_input)

        self.centerline_s_input = QDoubleSpinBox()
        self.centerline_s_input.setRange(0.0, 1.0)
        self.centerline_s_input.setSingleStep(0.001)
        self.centerline_s_input.setValue(0.01)
        self.centerline_s_input.setEnabled(False)
        control_layout.addWidget(QLabel("Centerline Smoothing"))
        control_layout.addWidget(self.centerline_s_input)

        self.surfacecurves_s_input = QDoubleSpinBox()
        self.surfacecurves_s_input.setRange(0.0, 1.0)
        self.surfacecurves_s_input.setSingleStep(0.001)
        self.surfacecurves_s_input.setValue(0.01)
        self.surfacecurves_s_input.setEnabled(False)
        control_layout.addWidget(QLabel("Surface Curves Smoothing"))
        control_layout.addWidget(self.surfacecurves_s_input)

        self.loop_smoothing_input = QDoubleSpinBox()
        self.loop_smoothing_input.setRange(0.0, 1.0)
        self.loop_smoothing_input.setSingleStep(0.001)
        self.loop_smoothing_input.setValue(0.0)
        self.loop_smoothing_input.setEnabled(False)
        control_layout.addWidget(QLabel("Loop Smoothing"))
        control_layout.addWidget(self.loop_smoothing_input)

        self.n_centerline_points_input = QSpinBox()
        self.n_centerline_points_input.setRange(50, 2000)
        self.n_centerline_points_input.setValue(500)
        self.n_centerline_points_input.setEnabled(False)
        control_layout.addWidget(QLabel("Centerline Points"))
        control_layout.addWidget(self.n_centerline_points_input)

        self.n_loop_points_input = QSpinBox()
        self.n_loop_points_input.setRange(50, 500)
        self.n_loop_points_input.setValue(100)
        self.n_loop_points_input.setEnabled(False)
        control_layout.addWidget(QLabel("Loop Points"))
        control_layout.addWidget(self.n_loop_points_input)

        self.n_subset_points_input = QSpinBox()
        self.n_subset_points_input.setRange(10, 100)
        self.n_subset_points_input.setValue(20)
        self.n_subset_points_input.setEnabled(False)
        control_layout.addWidget(QLabel("Subset Points"))
        control_layout.addWidget(self.n_subset_points_input)

        self.marching_record_step_input = QSpinBox()
        self.marching_record_step_input.setRange(1, 10)
        self.marching_record_step_input.setValue(5)
        self.marching_record_step_input.setEnabled(False)
        control_layout.addWidget(QLabel("Marching Record Step"))
        control_layout.addWidget(self.marching_record_step_input)

        # Export to Visualizer button (bottom of left panel)
        self.export_btn = QPushButton("Export to Visualizer")
        self.export_btn.clicked.connect(self.export_to_visualizer)
        self.export_btn.setEnabled(False)
        control_layout.addWidget(self.export_btn)

        control_layout.addStretch()
        left_panel = QWidget()
        left_panel.setLayout(control_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(left_panel)
        scroll_area.setMinimumWidth(450)
        scroll_area.setMaximumWidth(450)
        main_layout.addWidget(scroll_area, 1)

        # Plot Area
        self.plotter = self.pvqt_interactor()
        main_layout.addWidget(self.plotter.interactor, 4)

        # State
        self.input_file = None
        self.msh_file = "generated_mesh.msh"
        self.surf_poly = None
        self.raw_centerline_forward = None
        self.loopA = None
        self.loopB = None
        self.marching_record_forward = None
        self.final_centerline = None
        self.final_centerline_poly = None
        self.surface_curves = None

        # File type filter state
        self.accept_stl = True
        self.accept_stp = False
        self.accept_msh = False
        self.sizing_mode = "uniform"

    def pvqt_interactor(self):
        from pyvistaqt import QtInteractor
        return QtInteractor(self)

    def set_accept_stl(self):
        self.accept_stl = True
        self.accept_stp = False
        self.accept_msh = False
        self.stp_sizing_dropdown.setEnabled(False)
        self.element_size_input.setEnabled(False)
        self.max_element_size_factor_input.setEnabled(False)
        self.trim_points_input.setEnabled(False)
        self.centerline_s_input.setEnabled(False)
        self.surfacecurves_s_input.setEnabled(False)
        self.n_centerline_points_input.setEnabled(False)
        self.n_loop_points_input.setEnabled(False)
        self.n_subset_points_input.setEnabled(False)
        self.marching_record_step_input.setEnabled(False)

    def set_accept_stp(self):
        self.accept_stl = False
        self.accept_stp = True
        self.accept_msh = False
        self.stp_sizing_dropdown.setEnabled(True)
        self.element_size_input.setEnabled(True)
        self.max_element_size_factor_input.setEnabled(True)
        self.trim_points_input.setEnabled(True)
        self.centerline_s_input.setEnabled(True)
        self.surfacecurves_s_input.setEnabled(True)
        self.n_centerline_points_input.setEnabled(True)
        self.n_loop_points_input.setEnabled(True)
        self.n_subset_points_input.setEnabled(True)
        self.loop_smoothing_input.setEnabled(True)
        self.marching_record_step_input.setEnabled(True)

    def set_accept_msh(self):
        self.accept_stl = False
        self.accept_stp = False
        self.accept_msh = True
        # MSH files already have mesh, so disable mesh generation parameters
        self.stp_sizing_dropdown.setEnabled(False)
        self.element_size_input.setEnabled(False)
        self.max_element_size_factor_input.setEnabled(False)
        # But enable the processing parameters
        self.trim_points_input.setEnabled(True)
        self.centerline_s_input.setEnabled(True)
        self.surfacecurves_s_input.setEnabled(True)
        self.n_centerline_points_input.setEnabled(True)
        self.n_loop_points_input.setEnabled(True)
        self.n_subset_points_input.setEnabled(True)
        self.loop_smoothing_input.setEnabled(True)
        self.marching_record_step_input.setEnabled(True)
        self.n_loop_points_input.setEnabled(True)
        self.n_subset_points_input.setEnabled(True)
        self.marching_record_step_input.setEnabled(True)

    def load_file(self):
        file_dialog = QFileDialog()
        file_filter = ""
        if self.accept_stl:
            file_filter = "STL Files (*.stl)"
        elif self.accept_stp:
            file_filter = "STEP Files (*.stp *.step)"
        elif self.accept_msh:
            file_filter = "Mesh Files (*.msh)"
        else:
            file_filter = "All Files (*.*)"

        self.input_file, _ = file_dialog.getOpenFileName(self, "Open File", "", file_filter)
        if not self.input_file:
            return
        self.loaded_file.setText(f"Loaded: {os.path.basename(self.input_file)}")
        self.btn_process.setEnabled(True)
        self.feature_angle_input.setEnabled(True)
        
        # Automatically process the file after loading
        self.process_loaded_file()
    
    def process_loaded_file(self):
        """Process the loaded file: generate centerline, surface curves, and display everything"""
        import logging
        try:
            self.status_label.setText("Status: Processing file...")
            self.plotter.clear()
            
            # Read parameters
            self.element_size = self.element_size_input.value()
            self.max_element_size_factor = self.max_element_size_factor_input.value()
            self.feature_angle = self.feature_angle_input.value()
            self.trim_points = self.trim_points_input.value()
            self.centerline_s = self.centerline_s_input.value()
            self.surfacecurves_s = self.surfacecurves_s_input.value()
            self.loop_smoothing = self.loop_smoothing_input.value()
            self.n_centerline_points = self.n_centerline_points_input.value()
            self.n_loop_points = self.n_loop_points_input.value()
            self.n_subset_points = self.n_subset_points_input.value()

            # 1. Load surface mesh
            self.status_label.setText("Status: Loading and meshing file...")
            QApplication.processEvents()
            if self.input_file:
                self.surf_poly = self.helpers.load_surface_mesh(
                    self.input_file, self.msh_file,
                    self.stp_check(self.element_size, 0.15),
                    self.stp_check(self.max_element_size_factor, 2.0),
                    self.sizing_mode
                )
            else:
                logging.error(f"No input file selected")
                self.status_label.setText("Status: Error - No file selected")
                return

            # 2. Extract end loops
            self.status_label.setText("Status: Extracting coil end loops...")
            QApplication.processEvents()
            self.loopA, self.loopB = self.helpers.extract_coil_end_loops(
                self.surf_poly,
                self.stp_check(self.feature_angle, 75)
            )

            # 3. Generate centerline
            self.status_label.setText("Status: Computing centerline (this may take a moment)...")
            QApplication.processEvents()
            self.raw_centerline_forward, self.marching_record_forward = \
                self.helpers.compute_centerline_3d_mce(
                    self.surf_poly,
                    self.loopA,
                    self.loopB
                )
            if self.raw_centerline_forward is None:
                logging.error("Failed to compute centerline.")
                self.status_label.setText("Status: Error - Failed to compute centerline")
                return

            self.status_label.setText("Status: Processing centerline...")
            QApplication.processEvents()
            filtered_fwd = self.helpers.trim_end(self.raw_centerline_forward, self.stp_check(self.trim_points, 0))
            self.final_centerline = filtered_fwd
            self.final_centerline_poly = self.helpers.create_polyline(self.final_centerline, closed=False)

            # 4. Generate surface curves
            self.status_label.setText("Status: Refining end loops...")
            loopA_ordered = self.helpers.order_loop_points_pca(self.loopA.points)
            refined_loopA_pts = self.helpers.refine_loop(self.pv.PolyData(loopA_ordered), n_points=self.stp_check(self.n_loop_points, 200), smoothing=self.stp_check(self.loop_smoothing, 0), spline_degree=3)
            refined_loopA_poly = self.helpers.create_polyline(refined_loopA_pts, closed=True)

            loopB_ordered = self.helpers.order_loop_points_pca(self.loopB.points)
            refined_loopB_pts = self.helpers.refine_loop(self.pv.PolyData(loopB_ordered), n_points=self.stp_check(self.n_loop_points, 200), smoothing=self.stp_check(self.loop_smoothing, 0), spline_degree=3)
            refined_loopB_poly = self.helpers.create_polyline(refined_loopB_pts, closed=True)

            self.status_label.setText("Status: Building reference frames...")
            centerpoint_A = self.final_centerline[0]
            centerpoint_B = self.final_centerline[-1]
            contours_A = self.helpers.generate_intermediate_contours(refined_loopA_pts, centerpoint_A, n_contours=5)
            contours_B = self.helpers.generate_intermediate_contours(refined_loopB_pts, centerpoint_B, n_contours=5)

            n_vecs, x_vecs, y_vecs = self.helpers.build_no_roll_frames(self.final_centerline)

            self.status_label.setText("Status: Generating cross-sections...")
            cross_sections_scaffold = []
            total_sections = len(self.final_centerline)
            
            for i in range(total_sections):
                # Update progress every 10% of sections
                if i % max(1, total_sections // 10) == 0:
                    progress = int((i / total_sections) * 100)
                    self.status_label.setText(f"Status: Generating cross-sections... {progress}% complete")
                    # Force GUI update during long computation
                    QApplication.processEvents()
                
                if i == 0:
                    cross_sections_scaffold.append(refined_loopA_pts)
                    continue
                elif i == len(self.final_centerline) - 1:
                    cross_sections_scaffold.append(refined_loopB_pts)
                    continue
                center = self.final_centerline[i]
                n_i = n_vecs[i]
                x_i = x_vecs[i]
                y_i = y_vecs[i]
                sliced = self.helpers.slice_surface_at_point(self.surf_poly, center, n_i)
                if sliced is None or sliced.n_points < 3:
                    cross_sections_scaffold.append(None)
                    continue
                loops_sliced = sliced.split_bodies()
                if isinstance(loops_sliced, self.pv.MultiBlock):
                    slice_loop = max(loops_sliced, key=lambda lp: lp.length)
                else:
                    slice_loop = loops_sliced
                if slice_loop is None or slice_loop.n_points < 3:
                    cross_sections_scaffold.append(None)
                    continue
                raw_pts = slice_loop.points.copy()
                angles_indices = []
                for idx_pt, pt in enumerate(raw_pts):
                    v = pt - center
                    angle = np.arctan2(np.dot(v, y_i), np.dot(v, x_i))
                    angles_indices.append((angle, idx_pt))
                angles_indices.sort(key=lambda x: x[0])
                sorted_pts = raw_pts[[idx for (_, idx) in angles_indices]]
                sorted_pts = self.helpers.ensure_closed(sorted_pts)
                refined_pts = self.helpers.refine_loop(self.pv.PolyData(sorted_pts), n_points=self.stp_check(self.n_loop_points, 200), smoothing=self.stp_check(self.loop_smoothing, 0), spline_degree=3)
                cross_sections_scaffold.append(refined_pts)

            self.status_label.setText("Status: Preparing surface curve generation...")
            subset_points = self.helpers.select_evenly_spaced_subset(refined_loopA_pts, small_N=self.stp_check(self.n_subset_points, 20))
            subset_thetas = self.helpers.compute_theta_for_subset_points(subset_points, centerpoint_A, x_vecs[0], y_vecs[0])
            subset_r_initial = np.sqrt(np.sum((subset_points - centerpoint_A) ** 2, axis=1))

            self.status_label.setText("Status: Generating surface curves...")
            QApplication.processEvents()
            surface_curves = self.helpers.generate_surface_curves(
                cross_sections_scaffold=cross_sections_scaffold,
                centerline_points=self.final_centerline,
                n_vecs=n_vecs,
                x_vecs=x_vecs,
                y_vecs=y_vecs,
                subset_thetas=subset_thetas,
                subset_r_initial=subset_r_initial,
                subset_points=subset_points
            )

            self.status_label.setText("Status: Smoothing surface curves...")
            QApplication.processEvents()
            trimmed_surface_curves = []
            for curve in surface_curves:
                trimmed = self.helpers.trim_end(curve, self.stp_check(self.trim_points, 0))
                trimmed_surface_curves.append(trimmed)

            smoothed_surface_curves = [self.helpers.smooth_surface_curve(curve, s=self.stp_check(self.surfacecurves_s, 0.2), k=3, n_interp=self.stp_check(self.n_centerline_points, 200)) for curve in trimmed_surface_curves]

            # Store for potential export
            self.surface_curves = smoothed_surface_curves

            # 5. Display everything
            self.status_label.setText("Status: Rendering visualization...")
            QApplication.processEvents()
            self.plotter.add_mesh(self.surf_poly, color="lightblue", opacity=0.5, label="Surface Mesh")
            self.plotter.add_mesh(self.final_centerline_poly, color="magenta", line_width=3, label="Centerline")
            self.plotter.add_mesh(refined_loopA_poly, color="red", line_width=2, label="Loop A")
            self.plotter.add_mesh(refined_loopB_poly, color="green", line_width=2, label="Loop B")

            subset_poly = self.pv.PolyData(subset_points)
            self.plotter.add_mesh(subset_poly, color="red", point_size=5, render_points_as_spheres=True, label="Subset Points")

            for idx, curve in enumerate(smoothed_surface_curves):
                poly = self.helpers.create_polyline(curve, closed=False)
                self.plotter.add_mesh(poly, color="cyan", line_width=3, label=f"Surface Curve {idx}" if idx == 0 else None)

            for i in range(0, len(cross_sections_scaffold), 5):
                cs = cross_sections_scaffold[i]
                if cs is None:
                    continue
                cs_poly = self.helpers.create_polyline(cs, closed=True)
                self.plotter.add_mesh(cs_poly, color="blue", line_width=1, label=f"Cross Section {i}" if i == 0 else None)

            for contour in contours_A + contours_B:
                poly = self.helpers.create_polyline(contour, closed=True)
                self.plotter.add_mesh(poly, color="yellow", line_width=2, opacity=0.8)

            self.plotter.add_legend(bcolor="white")
            self.plotter.reset_camera()
                
            self.status_label.setText("Status: Processing complete. Centerline and surface curves generated.")
            
            # Enable buttons now that we have data
            self.btn_second_process.setEnabled(True)
            self.export_btn.setEnabled(True)
                
        except Exception as e:
            logging.exception("Error during file processing")
            self.status_label.setText("Status: Processing failed - see terminal for details")

    def clear_plot(self):
        self.plotter.clear()
        self.status_label.setText("Status: Plot Cleared")

    def stp_check(self, argument, default):
        return argument if self.accept_stp else default

    def generate_centerline(self):
        """Show marching record diagnostic view (only useful if data already exists)"""
        import logging
        try:
            # Check if data exists from file processing
            if not hasattr(self, 'final_centerline') or self.final_centerline is None:
                self.status_label.setText("Please load a file first.")
                return
            
            # Clear and show only the marching record visualization (diagnostic view)
            self.plotter.clear()
            self.helpers.plot_marching_record(
                self.surf_poly,
                self.final_centerline,
                self.loopA,
                self.loopB,
                self.marching_record_forward,
                step=self.stp_check(self.marching_record_step_input.value(), 5),
                plotter=self.plotter
            )

            self.status_label.setText("Marching record diagnostic view shown.")
            
            # Enable surface curves button since centerline data exists
            self.btn_second_process.setEnabled(True)

        except Exception as e:
            import logging
            logging.exception("Error showing marching record")
            self.status_label.setText("Error: See log in terminal")

    def generate_surface_curves(self):
        """Show full visualization (same as after file loading - for convenience)"""
        if not hasattr(self, 'surface_curves') or self.surface_curves is None:
            self.status_label.setText("Please load a file first.")
            return
        
        # Re-run the same visualization that was shown after file processing
        self.process_loaded_file()

    def export_to_visualizer(self):
        from PyQt5.QtWidgets import QMessageBox
        if self.final_centerline is None or self.surface_curves is None:
            QMessageBox.warning(self, "No Data", "No centerline/surface curves to export. Please generate them first.")
            return
        main_window = self.window()
        field_viz_tab = None
        if main_window and hasattr(main_window, 'tabs'):
            for i in range(main_window.tabs.count()):
                widget = main_window.tabs.widget(i)
                if widget.__class__.__name__ == 'MagneticFieldVisualizer':
                    field_viz_tab = widget
                    break
        if field_viz_tab:
            field_viz_tab.set_coil_data(self.final_centerline, self.surface_curves, self.surf_poly)
            for i in range(main_window.tabs.count()):
                if main_window.tabs.widget(i) is field_viz_tab:
                    main_window.tabs.setCurrentIndex(i)
                    break
            QMessageBox.information(self, "Export", "Data exported to Field Visualizer tab.")
        else:
            QMessageBox.information(self, "Export", "Data exported (but could not find Field Visualizer tab to update).")


class OptimizationTab(QWidget):
    """Combined tab for coil parameters, cross-section, volume, and population"""
    def __init__(self, field_viz_tab=None, parent=None):
        super().__init__(parent)
        self.field_viz_tab = field_viz_tab
        self.population_list = []
        self.performance_data = None
        self.selected_coil = None
        self.initUI()
        
    def initUI(self):
        # Main layout: left panel for controls, right panel for visualization
        main_layout = QHBoxLayout()
        
        # Left panel with tabs for parameters
        left_panel = QVBoxLayout()
        
        # Create the tabs for the different sections
        tabs = QTabWidget()
        tabs.setMaximumWidth(500)
        
        # Coil Parameters Tab
        coil_tab = self.create_coil_tab()
        tabs.addTab(coil_tab, "Coil Parameters")
        
        # Cross Section Tab - Hidden for now (redundant with r0 in parameters)
        # cross_tab = self.create_cross_tab()
        # tabs.addTab(cross_tab, "Cross Section")
        
        # Volume Tab
        volume_tab = self.create_volume_tab()
        tabs.addTab(volume_tab, "Volume")
        
        # Population Tab
        pop_tab = self.create_pop_tab()
        tabs.addTab(pop_tab, "Population")
        
        left_panel.addWidget(tabs)
        
        # Button layout
        btn_layout = QVBoxLayout()
        self.visualize_btn = QPushButton("Visualize Base Coil")
        self.generate_btn = QPushButton("Generate Population")
        self.export_btn = QPushButton("Export to Visualizer")
        
        btn_layout.addWidget(self.visualize_btn)
        btn_layout.addWidget(self.generate_btn)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addStretch()
        
        left_panel.addLayout(btn_layout)
        
        # Right panel for visualization
        right_panel = QVBoxLayout()
        self.vtk_widget = QtInteractor(self)
        self.vtk_widget.setMinimumWidth(600)
        right_panel.addWidget(self.vtk_widget)
        
        # Add both panels to main layout
        main_layout.addLayout(left_panel)
        main_layout.addLayout(right_panel, 1)
        
        # Connect signals
        self.visualize_btn.clicked.connect(self.visualize_base_coil)
        self.generate_btn.clicked.connect(self.generate_population)
        self.export_btn.clicked.connect(self.export_to_visualizer)
        
        self.setLayout(main_layout)
        
    def create_coil_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        title = QLabel("Base Coil Parameters")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        fields = [
            ('radius_y', base_coil_params['radius_y'], 
             'min_radius_y', default_coil_bounds['radius_y']['min'], 
             'max_radius_y', default_coil_bounds['radius_y']['max']),
            ('turns', base_coil_params['turns'], 
             'min_turns', default_coil_bounds['turns']['min'], 
             'max_turns', default_coil_bounds['turns']['max']),
            ('length', base_coil_params['length'], 
             'min_length', default_coil_bounds['length']['min'], 
             'max_length', default_coil_bounds['length']['max']),
            ('alpha', base_coil_params['alpha'], 
             'min_alpha', default_coil_bounds['alpha']['min'], 
             'max_alpha', default_coil_bounds['alpha']['max']),
            ('r0', base_coil_params['r0'], 
             'min_r0', default_coil_bounds['r0']['min'], 
             'max_r0', default_coil_bounds['r0']['max'])
        ]
        
        for field in fields:
            hbox = QHBoxLayout()
            label_text = field[0]
            if field[0] == 'r0':
                label_text = "r0 (Wire Radius)"
            hbox.addWidget(QLabel(label_text + ":"))
            hbox.addWidget(self.create_line_edit(field[0], str(field[1])))
            hbox.addWidget(QLabel("min:"))
            hbox.addWidget(self.create_line_edit(field[2], str(field[3])))
            hbox.addWidget(QLabel("max:"))
            hbox.addWidget(self.create_line_edit(field[4], str(field[5])))
            layout.addLayout(hbox)
        
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Min Spacing between turns:"))
        self.min_spacing_edit = QLineEdit("0.75")
        hbox.addWidget(self.min_spacing_edit)
        hbox.addWidget(QLabel("(x-distance between turns, 0.8 recommended)"))
        layout.addLayout(hbox)
        
        layout.addStretch()
        
        return tab
        
    def create_cross_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        title = QLabel("Cross Section Parameters")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Wire Radius (r₀):"))
        self.r0_edit = QLineEdit(str(base_cross_params['r0']))
        hbox.addWidget(self.r0_edit)
        hbox.addWidget(QLabel("min:"))
        self.min_r0_edit = QLineEdit(str(default_cross_bounds['r0']['min']))
        hbox.addWidget(self.min_r0_edit)
        hbox.addWidget(QLabel("max:"))
        self.max_r0_edit = QLineEdit(str(default_cross_bounds['r0']['max']))
        hbox.addWidget(self.max_r0_edit)
        layout.addLayout(hbox)
        
        note = QLabel("Note: Additional cross-section parameters can be added later.")
        note.setStyleSheet("font-size: 8px;")
        layout.addWidget(note)
        
        return tab
        
    def create_volume_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        title = QLabel("Volume Parameters (Cylindrical)")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        layout.addWidget(QLabel("Volume is centered at the coil's center (0,0,0)."))
        
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Cylinder Radius:"))
        self.vol_radius_edit = QLineEdit(str(default_volume['radius']))
        hbox.addWidget(self.vol_radius_edit)
        hbox.addWidget(QLabel("Length:"))
        self.vol_length_edit = QLineEdit(str(default_volume['length']))
        hbox.addWidget(self.vol_length_edit)
        layout.addLayout(hbox)
        
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Cylinder Axis (x,y,z):"))
        self.axis_x_edit = QLineEdit(str(default_volume['axis_x']))
        self.axis_y_edit = QLineEdit(str(default_volume['axis_y']))
        self.axis_z_edit = QLineEdit(str(default_volume['axis_z']))
        hbox.addWidget(self.axis_x_edit)
        hbox.addWidget(self.axis_y_edit)
        hbox.addWidget(self.axis_z_edit)
        layout.addLayout(hbox)
        
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Spacing:"))
        self.spacing_edit = QLineEdit(str(default_volume['spacing']))
        hbox.addWidget(self.spacing_edit)
        layout.addLayout(hbox)
        
        return tab
        
    def create_pop_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        title = QLabel("Population Settings")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Number of Coils:"))
        self.pop_size_edit = QLineEdit("50")
        hbox.addWidget(self.pop_size_edit)
        layout.addLayout(hbox)
        
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Enter Coil ID to Visualize:"))
        self.coil_id_edit = QLineEdit()
        self.view_coil_btn = QPushButton("View Selected Coil")
        hbox.addWidget(self.coil_id_edit)
        hbox.addWidget(self.view_coil_btn)
        layout.addLayout(hbox)
        
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Performance Plot:"))
        self.param_combo = QComboBox()
        self.param_combo.addItems(["none", "radius_y", "turns", "length", "alpha", "r0"])
        self.plot_btn = QPushButton("Show Performance Plot")
        hbox.addWidget(self.param_combo)
        hbox.addWidget(self.plot_btn)
        layout.addLayout(hbox)
        
        self.view_coil_btn.clicked.connect(self.view_selected_coil)
        self.plot_btn.clicked.connect(self.show_performance_plot)
        
        return tab
        
    def create_line_edit(self, name, default=""):
        le = QLineEdit(default)
        le.setObjectName(name)
        return le
        
    def get_params(self):
        try:
            # Get r0 from coil parameters tab
            r0_value = float(self.findChild(QLineEdit, 'r0').text())
            min_r0_value = float(self.findChild(QLineEdit, 'min_r0').text())
            max_r0_value = float(self.findChild(QLineEdit, 'max_r0').text())
            
            coil_params = {
                'radius_y': float(self.findChild(QLineEdit, 'radius_y').text()),
                'turns': float(self.findChild(QLineEdit, 'turns').text()),
                'length': float(self.findChild(QLineEdit, 'length').text()),
                'alpha': float(self.findChild(QLineEdit, 'alpha').text()),
                'r0': r0_value,
            }
            
            coil_bounds = {
                'radius_y': {
                    'min': float(self.findChild(QLineEdit, 'min_radius_y').text()),
                    'max': float(self.findChild(QLineEdit, 'max_radius_y').text())
                },
                'turns': {
                    'min': float(self.findChild(QLineEdit, 'min_turns').text()),
                    'max': float(self.findChild(QLineEdit, 'max_turns').text())
                },
                'length': {
                    'min': float(self.findChild(QLineEdit, 'min_length').text()),
                    'max': float(self.findChild(QLineEdit, 'max_length').text())
                },
                'alpha': {
                    'min': float(self.findChild(QLineEdit, 'min_alpha').text()),
                    'max': float(self.findChild(QLineEdit, 'max_alpha').text())
                },
                'r0': {
                    'min': min_r0_value,
                    'max': max_r0_value
                }
            }
            
            cross_params = {
                'r0': r0_value
            }
            
            cross_bounds = {
                'r0': {
                    'min': min_r0_value,
                    'max': max_r0_value
                }
            }
            
            volume = {
                'radius': float(self.vol_radius_edit.text()),
                'length': float(self.vol_length_edit.text()),
                'axis_x': float(self.axis_x_edit.text()),
                'axis_y': float(self.axis_y_edit.text()),
                'axis_z': float(self.axis_z_edit.text()),
                'spacing': float(self.spacing_edit.text())
            }
            
            min_spacing = float(self.min_spacing_edit.text())
            
            return {
                'coil_params': coil_params,
                'coil_bounds': coil_bounds,
                'cross_params': cross_params,
                'cross_bounds': cross_bounds,
                'volume': volume,
                'min_spacing': min_spacing
            }
            
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Invalid parameter value: {str(e)}")
            return None

    def visualize_base_coil(self):
        params_dict = self.get_params()
        if params_dict is None:
            return
        
        try:
            combined_params = params_dict['coil_params']
            volume = params_dict['volume']
            
            coil_pts = generate_base_coil(combined_params)
            coil_pts, _ = center_coil(coil_pts)
            surface_curves = generate_surface_curves(coil_pts, {'r0': combined_params['r0']})
            
            # Visualize in optimization tab
            self.plot_coil_in_optimization_tab(coil_pts, surface_curves, volume)
            
            # Also set in field visualization tab
            if self.field_viz_tab:
                self.field_viz_tab.set_coil_data(coil_pts, surface_curves)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Visualization failed: {str(e)}")

    def plot_coil_in_optimization_tab(self, coil_pts, surface_curves, volume):
        """Visualize the coil in the optimization tab's 3D viewer"""
        plotter = self.vtk_widget
        plotter.clear()
        
        # Create centerline
        centerline = pv.PolyData(coil_pts)
        centerline.lines = np.hstack([[coil_pts.shape[0]], np.arange(coil_pts.shape[0])])
        plotter.add_mesh(centerline, color="magenta", line_width=3, label="Centerline")
        
        # Create surface curves
        for curve in surface_curves:
            if len(curve) < 2:
                continue
            curve = np.array(curve, dtype=np.float32)
            curve_poly = pv.PolyData(curve)
            lines = np.hstack([[curve.shape[0]], np.arange(curve.shape[0])])
            curve_poly.lines = lines
            plotter.add_mesh(curve_poly, color="cyan", line_width=1, label="Surface Curves")
        
        # Create volume visualization
        origin = np.array([0, 0, 0])
        axis = np.array([volume['axis_x'], volume['axis_y'], volume['axis_z']])
        axis = axis / np.linalg.norm(axis)
        cylinder = pv.Cylinder(
            center=origin, 
            direction=axis, 
            radius=volume['radius'], 
            height=volume['length']
        )
        plotter.add_mesh(cylinder, color="green", opacity=0.3, label="Volume")
        
        plotter.add_axes()
        plotter.reset_camera()
        plotter.render()

    def generate_population(self):
        params_dict = self.get_params()
        if params_dict is None:
            return
        
        try:
            pop_size = int(self.pop_size_edit.text())
            if pop_size <= 0:
                raise ValueError("Population size must be positive")
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", str(e))
            return
        
        coil_params = params_dict['coil_params']
        coil_bounds = params_dict['coil_bounds']
        cross_params = params_dict['cross_params']
        cross_bounds = params_dict['cross_bounds']
        volume = params_dict['volume']
        min_spacing = params_dict['min_spacing']
        
        sample_points = get_volume_sample_points(volume)
        
        avg_Bx_list, var_Bx_list, coil_ids = [], [], []
        self.population_list = []
        
        for i in range(pop_size):
            valid = False
            iter_count = 0
            max_iter = 50
            
            while not valid and iter_count < max_iter:
                iter_count += 1
                new_center_params = randomize_params(coil_params, coil_bounds)
                new_cross_params = randomize_params(cross_params, cross_bounds)
                
                condition = ((new_center_params['length'] * np.sqrt(2/3)) - 
                            (2 * new_cross_params['r0'] * (new_center_params['turns'] + 0.5))
                            ) / (new_center_params['turns'] + 0.5)
                            
                if min_spacing <= condition:
                    valid = True
                    
            if not valid:
                continue
            
            new_center_params['r0'] = new_cross_params['r0']
            combined_params = new_center_params
            
            coil_pts = generate_base_coil(combined_params)
            coil_pts, _ = center_coil(coil_pts)
            avg_Bx, var_Bx = evaluate_coil(coil_pts, sample_points)
            
            avg_Bx_list.append(avg_Bx)
            var_Bx_list.append(var_Bx)
            coil_ids.append(i)
            
            self.population_list.append({
                'coil_id': i,
                'center_params': new_center_params,
                'cross_params': new_cross_params,
                'coil_points': coil_pts,
                'avg_Bx': avg_Bx,
                'var_Bx': var_Bx
            })
        
        self.performance_data = {"avg": avg_Bx_list, "var": var_Bx_list, "ids": coil_ids}
        
        plt.figure()
        plt.scatter(avg_Bx_list, var_Bx_list, c='blue', alpha=0.7)
        for i, txt in enumerate(coil_ids):
            plt.annotate(str(txt), (avg_Bx_list[i], var_Bx_list[i]), fontsize=8)
        plt.xlabel("Average |Bₓ|")
        plt.ylabel("Variance of Bₓ")
        plt.title("Coil Population Performance")
        plt.grid(True)
        plt.show(block=False)
        
        # Update field visualization with the first coil in the population
        if self.population_list:
            self.selected_coil = self.population_list[0]
            coil_pts = self.selected_coil['coil_points']
            surface_curves = generate_surface_curves(coil_pts, self.selected_coil['cross_params'])
            
            # Visualize in optimization tab
            self.plot_coil_in_optimization_tab(coil_pts, surface_curves, volume)
            
            # Also set in field visualization tab
            if self.field_viz_tab:
                self.field_viz_tab.set_coil_data(coil_pts, surface_curves)

    def view_selected_coil(self):
        if not self.population_list:
            QMessageBox.warning(self, "No Population", "No coil population available. Generate a population first.")
            return
        
        try:
            coil_id = int(self.coil_id_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please enter a valid integer Coil ID.")
            return
        
        selected = None
        for coil in self.population_list:
            if coil['coil_id'] == coil_id:
                selected = coil
                break
                
        if selected is None:
            QMessageBox.warning(self, "Not Found", f"Coil ID {coil_id} not found in the population.")
            return
        
        self.selected_coil = selected
        
        # Always plot the coil geometry
        try:
            params_dict = self.get_params()
            if not params_dict:
                return
                
            volume = params_dict['volume']
            coil_pts = selected['coil_points']
            surface_curves = generate_surface_curves(coil_pts, selected['cross_params'])
            
            # Visualize in optimization tab
            self.plot_coil_in_optimization_tab(coil_pts, surface_curves, volume)
            
            # Also set in field visualization tab
            if self.field_viz_tab:
                self.field_viz_tab.set_coil_data(coil_pts, surface_curves)
        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", str(e))
        
        param_info = f"Coil ID: {selected['coil_id']}\n\nCenterline Parameters:\n"
        for key, val in selected['center_params'].items():
            param_info += f"  {key}: {val}\n"
        param_info += "\nCross-Section Parameters:\n"
        for key, val in selected['cross_params'].items():
            param_info += f"  {key}: {val}\n"
        
        msg = QMessageBox()
        msg.setWindowFlags(Qt.WindowStaysOnTopHint)
        msg.setWindowTitle("Coil Parameters")
        msg.setText(param_info)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def show_performance_plot(self):
        if not self.performance_data:
            QMessageBox.warning(self, "No Data", "Generate a population first")
            return
            
        selected_param = self.param_combo.currentText()
        fig = plt.figure()
        
        if selected_param == "none":
            plt.scatter(self.performance_data["avg"], self.performance_data["var"], c='blue')
            plt.xlabel("Average |Bₓ|")
            plt.ylabel("Variance")
        else:
            ax = fig.add_subplot(111, projection='3d')
            z = [c['center_params'][selected_param] for c in self.population_list]
            ax.scatter(self.performance_data["avg"], self.performance_data["var"], z)
            ax.set_zlabel(selected_param)
        
        plt.title("Performance")
        plt.grid(True)
        fig.canvas.manager.window.activateWindow()
        plt.show(block=False)

    def export_to_visualizer(self):
        if self.selected_coil is None:
            QMessageBox.warning(self, "No Coil Selected", "Please generate a population and select a coil first.")
            return
        
        if not self.field_viz_tab:
            QMessageBox.warning(self, "Visualizer Not Available", "Field visualization tab is not initialized.")
            return
        
        # Set the coil data in the visualizer tab
        coil_pts = self.selected_coil['coil_points']
        surface_curves = generate_surface_curves(coil_pts, self.selected_coil['cross_params'])
        self.field_viz_tab.set_coil_data(coil_pts, surface_curves)
        
        # Switch to visualizer tab
        main_window = self.window()
        if main_window and hasattr(main_window, 'tabs'):
            main_window.tabs.setCurrentIndex(2)  # Switch to Field Visualization tab
        
        QMessageBox.information(
            self, "Export Successful", 
            "Coil exported to Field Visualizer tab.\n"
            "You can now compute and visualize magnetic fields."
        )


# ---------------------------
# Main Window
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("RF Coil Design Suite")
        self.setGeometry(100, 100, 1400, 900)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        
        # Tab 1: Mesh Processor
        self.mesh_processor_tab = MeshProcessorTab()
        self.tabs.addTab(self.mesh_processor_tab, "Mesh Processor")
        
        # Tab 2: Optimization
        self.field_viz_tab = MagneticFieldVisualizer()
        self.optimization_tab = OptimizationTab(self.field_viz_tab)
        self.tabs.addTab(self.optimization_tab, "Optimization")
        
        # Tab 3: Field Visualization
        self.tabs.addTab(self.field_viz_tab, "Field Visualization")
        
        layout.addWidget(self.tabs)
        
        # Exit button
        exit_btn = QPushButton("Exit")
        exit_btn.clicked.connect(self.close)
        layout.addWidget(exit_btn)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    print("Application started")
    sys.exit(app.exec_())