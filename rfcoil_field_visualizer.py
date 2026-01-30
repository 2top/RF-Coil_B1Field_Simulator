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
                             QGroupBox, QRadioButton, QFileDialog, QGridLayout, QDoubleSpinBox, QSpinBox, QButtonGroup,
                             QFormLayout, QSizePolicy, QCheckBox)
from PyQt5.QtCore import Qt
from pyvistaqt import QtInteractor


# ---------------------------
# Global Constants
# ---------------------------
MAX_INT = 2147483647
MAX_DOUBLE = sys.float_info.max

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
        self.current_view_mode = None  # Track current view: 'region', 'field', or 'basic'
        
        self.main_layout = QHBoxLayout(self)
        self.setLayout(self.main_layout)
        
        # Left panel with fixed width
        left_panel = QWidget()
        left_panel.setMinimumWidth(450)
        left_panel.setMaximumWidth(450)
        self.controls_layout = QVBoxLayout(left_panel)
        
        # Create tabs for better organization
        self.control_tabs = QTabWidget()
        
        # Tab 1: Region Setup (with Visualize button)
        region_tab = self.create_region_tab()
        self.control_tabs.addTab(region_tab, "Region Setup")
        
        # Tab 2: Field Computation (with Compute button)
        field_tab = self.create_field_computation_tab()
        self.control_tabs.addTab(field_tab, "Field Computation")
        
        self.controls_layout.addWidget(self.control_tabs)
        
        # Visualization controls - always visible at bottom
        self.controls_layout.addWidget(QLabel("<b>Visualization Controls:</b>"))
        
        # Field display mode
        display_layout = QHBoxLayout()
        display_layout.addWidget(QLabel("Display Mode:"))
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
        self.field_type_dropdown.currentIndexChanged.connect(self.plot_magnetic_field)
        display_layout.addWidget(self.field_type_dropdown)
        self.controls_layout.addLayout(display_layout)
        
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Vector Scale:"))
        self.vector_scale_input = QLineEdit("1.0")
        self.vector_scale_input.setFixedWidth(60)
        self.vector_scale_input.editingFinished.connect(self.plot_magnetic_field)
        scale_layout.addWidget(self.vector_scale_input)
        scale_layout.addStretch()
        self.controls_layout.addLayout(scale_layout)
        
        # Visibility checkboxes in compact grid
        vis_grid = QGridLayout()
        vis_grid.setHorizontalSpacing(5)
        vis_grid.setVerticalSpacing(3)
        
        self.hide_centerline_checkbox = QCheckBox("Hide Centerline")
        self.hide_centerline_checkbox.toggled.connect(self.refresh_current_view)
        vis_grid.addWidget(self.hide_centerline_checkbox, 0, 0)
        
        self.hide_surface_curves_checkbox = QCheckBox("Hide Surf. Curves")
        self.hide_surface_curves_checkbox.toggled.connect(self.refresh_current_view)
        vis_grid.addWidget(self.hide_surface_curves_checkbox, 0, 1)
        
        self.hide_coil_geometry_checkbox = QCheckBox("Hide Geometry")
        self.hide_coil_geometry_checkbox.toggled.connect(self.refresh_current_view)
        vis_grid.addWidget(self.hide_coil_geometry_checkbox, 1, 0)
        
        self.hide_region_checkbox = QCheckBox("Hide Region")
        self.hide_region_checkbox.toggled.connect(self.refresh_current_view)
        vis_grid.addWidget(self.hide_region_checkbox, 1, 1)
        
        self.hide_grid_checkbox = QCheckBox("Hide Grid")
        self.hide_grid_checkbox.toggled.connect(self.refresh_current_view)
        vis_grid.addWidget(self.hide_grid_checkbox, 2, 0)
        
        self.hide_axes_checkbox = QCheckBox("Hide Axes")
        self.hide_axes_checkbox.toggled.connect(self.refresh_current_view)
        vis_grid.addWidget(self.hide_axes_checkbox, 2, 1)
        
        self.controls_layout.addLayout(vis_grid)
        
        # Grid settings
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(QLabel("Grid:"))
        self.grid_spacing_input = QLineEdit("1")
        self.grid_spacing_input.setFixedWidth(40)
        self.grid_spacing_input.editingFinished.connect(self.refresh_current_view)
        grid_layout.addWidget(self.grid_spacing_input)
        grid_layout.addWidget(QLabel("Size:"))
        self.grid_side_length_input = QLineEdit("20")
        self.grid_side_length_input.setFixedWidth(40)
        self.grid_side_length_input.editingFinished.connect(self.refresh_current_view)
        grid_layout.addWidget(self.grid_side_length_input)
        grid_layout.addStretch()
        self.controls_layout.addLayout(grid_layout)
        
        self.controls_layout.addStretch()
        
        self.main_layout.addWidget(left_panel)
        
        self.vtk_widget = QtInteractor(self)
        self.vtk_widget.setMinimumWidth(600)
        self.main_layout.addWidget(self.vtk_widget, stretch=1)
        self.plotter = self.vtk_widget
        
        # Connect signals after all widgets are created
        self.setup_signal_connections()
        
        self.update_inputs()
    
    def create_region_tab(self) -> QWidget:
        """Create the Region Setup tab with region type and parameters"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        # Region type selector
        layout.addWidget(QLabel("<b>Region Type:</b>"))
        self.region_dropdown = QComboBox()
        self.region_dropdown.addItems(["Volume", "Plane", "Line"])
        self.region_dropdown.currentIndexChanged.connect(self.on_region_type_changed)
        layout.addWidget(self.region_dropdown)
        
        layout.addSpacing(10)
        
        # Volume region parameters
        self.volume_group = QGroupBox("Volume Region Parameters")
        vol_layout = QVBoxLayout()
        vol_layout.setSpacing(5)
        
        vol_grid = QGridLayout()
        vol_grid.setHorizontalSpacing(10)
        vol_grid.setVerticalSpacing(5)
        
        vol_grid.addWidget(QLabel("X Min:"), 0, 0)
        self.xmin_input = QLineEdit("-5")
        self.xmin_input.setFixedWidth(60)
        vol_grid.addWidget(self.xmin_input, 0, 1)
        vol_grid.addWidget(QLabel("X Max:"), 0, 2)
        self.xmax_input = QLineEdit("17")
        self.xmax_input.setFixedWidth(60)
        vol_grid.addWidget(self.xmax_input, 0, 3)
        
        vol_grid.addWidget(QLabel("Y Min:"), 1, 0)
        self.ymin_input = QLineEdit("0")
        self.ymin_input.setFixedWidth(60)
        vol_grid.addWidget(self.ymin_input, 1, 1)
        vol_grid.addWidget(QLabel("Y Max:"), 1, 2)
        self.ymax_input = QLineEdit("0")
        self.ymax_input.setFixedWidth(60)
        vol_grid.addWidget(self.ymax_input, 1, 3)
        
        vol_grid.addWidget(QLabel("Z Min:"), 2, 0)
        self.zmin_input = QLineEdit("-5")
        self.zmin_input.setFixedWidth(60)
        vol_grid.addWidget(self.zmin_input, 2, 1)
        vol_grid.addWidget(QLabel("Z Max:"), 2, 2)
        self.zmax_input = QLineEdit("17")
        self.zmax_input.setFixedWidth(60)
        vol_grid.addWidget(self.zmax_input, 2, 3)
        
        vol_layout.addLayout(vol_grid)
        
        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(QLabel("Spacing:"))
        self.points_spacing_input = QLineEdit("0.5")
        self.points_spacing_input.setFixedWidth(60)
        spacing_layout.addWidget(self.points_spacing_input)
        spacing_layout.addStretch()
        vol_layout.addLayout(spacing_layout)
        
        self.volume_group.setLayout(vol_layout)
        layout.addWidget(self.volume_group)
        
        # Plane region parameters
        self.plane_group = QGroupBox("Plane Region Parameters")
        plane_layout = QVBoxLayout()
        plane_layout.setSpacing(5)
        
        self.plane_origin_input = self.create_compact_input("Origin (x,y,z):", "0,0,0", plane_layout)
        self.plane_normal_input = self.create_compact_input("Normal (x,y,z):", "0,0,1", plane_layout)
        
        dims_layout = QHBoxLayout()
        dims_layout.addWidget(QLabel("Length:"))
        self.plane_length_input = QLineEdit("10")
        self.plane_length_input.setFixedWidth(50)
        dims_layout.addWidget(self.plane_length_input)
        dims_layout.addWidget(QLabel("Width:"))
        self.plane_width_input = QLineEdit("10")
        self.plane_width_input.setFixedWidth(50)
        dims_layout.addWidget(self.plane_width_input)
        dims_layout.addStretch()
        plane_layout.addLayout(dims_layout)
        
        self.plane_points_spacing_input = self.create_compact_input("Spacing:", "0.5", plane_layout)
        
        self.plane_group.setLayout(plane_layout)
        layout.addWidget(self.plane_group)
        
        # Line region parameters
        self.line_group = QGroupBox("Line Region Parameters")
        line_layout = QVBoxLayout()
        line_layout.setSpacing(5)
        
        self.line_start_input = self.create_compact_input("Start (x,y,z):", "0,0,0", line_layout)
        self.line_end_input = self.create_compact_input("End (x,y,z):", "1,1,1", line_layout)
        self.line_points_input = self.create_compact_input("Number of Points:", "10", line_layout)
        
        self.line_group.setLayout(line_layout)
        layout.addWidget(self.line_group)
        
        layout.addSpacing(10)
        
        # Exclusion controls
        layout.addWidget(QLabel("<b>Point Filtering:</b>"))
        self.exclude_interior_checkbox = QCheckBox("Exclude points inside surface mesh")
        self.exclude_interior_checkbox.setChecked(True)
        layout.addWidget(self.exclude_interior_checkbox)
        self.exclusion_distance_input = self.create_compact_input("Exclusion distance:", "0.25", layout)
        
        layout.addStretch()
        
        # Visualize Region button at bottom
        self.visualize_region_button = QPushButton("Visualize Region")
        self.visualize_region_button.clicked.connect(self.visualize_region)
        self.visualize_region_button.setToolTip("Preview the computation region")
        self.visualize_region_button.setStyleSheet("font-weight: bold; padding: 8px;")
        layout.addWidget(self.visualize_region_button)
        
        return tab
    
    def create_field_computation_tab(self) -> QWidget:
        """Create the Field Computation tab with field settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        # Field type selection
        layout.addWidget(QLabel("<b>Field Type:</b>"))
        self.field_compute_dropdown = QComboBox()
        self.field_compute_dropdown.addItems(["B-field (magnetic)", "E-field (electric)"])
        self.field_compute_dropdown.currentIndexChanged.connect(self.on_field_type_changed)
        layout.addWidget(self.field_compute_dropdown)
        
        # Frequency (for E-field)
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Frequency (MHz):"))
        self.freq_input = QLineEdit("400")
        self.freq_input.setFixedWidth(80)
        self.freq_input.setEnabled(False)
        freq_layout.addWidget(self.freq_input)
        freq_layout.addStretch()
        layout.addLayout(freq_layout)
        
        layout.addSpacing(10)
        
        # Current source selection
        layout.addWidget(QLabel("<b>Current Path:</b>"))
        source_layout = QHBoxLayout()
        self.surface_curves_radio = QRadioButton("Surface Curves")
        self.centerline_radio = QRadioButton("Centerline")
        self.surface_curves_radio.setChecked(True)
        source_layout.addWidget(self.surface_curves_radio)
        source_layout.addWidget(self.centerline_radio)
        source_layout.addStretch()
        layout.addLayout(source_layout)
        
        current_layout = QHBoxLayout()
        current_layout.addWidget(QLabel("Coil Current (A):"))
        self.current_input = QLineEdit("1.0")
        self.current_input.setFixedWidth(80)
        current_layout.addWidget(self.current_input)
        current_layout.addStretch()
        layout.addLayout(current_layout)
        
        layout.addSpacing(10)
        
        # Inductance calculation
        layout.addWidget(QLabel("<b>Inductance Calculation:</b>"))
        self.inductance_button = QPushButton("Compute Inductance from B-Field")
        self.inductance_button.clicked.connect(self.compute_inductance)
        layout.addWidget(self.inductance_button)
        
        note_lbl = QLabel("<i>Volume must fully enclose the coil</i>")
        note_lbl.setStyleSheet("font-size: 9px; color: gray;")
        layout.addWidget(note_lbl)
        
        ind_layout = QHBoxLayout()
        ind_layout.addWidget(QLabel("Inductance:"))
        self.inductance_value_lbl = QLabel("—")
        ind_layout.addWidget(self.inductance_value_lbl)
        ind_layout.addStretch()
        layout.addLayout(ind_layout)
        
        layout.addSpacing(10)
        
        # Export settings
        layout.addWidget(QLabel("<b>Export Settings:</b>"))
        export_layout = QHBoxLayout()
        export_layout.addWidget(QLabel("Base Name:"))
        self.export_basename_input = QLineEdit("field_export")
        self.export_basename_input.setFixedWidth(150)
        export_layout.addWidget(self.export_basename_input)
        export_layout.addStretch()
        layout.addLayout(export_layout)
        
        layout.addStretch()
        
        # Action buttons at bottom
        self.compute_field_button = QPushButton("Compute Field")
        self.compute_field_button.clicked.connect(self.compute_field)
        self.compute_field_button.setToolTip("Calculate electromagnetic field in the specified region")
        layout.addWidget(self.compute_field_button)
        
        self.export_button = QPushButton("Export Data")
        self.export_button.clicked.connect(self.export_data)
        self.export_button.setStyleSheet("font-weight: bold; padding: 8px;")
        self.export_button.setToolTip("Export computed field data to CSV file")
        layout.addWidget(self.export_button)
        
        return tab
    
    def create_compact_input(self, label_text: str, default_value: str, parent_layout) -> QLineEdit:
        """Create a compact horizontal input field"""
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel(label_text))
        input_field = QLineEdit(default_value)
        input_field.setFixedWidth(120)
        h_layout.addWidget(input_field)
        h_layout.addStretch()
        parent_layout.addLayout(h_layout)
        return input_field
    
    def on_field_type_changed(self):
        """Enable/disable frequency input based on field type"""
        is_efield = self.field_compute_dropdown.currentText() == "E-field (electric)"
        self.freq_input.setEnabled(is_efield)
    
    def setup_signal_connections(self):
        """Setup all signal connections after widgets are created"""
        self.exclude_interior_checkbox.toggled.connect(self.refresh_current_view)
        self.exclusion_distance_input.editingFinished.connect(self.refresh_current_view)

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
        # Respect the current view mode - don't automatically switch back to field view
        if self.current_view_mode == 'field' and self.last_B is not None:
            # Only refresh field if we're explicitly in field view mode
            self.plot_magnetic_field()
        elif self.current_view_mode == 'region':
            # Refresh region visualization
            self.visualize_region()
        else:
            # Default: refresh basic coil visualization
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
        self.current_view_mode = 'region'  # Set mode to region view
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

        self.current_view_mode = 'field'  # Set mode to field view
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

        self.current_view_mode = 'field'  # Set mode to field view
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
# Mesh Processor Tab
# ---------------------------

class MeshProcessorTab(QWidget):

    """Mesh processing tab with STL/STEP import, meshing, centerline, surface curve extraction, and visualization, matching convert.py logic and UI, with Export to Visualizer."""
    def __init__(self, parent=None):
        super().__init__(parent)
        import helpers
        self.helpers = helpers
        self.pv = pv
        
        self.setWindowTitle("Surface Mesh Processor")
        self.setGeometry(100, 100, 1400, 900)

        self.input_file = None
        self.msh_file = "generated_mesh.msh"
        self.source_file = None
        self.can_regenerate = False

        self.element_size = 0.15
        self.max_element_size_factor = 2.0
        self.feature_angle = 75
        self.trim_points = 0
        self.centerline_s = 0.01
        self.surfacecurves_s = 0.01
        self.loop_smoothing = 0.0
        self.n_centerline_points = 500
        self.n_loop_points = 100
        self.n_subset_points = 20
        self.marching_record_step = 5

        self.surf_poly = None
        self.raw_centerline_forward = None
        self.loopA = None
        self.marching_record_forward = None
        self.final_centerline = None
        self.final_centerline_poly = None

        self.accept_stl = True
        self.accept_stp = False
        self.accept_msh = False

        # Export state - not used currently
        self.export_dir = os.getcwd()
        self.export_basename = "output"

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        self.setLayout(main_layout)

        # Left side panel
        left_panel = QVBoxLayout()

        # Shared header (fixed across all tabs)
        self.loaded_file = QLabel("No file loaded")
        self.loaded_file.setStyleSheet("color: green;")
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: red;")
        self.status_label.setWordWrap(True)

        left_panel.addWidget(self.loaded_file)
        left_panel.addWidget(self.status_label)
        left_panel.addSpacing(8)

        # Tabs (store as member so it can be accessed later if needed)
        self.control_tabs = QTabWidget()
        self.control_tabs.setTabPosition(QTabWidget.West)

        # ---- File Tab ----
        file_tab = QWidget()
        file_layout = QFormLayout(file_tab)

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

        file_layout.addRow(QLabel("File Type Filter:"), self.stl_radio)
        file_layout.addRow("", self.stp_radio)
        file_layout.addRow("", self.msh_radio)

        file_layout.addRow(
            self._spacer_label(),
            self.btn_with_tooltip("Load File", self.load_file, "Load a .stp, .stl, or .msh file"),
        )

        # Can't apparently make a button that is hidden/shown easily, so it's in a container
        self.regen_container = QWidget()
        regen_layout = QHBoxLayout(self.regen_container)
        regen_layout.setContentsMargins(0, 0, 0, 0)
        self.regen_btn = self.btn_with_tooltip("Regenerate Mesh", self.regenerate_current_file, "Regenerate mesh with updated settings")
        regen_layout.addWidget(self.regen_btn)
        self.regen_container.setVisible(False)
        file_layout.addRow("", self.regen_container)

        file_layout.addRow(
            self._spacer_label(),
            self.btn_with_tooltip("Clean Plot", self.clear_plot, "Clear all meshes from the plotter"),
        )

        self.element_size_input = self._create_doublespinbox(
            0.01, self.element_size,
            lambda val: setattr(self, "element_size", val),
            tooltip="Base element size for meshing",
            enabled=False,
        )
        file_layout.addRow(QLabel("Element Size"), self.element_size_input)

        self.max_element_size_factor_input = self._create_doublespinbox(
            0.1, self.max_element_size_factor,
            lambda val: setattr(self, "max_element_size_factor", val),
            tooltip="Max element size multiplier",
            enabled=False,
        )
        file_layout.addRow(QLabel("Max Size Factor"), self.max_element_size_factor_input)

        self.workflow_btn = QCheckBox("Enable Step-by-Step Walkthrough")
        self.workflow_btn.setChecked(False)
        self.workflow_btn.setToolTip("Enable guided workflow for processing steps")
        file_layout.addRow(QLabel("Workflow"), self.workflow_btn)

        # ---- Centerline Tab ----
        center_tab = QWidget()
        center_layout = QFormLayout(center_tab)

        self.feature_angle_input = self._create_spinbox(
            1, self.feature_angle,
            lambda val: setattr(self, "feature_angle", val),
            tooltip="Feature angle determines where the end loops of the coil are defined",
            enabled=True,
        )
        center_layout.addRow(QLabel("Feature Angle"), self.feature_angle_input)

        self.centerline_s_input = self._create_doublespinbox(
            0.001, self.centerline_s,
            lambda val: setattr(self, "centerline_s", val),
            tooltip="Centerline smoothing parameter",
            enabled=True,
        )
        center_layout.addRow(QLabel("Centerline Smooth (s)"), self.centerline_s_input)

        self.n_centerline_points_input = self._create_spinbox(
            10, self.n_centerline_points,
            lambda val: setattr(self, "n_centerline_points", val),
            tooltip="Number of points in final centerline",
            enabled=True,
        )
        center_layout.addRow(QLabel("Centerline Points"), self.n_centerline_points_input)

        self.marching_record_step_input = self._create_spinbox(
            1, self.marching_record_step,
            lambda val: setattr(self, "marching_record_step", val),
            tooltip="Recording step for marching algorithm",
            enabled=True,
        )

        center_layout.addRow(QLabel("Marching Record Step"), self.marching_record_step_input)

        self.trim_points_input = self._create_spinbox(
            1, self.trim_points,
            lambda val: setattr(self, "trim_points", val),
            tooltip="Trim N points off loop B end of extracted coil mesh centerline",
            enabled=True,
        )
        center_layout.addRow(QLabel("Trim Points"), self.trim_points_input)

        self.generate_centerline_btn = self.btn_with_tooltip("Generate Centerline", self.generate_centerline, "Generate centerline for mesh")
        self.generate_centerline_btn.setEnabled(False)
        center_layout.addRow(self._spacer_label(), self.generate_centerline_btn)

        # ---- Surface Curves Tab ----
        sc_tab = QWidget()
        sc_layout = QFormLayout(sc_tab)

        self.surfacecurves_s_input = self._create_doublespinbox(
            0.001, self.surfacecurves_s,
            lambda val: setattr(self, "surfacecurves_s", val),
            tooltip="Surface curves smoothing parameter",
            enabled=True,
        )
        sc_layout.addRow(QLabel("Surface Curves Smooth (s)"), self.surfacecurves_s_input)

        self.loop_smoothing_input = self._create_doublespinbox(
            0.1, self.loop_smoothing,
            lambda val: setattr(self, "loop_smoothing", val),
            tooltip="Loop smoothing strength",
            enabled=True,
        )
        sc_layout.addRow(QLabel("Loop Smoothing"), self.loop_smoothing_input)

        self.n_loop_points_input = self._create_spinbox(
            1, self.n_loop_points,
            lambda val: setattr(self, "n_loop_points", val),
            tooltip="Number of points in each loop",
            enabled=True,
        )
        sc_layout.addRow(QLabel("Loop Points"), self.n_loop_points_input)

        self.n_subset_points_input = self._create_spinbox(
            1, self.n_subset_points,
            lambda val: setattr(self, "n_subset_points", val),
            tooltip="Number of subset points",
            enabled=True,
        )
        sc_layout.addRow(QLabel("Subset Points"), self.n_subset_points_input)

        self.surface_curves_btn = self.btn_with_tooltip("Generate Surface Curves", self.generate_surface_curves, "Generate surface curves from centerline")
        self.surface_curves_btn.setEnabled(False)
        sc_layout.addRow(self._spacer_label(), self.surface_curves_btn)

        # ---- Export Tab ----
        export_tab = QWidget()
        export_layout = QFormLayout(export_tab)

        self.export_btn = self.btn_with_tooltip("Export to Visualizer", self.export_to_visualizer, "Export processed mesh and centerline to Field Visualizer")
        self.export_btn.setEnabled(False)
        export_layout.addRow(self._spacer_label(), self.export_btn)

        # Wrap tabs in scroll areas
        def wrap_in_scroll(widget: QWidget) -> QScrollArea:
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setWidget(widget)
            return sa

        self.control_tabs.addTab(wrap_in_scroll(file_tab), "File")
        self.control_tabs.addTab(wrap_in_scroll(center_tab), "Centerline")
        self.control_tabs.addTab(wrap_in_scroll(sc_tab), "Surface Curves")
        self.control_tabs.addTab(wrap_in_scroll(export_tab), "Export")

        # Start on File tab
        self.control_tabs.setCurrentIndex(0)

        left_panel.addWidget(self.control_tabs)
        left_panel.setStretchFactor(self.control_tabs, 1)

        # Keep the left panel from eating the whole window :D
        self.control_tabs.setMinimumWidth(340)
        self.control_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(480)
        main_layout.addWidget(left_widget, 2)

        self.plotter = QtInteractor(self)
        main_layout.addWidget(self.plotter, 5)

    # ------------------------- WIDGET FACTORIES ------------------------- #
    def _create_spinbox(self, step, value, callback, tooltip=None, enabled=False, max_value=MAX_INT, min_value=0):
        spin = QSpinBox()
        spin.setRange(min_value, max_value)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.valueChanged.connect(callback)
        spin.setEnabled(enabled)
        if tooltip:
            spin.setToolTip(tooltip)
        return spin

    def _create_doublespinbox(self, step, value, callback, tooltip=None, enabled=False, max_value=MAX_DOUBLE, min_value=0.0):
        spin = QDoubleSpinBox()
        spin.setRange(min_value, max_value)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.valueChanged.connect(callback)
        spin.setEnabled(enabled)
        if tooltip:
            spin.setToolTip(tooltip)
        return spin

    def _spacer_label(self):
        return QLabel("")

    def btn_with_tooltip(self, text, slot, tooltip, visible=True):
        btn = QPushButton(text)
        btn.clicked.connect(slot)
        btn.setToolTip(tooltip)
        # Ensure buttons fit/scale within left panel
        sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.setSizePolicy(sp)
        btn.setMinimumHeight(34)
        btn.setVisible(visible)
        return btn
    
    # ------------------------ HELPER FUNCTIONS ------------------------- #

    def pvqt_interactor(self):
        from pyvistaqt import QtInteractor
        return QtInteractor(self)

    def set_accept_stl(self):
        self.accept_stl = True
        self.accept_stp = False
        self.accept_msh = False
        self.element_size_input.setEnabled(False)
        self.max_element_size_factor_input.setEnabled(False)

    def set_accept_stp(self):
        self.accept_stl = False
        self.accept_stp = True
        self.accept_msh = False
        self.element_size_input.setEnabled(True)
        self.max_element_size_factor_input.setEnabled(True)

    def set_accept_msh(self):
        self.accept_stl = False
        self.accept_stp = False
        self.accept_msh = True
        self.element_size_input.setEnabled(False)
        self.max_element_size_factor_input.setEnabled(False)

    def clear_plot(self):
        self.plotter.clear()
        self.status_label.setText("Status: Plot Cleared")
        
    def stp_check(self, argument, default):
        return argument if (self.accept_stp or self.accept_msh) else default

    # ------------------------- MESH FILE CHOOSER ------------------------- #
    
    def default_mesh_path_for_input(self, input_path: str) -> str:
        base_dir = os.path.dirname(input_path)
        stem = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(base_dir, f"{stem}.msh")

    def prompt_existing_mesh_action(self, msh_path: str):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Mesh already exists")
        msg.setText(f"A mesh file already exists:\n\n{msh_path}\n\nWhat would you like to do?")

        btn_load = msg.addButton("Load existing mesh", QMessageBox.AcceptRole)
        btn_load.setToolTip("Load the existing mesh file")

        btn_rename = msg.addButton("Edit name…", QMessageBox.ActionRole)
        btn_rename.setToolTip("Choose a different name for the new mesh file")

        btn_override = msg.addButton("Override", QMessageBox.DestructiveRole)
        btn_override.setToolTip("Regenerate the mesh and overwrite the existing file")

        btn_cancel = msg.addButton("Cancel", QMessageBox.RejectRole) # This is used, albeit implicitly
        btn_cancel.setToolTip("Cancel the operation")

        msg.exec() 

        clicked = msg.clickedButton()
        if clicked == btn_load:
            return "load"
        if clicked == btn_rename:
            return "rename"
        if clicked == btn_override:
            return "override"
        return None

    def choose_mesh_output_path(self, suggested_path: str) -> str | None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Choose Mesh Output (.msh)",
            suggested_path,
            "Gmsh Mesh (*.msh);;All Files (*)"
        )
        if not path:
            return None
        if not path.lower().endswith(".msh"):
            path += ".msh"
        return path

    # ------------------------- CORE FUNCTIONALITY ------------------------- #

    def load_file(self):
        file_dialog = QFileDialog()
        file_filter = ""

        self.element_size = self.element_size_input.value()
        self.max_element_size_factor = self.max_element_size_factor_input.value()
        
        if self.accept_stl:
            file_filter = "STL Files (*.stl)"
        elif self.accept_stp:
            file_filter = "STEP Files (*.stp *.step)"
        elif self.accept_msh:
            file_filter = "MSH Files (*.msh *.mesh)"
        else:
            file_filter = "" # This should never happen, but just in case
            logging.error("No file type selected. Please select a file type before loading a file.")
        
        previous_file = self.input_file 
        self.input_file, _ = file_dialog.getOpenFileName(self, "Open File", "", file_filter)
        if not self.input_file:
            return
        
        if self.accept_stp:
            self.source_file = self.input_file 
            self.can_regenerate = True
        else:
            self.source_file = None
            self.can_regenerate = False
        
        # Ensure that if a new file is loaded, the centerline and surface curves buttons are disabled until re-processed
        if previous_file != self.input_file:
            self.generate_centerline_btn.setEnabled(False)
            self.surface_curves_btn.setEnabled(False)
            self.export_btn.setEnabled(False)

            self.generate_centerline_btn.setText("Generate Centerline")
            self.surface_curves_btn.setText("Generate Surface Curves")

        self.loaded_file.setText(f"Loaded: {os.path.basename(self.input_file)}")
        
        # Display Mesh visualization
        try:
            self.clear_plot()

            self.msh_file = self.default_mesh_path_for_input(self.input_file)

            # Only do this logic for STP/STL files
            if self.accept_stp or self.accept_stl:
                if os.path.exists(self.msh_file):
                    while os.path.exists(self.msh_file):
                        action = self.prompt_existing_mesh_action(self.msh_file)
                        if action is None:
                            return  # user cancelled

                        if action == "rename":
                            new_path = self.choose_mesh_output_path(self.msh_file)
                            if new_path is None:
                                return  # user cancelled
                            self.msh_file = new_path
                            self.generate_mesh(self.msh_file)
                            break # QFileDialog will ensure no overwrite

                        elif action == "load":
                            self.load_existing_msh(self.msh_file)
                            break

                        elif action == "override":
                            self.generate_mesh(self.msh_file)
                            break
                else: # it ain't there, so just generate normally
                    self.generate_mesh(self.msh_file)
            else: # MSH file selected directly
                self.load_existing_msh(self.input_file)

            self.regen_container.setVisible(self.can_regenerate)
            if self.workflow_btn.isChecked():
                self.control_tabs.setCurrentIndex(1)
            return

        except Exception as e:
            logging.exception("Error during file load")
            self.status_label.setText("Error: See log in terminal")

    def regenerate_current_file(self):
        try: 
            self.element_size = self.element_size_input.value()
            self.max_element_size_factor = self.max_element_size_factor_input.value()
            self.clear_plot()
            self.status_label.setText("Status: Regenerating mesh...")
            self.surf_poly = self.helpers.load_surface_mesh(
                self.source_file,
                self.msh_file,
                self.element_size,
                self.max_element_size_factor,
                self.status_label
            )
            self.status_label.setText("Status: Regenerated mesh from current file")
            self.update_plot()
            if self.workflow_btn.isChecked():
                self.control_tabs.setCurrentIndex(1) 
        except Exception as e:
            logging.exception("Error during file load")
            self.status_label.setText("Error: See log in terminal")
    
    def generate_mesh(self, msh_path: str):
        # Generate/Load mesh from an STP/STL file 
        self.status_label.setText("Status: Generating mesh...")
        self.surf_poly = self.helpers.load_surface_mesh(
            self.input_file,
            msh_path or self.msh_file,
            self.stp_check(self.element_size, 0.15),
            self.stp_check(self.max_element_size_factor, 2.0),
            self.status_label
        )
        if self.surf_poly is None or self.surf_poly.n_points == 0:
            self.status_label.setText("Error: Mesh generation failed")
            return

        self.loaded_file.setText(f"Loaded: {os.path.basename(self.msh_file)}")
        self.status_label.setText("Status: Mesh generated successfully")
        self.update_plot()

    def load_existing_msh(self, msh_path: str):
        self.status_label.setText("Status: Loading existing mesh...")

        # Ensure that the program is using accept_msh mode
        if self.accept_stp or self.accept_stl:
            self.msh_radio.setChecked(True)
            self.accept_msh = True # Tbh this might already be set by msh_radio, but just to be safe
            self.accept_stp = False
            self.accept_stl = False

        self.surf_poly = self.helpers.load_surface_mesh(
            msh_path,   # treat the .msh as the "input"
            msh_path,   # and the mesh path
            self.stp_check(self.element_size, 0.15),
            self.stp_check(self.max_element_size_factor, 2.0),
            self.status_label
        )
        if self.surf_poly is None or self.surf_poly.n_points == 0:
            self.status_label.setText("Error: Failed to load surface mesh")
            return

        self.loaded_file.setText(f"Loaded: {os.path.basename(msh_path)}")
        self.status_label.setText("Status: Loaded existing mesh")
        self.update_plot()

    def update_plot(self):
            # Draw immediately in the embedded QtInteractor
            self.status_label.setText("Status: Displaying mesh...")
            self.plotter.clear()
            self.plotter.add_mesh(
                self.surf_poly,
                color="lightsteelblue",
                show_edges=True,
                smooth_shading=True,
                label="Loaded Surface"
            )
            try:
                # avoid stacking legends across loads
                self.plotter.remove_legend()
            except Exception:
                pass
            self.plotter.reset_camera()
            self.plotter.render()

            self.status_label.setText("Status: File loaded and displayed")
            self.generate_centerline_btn.setEnabled(True)

    # ------------------------- CENTERLINE CACHING AND COMPUTING ------------------------- #
    def _ensure_centerline_cache(self):
        """
        Initialize cache containers used to avoid recomputing expensive steps.
        Being called lazily to avoid unnecessary memory usage.
        To be clear, this is just storing the values in memory; NOT serializing to disk.
        """
        if not hasattr(self, "_cl_cache"):
            self._cl_cache = {
                "surf_poly_id": None,

                # Stage 1: loops
                "loops": None,
                "loops_sig": None,

                # Stage 2: marching + raw centerline
                "marching_record_forward": None,
                "raw_centerline_forward": None,
                "raw_sig": None,

                # Stage 3: post-processed centerline + polyline
                "smoothed_fwd": None,
                "final_centerline": None,
                "final_centerline_poly": None,
                "post_sig": None,
            }

    def _invalidate_centerline_cache(self, stage: str = "all"):
        """
        stage in {"all", "loops", "raw", "post"}.
        Invalidates stage and all downstream stages.
        """
        self._ensure_centerline_cache()

        if stage in ("all", "loops"):
            self._cl_cache["loops"] = None
            self._cl_cache["loops_sig"] = None
            stage = "raw"

        if stage in ("all", "raw"):
            self._cl_cache["marching_record_forward"] = None
            self._cl_cache["raw_centerline_forward"] = None
            self._cl_cache["raw_sig"] = None
            stage = "post"

        if stage in ("all", "post"):
            self._cl_cache["smoothed_fwd"] = None
            self._cl_cache["final_centerline"] = None
            self._cl_cache["final_centerline_poly"] = None
            self._cl_cache["post_sig"] = None

    def _read_centerline_params_from_ui(self):
        """
        Reads current GUI values into instance vars
        """
        self.trim_points = self.trim_points_input.value()
        self.feature_angle = self.feature_angle_input.value()
        self.marching_record_step = self.marching_record_step_input.value()
        self.centerline_s = self.centerline_s_input.value()
        self.n_centerline_points = self.n_centerline_points_input.value()

        if self.trim_points > self.n_centerline_points - 2:
            self.trim_points = self.n_centerline_points - 2
            logging.warning("Trim points exceeds centerline points; adjusting trim points to %d", self.trim_points)

    def _sync_cache_mesh_identity(self):
        """
        If the mesh changed (new file loaded), invalidate everything.
        """
        self._ensure_centerline_cache()
        current_id = id(self.surf_poly) if self.surf_poly is not None else None
        if self._cl_cache["surf_poly_id"] != current_id:
            self._cl_cache["surf_poly_id"] = current_id
            self._invalidate_centerline_cache("all")

    def _compute_end_loops_cached(self):
        """
        Stage 1: end loop extraction.
        Cached on (surf_poly_id, feature_angle).
        """
        self._ensure_centerline_cache()
        sig = (self._cl_cache["surf_poly_id"], float(self.feature_angle))

        if self._cl_cache["loops"] is not None and self._cl_cache["loops_sig"] == sig:
            self.loopA, self.loopB = self._cl_cache["loops"]
            return self.loopA, self.loopB

        self.status_label.setText("Status: Extracting end loops...")
        QApplication.processEvents()
        loopA, loopB = self.helpers.extract_coil_end_loops(self.surf_poly, self.feature_angle, self.status_label)

        if loopA is None or loopB is None:
            self._invalidate_centerline_cache("loops")
            return None, None

        self.loopA, self.loopB = loopA, loopB
        self._cl_cache["loops"] = (loopA, loopB)
        self._cl_cache["loops_sig"] = sig

        self._invalidate_centerline_cache("raw")
        return loopA, loopB

    def _compute_raw_centerline_cached(self):
        """
        Stage 2: marching rings + raw centerline.
        Cached on (surf_poly_id, loops_sig).
        """
        self._ensure_centerline_cache()

        loopA, loopB = self._compute_end_loops_cached()
        if loopA is None or loopB is None:
            return None, None

        sig = (
            self._cl_cache["surf_poly_id"],
            self._cl_cache["loops_sig"],
        )

        if (
            self._cl_cache["raw_centerline_forward"] is not None
            and self._cl_cache["marching_record_forward"] is not None
            and self._cl_cache["raw_sig"] == sig
        ):
            self.raw_centerline_forward = self._cl_cache["raw_centerline_forward"]
            self.marching_record_forward = self._cl_cache["marching_record_forward"]
            return self.raw_centerline_forward, self.marching_record_forward

        # Compute marching rings
        self.status_label.setText("Status: Computing marching rings...")
        QApplication.processEvents()
        moving_sections = self.helpers.compute_marching_rings(self.surf_poly, loopA, loopB, self.status_label)

        # Compute raw centerline from marching rings
        self.status_label.setText("Status: Computing raw centerline...")
        QApplication.processEvents()
        vertices = self.surf_poly.points
        centers = []
        centers.append(loopA.points.mean(axis=0))
        total_sections = len(moving_sections)
        for i, section in enumerate(moving_sections):
            if i % max(1, total_sections // 10) == 0:
                    progress = int((i / total_sections) * 100)
                    self.status_label.setText(f"Status: Generating cross-sections... {progress}% complete")
                    QApplication.processEvents()
            if section:
                coords = np.array([vertices[v] for v in section])
                centers.append(coords.mean(axis=0))
        centers.append(loopB.points.mean(axis=0))

        raw_centerline = np.vstack(centers) if centers else None
        if raw_centerline is None:
            logging.error("Failed to compute raw centerline from marching rings.")
            self._invalidate_centerline_cache("raw")
            return None, None

        self.raw_centerline_forward = raw_centerline
        self.marching_record_forward = moving_sections

        self._cl_cache["raw_centerline_forward"] = raw_centerline
        self._cl_cache["marching_record_forward"] = moving_sections
        self._cl_cache["raw_sig"] = sig

        self._invalidate_centerline_cache("post")
        return raw_centerline, moving_sections

    def _postprocess_and_plot_centerline_cached(self):
        """
        Stage 3: smooth + trim + polyline + plotting.
        Cached on (raw_sig, trim_points, centerline_s, n_centerline_points, marching_record_step).
        """
        self._ensure_centerline_cache()

        raw_centerline, marching_record = self._compute_raw_centerline_cached()
        if raw_centerline is None or marching_record is None:
            return False

        sig = (
            self._cl_cache["raw_sig"],
            int(self.trim_points),
            float(self.centerline_s),
            int(self.n_centerline_points),
            int(self.marching_record_step),
        )

        if self._cl_cache["final_centerline"] is None or self._cl_cache["post_sig"] != sig:

            self.status_label.setText("Status: Smoothing centerline...")
            QApplication.processEvents()
            smoothed = self.helpers.smooth_centerline(
                raw_centerline,
                s=self.centerline_s,
                k=3,
                n_interp=self.n_centerline_points,
            )

            self.status_label.setText("Status: Trimming centerline...")
            QApplication.processEvents()
            filtered = self.helpers.trim_end(smoothed, self.trim_points)

            # Polyline
            self.status_label.setText("Status: Finalizing centerline...")
            QApplication.processEvents()
            final_poly = self.helpers.create_polyline(filtered, closed=False)

            # Store on instance
            self.smoothed_fwd = smoothed
            self.final_centerline = filtered
            self.final_centerline_poly = final_poly

            # Cache
            self._cl_cache["smoothed_fwd"] = smoothed
            self._cl_cache["final_centerline"] = filtered
            self._cl_cache["final_centerline_poly"] = final_poly
            self._cl_cache["post_sig"] = sig
        else:
            # Restore cached post outputs
            self.smoothed_fwd = self._cl_cache["smoothed_fwd"]
            self.final_centerline = self._cl_cache["final_centerline"]
            self.final_centerline_poly = self._cl_cache["final_centerline_poly"]

        # Plot results
        self.status_label.setText("Status: Plotting marching record...")
        QApplication.processEvents()
        logging.info("Plotting marching record...")

        self.helpers.plot_marching_record(
            self.surf_poly,
            self.final_centerline,
            self.loopA,
            self.loopB,
            self.marching_record_forward,
            step=self.marching_record_step,
            plotter=self.plotter,
        )

        self.status_label.setText("Current marching record shown. Please confirm before continuing.")
        return True

    def generate_centerline(self):
        try:
            if not self.input_file or self.surf_poly is None:
                self.status_label.setText("Status: No input file loaded!")
                return

            # Clear existing plots
            self.plotter.clear()

            # Read GUI parameters into instance vars
            self._read_centerline_params_from_ui()

            # Detect mesh change and invalidate caches if needed
            self._sync_cache_mesh_identity()

            # Run pipeline (cached)
            ok = self._postprocess_and_plot_centerline_cached()
            if not ok:
                self.status_label.setText(
                    "Could not identify exactly two end loops. Try lowering feature angle or check mesh."
                )
                return

            self.generate_centerline_btn.setText("Regenerate Centerline")
            self.status_label.setText("Status: Centerline generated successfully.")

            if self.workflow_btn.isChecked():  
                self.control_tabs.setCurrentIndex(2)
            self.surface_curves_btn.setEnabled(True)

        except Exception:
            logging.exception("Error during centerline generation")
            self.status_label.setText("Error: See log in terminal")

    # ---------------------- SURFACE CURVES FUNCTIONALITY ---------------------- #
    def generate_surface_curves(self):
        try: 
            if not self.input_file:
                self.status_label.setText("No input file loaded!")
                return
             
            self.plotter.clear()  # Clear previous scene if any

            # Read GUI parameters
            self.trim_points = self.trim_points_input.value()
            self.centerline_s = self.centerline_s_input.value()
            self.surfacecurves_s = self.surfacecurves_s_input.value()
            self.loop_smoothing = self.loop_smoothing_input.value()
            self.n_centerline_points = self.n_centerline_points_input.value()
            self.n_loop_points = self.n_loop_points_input.value()
            self.n_subset_points = self.n_subset_points_input.value()

            self.status_label.setText("Status: Refining end loops...")
            loopA_ordered = self.helpers.order_loop_points_pca(self.loopA.points)
            refined_loopA_pts = self.helpers.refine_loop(self.pv.PolyData(loopA_ordered), n_points=self.n_loop_points, smoothing=self.loop_smoothing, spline_degree=3)
            refined_loopA_poly = self.helpers.create_polyline(refined_loopA_pts, closed=True)

            loopB_ordered = self.helpers.order_loop_points_pca(self.loopB.points)
            refined_loopB_pts = self.helpers.refine_loop(self.pv.PolyData(loopB_ordered), n_points=self.n_loop_points, smoothing=self.loop_smoothing, spline_degree=3)
            refined_loopB_poly = self.helpers.create_polyline(refined_loopB_pts, closed=True)

            self.status_label.setText("Status: Building reference frames...")
            centerpoint_A = self.final_centerline[0]
            centerpoint_B = self.final_centerline[-1]
            # contours_A = self.helpers.generate_intermediate_contours(refined_loopA_pts, centerpoint_A, n_contours=5)
            # contours_B = self.helpers.generate_intermediate_contours(refined_loopB_pts, centerpoint_B, n_contours=5)

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
                refined_pts = self.helpers.refine_loop(self.pv.PolyData(sorted_pts), n_points=self.n_loop_points, smoothing=self.loop_smoothing, spline_degree=3)
                cross_sections_scaffold.append(refined_pts)

            self.status_label.setText("Status: Preparing surface curve generation...")
            subset_points = self.helpers.select_evenly_spaced_subset(refined_loopA_pts, small_N=self.n_subset_points)
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

            # Both trim and smooth 
            self.status_label.setText("Status: Smoothing surface curves...")
            QApplication.processEvents()
            trimmed_surface_curves = []
            for curve in surface_curves:
                trimmed = self.helpers.trim_end(curve, self.trim_points)
                trimmed_surface_curves.append(trimmed)

            smoothed_surface_curves = [self.helpers.smooth_surface_curve(curve, s=self.surfacecurves_s, k=3, n_interp=self.n_centerline_points) for curve in trimmed_surface_curves]

            # Store for potential export
            self.surface_curves = smoothed_surface_curves

            # Display everything
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

            # for contour in contours_A + contours_B:
            #     poly = self.helpers.create_polyline(contour, closed=True)
            #     self.plotter.add_mesh(poly, color="yellow", line_width=2, opacity=0.8)

            self.plotter.add_legend(bcolor="white")
            self.plotter.reset_camera()
                
            self.status_label.setText("Status: Processing complete. Finished centerline and surface curves generated.")

            # Store for export and enable button
            self.trimmed_surface_curves = trimmed_surface_curves

            self.surface_curves_btn.setText("Regenerate Surface Curves")
            
            if self.workflow_btn.isChecked():
                self.control_tabs.setCurrentIndex(3)
            self.export_btn.setEnabled(True)

        except Exception as e:
            logging.exception("Error during surface curve generation")
            self.status_label.setText("Error: See log in terminal")

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
        
        # Left panel with scroll area for controls
        left_panel = QWidget()
        left_panel.setMinimumWidth(500)
        left_panel.setMaximumWidth(500)
        left_layout = QVBoxLayout(left_panel)
        
        # Create the tabs for the different sections
        tabs = QTabWidget()
        
        # Coil Parameters Tab
        coil_tab = self.create_coil_tab()
        tabs.addTab(coil_tab, "Coil Parameters")
        
        # Volume Tab
        volume_tab = self.create_volume_tab()
        tabs.addTab(volume_tab, "Volume")
        
        # Population Tab
        pop_tab = self.create_pop_tab()
        tabs.addTab(pop_tab, "Population")
        
        left_layout.addWidget(tabs)
        
        # Actions section at the bottom
        left_layout.addWidget(QLabel("<b>Actions:</b>"))
        
        self.visualize_btn = QPushButton("Visualize Base Coil")
        self.visualize_btn.setToolTip("Visualize the base coil with current parameters")
        left_layout.addWidget(self.visualize_btn)
        
        self.generate_btn = QPushButton("Generate Population")
        self.generate_btn.setToolTip("Generate a population of coils within specified bounds")
        left_layout.addWidget(self.generate_btn)
        
        self.export_btn = QPushButton("Export to Visualizer")
        self.export_btn.setToolTip("Export selected coil to Field Visualization tab")
        self.export_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        left_layout.addWidget(self.export_btn)
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: red; font-size: 10px; padding-top: 5px;")
        left_layout.addWidget(self.status_label)
        
        left_layout.addStretch()
        
        # Right panel for visualization
        right_panel = QVBoxLayout()
        right_label = QLabel("<b>3D Visualization</b>")
        right_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(right_label)
        
        self.vtk_widget = QtInteractor(self)
        self.vtk_widget.setMinimumWidth(600)
        right_panel.addWidget(self.vtk_widget)
        
        # Add both panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addLayout(right_panel, 1)
        
        # Connect signals
        self.visualize_btn.clicked.connect(self.visualize_base_coil)
        self.generate_btn.clicked.connect(self.generate_population)
        self.export_btn.clicked.connect(self.export_to_visualizer)
        
        self.setLayout(main_layout)
        
    def create_coil_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        title = QLabel("<b>Base Coil Parameters</b>")
        title.setStyleSheet("font-size: 13px; padding-bottom: 5px;")
        layout.addWidget(title)
        
        # Create grid layout for parameters
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        
        fields = [
            ('radius_y', 'Radius Y', base_coil_params['radius_y'], 
             default_coil_bounds['radius_y']['min'], default_coil_bounds['radius_y']['max']),
            ('turns', 'Turns', base_coil_params['turns'], 
             default_coil_bounds['turns']['min'], default_coil_bounds['turns']['max']),
            ('length', 'Length', base_coil_params['length'], 
             default_coil_bounds['length']['min'], default_coil_bounds['length']['max']),
            ('alpha', 'Alpha', base_coil_params['alpha'], 
             default_coil_bounds['alpha']['min'], default_coil_bounds['alpha']['max']),
            ('r0', 'r0 (Wire Radius)', base_coil_params['r0'], 
             default_coil_bounds['r0']['min'], default_coil_bounds['r0']['max'])
        ]
        
        row = 0
        for field_name, label_text, default_val, min_val, max_val in fields:
            # Parameter name label
            grid.addWidget(QLabel(label_text + ":"), row, 0)
            
            # Value input
            grid.addWidget(self.create_line_edit(field_name, str(default_val)), row, 1)
            
            # Min label and input
            grid.addWidget(QLabel("Min:"), row, 2)
            grid.addWidget(self.create_line_edit(f'min_{field_name}', str(min_val)), row, 3)
            
            # Max label and input
            grid.addWidget(QLabel("Max:"), row, 4)
            grid.addWidget(self.create_line_edit(f'max_{field_name}', str(max_val)), row, 5)
            
            row += 1
        
        layout.addLayout(grid)
        
        # Min spacing section
        layout.addWidget(QLabel("<b>Constraint:</b>"))
        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(QLabel("Min Spacing (x-distance):"))
        self.min_spacing_edit = QLineEdit("0.75")
        self.min_spacing_edit.setFixedWidth(60)
        spacing_layout.addWidget(self.min_spacing_edit)
        # spacing_layout.addWidget(QLabel("(0.8 recommended)"))
        spacing_layout.addStretch()
        layout.addLayout(spacing_layout)
        
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
        layout.setSpacing(10)
        
        title = QLabel("<b>Volume Parameters (Cylindrical)</b>")
        title.setStyleSheet("font-size: 13px; padding-bottom: 5px;")
        layout.addWidget(title)
        
        info = QLabel("Volume is centered at the coil's center (0,0,0).")
        info.setStyleSheet("font-style: italic; color: gray; padding-bottom: 10px;")
        layout.addWidget(info)
        
        # Cylinder dimensions
        layout.addWidget(QLabel("<b>Cylinder Dimensions:</b>"))
        dims_layout = QHBoxLayout()
        dims_layout.addWidget(QLabel("Radius:"))
        self.vol_radius_edit = QLineEdit(str(default_volume['radius']))
        self.vol_radius_edit.setFixedWidth(80)
        dims_layout.addWidget(self.vol_radius_edit)
        dims_layout.addSpacing(20)
        dims_layout.addWidget(QLabel("Length:"))
        self.vol_length_edit = QLineEdit(str(default_volume['length']))
        self.vol_length_edit.setFixedWidth(80)
        dims_layout.addWidget(self.vol_length_edit)
        dims_layout.addStretch()
        layout.addLayout(dims_layout)
        
        # Cylinder axis
        layout.addWidget(QLabel("<b>Cylinder Axis (x, y, z):</b>"))
        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("x:"))
        self.axis_x_edit = QLineEdit(str(default_volume['axis_x']))
        self.axis_x_edit.setFixedWidth(80)
        axis_layout.addWidget(self.axis_x_edit)
        axis_layout.addWidget(QLabel("y:"))
        self.axis_y_edit = QLineEdit(str(default_volume['axis_y']))
        self.axis_y_edit.setFixedWidth(80)
        axis_layout.addWidget(self.axis_y_edit)
        axis_layout.addWidget(QLabel("z:"))
        self.axis_z_edit = QLineEdit(str(default_volume['axis_z']))
        self.axis_z_edit.setFixedWidth(80)
        axis_layout.addWidget(self.axis_z_edit)
        axis_layout.addStretch()
        layout.addLayout(axis_layout)
        
        # Sampling spacing
        layout.addWidget(QLabel("<b>Sampling:</b>"))
        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(QLabel("Point Spacing:"))
        self.spacing_edit = QLineEdit(str(default_volume['spacing']))
        self.spacing_edit.setFixedWidth(80)
        spacing_layout.addWidget(self.spacing_edit)
        spacing_layout.addStretch()
        layout.addLayout(spacing_layout)
        
        layout.addStretch()
        
        return tab
        
    def create_pop_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        title = QLabel("<b>Population Settings</b>")
        title.setStyleSheet("font-size: 13px; padding-bottom: 5px;")
        layout.addWidget(title)
        
        # Population size
        layout.addWidget(QLabel("<b>Generation:</b>"))
        pop_layout = QHBoxLayout()
        pop_layout.addWidget(QLabel("Number of Coils:"))
        self.pop_size_edit = QLineEdit("50")
        self.pop_size_edit.setFixedWidth(80)
        pop_layout.addWidget(self.pop_size_edit)
        pop_layout.addStretch()
        layout.addLayout(pop_layout)
        
        layout.addSpacing(10)
        
        # Coil selection
        layout.addWidget(QLabel("<b>Coil Selection:</b>"))
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("Coil ID:"))
        self.coil_id_edit = QLineEdit()
        self.coil_id_edit.setFixedWidth(80)
        self.coil_id_edit.setPlaceholderText("Enter ID")
        select_layout.addWidget(self.coil_id_edit)
        self.view_coil_btn = QPushButton("View Selected Coil")
        select_layout.addWidget(self.view_coil_btn)
        select_layout.addStretch()
        layout.addLayout(select_layout)
        
        layout.addSpacing(10)
        
        # Performance plot
        layout.addWidget(QLabel("<b>Performance Analysis:</b>"))
        plot_layout = QHBoxLayout()
        plot_layout.addWidget(QLabel("Parameter:"))
        self.param_combo = QComboBox()
        self.param_combo.addItems(["none", "radius_y", "turns", "length", "alpha", "r0"])
        self.param_combo.setFixedWidth(120)
        plot_layout.addWidget(self.param_combo)
        self.plot_btn = QPushButton("Show Performance Plot")
        plot_layout.addWidget(self.plot_btn)
        plot_layout.addStretch()
        layout.addLayout(plot_layout)
        
        layout.addStretch()
        
        # Connect signals
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
            self.status_label.setText("Status: Generating base coil...")
            QApplication.processEvents()
            
            combined_params = params_dict['coil_params']
            volume = params_dict['volume']
            
            coil_pts = generate_base_coil(combined_params)
            coil_pts, _ = center_coil(coil_pts)
            
            self.status_label.setText("Status: Generating surface curves...")
            QApplication.processEvents()
            surface_curves = generate_surface_curves(coil_pts, {'r0': combined_params['r0']})
            
            self.status_label.setText("Status: Rendering visualization...")
            QApplication.processEvents()
            
            # Visualize in optimization tab
            self.plot_coil_in_optimization_tab(coil_pts, surface_curves, volume)
            
            # Also set in field visualization tab
            if self.field_viz_tab:
                self.field_viz_tab.set_coil_data(coil_pts, surface_curves)
            
            self.status_label.setText("Status: Base coil visualization complete!")
            
        except Exception as e:
            self.status_label.setText(f"Status: Error - {str(e)}")
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
        
        self.status_label.setText("Status: Initializing population generation...")
        QApplication.processEvents()
        
        coil_params = params_dict['coil_params']
        coil_bounds = params_dict['coil_bounds']
        cross_params = params_dict['cross_params']
        cross_bounds = params_dict['cross_bounds']
        volume = params_dict['volume']
        min_spacing = params_dict['min_spacing']
        
        self.status_label.setText("Status: Generating sample points...")
        QApplication.processEvents()
        sample_points = get_volume_sample_points(volume)
        
        avg_Bx_list, var_Bx_list, coil_ids = [], [], []
        self.population_list = []
        
        for i in range(pop_size):
            # Update status every 5 coils or at specific milestones
            if i % max(1, pop_size // 10) == 0 or i == 0:
                progress = int((i / pop_size) * 100)
                self.status_label.setText(f"Status: Generating coil {i+1}/{pop_size} ({progress}% complete)...")
                QApplication.processEvents()
            
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
        
        self.status_label.setText("Status: Creating performance plot...")
        QApplication.processEvents()
        
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
        
        self.status_label.setText("Status: Visualizing first coil...")
        QApplication.processEvents()
        
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
        
        self.status_label.setText(f"Status: Population generation complete! Generated {len(self.population_list)} coils.")

    def view_selected_coil(self):
        if not self.population_list:
            QMessageBox.warning(self, "No Population", "No coil population available. Generate a population first.")
            return
        
        try:
            coil_id = int(self.coil_id_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please enter a valid integer Coil ID.")
            return
        
        self.status_label.setText(f"Status: Searching for coil ID {coil_id}...")
        QApplication.processEvents()
        
        selected = None
        for coil in self.population_list:
            if coil['coil_id'] == coil_id:
                selected = coil
                break
                
        if selected is None:
            self.status_label.setText(f"Status: Coil ID {coil_id} not found")
            QMessageBox.warning(self, "Not Found", f"Coil ID {coil_id} not found in the population.")
            return
        
        self.selected_coil = selected
        
        # Always plot the coil geometry
        try:
            self.status_label.setText(f"Status: Loading coil ID {coil_id}...")
            QApplication.processEvents()
            
            params_dict = self.get_params()
            if not params_dict:
                return
                
            volume = params_dict['volume']
            coil_pts = selected['coil_points']
            
            self.status_label.setText(f"Status: Generating surface curves for coil ID {coil_id}...")
            QApplication.processEvents()
            surface_curves = generate_surface_curves(coil_pts, selected['cross_params'])
            
            self.status_label.setText(f"Status: Rendering coil ID {coil_id}...")
            QApplication.processEvents()
            
            # Visualize in optimization tab
            self.plot_coil_in_optimization_tab(coil_pts, surface_curves, volume)
            
            # Also set in field visualization tab
            if self.field_viz_tab:
                self.field_viz_tab.set_coil_data(coil_pts, surface_curves)
            
            self.status_label.setText(f"Status: Coil ID {coil_id} loaded successfully!")
            
        except Exception as e:
            self.status_label.setText(f"Status: Error loading coil - {str(e)}")
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