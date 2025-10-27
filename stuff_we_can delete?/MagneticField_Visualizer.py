import numpy as np
import pyvista as pv
import os
import signal
import csv

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QCheckBox
)
from PyQt5.QtCore import QCoreApplication

##############################################################################
# Biot-Savart Computation for an Open Coil
##############################################################################
def compute_biot_savart_field(observation_points, coil_points, current=1.0, mu0=4*np.pi*1e-7):
    """
    Approximate the magnetic field B (in Tesla) at the given observation_points
    due to a piecewise-linear coil defined by coil_points (open, end-to-end).

    - observation_points: (M,3) array of coordinates
    - coil_points: (N,3) array of coil coordinates (defining N-1 line segments)
    - current: scalar (A)
    - mu0: vacuum permeability (4e-7*pi in SI)

    Returns: (M,3) array of B-field vectors (Tesla).
    """
    n_seg = len(coil_points) - 1
    if n_seg < 1:
        return np.zeros((len(observation_points), 3), dtype=np.float64)

    # Prepare output array
    B = np.zeros((len(observation_points), 3), dtype=np.float64)

    for i in range(n_seg):
        r1 = coil_points[i]
        r2 = coil_points[i+1]
        dl = r2 - r1
        # Approximation by midpoint of the segment
        mid = 0.5 * (r1 + r2)

        # Vector from midpoint of segment to each observation point
        r_vec = observation_points - mid  # (M, 3)
        cross_vals = np.cross(dl, r_vec)  # (M, 3)
        r_norm = np.linalg.norm(r_vec, axis=1)**3  # (M, )

        valid = r_norm > 1e-20
        B[valid] += cross_vals[valid] / r_norm[valid, np.newaxis]

    # Multiply by constant factor
    B *= (mu0 * current) / (4.0 * np.pi)
    return B

class MagneticFieldVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Magnetic Field Visualizer")
        self.setGeometry(100, 100, 500, 600)

        # Enable Ctrl+C interrupt
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 1) Dropdown for region type
        self.layout.addWidget(QLabel("Select Region Type:"))
        self.region_dropdown = QComboBox()
        self.region_dropdown.addItems(["Volume", "Plane", "Line"])
        self.region_dropdown.currentIndexChanged.connect(self.update_inputs)
        self.layout.addWidget(self.region_dropdown)

        # 2) Region-specific inputs
        self.volume_inputs = []
        self.xmin_input = self.create_input("Xmin:", "-2", self.volume_inputs)
        self.xmax_input = self.create_input("Xmax:", "2", self.volume_inputs)
        self.ymin_input = self.create_input("Ymin:", "-2", self.volume_inputs)
        self.ymax_input = self.create_input("Ymax:", "2", self.volume_inputs)
        self.zmin_input = self.create_input("Zmin:", "-2", self.volume_inputs)
        self.zmax_input = self.create_input("Zmax:", "2", self.volume_inputs)
        self.grid_spacing_input = self.create_input("Grid spacing:", "0.5", self.volume_inputs)

        self.plane_inputs = []
        self.plane_origin_input = self.create_input("Plane Origin (x,y,z):", "0,0,0", self.plane_inputs)
        self.plane_normal_input = self.create_input("Plane Normal Vector (x,y,z):", "0,0,1", self.plane_inputs)
        self.plane_length_input = self.create_input("Plane Length:", "10", self.plane_inputs)
        self.plane_width_input = self.create_input("Plane Width:", "10", self.plane_inputs)
        self.plane_grid_spacing_input = self.create_input("Grid spacing:", "0.5", self.plane_inputs)

        self.line_inputs = []
        self.line_start_input = self.create_input("Line Start (x,y,z):", "0,0,0", self.line_inputs)
        self.line_end_input = self.create_input("Line End (x,y,z):", "1,1,1", self.line_inputs)
        self.line_points_input = self.create_input("Number of Points:", "10", self.line_inputs)

        # 3) Visualize region button
        self.visualize_region_button = QPushButton("Visualize Region")
        self.visualize_region_button.clicked.connect(self.visualize_region)
        self.layout.addWidget(self.visualize_region_button)

        # 4) Checkbox to exclude interior points
        self.exclude_interior_checkbox = QCheckBox("Exclude points inside surface mesh")
        self.exclude_interior_checkbox.setChecked(True)  # Default ON
        self.layout.addWidget(self.exclude_interior_checkbox)

        # 5) Additional controls for B-field
        self.layout.addWidget(QLabel("Coil Current (A):"))
        self.current_input = QLineEdit("1.0")
        self.layout.addWidget(self.current_input)

        self.layout.addWidget(QLabel("Field to Visualize:"))
        self.field_type_dropdown = QComboBox()
        self.field_type_dropdown.clear()
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
        self.layout.addWidget(self.field_type_dropdown)

        self.layout.addWidget(QLabel("Vector Scale (for 'Vector' display):"))
        self.vector_scale_input = QLineEdit("1.0")
        self.layout.addWidget(self.vector_scale_input)

        self.layout.addWidget(QLabel("Color Range Min:"))
        self.color_range_min_input = QLineEdit("")
        self.layout.addWidget(self.color_range_min_input)

        self.layout.addWidget(QLabel("Color Range Max:"))
        self.color_range_max_input = QLineEdit("")
        self.layout.addWidget(self.color_range_max_input)

        # 6) Compute field button
        self.compute_field_button = QPushButton("Compute Magnetic Field")
        self.compute_field_button.clicked.connect(self.compute_magnetic_field)
        self.layout.addWidget(self.compute_field_button)

        # 7) Export data controls
        self.layout.addWidget(QLabel("Export Base File Name:"))
        self.export_basename_input = QLineEdit("field_export")
        self.layout.addWidget(self.export_basename_input)

        self.export_button = QPushButton("Export Data")
        self.export_button.clicked.connect(self.export_data)
        self.layout.addWidget(self.export_button)

        # 8) Exit button
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.exit_application)
        self.layout.addWidget(self.exit_button)

        # Load the surface mesh and centerline
        self.load_data()

        # Setup a 3D Plotter
        self.plotter = pv.Plotter()

        # Show only Volume inputs by default
        self.update_inputs()

        # Data holders for the last calculation
        self.last_region_points = None
        self.last_B = None
        self.last_computed_region_type = None

        # Keep track of how many times each region was computed
        self.region_counts = {"Volume": 0, "Plane": 0, "Line": 0}
        # Keep track of the last set of points used per region (so we can detect changes)
        self.last_region_points_by_type = {"Volume": None, "Plane": None, "Line": None}

        # Re-plot if user changes display type
        self.field_type_dropdown.currentIndexChanged.connect(self.plot_magnetic_field)


    def exit_application(self):
        QCoreApplication.instance().quit()

    def load_data(self):
        """Load surface mesh and centerline data (if available)."""
        surface_mesh_file = "surface_mesh.vtk"
        centerline_file = "averaged_centerline.npy"

        if os.path.exists(surface_mesh_file):
            self.surf_poly = pv.read(surface_mesh_file)
        else:
            print(f"[ERROR] Surface mesh file '{surface_mesh_file}' not found.")
            self.surf_poly = None

        if os.path.exists(centerline_file):
            centerline_points = np.load(centerline_file).astype(np.float32)
            self.centerline = pv.PolyData()
            self.centerline.points = centerline_points
            self.centerline.lines = np.hstack([[len(centerline_points)], np.arange(len(centerline_points))])
        else:
            print(f"[ERROR] Centerline file '{centerline_file}' not found.")
            self.centerline = None

    def create_input(self, label_text, default_value, input_list):
        label = QLabel(label_text)
        input_field = QLineEdit()
        input_field.setText(default_value)
        self.layout.addWidget(label)
        self.layout.addWidget(input_field)
        input_list.append((label, input_field))
        return input_field

    def update_inputs(self):
        region_type = self.region_dropdown.currentText()
        # Hide all
        for inputs in [self.volume_inputs, self.plane_inputs, self.line_inputs]:
            for label, field in inputs:
                label.hide()
                field.hide()

        # Show only the relevant ones
        if region_type == "Volume":
            for label, field in self.volume_inputs:
                label.show()
                field.show()
        elif region_type == "Plane":
            for label, field in self.plane_inputs:
                label.show()
                field.show()
        elif region_type == "Line":
            for label, field in self.line_inputs:
                label.show()
                field.show()

    ########################################################################
    # Method to filter out interior points
    ########################################################################
    def filter_interior_points(self, points):
        """Return only the points outside self.surf_poly (if loaded)."""
        if points is None or len(points) == 0:
            return points
        if self.surf_poly is None:
            # No surface, do nothing
            return points

        test_poly = pv.PolyData(points)
        # Remove 'tolerance' argument if your PyVista/VTK doesn't support it
        enclosed_result = test_poly.select_enclosed_points(self.surf_poly, check_surface=True)
        inside_mask = enclosed_result["SelectedPoints"]  # 1=inside, 0=outside
        outside_points = test_poly.points[inside_mask == 0]
        n_interior = np.count_nonzero(inside_mask == 1)
        if n_interior > 0:
            print(f"[INFO] Removed {n_interior} interior points. {len(outside_points)} remain.")
        return outside_points

    ########################################################################
    # Region Visualization
    ########################################################################
    def visualize_region(self):
        region_type = self.region_dropdown.currentText()
        self.plotter.clear()

        # Base geometry
        if self.surf_poly:
            self.plotter.add_mesh(self.surf_poly, color="lightblue", opacity=0.5, label="Surface Mesh")
        if self.centerline:
            self.plotter.add_mesh(self.centerline, color="magenta", line_width=3, label="Centerline")

        # Add coordinate axes and a 1-unit grey grid
        self.add_coordinate_system()
        self.add_grey_points()

        if region_type == "Volume":
            self.visualize_volume_region()
        elif region_type == "Plane":
            self.visualize_plane_region()
        elif region_type == "Line":
            self.visualize_line_region()

        self.plotter.show(title=f"Region Visualization: {region_type}")

    def visualize_volume_region(self):
        xmin, xmax = float(self.xmin_input.text()), float(self.xmax_input.text())
        ymin, ymax = float(self.ymin_input.text()), float(self.ymax_input.text())
        zmin, zmax = float(self.zmin_input.text()), float(self.zmax_input.text())
        spacing = float(self.grid_spacing_input.text())

        grid_points = []
        x_vals = np.arange(xmin, xmax + spacing, spacing)
        y_vals = np.arange(ymin, ymax + spacing, spacing)
        z_vals = np.arange(zmin, zmax + spacing, spacing)
        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    grid_points.append([x, y, z])
        grid_points = np.array(grid_points, dtype=np.float32)

        if self.exclude_interior_checkbox.isChecked():
            grid_points = self.filter_interior_points(grid_points)

        grid_poly = pv.PolyData(grid_points)
        self.plotter.add_mesh(grid_poly, color="orange", point_size=4,
                              render_points_as_spheres=True, label="Grid Points")

        # Bounding box corners
        corners = np.array([
            [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
        ], dtype=np.float32)
        corner_poly = pv.PolyData(corners)
        self.plotter.add_mesh(corner_poly, color="red", point_size=8,
                              render_points_as_spheres=True, label="Bounding Box")

    def visualize_plane_region(self):
        origin = np.array([float(x) for x in self.plane_origin_input.text().split(",")], dtype=np.float32)
        normal = np.array([float(x) for x in self.plane_normal_input.text().split(",")], dtype=np.float32)
        length = float(self.plane_length_input.text())
        width = float(self.plane_width_input.text())
        spacing = float(self.plane_grid_spacing_input.text())

        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-12:
            raise ValueError("[ERROR] Normal vector cannot be zero.")
        normal /= norm_len

        # Two orthogonal vectors spanning the plane
        u = np.cross(normal, [1, 0, 0])
        if np.linalg.norm(u) < 1e-12:
            u = np.cross(normal, [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        v /= np.linalg.norm(v)

        plane_points = []
        for x in np.arange(-length / 2, length / 2 + spacing, spacing):
            for y in np.arange(-width / 2, width / 2 + spacing, spacing):
                point = origin + x*u + y*v
                plane_points.append(point)
        plane_points = np.array(plane_points, dtype=np.float32)

        if self.exclude_interior_checkbox.isChecked():
            plane_points = self.filter_interior_points(plane_points)

        plane_poly = pv.PolyData(plane_points)
        self.plotter.add_mesh(plane_poly, color="orange", point_size=4,
                              render_points_as_spheres=True, label="Plane Points")

        # Corner markers
        corners = [
            origin - (length/2)*u - (width/2)*v,
            origin + (length/2)*u - (width/2)*v,
            origin + (length/2)*u + (width/2)*v,
            origin - (length/2)*u + (width/2)*v,
        ]
        corners = np.array(corners, dtype=np.float32)
        corner_poly = pv.PolyData(corners)
        self.plotter.add_mesh(corner_poly, color="red", point_size=8,
                              render_points_as_spheres=True, label="Plane Corners")

        # Normal arrow
        normal_arrow = pv.Arrow(start=origin, direction=normal, scale=1.0)
        self.plotter.add_mesh(normal_arrow, color="red", label="Normal Vector")

    def visualize_line_region(self):
        start_point = np.array([float(x) for x in self.line_start_input.text().split(",")], dtype=np.float32)
        end_point = np.array([float(x) for x in self.line_end_input.text().split(",")], dtype=np.float32)
        num_points = int(self.line_points_input.text())

        line_points = np.linspace(start_point, end_point, num_points)
        if self.exclude_interior_checkbox.isChecked():
            line_points = self.filter_interior_points(line_points)

        if len(line_points) > 1:
            line_poly = pv.PolyData(line_points)
            connectivity = np.hstack([[len(line_points)], np.arange(len(line_points))])
            line_poly.lines = connectivity
            self.plotter.add_mesh(line_poly, color="orange", line_width=3, label="Line Region")

            endpoints = np.array([line_points[0], line_points[-1]], dtype=np.float32)
            epoly = pv.PolyData(endpoints)
            self.plotter.add_mesh(epoly, color="red", point_size=8,
                                  render_points_as_spheres=True, label="Line Endpoints")
        else:
            line_poly = pv.PolyData(line_points)
            self.plotter.add_mesh(line_poly, color="orange", point_size=6,
                                  render_points_as_spheres=True, label="Line Points")

    ########################################################################
    # Helpers for reference geometry
    ########################################################################
    def add_grey_points(self):
        grid_points = []
        for x in range(-10, 11):
            for y in range(-10, 11):
                for z in range(-10, 11):
                    grid_points.append([x, y, z])
        grid_points = np.array(grid_points, dtype=np.float32)
        grid_poly = pv.PolyData(grid_points)
        self.plotter.add_mesh(grid_poly, color="grey", point_size=2,
                              render_points_as_spheres=True, label="1-Unit Grid")

    def add_coordinate_system(self):
        origin = np.array([0, 0, 0])
        axes = {
            'X': ([1, 0, 0], 'red'),
            'Y': ([0, 1, 0], 'green'),
            'Z': ([0, 0, 1], 'blue')
        }
        for label, (direction, color) in axes.items():
            arrow = pv.Arrow(start=origin, direction=direction, scale=1.0)
            self.plotter.add_mesh(arrow, color=color, label=label)
            tip = origin + np.array(direction)*1.1
            self.plotter.add_point_labels(
                tip.reshape(1, -1), [label], font_size=10, text_color=color
            )

    ########################################################################
    # Field Computation
    ########################################################################
    def compute_magnetic_field(self):
        """
        Compute Biot-Savart for the region. Each new region_points or region_type
        increments a counter for exporting.
        """
        region_type = self.region_dropdown.currentText()
        print(f"[INFO] Computing magnetic field for region type: {region_type}")

        # Current
        try:
            current = float(self.current_input.text())
        except ValueError:
            print("[ERROR] Invalid current value, reverting to 1.0 A")
            current = 1.0

        # Gather region points
        region_points = self.get_region_points(region_type)
        if region_points is None or len(region_points) == 0:
            print("[WARNING] No region points found. Cannot compute B-field.")
            return

        # Check coil validity
        if self.centerline is None or self.centerline.n_points < 2:
            print("[ERROR] Invalid or missing centerline. Cannot compute B-field.")
            return

        coil_points = self.centerline.points
        B = compute_biot_savart_field(region_points, coil_points, current=current)

        # Store them for re-plotting
        self.last_region_points = region_points
        self.last_B = B
        self.last_computed_region_type = region_type

        # Detect if these region_points differ from the last used for that region type
        if self.points_changed(self.last_region_points_by_type[region_type], region_points):
            self.region_counts[region_type] += 1
            self.last_region_points_by_type[region_type] = region_points

        print(f"[INFO] Successfully computed B-field for {len(region_points)} points.")

        self.plot_magnetic_field()

    def plot_magnetic_field(self):
        """
        Re-visualize the last B field (if any), adding vector arrows or color maps.
        """
        if self.last_B is None or self.last_region_points is None:
            print("[INFO] No computed B-field to plot yet.")
            return

        # Color range
        cmin = self.parse_float_or_none(self.color_range_min_input.text())
        cmax = self.parse_float_or_none(self.color_range_max_input.text())
        color_limits = (cmin, cmax) if (cmin is not None and cmax is not None and cmax > cmin) else None

        B = self.last_B
        region_points = self.last_region_points

        # Clear & re-add base geometry
        self.plotter.clear()
        if self.surf_poly:
            self.plotter.add_mesh(self.surf_poly, color="lightblue", opacity=0.5, label="Surface Mesh")
        if self.centerline:
            self.plotter.add_mesh(self.centerline, color="magenta", line_width=3, label="Centerline")
        self.add_coordinate_system()
        self.add_grey_points()

        # Show the region points in orange again
        self.visualize_region_by_type(region_points, self.last_computed_region_type)

        # Prepare field data
        field_poly = pv.PolyData(region_points)
        Bmag = np.linalg.norm(B, axis=1)

        try:
            vector_scale = float(self.vector_scale_input.text())
        except ValueError:
            vector_scale = 1.0

        field_type = self.field_type_dropdown.currentText()

        # Vector- Magnitude (Color), Direction(Arrow)
        if field_type == "Vector- Magnitude (Color), Direction(Arrow)":
            field_poly["B"] = B
            field_poly["Bmag"] = Bmag
            arrows = field_poly.glyph(orient="B", scale=False, factor=vector_scale)
            arrows["Bmag"] = np.repeat(Bmag, arrows.n_points // region_points.shape[0])
            self.plotter.add_mesh(arrows, scalars="Bmag", cmap="jet",
                                  clim=color_limits,
                                  scalar_bar_args={"title": "|B| (T)"})

        elif field_type == "Vector, Magnitude & Direction":
            max_mag = Bmag.max() if Bmag.size else 1.0
            if max_mag < 1e-20:
                max_mag = 1.0
            scale_factor = vector_scale / max_mag
            field_poly["B"] = B
            field_poly["Bmag"] = Bmag
            arrows = field_poly.glyph(orient="B", scale="Bmag", factor=scale_factor)
            arrows["Bmag"] = np.repeat(Bmag, arrows.n_points // region_points.shape[0])
            self.plotter.add_mesh(arrows, scalars="Bmag", cmap="jet",
                                  clim=color_limits,
                                  scalar_bar_args={"title": "|B| (T)"})

        elif field_type == "Magnitude":
            field_poly["Bmag"] = Bmag
            self.plotter.add_mesh(field_poly, scalars="Bmag", cmap="jet",
                                  point_size=5, render_points_as_spheres=True,
                                  scalar_bar_args={"title": "|B| (T)"},
                                  clim=color_limits)

        elif field_type == "Bx (Color Only)":
            bx_abs = np.abs(B[:, 0])
            field_poly["BxAbs"] = bx_abs
            self.plotter.add_mesh(field_poly, scalars="BxAbs", cmap="jet",
                                  point_size=5, render_points_as_spheres=True,
                                  scalar_bar_args={"title": "|Bx| (T)"},
                                  clim=color_limits)

        elif field_type == "Bx (Arrow)":
            bx = B[:, 0]
            bx_abs = np.abs(bx)
            max_bx = bx_abs.max() if bx_abs.size else 1.0
            if max_bx < 1e-20:
                max_bx = 1.0
            scale_factor = vector_scale / max_bx

            bx_dir = np.zeros_like(B)
            bx_dir[:, 0] = np.sign(bx)
            field_poly["Bx_dir"] = bx_dir
            field_poly["Bx_len"] = bx_abs

            arrows = field_poly.glyph(orient="Bx_dir", scale="Bx_len", factor=scale_factor)
            arrows["BxAbs"] = np.repeat(bx_abs, arrows.n_points // region_points.shape[0])
            self.plotter.add_mesh(arrows, scalars="BxAbs", cmap="jet",
                                  scalar_bar_args={"title": "|Bx| (T)"},
                                  clim=color_limits)

        elif field_type == "By (Color Only)":
            by_abs = np.abs(B[:, 1])
            field_poly["ByAbs"] = by_abs
            self.plotter.add_mesh(field_poly, scalars="ByAbs", cmap="jet",
                                  point_size=5, render_points_as_spheres=True,
                                  scalar_bar_args={"title": "|By| (T)"},
                                  clim=color_limits)

        elif field_type == "By (Arrow)":
            by = B[:, 1]
            by_abs = np.abs(by)
            max_by = by_abs.max() if by_abs.size else 1.0
            if max_by < 1e-20:
                max_by = 1.0
            scale_factor = vector_scale / max_by

            by_dir = np.zeros_like(B)
            by_dir[:, 1] = np.sign(by)
            field_poly["By_dir"] = by_dir
            field_poly["By_len"] = by_abs

            arrows = field_poly.glyph(orient="By_dir", scale="By_len", factor=scale_factor)
            arrows["ByAbs"] = np.repeat(by_abs, arrows.n_points // region_points.shape[0])
            self.plotter.add_mesh(arrows, scalars="ByAbs", cmap="jet",
                                  scalar_bar_args={"title": "|By| (T)"},
                                  clim=color_limits)

        elif field_type == "Bz (Color Only)":
            bz_abs = np.abs(B[:, 2])
            field_poly["BzAbs"] = bz_abs
            self.plotter.add_mesh(field_poly, scalars="BzAbs", cmap="jet",
                                  point_size=5, render_points_as_spheres=True,
                                  scalar_bar_args={"title": "|Bz| (T)"},
                                  clim=color_limits)

        elif field_type == "Bz (Arrow)":
            bz = B[:, 2]
            bz_abs = np.abs(bz)
            max_bz = bz_abs.max() if bz_abs.size else 1.0
            if max_bz < 1e-20:
                max_bz = 1.0
            scale_factor = vector_scale / max_bz

            bz_dir = np.zeros_like(B)
            bz_dir[:, 2] = np.sign(bz)
            field_poly["Bz_dir"] = bz_dir
            field_poly["Bz_len"] = bz_abs

            arrows = field_poly.glyph(orient="Bz_dir", scale="Bz_len", factor=scale_factor)
            arrows["BzAbs"] = np.repeat(bz_abs, arrows.n_points // region_points.shape[0])
            self.plotter.add_mesh(arrows, scalars="BzAbs", cmap="jet",
                                  scalar_bar_args={"title": "|Bz| (T)"},
                                  clim=color_limits)

        self.plotter.show(title=f"B-field Visualization: {field_type}")

    def visualize_region_by_type(self, region_points, region_type):
        """
        Re-add the region to the plotter after clearing,
        using 'region_points' directly in orange.
        """
        if region_type == "Volume":
            grid_poly = pv.PolyData(region_points)
            self.plotter.add_mesh(grid_poly, color="orange", point_size=4,
                                  render_points_as_spheres=True, label="Grid Points")
        elif region_type == "Plane":
            plane_poly = pv.PolyData(region_points)
            self.plotter.add_mesh(plane_poly, color="orange", point_size=4,
                                  render_points_as_spheres=True, label="Plane Points")
        elif region_type == "Line":
            if len(region_points) > 1:
                line_poly = pv.PolyData(region_points)
                connectivity = np.hstack([[len(region_points)], np.arange(len(region_points))])
                line_poly.lines = connectivity
                self.plotter.add_mesh(line_poly, color="orange", line_width=3, label="Line Region")
                # Endpoints in red
                endpoints = np.array([region_points[0], region_points[-1]], dtype=np.float32)
                epoly = pv.PolyData(endpoints)
                self.plotter.add_mesh(epoly, color="red", point_size=8,
                                      render_points_as_spheres=True, label="Line Endpoints")
            else:
                line_poly = pv.PolyData(region_points)
                self.plotter.add_mesh(line_poly, color="orange", point_size=6,
                                      render_points_as_spheres=True, label="Line Points")

    def get_region_points(self, region_type):
        """
        Return an (M,3) array of region points (Volume, Plane, or Line).
        """
        region_points = []
        if region_type == "Volume":
            xmin, xmax = float(self.xmin_input.text()), float(self.xmax_input.text())
            ymin, ymax = float(self.ymin_input.text()), float(self.ymax_input.text())
            zmin, zmax = float(self.zmin_input.text()), float(self.zmax_input.text())
            spacing = float(self.grid_spacing_input.text())
            x_vals = np.arange(xmin, xmax + spacing, spacing)
            y_vals = np.arange(ymin, ymax + spacing, spacing)
            z_vals = np.arange(zmin, zmax + spacing, spacing)
            for x in x_vals:
                for y in y_vals:
                    for z in z_vals:
                        region_points.append([x, y, z])

        elif region_type == "Plane":
            origin = np.array([float(x) for x in self.plane_origin_input.text().split(",")], dtype=np.float32)
            normal = np.array([float(x) for x in self.plane_normal_input.text().split(",")], dtype=np.float32)
            length = float(self.plane_length_input.text())
            width = float(self.plane_width_input.text())
            spacing = float(self.plane_grid_spacing_input.text())

            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-12:
                print("[ERROR] Normal vector is zero.")
                return None
            normal /= norm_len

            u = np.cross(normal, [1, 0, 0])
            if np.linalg.norm(u) < 1e-12:
                u = np.cross(normal, [0, 1, 0])
            u /= np.linalg.norm(u)
            v = np.cross(normal, u)
            v /= np.linalg.norm(v)

            for xx in np.arange(-length/2, length/2 + spacing, spacing):
                for yy in np.arange(-width/2, width/2 + spacing, spacing):
                    pt = origin + xx*u + yy*v
                    region_points.append(pt)

        elif region_type == "Line":
            start_point = np.array([float(x) for x in self.line_start_input.text().split(",")], dtype=np.float32)
            end_point = np.array([float(x) for x in self.line_end_input.text().split(",")], dtype=np.float32)
            num_points = int(self.line_points_input.text())
            region_points = np.linspace(start_point, end_point, num_points)

        region_points = np.array(region_points, dtype=np.float32)

        # Exclude interior if needed
        if self.exclude_interior_checkbox.isChecked() and self.surf_poly is not None:
            region_points = self.filter_interior_points(region_points)

        return region_points

    ########################################################################
    # Export Functionality
    ########################################################################
    def export_data(self):
        """
        Writes out the last computed field to a CSV with name like:
        "<basename>-volume1.csv" or "coil-plane2.csv".
        If no field has been computed, it does nothing.
        """
        if self.last_region_points is None or self.last_B is None or self.last_computed_region_type is None:
            print("[WARNING] No computed field data to export. Please compute first.")
            return

        region_type = self.last_computed_region_type
        count = self.region_counts[region_type]  # e.g. volume2, plane1, etc.

        base_name = self.export_basename_input.text().strip()
        if not base_name:
            base_name = "field_export"

        # Final file name, e.g. "test-volume1.csv"
        file_name = f"{base_name}-{region_type.lower()}{count}.csv"

        # Prepare the data for CSV: x,y,z,Bx,By,Bz
        data = np.column_stack((self.last_region_points, self.last_B))

        # Use numpy.savetxt for convenience
        header = "x,y,z,Bx,By,Bz"
        np.savetxt(file_name, data, delimiter=",", header=header, comments="")

        print(f"[INFO] Exported field data to '{file_name}'.")

    ########################################################################
    # Simple utility
    ########################################################################
    @staticmethod
    def parse_float_or_none(txt):
        try:
            return float(txt)
        except ValueError:
            return None

    def points_changed(self, old_points, new_points):
        """Return True if old_points differ from new_points in length or values."""
        if old_points is None:
            return True
        if len(old_points) != len(new_points):
            return True
        # Compare positions
        return not np.allclose(old_points, new_points, rtol=1e-7, atol=1e-12)

########################################################################
# Main
########################################################################
if __name__ == "__main__":
    app = QApplication([])
    window = MagneticFieldVisualizer()
    window.show()
    app.exec_()
