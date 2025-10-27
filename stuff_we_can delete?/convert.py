import sys
import os
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QWidget, QLabel, QDoubleSpinBox, 
    QSpinBox, QRadioButton, QButtonGroup,
    QComboBox, QTabWidget
)
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
import helpers

class PyQT(QMainWindow):
    def __init__(self):
        '''
        Function: __init__
        Description: Loads the basic elements of the PyQT screen and gives default values. 
        '''
        super().__init__()
        self.setWindowTitle("Coil Surface Curve Analyzer")
        self.setGeometry(100, 100, 1400, 900)

        '''
        input_file: File that is inputed by the user. This file can either be a .stp or a .stl file.
        msh_file: The mesh file that is auto assigned as "generated_msh.msh" 
        '''
        self.input_file = None
        self.msh_file = "generated_mesh.msh"

        '''
        DEFAULT BUTTON VALUES
        '''
        self.element_size = 0.15
        self.max_element_size_factor = 2.0
        self.feature_angle = 65
        self.trim_points = 0
        self.centerline_s = 0.01
        self.surfacecurves_s = 0.01
        self.loop_smoothing = 0.0
        self.n_centerline_points = 500
        self.n_loop_points = 100
        self.n_subset_points = 20
        self.marching_record_step = 5

        '''
        BACKEND VALUES (user never uses or sees)
        '''
        self.surf_poly = None
        self.raw_centerline_forward = None
        self.loopA = None
        self.marching_record_forward = None
        self.final_centerline = None
        self.final_centerline_poly = None

        # File type filter state
        self.accept_stl = True
        self.accept_stp = False

        # Uniform vs Curvature
        self.sizing_mode = "uniform"

        self.init_ui()

    
    def init_ui(self):
       
        # Layouts
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        control_layout = QVBoxLayout()

        # Left control panel
        self.btn_load_file = QPushButton("Load File")
        self.btn_load_file.clicked.connect(self.load_file)
        self.btn_load_file.setToolTip("Load a .stp or .stl file")
        self.loaded_file = QLabel("No file loaded")
        self.loaded_file.setStyleSheet("color: green;")

        self.btn_process = QPushButton("Generate Centerline")
        self.btn_process.clicked.connect(self.generate_centerline)
        self.btn_process.setEnabled(False)
        self.btn_process.setToolTip("Generate a centerline for loaded coil")

        self.btn_second_process = QPushButton("Generate Surface Curves")
        self.btn_second_process.clicked.connect(self.generate_surface_curves)
        self.btn_second_process.setToolTip("Generate surface curves for centerline")
        
        self.btn_clear = QPushButton("Clear Plot")
        self.btn_clear.clicked.connect(self.clear_plot) 
        self.btn_clear.setToolTip("Clears the current plot")

        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: red;")

        # Radio buttons to choose file type
        self.stl_radio = QRadioButton("Accept STL Files")
        self.stl_radio.setChecked(True)
        self.stl_radio.toggled.connect(self.set_accept_stl)

        self.stp_radio = QRadioButton("Accept STP Files")
        self.stp_radio.toggled.connect(self.set_accept_stp)

        filetype_group = QButtonGroup()
        filetype_group.addButton(self.stl_radio)
        filetype_group.addButton(self.stp_radio)

        # Add left panel file controls
        control_layout.addWidget(QLabel("File Type Filter:"))
        control_layout.addWidget(self.stl_radio)
        control_layout.addWidget(self.stp_radio)

        control_layout.addWidget(self.btn_load_file)
        control_layout.addWidget(self.loaded_file)
        control_layout.addWidget(self.btn_process)
        control_layout.addWidget(self.btn_second_process)
        control_layout.addWidget(self.btn_clear)
        control_layout.addWidget(self.status_label)

        # Dropdown for STP sizing mode
        self.stp_sizing_dropdown = QComboBox()
        self.stp_sizing_dropdown.addItems(["curvature", "uniform"])
        self.stp_sizing_dropdown.setCurrentText("uniform")
        self.stp_sizing_dropdown.currentTextChanged.connect(lambda val: setattr(self, 'sizing_mode', val))
        self.stp_sizing_dropdown.setToolTip("Choose between Uniform and Curvature Modes for STP Files")
        self.stp_sizing_dropdown.setEnabled(False)
        control_layout.addWidget(QLabel("Sizing Mode:"))
        control_layout.addWidget(self.stp_sizing_dropdown)
        
        # Element size input
        self.element_size_input = self._create_doublespinbox(
                0.01, 10.0, 0.01, self.element_size,
                lambda val: setattr(self, 'element_size', val),
                enabled=False
            )
        control_layout.addWidget(QLabel("Element Size"))
        control_layout.addWidget(self.element_size_input)


        # Max element size factor input
        self.max_element_size_factor_input = self._create_doublespinbox(
                1.0, 5.0, 0.01, self.max_element_size_factor,
                lambda val: setattr(self, 'max_element_size_factor', val),
                enabled=False
            )
        control_layout.addWidget(QLabel("Max Element Size Factor"))
        control_layout.addWidget(self.max_element_size_factor_input)

        # Feature angle input
        self.feature_angle_input = self._create_spinbox(
                0, 180, 1, self.feature_angle,
                lambda val: setattr(self, 'feature_angle', val),
                enabled = True
            )
        control_layout.addWidget(QLabel("Feature Angle"))
        control_layout.addWidget(self.feature_angle_input)
        self.feature_angle_input.setToolTip("Feature Angle: Determines the angle that defines the coil end")

        # Trim points input
        self.trim_points_input = self._create_spinbox(
                0, 1000, 1, self.trim_points, 
                lambda val: setattr(self, 'trim_points', val),
                enabled=False
            )
        control_layout.addWidget(QLabel("Trim Points"))
        control_layout.addWidget(self.trim_points_input)
        self.trim_points_input.setToolTip("Trim Points: Trims n points from the end, excluding the endpoint")
        
        # Centerline smoothing input
        self.centerline_s_input = self._create_doublespinbox(
                0.0, 1.0, 0.001, self.centerline_s,
                lambda val: setattr(self, 'centerline_s', val),
                enabled=False
            )
        control_layout.addWidget(QLabel("Centerline Smoothing"))
        control_layout.addWidget(self.centerline_s_input)

        # Surface curves smoothing input
        self.surfacecurves_s_input = self._create_doublespinbox(
                0.0, 1.0, 0.001, self.surfacecurves_s,
                lambda val: setattr(self, 'surfacecurves_s', val),
                enabled=False
            )
        control_layout.addWidget(QLabel("Surface Curves Smoothing"))
        control_layout.addWidget(self.surfacecurves_s_input)

        # Loop smoothing input
        self.loop_smoothing_input = self._create_doublespinbox(
                0.0, 1.0, 0.001, self.loop_smoothing,
                lambda val: setattr(self, 'loop_smoothing', val),
                enabled=False
            )
        control_layout.addWidget(QLabel("Loop Smoothing"))
        control_layout.addWidget(self.loop_smoothing_input)

        # Number of centerline points input
        self.n_centerline_points_input = self._create_spinbox(
                50, 2000, 1, self.n_centerline_points,
                lambda val: setattr(self, 'n_centerline_points', val),
                enabled=False
            )
        control_layout.addWidget(QLabel("Centerline Points"))
        control_layout.addWidget(self.n_centerline_points_input)

        # Number of loop points input
        self.n_loop_points_input = self._create_spinbox(
                50, 500, 1, self.n_loop_points,
                lambda val: setattr(self, 'n_loop_points', val),
                enabled=False
            )
        control_layout.addWidget(QLabel("Loop Points"))
        control_layout.addWidget(self.n_loop_points_input)

        # Number of subset points input
        self.n_subset_points_input = self._create_spinbox(
                10, 100, 1, self.n_subset_points,
                lambda val: setattr(self, 'n_subset_points', val),
                enabled=False
            )
        control_layout.addWidget(QLabel("Subset Points"))
        control_layout.addWidget(self.n_subset_points_input)

        # Marching Record Step input
        self.marching_record_step_input = self._create_spinbox(
                1, 10, 1, self.marching_record_step,
                lambda val: setattr(self, 'marching_record_step', val),
                enabled=False
            )
        control_layout.addWidget(QLabel("Marching Record Step"))
        control_layout.addWidget(self.marching_record_step_input)

        control_layout.addStretch()
        left_panel = QWidget()
        left_panel.setLayout(control_layout)
        main_layout.addWidget(left_panel, 1)

        # Plot Area
        self.plotter = QtInteractor(self)
        main_layout.addWidget(self.plotter.interactor, 4)

    def _create_spinbox(self, min_val, max_val, step, value, callback, tooltip=None, enabled=False):
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.valueChanged.connect(callback)
        spin.setEnabled(enabled)
        if tooltip:
            spin.setToolTip(tooltip)
        return spin

    def _create_doublespinbox(self, min_val, max_val, step, value, callback, tooltip=None, enabled=False):
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.valueChanged.connect(callback)
        spin.setEnabled(enabled)
        if tooltip:
            spin.setToolTip(tooltip)
        return spin

    def set_accept_stl(self):
        self.accept_stl = True
        self.accept_stp = False
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
        self.stp_sizing_dropdown.setEnabled(True)
        self.element_size_input.setEnabled(True)
        self.max_element_size_factor_input.setEnabled(True)
        self.trim_points_input.setEnabled(True)
        self.centerline_s_input.setEnabled(True)
        self.surfacecurves_s_input.setEnabled(True)
        self.n_centerline_points_input.setEnabled(True)
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
        else:
            file_filter = "Mesh Files (*.msh)" # This option should not be reached 

        self.input_file, _ = file_dialog.getOpenFileName(self, "Open File", "", file_filter)
        if not self.input_file:
            return
        self.loaded_file.setText(f"Loaded: {os.path.basename(self.input_file)}")
        self.btn_process.setEnabled(True)

    def clear_plot(self):
        """Clears the PyVista plot."""
        self.plotter.clear()  # Clears all mesh and objects from the plotter
        self.status_label.setText("Status: Plot Cleared")  # Update the status label

    # Helper function that returns the argument if the file is .stp, but the default value if .stl
    def stp_check(self, argument, default):
        return argument if self.accept_stp else default

    def generate_centerline(self):
        try:
            # Clear existing plots
            self.plotter.clear()

            # Read GUI parameters
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

            # Load surface mesh
            if self.input_file:
                self.surf_poly = helpers.load_surface_mesh(
                    self.input_file, self.msh_file,
                    self.stp_check(self.element_size, 0.15),
                    self.stp_check(self.max_element_size_factor, 2.0),
                    self.sizing_mode
                )
            else:
                logging.error(f"Unsupported File format: {self.input_file}")
                return

            # Extract end loops
            logging.info("Extracting coil end loops...")
            self.loopA, self.loopB = helpers.extract_coil_end_loops(
                self.surf_poly,
                self.stp_check(self.feature_angle, 75)
            )

            # Compute raw centerline
            logging.info("Computing forward centerline (loopA -> loopB)...")
            self.raw_centerline_forward, self.marching_record_forward = \
                helpers.compute_centerline_3d_mce(
                    self.surf_poly,
                    self.loopA,
                    self.loopB
                )
            if self.raw_centerline_forward is None:
                logging.error("Failed to compute centerline.")
                return

            # Filter the raw centerline
            filtered_fwd = helpers.trim_end(self.raw_centerline_forward, self.stp_check(self.trim_points, 0))

            # Store and plot final centerline using trimmed data
            self.final_centerline = filtered_fwd
            self.final_centerline_poly = helpers.create_polyline(
                self.final_centerline,
                closed=False
            )

            # Plot marching record for user confirmation
            logging.info("Plotting marching record...")
            helpers.plot_marching_record(
                self.surf_poly,
                self.final_centerline,
                self.loopA,
                self.loopB,
                self.marching_record_forward,
                step=self.stp_check(self.marching_record_step, 5),
                plotter=self.plotter
            )

            # Enable next step in UI
            # self.btn_second_process.setEnabled(True)
            self.status_label.setText(
                "Current marching record shown. Please confirm before continuing."
            )

        except Exception as e:
            logging.exception("Error during centerline generation")
            self.status_label.setText("Error: See log in terminal")

    def generate_surface_curves(self):
        try: 
            '''
            Run the second part of the analysis right after displaying the current version to the user
            '''
            self.plotter.clear()  # Clear previous scene if any

            # Read GUI parameters
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

            logging.info("Refining end loop A...")
            loopA_ordered = helpers.order_loop_points_pca(self.loopA.points)
            refined_loopA_pts = helpers.refine_loop(pv.PolyData(loopA_ordered), n_points=self.stp_check(self.n_loop_points, 200), smoothing=self.stp_check(self.loop_smoothing, 0), spline_degree=3)
            refined_loopA_poly = helpers.create_polyline(refined_loopA_pts, closed=True)
            
            logging.info("Refining end loop B...")
            loopB_ordered = helpers.order_loop_points_pca(self.loopB.points)
            refined_loopB_pts = helpers.refine_loop(pv.PolyData(loopB_ordered), n_points=self.stp_check(self.n_loop_points, 200), smoothing=self.stp_check(self.loop_smoothing, 0), spline_degree=3)
            refined_loopB_poly = helpers.create_polyline(refined_loopB_pts, closed=True)

            centerpoint_A = self.final_centerline[0]
            centerpoint_B = self.final_centerline[-1]
            contours_A = helpers.generate_intermediate_contours(refined_loopA_pts, centerpoint_A, n_contours=5)
            contours_B = helpers.generate_intermediate_contours(refined_loopB_pts, centerpoint_B, n_contours=5)

            logging.info("Building no-roll frames along the centerline...")
            n_vecs, x_vecs, y_vecs = helpers.build_no_roll_frames(self.final_centerline)

            logging.info("Generating scaffold cross sections along the centerline...")
            cross_sections_scaffold = []
            for i in range(len(self.final_centerline)):
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
                sliced = helpers.slice_surface_at_point(self.surf_poly, center, n_i)
                if sliced is None or sliced.n_points < 3:
                    cross_sections_scaffold.append(None)
                    continue
                loops_sliced = sliced.split_bodies()
                if isinstance(loops_sliced, pv.MultiBlock):
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
                sorted_pts = helpers.ensure_closed(sorted_pts)
                refined_pts = helpers.refine_loop(pv.PolyData(sorted_pts), n_points=self.stp_check(self.n_loop_points, 200), smoothing=self.stp_check(self.loop_smoothing, 0), spline_degree=3)
                cross_sections_scaffold.append(refined_pts)

            logging.info("Selecting evenly spaced subset points from refined Loop A...")
            subset_points = helpers.select_evenly_spaced_subset(refined_loopA_pts, small_N=self.stp_check(self.n_subset_points, 20))
            subset_thetas = helpers.compute_theta_for_subset_points(subset_points, centerpoint_A, x_vecs[0], y_vecs[0])
            subset_r_initial = np.sqrt(np.sum((subset_points - centerpoint_A) ** 2, axis=1))

            logging.info("Generating surface curves...")
            surface_curves = helpers.generate_surface_curves(
                cross_sections_scaffold=cross_sections_scaffold,
                centerline_points=self.final_centerline,
                n_vecs=n_vecs,
                x_vecs=x_vecs,
                y_vecs=y_vecs,
                subset_thetas=subset_thetas,
                subset_r_initial=subset_r_initial,
                subset_points=subset_points
            )
            
            # Trim and filter the surface curves using the same parameters as for the centerline.
            trimmed_surface_curves = []
            for curve in surface_curves:
                trimmed = helpers.trim_end(curve, self.stp_check(self.trim_points, 0))
                trimmed_surface_curves.append(trimmed)
            
            smoothed_surface_curves = [helpers.smooth_surface_curve(curve, s=self.stp_check(self.surfacecurves_s, 0.2), k=3, n_interp=self.stp_check(self.n_centerline_points, 200)) for curve in trimmed_surface_curves]

            # -----------------------------------------------------------------------------
            # Visualization via PyVista
            # -----------------------------------------------------------------------------
            self.plotter.add_mesh(self.surf_poly, color="lightblue", opacity=0.5, label="Surface Mesh")
            self.plotter.add_mesh(self.final_centerline_poly, color="magenta", line_width=3, label="Centerline")
            self.plotter.add_mesh(refined_loopA_poly, color="red", line_width=2, label="Loop A")
            self.plotter.add_mesh(refined_loopB_poly, color="green", line_width=2, label="Loop B")

            subset_poly = pv.PolyData(subset_points)
            self.plotter.add_mesh(subset_poly, color="red", point_size=5, render_points_as_spheres=True, label="Subset Points")

            raw_points = np.vstack([curve for curve in surface_curves if len(curve) > 0])
            raw_points_poly = pv.PolyData(raw_points)
            self.plotter.add_mesh(raw_points_poly, color="red", point_size=2, render_points_as_spheres=True, label="Raw Surface Curve Points")

            for idx, curve in enumerate(smoothed_surface_curves):
                poly = helpers.create_polyline(curve, closed=False)
                self.plotter.add_mesh(poly, color="cyan", line_width=3, label=f"Surface Curve {idx}" if idx == 0 else None)

            for i in range(0, len(cross_sections_scaffold), 5):
                cs = cross_sections_scaffold[i]
                if cs is None:
                    continue
                cs_poly = helpers.create_polyline(cs, closed=True)
                self.plotter.add_mesh(cs_poly, color="blue", line_width=1, label=f"Cross Section {i}" if i == 0 else None)

            for contour in contours_A + contours_B:
                poly = helpers.create_polyline(contour, closed=True)
                self.plotter.add_mesh(poly, color="yellow", line_width=2, opacity=0.8)

            self.plotter.add_legend(bcolor="white")
            self.plotter.reset_camera()
            self.status_label.setText("Surface Curves Generated")

        except Exception as e:
            logging.exception("Error during surface curve generation")
            self.status_label.setText("Error: See log in terminal")

if __name__ == "__main__":
    print("STARTING PROGRAM...")

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    app = QApplication(sys.argv)
    window = PyQT()
    window.show()
    sys.exit(app.exec_())