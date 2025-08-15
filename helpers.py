import os
import sys
import logging
from collections import defaultdict
from typing import Tuple, List, Optional

import numpy as np
import gmsh
import meshio
import pyvista as pv
from scipy.interpolate import splprep, splev, interp1d
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QMessageBox


# -----------------------------------------------------------------------------
# Constants (instead of magic numbers)
# -----------------------------------------------------------------------------
ENDPOINT_WEIGHT: float = 1000.0  # Weight applied to endpoints during spline smoothing.
TOL: float = 1e-6               # Tolerance for floating point comparisons.
EPS: float = 1e-14              # A very small number to avoid division-by-zero.

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def ensure_closed(curve: np.ndarray, tol: float = TOL) -> np.ndarray:
    """
    Ensure that a curve (an array of shape (N,3)) is closed by appending the first
    point at the end if necessary.
    """
    if not np.allclose(curve[0], curve[-1], atol=tol):
        curve = np.vstack([curve, curve[0]])
    return curve

def create_polyline(points: np.ndarray, closed: bool = False) -> pv.PolyData:
    """
    Create a PyVista PolyData polyline from an array of points.
    
    Parameters:
        points: Array of 3D points.
        closed: If True, ensures that the polyline is closed (first and last point are identical).
                For open curves (like the centerline or surface curves) use closed=False.
    
    Returns:
        A PyVista PolyData representing the polyline.
    """
    if closed:
        points = ensure_closed(points)
    n_points = len(points)
    connectivity = np.hstack([[n_points], np.arange(n_points)])
    poly = pv.PolyData()
    poly.points = points
    poly.lines = connectivity
    return poly

# -----------------------------------------------------------------------------
# Mesh Loading / Creation Functions
# -----------------------------------------------------------------------------
def mesh_step_file(step_filename: str,
                   mesh_filename: str = "generated_mesh.msh",
                   element_size: float = 0.2,
                   size_factor: float = 1.0,
                   sizing_mode: str = "uniform") -> None:
    """
    Mesh a STEP file using Gmsh with optional curvature refinement.

    Parameters:
        step_filename: The input STEP file.
        mesh_filename: The temporary output mesh file.
        element_size: The minimum element size.
        size_factor: Multiplier for maximum element size.
        sizing_mode: Either 'uniform' or 'curvature' for adaptive refinement.
    """
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.open(step_filename)
        if sizing_mode == "uniform":
            # Classic global parameters
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax",
                                   element_size * size_factor)

        elif sizing_mode == "curvature":
            # --------------------------------------------------------------
            # Curvature-controlled mesh size (Gmsh ≥ 4.10)
            # --------------------------------------------------------------
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints",         0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

            # ↓ Target # of elements per full 2π bend (tune as you like)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 25)

            # Hard limits
            gmsh.option.setNumber("Mesh.MeshSizeMin",  element_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax",  element_size * size_factor)


        gmsh.model.mesh.generate(3)
        gmsh.write(mesh_filename)
        logging.info(f"Meshed '{step_filename}' to '{mesh_filename}' (mode={sizing_mode}, size=[{element_size}, {element_size * size_factor}])")

    except Exception as e:
        logging.error(f"Gmsh meshing failed for file {step_filename}: {e}")
    finally:
        gmsh.finalize()

def load_msh_as_pv_mesh(msh_filename: str = "generated_mesh.msh") -> pv.UnstructuredGrid:
    """
    Convert a Gmsh .msh file to a PyVista UnstructuredGrid.
    
    Parameters:
        msh_filename: The filename of the .msh file.
    
    Returns:
        A PyVista UnstructuredGrid containing the volumetric mesh.
    """
    try:
        mesh = meshio.read(msh_filename)
    except Exception as e:
        logging.error(f"Error reading mesh file {msh_filename}: {e}")

    points = mesh.points
    cells = []
    cell_types = []
    # Only consider volumetric cells.
    for cell_block in mesh.cells:
        ctype = cell_block.type
        if ctype in ["tetra", "hexahedron", "wedge", "pyramid"]:
            for cell in cell_block.data:
                cells.append(np.concatenate(([len(cell)], cell)))
            if ctype == "tetra":
                vtk_cell_type = 10
            elif ctype == "hexahedron":
                vtk_cell_type = 12
            elif ctype == "wedge":
                vtk_cell_type = 13
            elif ctype == "pyramid":
                vtk_cell_type = 14
            cell_types.extend([vtk_cell_type] * len(cell_block.data))
    if len(cells) == 0:
        logging.error("No 3D volumetric cells found in the mesh.")
    cells = np.hstack(cells)
    cell_types = np.array(cell_types)
    pv_grid = pv.UnstructuredGrid(cells, cell_types, points)
    return pv_grid

def extract_surface_mesh_from_volume(pv_grid: pv.UnstructuredGrid) -> pv.PolyData:
    """
    Extract the surface mesh from a volumetric PyVista grid.
    
    Parameters:
        pv_grid: A PyVista UnstructuredGrid.
    
    Returns:
        The surface mesh as a PyVista PolyData object.
    """
    return pv_grid.extract_surface()

def load_surface_mesh(input_filename: str, mesh_filename: str, element_size: float, size_factor: float, sizing_mode="uniform") -> pv.PolyData:
    """
    Load a surface mesh from an input CAD file. If the file is a STEP file (.stp or .step),
    it is meshed with Gmsh. If it is an STL file (.stl), it is loaded directly.
    
    Parameters:
        input_filename: The input file (.stp/.step or .stl).
        mesh_filename: Temporary mesh filename (used for STEP files).
        element_size: Mesh element size for STEP files.
    
    Returns:
        A PyVista PolyData representing the surface mesh.
    """
    ext = os.path.splitext(input_filename)[1].lower()
    if ext == ".stl":
        try:
            surf_poly = pv.read(input_filename)
            logging.info(f"Loaded surface mesh from STL file: {input_filename}")

            vertices = surf_poly.points
            faces    = surf_poly.faces.reshape((-1, 4))[:, 1:]
            write_msh_direct_from_stl(mesh_filename, vertices, faces)

            return surf_poly
        except Exception as e:
            logging.error(f"Failed to load STL file '{input_filename}': {e}")
    elif ext in [".stp", ".step"]:
        mesh_step_file(input_filename, mesh_filename, element_size, size_factor, sizing_mode)
        try:
            pv_grid = load_msh_as_pv_mesh(mesh_filename)
            surf_poly = extract_surface_mesh_from_volume(pv_grid)
            return surf_poly
        except Exception as e:
            logging.error(f"Failed to load mesh from '{mesh_filename}': {e}")
    else:
        logging.error("Unsupported file format. Please use a .stp/.step or .stl file.")

def write_msh_direct_from_stl(fname, vertices, faces):
    with open(fname, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        
        # ---- Nodes ----
        f.write("$Nodes\n%d\n" % len(vertices))
        for idx, (x, y, z) in enumerate(vertices, start=1):
            f.write(f"{idx} {x:.17g} {y:.17g} {z:.17g}\n")
        f.write("$EndNodes\n")
        
        # ---- Elements ----  (type 2 = triangle)
        f.write("$Elements\n%d\n" % len(faces))
        for eid, (n1, n2, n3) in enumerate(faces, start=1):
            f.write(f"{eid} 2 0 {n1+1} {n2+1} {n3+1}\n")
        f.write("$EndElements\n")


# -----------------------------------------------------------------------------
# Centerline and End Loop Extraction Functions
# -----------------------------------------------------------------------------

def extract_coil_end_loops(surf_poly, feature_angle=55.0):
    edges_poly = surf_poly.extract_feature_edges(
        boundary_edges=True,
        feature_edges=True,
        manifold_edges=False,
        feature_angle=feature_angle
    )
    split_edges = edges_poly.split_bodies()
    if isinstance(split_edges, pv.MultiBlock):
        loops = [split_edges[i] for i in range(len(split_edges))]
    else:
        loops = [split_edges]
    n_loops = len(loops)
    if n_loops != 2:
        plotter = pv.Plotter()
        plotter.add_mesh(surf_poly, color="blue", opacity=0.5, label="Surface Mesh")
        for i, loop in enumerate(loops):
            plotter.add_mesh(loop, color="red", line_width=4, label=f"Loop {i}")
        plotter.show(title="Error: Surface Mesh + Detected Loops")
        raise ValueError(
            f"Expected exactly 2 end loops, but found {n_loops}. "
            "Please check your geometry or feature_angle."
        )
    return loops

def compute_centerline_3d_mce(surf_poly: pv.PolyData, loopA: pv.PolyData, loopB: pv.PolyData) -> Tuple[Optional[np.ndarray], List[set]]:
    """
    Compute the centerline using a modified marching algorithm.
    
    Parameters:
        surf_poly: The surface mesh.
        loopA: The starting end loop (moving set).
        loopB: The reference end loop.
    
    Returns:
        A tuple of (centerline_points, marching_record), where centerline_points is an (N,3) array.
    """
    pv_faces = surf_poly.faces.reshape((-1, 4))[:, 1:]
    vertices = surf_poly.points
    faces = pv_faces

    loopA_plane_centroid, loopA_plane_normal = compute_best_fit_plane(loopA.points)
    loopB_plane_centroid, loopB_plane_normal = compute_best_fit_plane(loopB.points)

    endA_vertex_indices = select_vertices_near_plane(surf_poly, loopA_plane_centroid, loopA_plane_normal, TOL)
    logging.info(f"Found {len(endA_vertex_indices)} on end A out of {len(vertices)} vertices in the mesh based on plane tolerance.")

    endB_vertex_indices = select_vertices_near_plane(surf_poly, loopB_plane_centroid, loopB_plane_normal, TOL)
    logging.info(f"Found {len(endB_vertex_indices)} on end B out of {len(vertices)} vertices in the mesh based on plane tolerance.")

    all_vertex_indices = set(range(len(vertices)))

    # Combine the indices from both ends.
    inactive_vertex_indices = set(endA_vertex_indices).union(set(endB_vertex_indices))

    # The active vertex indices are those not in the inactive set.
    active_vertex_indices = all_vertex_indices - inactive_vertex_indices

    logging.info(f"Active vertex count: {len(active_vertex_indices)} out of {len(vertices)} total vertices.")

    # Now, build the active face set.
    # Each face is given by an array of vertex indices (from pv_faces).
    active_face_indices = set()
    for f_idx, face in enumerate(pv_faces):
        # If any vertex in this face is in the active vertex set, consider the face active.
        if any(v in active_vertex_indices for v in face):
            active_face_indices.add(f_idx)

    logging.info(f"Active face count: {len(active_face_indices)} out of {len(pv_faces)} total faces.")

    edge_to_faces = defaultdict(list)
    for f_idx, tri in enumerate(faces):
        for edge in [tuple(sorted((tri[0], tri[1]))),
                    tuple(sorted((tri[1], tri[2]))),
                    tuple(sorted((tri[2], tri[0])))]:
            edge_to_faces[edge].append(f_idx)

    def find_vertex_indices_in_polydata(src_poly: pv.PolyData, all_points: np.ndarray) -> set:
        coord_to_index = {tuple(pt): i for i, pt in enumerate(all_points)}
        indices = [coord_to_index.get(tuple(pt), -1) for pt in src_poly.points]
        return {i for i in indices if i >= 0}

    V_mov = find_vertex_indices_in_polydata(loopA, vertices)
    V_ref = find_vertex_indices_in_polydata(loopB, vertices)

    visited = set()
    moving_sections = []
    marching_record = []
    current_moving = set(V_mov)

    def get_external_edges_and_vertices(faces_subset: set) -> Tuple[List[tuple], set]:
        e2f = defaultdict(list)
        for f_idx in faces_subset:
            tri = faces[f_idx]
            for edge in [tuple(sorted((tri[0], tri[1]))),
                        tuple(sorted((tri[1], tri[2]))),
                        tuple(sorted((tri[2], tri[0])))]:
                e2f[edge].append(f_idx)
        E_ext = [e for e, flist in e2f.items() if len(flist) == 1]
        V_ext = set()
        for v1, v2 in E_ext:
            V_ext.add(v1)
            V_ext.add(v2)
        return E_ext, V_ext
    
    while True:
        visited |= current_moving
        af = active_face_indices
        new_active_faces = {f_idx for f_idx in af if not (faces[f_idx][0] in visited or faces[f_idx][1] in visited or faces[f_idx][2] in visited)}
        if not new_active_faces:
            break
        _, V_ext_new = get_external_edges_and_vertices(new_active_faces)
        new_moving = V_ext_new - V_ref
        if not new_moving:
            break
        marching_record.append(new_moving.copy())
        moving_sections.append(current_moving)
        current_moving = new_moving
        active_faces = new_active_faces

    centerline_points = []
    for section in moving_sections:
        if section:
            coords = np.array([vertices[v] for v in section])
            centerline_points.append(coords.mean(axis=0))
    if V_ref:
        coords_ref = np.array([vertices[v] for v in V_ref])
        centerline_points.append(coords_ref.mean(axis=0))
    if centerline_points:
        centerline_points = np.vstack(centerline_points)
    else:
        centerline_points = None
    return centerline_points, marching_record

def trim_end(raw_points: np.ndarray, n_trim: int) -> np.ndarray:
    """
    Trim a user-defined number of points from the *end* of the raw computed centerline or surface curve,
    but always keep the last point (the endloop center).

    Parameters:
        raw_points: The raw centerline as an (N,3) array.
        n_trim: Number of interior points to trim from the end.

    Returns:
        A new centerline array with the last `n_trim` interior points removed,
        but with the endpoint re-added.
    """
    N = len(raw_points)
    if N < 3:
        return raw_points.copy()

    if N - n_trim < 2:
        return raw_points.copy()

    trimmed = raw_points[0:N - n_trim]

    if not np.allclose(trimmed[-1], raw_points[-1], atol=1e-6):
        trimmed = np.vstack([trimmed, raw_points[-1]])
    return trimmed

# -----------------------------------------------------------------------------
# End Loop and Intermediate Contour Functions
# -----------------------------------------------------------------------------
def compute_best_fit_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the best-fit plane for a set of points using PCA.
    
    Parameters:
        points: An (N, 3) numpy array of points.
    
    Returns:
        A tuple (centroid, normal) where centroid is the mean of the points and
        normal is the unit normal of the best-fit plane.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    # Compute the singular value decomposition.
    U, S, Vt = np.linalg.svd(centered)
    # The normal is the eigenvector corresponding to the smallest singular value.
    normal = Vt[-1]
    # Ensure it's a unit vector.
    normal /= np.linalg.norm(normal)
    return centroid, normal

def select_vertices_near_plane(surf_poly: pv.PolyData, plane_point: np.ndarray, plane_normal: np.ndarray, tol: float) -> np.ndarray:
    """
    Identify the indices of vertices in the surface mesh that are within a
    tol from the given plane.
    
    Parameters:
        surf_poly: The surface mesh (PyVista PolyData).
        plane_point: A point on the plane (e.g. the centroid of the endloop).
        plane_normal: The unit normal of the plane.
        tol: Distance tolerance below which vertices are considered to lie on the plane.
    
    Returns:
        A boolean mask (or set) of vertex indices that in the flat region.
    """
    vertices = surf_poly.points  # (N, 3) array
    # Compute the signed distance of each vertex from the plane.
    # The formula is: distance = dot(vertex - plane_point, plane_normal)
    distances = np.dot(vertices - plane_point, plane_normal)
    # We want vertices that are farther than tol from the plane (in absolute value).
    valid_mask = np.abs(distances) < tol
    # Return indices (or the mask) of valid vertices.
    valid_indices = np.where(valid_mask)[0]
    return valid_indices

def order_loop_points_pca(points: np.ndarray) -> np.ndarray:
    """
    Order loop points by projecting onto the first two principal components.
    
    Parameters:
        points: Array of loop points.
    
    Returns:
        An ordered (and closed) array of points.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:2].T
    projected = centered.dot(basis)
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    sorted_indices = np.argsort(angles)
    ordered_points = points[sorted_indices]
    return ensure_closed(ordered_points)

def refine_loop(loop_poly: pv.PolyData, n_points: int = 200, smoothing: float = 0.0, spline_degree: int = 3) -> np.ndarray:
    """
    Refine a loop using a smoothing spline.
    
    Parameters:
        loop_poly: The input loop as a PyVista PolyData.
        n_points: The number of points desired.
        smoothing: Smoothing factor.
        spline_degree: Degree of the spline.
    
    Returns:
        A (n_points, 3) array of refined loop points.
    """
    pts = loop_poly.points.copy()
    pts = ensure_closed(pts)
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_arc = np.concatenate(([0], np.cumsum(seg_lengths)))
    total_length = cum_arc[-1]
    if total_length < EPS:
        return pts
    u = cum_arc / total_length
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], u=u, s=smoothing, k=spline_degree, per=True)
    except Exception as e:
        logging.error(f"Refining loop failed during splprep: {e}")
        return pts
    dense_n = 1000
    u_dense = np.linspace(0, 1, dense_n)
    x_dense, y_dense, z_dense = splev(u_dense, tck)
    dense_points = np.vstack((x_dense, y_dense, z_dense)).T
    diffs_dense = np.diff(dense_points, axis=0)
    seg_dense = np.linalg.norm(diffs_dense, axis=1)
    cum_dense = np.concatenate(([0], np.cumsum(seg_dense)))
    uniform_arc = np.linspace(0, cum_dense[-1], n_points)
    refined_points = np.empty((n_points, 3))
    for dim in range(3):
        refined_points[:, dim] = np.interp(uniform_arc, cum_dense, dense_points[:, dim])
    return refined_points

def generate_intermediate_contours(refined_points: np.ndarray, centerpoint: np.ndarray, n_contours: int = 5) -> List[np.ndarray]:
    """
    Generate intermediate contours between the refined loop and the centerpoint.
    
    Parameters:
        refined_points: The refined end-loop points.
        centerpoint: The center point (of one end).
        n_contours: Number of intermediate contours.
    
    Returns:
        A list of contour arrays.
    """
    alphas = np.linspace(0, 1, n_contours + 2)[1:-1]
    contours = []
    for a in alphas:
        contour = centerpoint + a * (refined_points - centerpoint)
        contours.append(ensure_closed(contour))
    return contours

def visualize_contours(contours: List[np.ndarray], plotter: Optional[pv.Plotter] = None) -> pv.Plotter:
    """
    Visualize a list of contours using PyVista.
    
    Parameters:
        contours: List of contour point arrays.
        plotter: An existing PyVista Plotter (or None to create a new one).
    
    Returns:
        The PyVista Plotter with the contours added.
    """
    if plotter is None:
        plotter = pv.Plotter()
    colors = ['yellow', 'orange', 'green', 'cyan', 'blue', 'purple']
    for i, pts in enumerate(contours):
        poly = create_polyline(pts, closed=True)
        plotter.add_mesh(poly, color=colors[i % len(colors)], line_width=2)
    return plotter

# -----------------------------------------------------------------------------
# "No Roll" Frame and Surface Curve Generation Functions
# -----------------------------------------------------------------------------
def rodrigues(k: np.ndarray, theta: float) -> np.ndarray:
    """
    Rodrigues' rotation formula: Compute the rotation matrix for rotating
    around unit vector k by angle theta.
    """
    K = np.array([[0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0]], dtype=float)
    I = np.eye(3)
    return I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def minimal_rotation_matrix(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute the minimal rotation matrix that rotates vector u to vector v.
    
    Parameters:
        u: Source vector.
        v: Destination vector.
    
    Returns:
        A 3x3 rotation matrix.
    """
    nu = u / (np.linalg.norm(u) + EPS)
    nv = v / (np.linalg.norm(v) + EPS)
    dot_val = np.dot(nu, nv)
    if dot_val > 1.0 - 1e-12:
        return np.eye(3)
    if dot_val < -1.0 + 1e-12:
        perp = np.array([1, 0, 0], dtype=float)
        if abs(np.dot(perp, nu)) > 0.9:
            perp = np.array([0, 1, 0], dtype=float)
        axis = np.cross(nu, perp)
        axis /= (np.linalg.norm(axis) + EPS)
        return rodrigues(axis, np.pi)
    axis = np.cross(nu, nv)
    axis /= (np.linalg.norm(axis) + EPS)
    angle = np.arccos(np.clip(dot_val, -1, 1))
    return rodrigues(axis, angle)

def slice_surface_at_point(surf_poly: pv.PolyData, point: np.ndarray, normal: np.ndarray) -> Optional[pv.PolyData]:
    """
    Slice the surface mesh with a plane defined by a point and a normal.
    If multiple loops result, return the one whose centroid is closest to the point.
    
    Parameters:
        surf_poly: The surface mesh.
        point: A point on the slicing plane.
        normal: The normal vector of the slicing plane.
    
    Returns:
        The sliced loop as a PolyData, or None if the intersection is insufficient.
    """
    sliced = surf_poly.slice(origin=point, normal=normal)
    if sliced.n_points < 3:
        return None
    loops = sliced.split_bodies()
    if isinstance(loops, pv.MultiBlock):
        min_dist = float('inf')
        closest_loop = None
        for i in range(len(loops)):
            loop = loops[i]
            dist = np.linalg.norm(loop.center - point)
            if dist < min_dist:
                min_dist = dist
                closest_loop = loop
        return closest_loop
    else:
        return loops

def compute_local_tangent(centerline: np.ndarray, i: int) -> np.ndarray:
    """
    Compute the local tangent vector at index i of the centerline.
    """
    n = len(centerline)
    if n < 2:
        return np.array([0, 0, 1], dtype=float)
    if i == 0:
        return centerline[1] - centerline[0]
    elif i == n - 1:
        return centerline[-1] - centerline[-2]
    else:
        return centerline[i + 1] - centerline[i - 1]

def build_no_roll_frames(centerline_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build local frames along the centerline with minimal rotation ("no roll").
    
    Returns:
        n_vecs: Array of tangents (local z-axis) for each centerline point.
        x_vecs: Array of x-axis vectors.
        y_vecs: Array of y-axis vectors.
    """
    n_pts = len(centerline_points)
    tangents = []
    for i in range(n_pts):
        tvec = compute_local_tangent(centerline_points, i)
        tn = tvec / (np.linalg.norm(tvec) + EPS)
        tangents.append(tn)
    tangents = np.array(tangents)
    n_vecs = np.zeros_like(tangents)
    x_vecs = np.zeros_like(tangents)
    y_vecs = np.zeros_like(tangents)
    n_vecs[0] = tangents[0]
    guess = np.array([1, 0, 0], dtype=float)
    if abs(np.dot(guess, n_vecs[0])) > 0.9:
        guess = np.array([0, 1, 0], dtype=float)
    x0 = guess - (np.dot(guess, n_vecs[0])) * n_vecs[0]
    x0 /= (np.linalg.norm(x0) + EPS)
    x_vecs[0] = x0
    y_vecs[0] = np.cross(n_vecs[0], x_vecs[0])
    for i in range(n_pts - 1):
        n_i = n_vecs[i]
        x_i = x_vecs[i]
        n_next = tangents[i + 1]
        R = minimal_rotation_matrix(n_i, n_next)
        x_next = R @ x_i
        x_next -= (np.dot(x_next, n_next)) * n_next
        norm_xn = np.linalg.norm(x_next)
        if norm_xn < EPS:
            x_next = x_i
        else:
            x_next /= norm_xn
        n_vecs[i + 1] = n_next
        x_vecs[i + 1] = x_next
        y_vecs[i + 1] = np.cross(n_next, x_next)
    return n_vecs, x_vecs, y_vecs

def select_evenly_spaced_subset(loop_points: np.ndarray, small_N: int = 20) -> np.ndarray:
    """
    Select a subset of evenly spaced points from a cyclic loop.
    
    Parameters:
        loop_points: The ordered (closed) loop points.
        small_N: Number of points to select.
    
    Returns:
        An array of selected points.
    """
    n_total = len(loop_points)
    indices = np.linspace(0, n_total - 1, small_N, endpoint=False, dtype=int)
    return loop_points[indices]

def compute_theta_for_subset_points(subset_points: np.ndarray, centerpoint: np.ndarray, x_i: np.ndarray, y_i: np.ndarray) -> np.ndarray:
    """
    Compute the theta (azimuthal) angles for the subset points based on a local frame.
    """
    v = subset_points - centerpoint
    xvals = np.dot(v, x_i)
    yvals = np.dot(v, y_i)
    thetas = np.arctan2(yvals, xvals)
    return thetas

def interpolate_r_at_theta(scaffold_thetas: np.ndarray, scaffold_rs: np.ndarray, target_thetas: np.ndarray) -> np.ndarray:
    """
    Interpolate the radial distances (r) at the target theta angles based on scaffold data.
    """
    sorted_indices = np.argsort(scaffold_thetas)
    scaffold_thetas_sorted = scaffold_thetas[sorted_indices]
    scaffold_rs_sorted = scaffold_rs[sorted_indices]
    scaffold_thetas_extended = np.concatenate((scaffold_thetas_sorted, scaffold_thetas_sorted + 2 * np.pi))
    scaffold_rs_extended = np.concatenate((scaffold_rs_sorted, scaffold_rs_sorted))
    unique_thetas, unique_indices = np.unique(scaffold_thetas_extended, return_index=True)
    scaffold_thetas_unique = scaffold_thetas_extended[unique_indices]
    scaffold_rs_unique = scaffold_rs_extended[unique_indices]
    interpolator = interp1d(scaffold_thetas_unique, scaffold_rs_unique, kind='linear', fill_value="extrapolate")
    target_thetas = np.mod(target_thetas, 2 * np.pi)
    return interpolator(target_thetas)

def generate_surface_curves(
    cross_sections_scaffold: List[Optional[np.ndarray]],
    centerline_points: np.ndarray,
    n_vecs: np.ndarray,
    x_vecs: np.ndarray,
    y_vecs: np.ndarray,
    subset_thetas: np.ndarray,
    subset_r_initial: np.ndarray,
    subset_points: np.ndarray
) -> List[np.ndarray]:
    """
    Generate surface curves by keeping theta fixed and adjusting r based on scaffold cross sections.
    
    Parameters:
        cross_sections_scaffold: List of scaffold cross-section arrays (one per centerline point).
        centerline_points: The centerline points.
        n_vecs, x_vecs, y_vecs: Local frame vectors along the centerline.
        subset_thetas: Fixed theta angles from the first cross-section.
        subset_r_initial: Initial radial distances.
        subset_points: Starting subset points (from the first end loop).
    
    Returns:
        A list of surface curves (each a (M,3) array where M is the number of centerline points).
    """
    small_N = len(subset_thetas)
    M = len(centerline_points)
    surface_curves = [[] for _ in range(small_N)]
    for i in range(M):
        scaffold_cs = cross_sections_scaffold[i]
        center = centerline_points[i]
        n_i = n_vecs[i]
        x_i = x_vecs[i]
        y_i = y_vecs[i]
        if i == 0:
            for k in range(small_N):
                surface_curves[k].append(subset_points[k])
            continue
        if scaffold_cs is None:
            continue
        v_scaffold = scaffold_cs - center
        x_scaffold = np.dot(v_scaffold, x_i)
        y_scaffold = np.dot(v_scaffold, y_i)
        scaffold_thetas = np.arctan2(y_scaffold, x_scaffold)
        scaffold_rs = np.sqrt(x_scaffold ** 2 + y_scaffold ** 2)
        scaffold_thetas = np.mod(scaffold_thetas, 2 * np.pi)
        target_rs = interpolate_r_at_theta(scaffold_thetas, scaffold_rs, subset_thetas)
        for k in range(small_N):
            r_new = target_rs[k]
            point_new = center + r_new * (np.cos(subset_thetas[k]) * x_i + np.sin(subset_thetas[k]) * y_i)
            surface_curves[k].append(point_new)
    surface_curves = [np.array(curve) for curve in surface_curves]
    return surface_curves

def smooth_surface_curve(curve: np.ndarray, s: float = 0.1, k: int = 3, n_interp: int = 200) -> np.ndarray:
    """
    Smooth a surface curve using spline interpolation with fixed endpoint weighting.
    
    Parameters:
        curve: Array of 3D points representing the curve.
        s: Smoothing factor.
        k: Spline degree.
        n_interp: Number of interpolated points.
    
    Returns:
        Smoothed curve as an (n_interp,3) array.
    """
    if len(curve) < 3:
        return curve.copy()
    x, y, z = curve.T
    diffs = np.diff(curve, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    t = np.concatenate(([0], np.cumsum(seg_lengths)))
    total_length = t[-1]
    if total_length < EPS:
        return curve.copy()
    t /= total_length
    weights = np.ones(len(x))
    weights[0] = ENDPOINT_WEIGHT
    weights[-1] = ENDPOINT_WEIGHT
    try:
        tck, _ = splprep([x, y, z], u=t, s=s, k=k, w=weights)
        u_new = np.linspace(0, 1, n_interp)
        x_new, y_new, z_new = splev(u_new, tck)
        smoothed_curve = np.vstack((x_new, y_new, z_new)).T
        smoothed_curve[0] = curve[0]
        smoothed_curve[-1] = curve[-1]
        return smoothed_curve
    except Exception as e:
        logging.error(f"Surface curve smoothing failed: {e}")
        return curve.copy()


# -----------------------------------------------------------------------------
#  Function to Plot Marching Record (every 5th marching set) Along with Raw Centerline and Surface Mesh
# -----------------------------------------------------------------------------
def plot_marching_record(surface_mesh: pv.PolyData, raw_centerline: np.ndarray, loopA: pv.PolyData, loopB: pv.PolyData, marching_record: List[set], step: int = 5, plotter: pv.Plotter = None) -> None:
    """
    Plot the surface mesh, the raw centerline, and every nth marching record.
    
    Parameters:
        surface_mesh: The PyVista PolyData of the surface mesh.
        raw_centerline: An (N,3) numpy array of the raw centerline points.
        marching_record: A list where each element is a set of vertex indices from the marching algorithm.
        step: Plot every nth marching record.
    """
    external_plotter = plotter is not None
    if not external_plotter:
        plotter = pv.Plotter()

    plotter.add_mesh(surface_mesh, color='lightgray', opacity=0.5, label='Surface Mesh')
    plotter.add_mesh(pv.PolyData(raw_centerline), color='blue', line_width=3, label='Raw Centerline')
    plotter.add_mesh(loopA, color="red", line_width=2, label="Loop A")
    plotter.add_mesh(loopB, color="green", line_width=2, label="Loop B")

    for idx, v_set in enumerate(marching_record):
        if idx % step == 0:
            pts = np.array([surface_mesh.points[v] for v in v_set])
            plotter.add_points(pts, color='red', point_size=8, render_points_as_spheres=True,
                               label=f'Marching Record {idx}' if idx == 0 else None)

    plotter.add_legend()
    plotter.reset_camera()
    plotter.render()
    plotter.show()
    