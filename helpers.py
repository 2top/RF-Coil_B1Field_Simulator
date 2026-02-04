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
from PyQt5.QtWidgets import QApplication, QLabel


# -----------------------------------------------------------------------------
# Constants
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
                   mesh_filename: str,
                   element_size: float = 0.2,
                   size_factor: float = 1.0,
                   status_label: QLabel = None) -> None:
    """
    Mesh a STEP file using Gmsh with optional curvature refinement.

    Parameters:
        step_filename: The input STEP file.
        mesh_filename: The temporary output mesh file.
        element_size: The minimum element size.
        size_factor: Multiplier for maximum element size.
    """
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(step_filename)
        # Classic global parameters
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size * size_factor)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)

        status_label.setText(f"Meshing STP file '{os.path.basename(step_filename)}'...")
        QApplication.processEvents()

        gmsh.logger.start()

        gmsh.model.mesh.generate(3)

        for i, msg in enumerate(gmsh.logger.get()):
            if i % 20 == 0:
                status_label.setText(msg.strip())
            QApplication.processEvents()

        gmsh.write(mesh_filename)
        logging.info(f"Meshed '{step_filename}' to '{mesh_filename}' size=[{element_size}, {element_size * size_factor}])")

    except Exception as e:
        logging.error(f"Gmsh meshing failed for file {step_filename}: {e}")
        status_label.setText(f"Error: Gmsh meshing failed: {e}")
    finally:
        gmsh.finalize()

def load_msh_as_pv_mesh(msh_filename: str) -> pv.UnstructuredGrid:
    """
    Load a Gmsh .msh into either:
      - pv.UnstructuredGrid (if volumetric cells exist), or
      - pv.PolyData (if only surface triangles exist)
    """
    mesh = meshio.read(msh_filename)
    points = mesh.points

    # --- Try volumetric first ---
    vol_cells = []
    vol_types = []

    for block in mesh.cells:
        ctype = block.type
        if ctype in ["tetra", "hexahedron", "wedge", "pyramid"]:
            for cell in block.data:
                vol_cells.append(np.concatenate(([len(cell)], cell)))

            vtk_map = {"tetra": 10, "hexahedron": 12, "wedge": 13, "pyramid": 14}
            vol_types.extend([vtk_map[ctype]] * len(block.data))

    if vol_cells:
        cells = np.hstack(vol_cells)
        cell_types = np.array(vol_types)
        return pv.UnstructuredGrid(cells, cell_types, points)

    # --- Fallback: surface triangles ---
    tri = None
    for block in mesh.cells:
        if block.type in ["triangle", "tri"]:
            tri = block.data
            break

    if tri is None or len(tri) == 0:
        raise ValueError(f"No volumetric cells AND no triangle surface cells found in {msh_filename}")

    # PyVista PolyData expects faces as: [3, i0, i1, i2, 3, j0, j1, j2, ...]
    faces = np.hstack([np.full((tri.shape[0], 1), 3), tri]).astype(np.int64).ravel()
    return pv.PolyData(points, faces)

def extract_surface_mesh_from_volume(pv_grid: pv.UnstructuredGrid) -> pv.PolyData:
    """
    Extract the surface mesh from a volumetric PyVista grid.
    
    Parameters:
        pv_grid: A PyVista UnstructuredGrid.
    
    Returns:
        The surface mesh as a PyVista PolyData object.
    """
    return pv_grid.extract_surface()

def write_msh_direct_from_stl(mesh_name, vertices, faces):
    with open(mesh_name, "w") as f:
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

def load_surface_mesh(input_filename: str, mesh_filename: str, element_size: float, size_factor: float, status_label: QLabel) -> pv.PolyData:
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
    if not os.path.isfile(input_filename):
        logging.error(f"Input file '{input_filename}' does not exist.")
        return None
    
    ext = os.path.splitext(input_filename)[1].lower()
    if ext == ".stl":
        try:
            surf_poly = pv.read(input_filename)
            logging.info(f"Loaded surface mesh from STL file: {input_filename}")
            status_label.setText(f"Converting STL to mesh: {os.path.basename(input_filename)}")
            QApplication.processEvents()
            vertices = surf_poly.points
            faces    = surf_poly.faces.reshape((-1, 4))[:, 1:]
            write_msh_direct_from_stl(mesh_filename, vertices, faces)
            return surf_poly
        except Exception as e:
            logging.error(f"Failed to load STL file '{input_filename}': {e}")
    elif ext in [".stp", ".step"]:
        mesh_step_file(input_filename, mesh_filename, element_size, size_factor, status_label)
        grid_or_poly = load_msh_as_pv_mesh(mesh_filename)
        if isinstance(grid_or_poly, pv.UnstructuredGrid):
            surf_poly = grid_or_poly.extract_surface()
        else:
            surf_poly = grid_or_poly
        return surf_poly
    elif ext in [".msh", ".mesh"]:
        try:
            status_label.setText(f"Loading .msh file: {os.path.basename(input_filename)}")
            QApplication.processEvents()
            pv_grid = load_msh_as_pv_mesh(mesh_filename)
            surf_poly = extract_surface_mesh_from_volume(pv_grid)
            return surf_poly
        except Exception as e:
            logging.error(f"Failed to load .msh file '{input_filename}': {e}")
            status_label.setText(f"Error: Failed to load .msh file: {e}")
    else:
        logging.error("Unsupported file format. Please use a .stp/.stl/.msh file.")


# -----------------------------------------------------------------------------
# Centerline and End Loop Extraction Functions
# -----------------------------------------------------------------------------

def _pca_axis(points: np.ndarray) -> np.ndarray:
    """
    _pca_axis computes the principal axis of a set of 3D points using PCA.
    What does this mean?:
    This function utilizes Principal Component Analysis (PCA) to determine the main direction.
    PCA states that principal axes are orthogonal for most forms and eliminate the need for complex calculations in cross products. 
    Using SVD (Singular Value Decomposition), we can find the direction of maximum variance. 
    Maximum variance will give us the longest axis of the coil... hopefully.
    """
    pts = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    axis = vt[0]
    axis = axis / np.linalg.norm(axis)
    return axis

def _component_perimeter(ds) -> float:
    """
    Robustly estimate 'perimeter'/length for a loop-like dataset.
    Works for PolyData and UnstructuredGrid.
    """
    if ds is None or ds.n_points == 0:
        return 0.0

    # Coerce to PolyData
    poly = ds if isinstance(ds, pv.PolyData) else ds.extract_surface()

    if poly is None or poly.n_points == 0 or poly.n_cells == 0:
        return 0.0

    # Ensure we have line-like geometry; if not, fall back to edges
    try:
        has_lines = poly.lines is not None and len(poly.lines) > 0
    except Exception:
        has_lines = False

    edge_poly = poly if has_lines else poly.extract_all_edges()

    if edge_poly is None or edge_poly.n_cells == 0:
        return 0.0

    sizes = edge_poly.compute_cell_sizes(length=True)
    # 'Length' is the per-cell length; sum it up
    return float(np.sum(sizes["Length"]))

def _pick_best_two_loops(loops: list[pv.PolyData], axis: np.ndarray):
    if len(loops) < 2:
        return None, None

    # Compute perimeter/length scores
    scored = [(lp, _component_perimeter(lp), lp.n_points) for lp in loops]
    scored.sort(key=lambda x: x[1], reverse=True)

    # print(f"Scores(pre sort): {scored}")

    # Reject tiny components (This may want to have tunable thresholds... haven't decided yet)
    MIN_PTS = 10          # keep loops with at least 10 points
    MIN_FRAC = 0.50       # keep loops with at least 50% of the max length

    max_len = scored[0][1] if scored else 0.0
    filtered = [(lp, ln) for (lp, ln, npts) in scored if npts >= MIN_PTS and ln >= MIN_FRAC * max_len]

    # print(f"Filtered(post sort): {filtered}")

    # If filtering leaves fewer than 2, fall back to the top-2 by length
    if len(filtered) < 2:
        top2 = [scored[i][0] for i in range(min(2, len(scored)))]
        return (top2[0], top2[1]) if len(top2) == 2 else (None, None)

    # From filtered candidates, choose the two farthest apart along axis
    candidates = [lp for lp, _ in filtered[: min(10, len(filtered))]]
    projs = [float(np.dot(lp.points.mean(axis=0), axis)) for lp in candidates]

    best_pair = None
    best_dist = -1.0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            d = abs(projs[i] - projs[j])
            if d > best_dist:
                best_dist = d
                best_pair = (candidates[i], candidates[j])

    return best_pair if best_pair else (None, None)

def _split_components(ds) -> list[pv.PolyData]:
    """
    Split a dataset into connected components (PolyData), returning a list of PolyData objects.
    This is to create "candidate loops" from edge extraction, in case of multiple loops being 
    found or fragmentation.
    """
    if ds is None or ds.n_points == 0:
        return []
    mb = ds.split_bodies()
    blocks = []
    if isinstance(mb, pv.MultiBlock):
        for i in range(len(mb)):
            b = mb[i]
            if b is None or b.n_points == 0:
                continue
            blocks.append(b if isinstance(b, pv.PolyData) else b.extract_surface())
    else:
        if mb is not None and mb.n_points > 0:
            blocks.append(mb if isinstance(mb, pv.PolyData) else mb.extract_surface())
    return blocks

def _slice_loops_near_ends(surf_poly: pv.PolyData, axis: np.ndarray, frac: float = 0.01) -> tuple[pv.PolyData, pv.PolyData] | tuple[None, None]:
    """
    Fallback for watertight meshes (no boundary edges).
    Slice near the two extremes along the principal axis and extract intersection polylines.
    Literally the last thing to try if all else fails, because it is less robust and not
    all coils will be capped nicely. 
    """
    pts = surf_poly.points
    t = pts @ axis
    tmin, tmax = float(t.min()), float(t.max())
    span = tmax - tmin
    if span <= 0:
        return None, None

    # pick planes slightly inboard from extremes
    d = max(frac * span, 1e-6)
    o1 = axis * (tmin + d)
    o2 = axis * (tmax - d)

    # Slice returns polylines where mesh intersects the plane
    s1 = surf_poly.slice(normal=axis, origin=o1).clean(tolerance=1e-6)
    s2 = surf_poly.slice(normal=axis, origin=o2).clean(tolerance=1e-6)

    # Sometimes slice yields multiple polylines; pick the largest by perimeter
    c1 = _split_components(s1)
    c2 = _split_components(s2)
    if not c1 or not c2:
        return None, None

    c1.sort(key=_component_perimeter, reverse=True)
    c2.sort(key=_component_perimeter, reverse=True)
    return c1[0], c2[0]

def extract_coil_end_loops(
    surf_poly: pv.PolyData,
    angle_threshold: float = 75.0,
    clean_tolerance: float = 1e-6,
    prefer_boundary_only: bool = True,
) -> tuple[pv.PolyData, pv.PolyData]:
    """
      1) Try boundary edges only
      2) Try boundary + feature edges
      3) If no boundary edges (watertight), fallback to slicing near ends
    """
    if surf_poly is None or surf_poly.n_points == 0:
        raise ValueError("surf_poly is empty.")

    axis = _pca_axis(surf_poly.points)

    # --- Pass 1: Boundary edges only ---
    if prefer_boundary_only:
        edges = surf_poly.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            feature_angle=angle_threshold
        ).clean(tolerance=clean_tolerance) 

        loops = _split_components(edges)

        if len(loops) == 2:
            return loops[0], loops[1]

        if len(loops) > 2:
            loopA, loopB = _pick_best_two_loops(loops, axis)
            if loopA is not None and loopB is not None:
                return loopA, loopB

        # If prefer_boundary_only is True but we didn't find two loops, continue to Pass 2.
        # ... We will almost always use pass 2 and not pass 1

    # --- Pass 2: Boundary + feature edges (to catch fragmented ends) ---
    edges = surf_poly.extract_feature_edges(
        boundary_edges=True,
        feature_edges=True,
        manifold_edges=False,
        feature_angle=angle_threshold
    ).clean(tolerance=clean_tolerance)

    loops = _split_components(edges)

    # This should show available loops. The program should be using the best two below.
    # for i, lp in enumerate(loops):
    #     print(f"[DEBUG] loop {i}: n_points={lp.n_points}, n_cells={lp.n_cells}")

    if len(loops) == 2:
        return loops[0], loops[1]

    if len(loops) > 2:
        print("TOO MANY LOOPS")
        loopA, loopB = _pick_best_two_loops(loops, axis)
        if loopA is not None and loopB is not None:
            return loopA, loopB

    # --- Pass 3 fallback: watertight mesh or bad boundaries -> slice near ends ---
    loopA, loopB = _slice_loops_near_ends(surf_poly, axis)
    if loopA is not None and loopB is not None:
        return loopA, loopB

    raise ValueError(
        "Could not identify two end loops. "
        "Try: (1) repairing STL to ensure open ends, "
        "(2) increasing mesh resolution, "
        "(3) adjusting angle_threshold, "
        "(4) verify the mesh is not watertight if you expect open ends."
    )

def split_connected_loops(poly: pv.PolyData) -> list[pv.PolyData]:
    """
    Split a (polyline) PolyData into connected components (loops/curves),
    tolerating PyVista versions where RegionId may be in point_data instead of cell_data.
    Returns a list of PolyData, one per connected component.
    """
    if poly is None or poly.n_cells == 0:
        return []

    # First try standard cell-based connectivity
    labeled = poly.connectivity(point_data=False)
    rid_cell = labeled.cell_data.get('RegionId', None)

    loops = []
    if rid_cell is not None:
        # Split by cell RegionId
        rids = np.unique(rid_cell)
        for rid in rids:
            # Select cells with this region id
            mask = (rid_cell == rid)
            cell_ids = np.where(mask)[0]
            if cell_ids.size == 0:
                continue
            sub = labeled.extract_cells(cell_ids)
            if sub is not None and sub.n_cells > 0 and sub.n_points >= 2:
                loops.append(sub)

        return loops

    # Fallback: point-based connectivity
    labeled = poly.connectivity(point_data=True)
    rid_point = labeled.point_data.get('RegionId', None)
    if rid_point is None:
        # Last resort: return the input as single component
        return [poly]

    rids = np.unique(rid_point)
    for rid in rids:
        pmask = (rid_point == rid)
        sub = labeled.extract_points(pmask, adjacent_cells=True)
        if sub is not None and sub.n_cells > 0 and sub.n_points >= 2:
            loops.append(sub)

    return loops

def map_poly_points_to_surface_indices(
    loop_poly: pv.PolyData,
    surf_poly: pv.PolyData,
    tol: float | None = None
) -> set[int]:
    """
    Map loop_poly points to nearest vertex indices in surf_poly, with tolerance.
    This replaces brittle exact coordinate matching.

    tol: maximum allowed distance from a loop point to its mapped surface vertex.
         If None, uses a scale-aware default based on surf_poly bbox diagonal.
    """
    if loop_poly is None or loop_poly.n_points == 0:
        return set()
    if surf_poly is None or surf_poly.n_points == 0:
        return set()

    surf_pts = np.asarray(surf_poly.points)
    loop_pts = np.asarray(loop_poly.points)

    if tol is None:
        xmin, xmax, ymin, ymax, zmin, zmax = surf_poly.bounds
        diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
        tol = max(1e-6, 1e-4 * diag)  # adjust to 1e-3*diag if still failing

    # Fast path if SciPy exists
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(surf_pts)
        dists, idxs = tree.query(loop_pts, k=1)
        # print(f"[DEBUG] map distances: min={dists.min():.3e} med={np.median(dists):.3e} max={dists.max():.3e} tol={tol:.3e}") # If tol of mid and max are >> tol, then mapping is failing.
        return {int(i) for d, i in zip(dists, idxs) if d <= tol}
    except Exception:
        # Fallback using PyVista's closest point
        out = set()
        for p in loop_pts:
            i = int(surf_poly.find_closest_point(p))
            if float(np.linalg.norm(surf_pts[i] - p)) <= tol:
                out.add(i)
        return out


def compute_marching_rings(surf_poly: pv.PolyData,
                           loopA: pv.PolyData,
                           loopB: pv.PolyData,
                           status_label= QLabel) -> list[set]:
    """
    Return marching_record (list of vertex-index sets) without collapsing
    them into center points.
    """
    status_label.setText("Computing marching rings...")
    QApplication.processEvents()
    pv_faces = surf_poly.faces.reshape((-1, 4))[:, 1:]
    vertices = surf_poly.points

    status_label.setText("Mapping end loops to surface vertices...")
    QApplication.processEvents()
    loopA_plane_centroid, loopA_plane_normal = compute_best_fit_plane(loopA.points)
    loopB_plane_centroid, loopB_plane_normal = compute_best_fit_plane(loopB.points)

    status_label.setText("Selecting end vertices near planes...")
    QApplication.processEvents()
    endA_vertex_indices = select_vertices_near_plane(
        surf_poly, loopA_plane_centroid, loopA_plane_normal, TOL
    )
    endB_vertex_indices = select_vertices_near_plane(
        surf_poly, loopB_plane_centroid, loopB_plane_normal, TOL
    )
    
    status_label.setText("Identifying active vertices...")
    QApplication.processEvents()
    all_vertex_indices = set(range(len(vertices)))
    inactive_vertex_indices = set(endA_vertex_indices).union(set(endB_vertex_indices))
    active_vertex_indices = all_vertex_indices - inactive_vertex_indices

    pv_faces_set = set()
    for f_idx, face in enumerate(pv_faces):
        if any(v in active_vertex_indices for v in face):
            pv_faces_set.add(f_idx)

    status_label.setText("Mapping loop points to surface vertices...")
    QApplication.processEvents()
    V_mov = map_poly_points_to_surface_indices(loopA, surf_poly, tol=None)
    V_ref = map_poly_points_to_surface_indices(loopB, surf_poly, tol=None)

    # This should tell you whether mapping is working. If the numbers are extremely small (i.e. loops aren't being found), that may be why something fails later.
    # print(
    #     f"[DEBUG] marching rings mapping:"
    #     f" V_mov={len(V_mov)}"
    #     f" V_ref={len(V_ref)}"
    # )

    if len(V_mov) == 0 or len(V_ref) == 0:
        raise ValueError(
            f"Could not map loop points to surface vertices "
            f"(V_mov={len(V_mov)}, V_ref={len(V_ref)}). "
            "This is typically caused by edge cleaning/feature extraction altering point coordinates. "
            "Increase mapping tolerance or reduce clean_tolerance / prefer boundary-only loops."
        )

    def _external_edges_and_vertices(faces_subset: set):
        e2f = defaultdict(list)
        for f_idx in faces_subset:
            tri = pv_faces[f_idx]
            for edge in [tuple(sorted((tri[0], tri[1]))),
                         tuple(sorted((tri[1], tri[2]))),
                         tuple(sorted((tri[2], tri[0])))]:
                e2f[edge].append(f_idx)
        E_ext = [e for e, flist in e2f.items() if len(flist) == 1]
        V_ext = {v for e in E_ext for v in e}
        return E_ext, V_ext

    visited = set()
    marching_record = []
    current_moving = set(V_mov)
    total_len = len(pv_faces_set)

    while True:
        visited |= current_moving
        new_active_faces = {
            f_idx for f_idx in pv_faces_set
            if not any(v in visited for v in pv_faces[f_idx])
        }
        if not new_active_faces:
            break

        _, V_ext_new = _external_edges_and_vertices(new_active_faces)

        new_moving = V_ext_new - V_ref
        if not new_moving:
            break

        marching_record.append(new_moving.copy())
        current_moving = new_moving

        if len(marching_record) % 10 == 0:
            status_label.setText(
                f"Computing marching rings... "
                f"processed {len(marching_record)} rings, "
                f"{len(new_active_faces)}/{total_len} faces remaining."
            )
            QApplication.processEvents()

    return marching_record

def vertex_set_to_points(surf_poly: pv.PolyData, vset: set) -> np.ndarray:
    """
    Map a set of vertex indices to xyz points.
    This function is used to help display the data in meshing.
    """
    verts = surf_poly.points
    return np.array([verts[v] for v in vset], dtype=float)


def ring_to_polyline(ring_points: np.ndarray,
                     n_points: int = 200,
                     smoothing: float = 0.0) -> pv.PolyData:
    """
    Turn a noisy ring of points into an ordered, smooth closed polyline,
    reusing existing helpers (order_loop_points_pca, refine_loop, create_polyline).
    """
    ordered = order_loop_points_pca(ring_points)
    refined = refine_loop(pv.PolyData(ordered), n_points=n_points,
                          smoothing=smoothing, spline_degree=3)
    return create_polyline(refined, closed=True)

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

def smooth_centerline(centerline_points: np.ndarray, s: float = 1.0, k: int = 3, n_interp: int = 200) -> np.ndarray:
    """
    Smooth the centerline using spline interpolation with endpoint weighting.
    
    Parameters:
        centerline_points: Array of centerline points.
        s: Smoothing factor.
        k: Spline degree.
        n_interp: Number of interpolated points.
    
    Returns:
        Smoothed centerline points.
    """
    if centerline_points.shape[0] < 3:
        return centerline_points.copy()
    x, y, z = centerline_points.T
    weights = np.ones(centerline_points.shape[0])
    weights[0] = ENDPOINT_WEIGHT
    weights[-1] = ENDPOINT_WEIGHT
    diffs = np.diff(centerline_points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    t = np.concatenate(([0], np.cumsum(seg_lengths)))
    total_length = t[-1]
    if total_length < EPS:
        return centerline_points.copy()
    t /= total_length
    try:
        tck, _ = splprep([x, y, z], u=t, w=weights, s=s, k=k)
        u_new = np.linspace(0, 1, n_interp)
        x_new, y_new, z_new = splev(u_new, tck)
        smoothed_points = np.vstack((x_new, y_new, z_new)).T
        return smoothed_points
    except Exception as e:
        logging.error(f"Centerline smoothing failed: {e}")
        return centerline_points.copy()

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
    plotter.add_mesh(pv.PolyData(raw_centerline), color='blue', line_width=3, label='Centerline')
    plotter.add_mesh(loopA, color="red", line_width=2, label="Start Loop")
    plotter.add_mesh(loopB, color="green", line_width=2, label="End Loop")

    for idx, v_set in enumerate(marching_record):
        if idx % step == 0:
            pts = np.array([surface_mesh.points[v] for v in v_set])
            plotter.add_points(pts, color='red', point_size=8, render_points_as_spheres=True,
                               label=f'Marching Record {idx}' if idx == 0 else None)

    plotter.add_legend()
    plotter.reset_camera()
    plotter.render()
    plotter.show()