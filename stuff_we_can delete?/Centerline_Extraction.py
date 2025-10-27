#!/usr/bin/env python3
import os
import gmsh
import numpy as np
import meshio
import pyvista as pv
from collections import defaultdict
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

##############################################################################
# 1. Mesh the STEP file with Gmsh
##############################################################################
def mesh_step_file(step_filename, mesh_filename="ribbonwire_v1.msh", element_size=0.2):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.open(step_filename)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size)
    gmsh.model.mesh.generate(3)
    gmsh.write(mesh_filename)
    gmsh.finalize()

##############################################################################
# 2. Convert Gmsh output to PyVista UnstructuredGrid
##############################################################################
def load_msh_as_pv_mesh(msh_filename="ribbonwire_v1.msh"):
    mesh = meshio.read(msh_filename)
    points = mesh.points
    cells = []
    cell_types = []
    for c in mesh.cells:
        ctype = c.type
        if ctype in ["tetra", "hexahedron", "wedge", "pyramid"]:
            for cell_block in c.data:
                ca = np.concatenate(([len(cell_block)], cell_block))
                cells.append(ca)
            if ctype == "tetra":
                cell_type_id = 10
            elif ctype == "hexahedron":
                cell_type_id = 12
            elif ctype == "wedge":
                cell_type_id = 13
            elif ctype == "pyramid":
                cell_type_id = 14
            cell_types.extend([cell_type_id]*len(c.data))
    if len(cells) == 0:
        raise ValueError("No 3D volumetric cells found in the mesh.")
    cells = np.hstack(cells)
    cell_types = np.array(cell_types)
    pv_grid = pv.UnstructuredGrid(cells, cell_types, points)
    return pv_grid

def extract_surface_mesh_from_volume(pv_grid):
    return pv_grid.extract_surface()

##############################################################################
# 3. Extract Coil End Loops (feature edges)
##############################################################################
def extract_coil_end_loops(surf_poly, feature_angle=65.0):
    edges_poly = surf_poly.extract_feature_edges(
        boundary_edges=True,
        feature_edges=True,
        manifold_edges=False,
        feature_angle=feature_angle
    )
    split_edges = edges_poly.split_bodies()
    loops = []
    if isinstance(split_edges, pv.MultiBlock):
        loops = [split_edges[i] for i in range(len(split_edges))]
    else:
        loops = [split_edges]
    n_loops = len(loops)
    print(f"[INFO] Found {n_loops} disconnected edge loop(s).")
    if n_loops != 2:
        plotter = pv.Plotter()
        plotter.add_mesh(surf_poly, color="blue", opacity=0.5, label="Surface Mesh")
        for i, loop in enumerate(loops):
            centroid = np.mean(loop.points, axis=0)
            print(f"  Loop {i}: {loop.n_points} points, Centroid = {centroid}")
            plotter.add_mesh(loop, color="red", line_width=4, label=f"Loop {i}")
        plotter.show(title="Debug: Surface Mesh + Detected Loops")
        raise ValueError(
            f"Expected exactly 2 end loops, but found {n_loops}. "
            "Please check your geometry or feature_angle."
        )
    return loops

##############################################################################
# 4. 3D-MCE-like Centerline Extraction Function
##############################################################################
def compute_centerline_3d_mce(surf_poly, loopA, loopB):
    """
    Modified version of the centerline extraction function that records
    the indices of surface mesh vertices involved in each marching step.
    """
    pv_faces = surf_poly.faces.reshape((-1, 4))[:, 1:]
    vertices = surf_poly.points
    faces = pv_faces
    edge_to_faces = defaultdict(list)
    for f_idx, tri in enumerate(faces):
        i1, i2, i3 = tri
        for edge in [tuple(sorted((i1, i2))),
                     tuple(sorted((i2, i3))),
                     tuple(sorted((i3, i1)))]:
            edge_to_faces[edge].append(f_idx)

    def find_vertex_indices_in_polydata(src_poly, all_points):
        idxs = []
        coord_to_index = {tuple(pt): i for i, pt in enumerate(all_points)}
        for pt in src_poly.points:
            idxs.append(coord_to_index[tuple(pt)])
        return set(idxs)

    # Use loopA as moving, loopB as reference.
    V_mov = find_vertex_indices_in_polydata(loopA, vertices)
    V_ref = find_vertex_indices_in_polydata(loopB, vertices)
    
    visited = set()
    moving_sections = []
    marching_record = []
    current_moving = set(V_mov)
    active_faces = set(range(len(faces)))
    def get_external_edges_and_vertices(faces_subset):
        e2f = defaultdict(list)
        for f_idx in faces_subset:
            tri = faces[f_idx]
            for edge in [tuple(sorted((tri[0], tri[1]))),
                         tuple(sorted((tri[1], tri[2]))),
                         tuple(sorted((tri[2], tri[0])))]:
                e2f[edge].append(f_idx)
        E_ext = [e for e, flist in e2f.items() if len(flist)==1]
        V_ext = set()
        for (v1, v2) in E_ext:
            V_ext.add(v1)
            V_ext.add(v2)
        return E_ext, V_ext

    while True:
        visited |= current_moving
        new_active_faces = set()
        for f_idx in active_faces:
            tri = faces[f_idx]
            if not (tri[0] in visited or tri[1] in visited or tri[2] in visited):
                new_active_faces.add(f_idx)
        if not new_active_faces:
            break
        E_ext_new, V_ext_new = get_external_edges_and_vertices(new_active_faces)
        new_moving = V_ext_new - V_ref
        if len(new_moving) == 0:
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
    centerline_points = np.vstack(centerline_points) if centerline_points else None
    return centerline_points, marching_record

##############################################################################
# 5. Centerline Filtering and Smoothing
##############################################################################
def filter_centerline_by_angle(raw_points, angle_threshold_degrees=30.0):
    if raw_points is None or len(raw_points) < 3:
        return raw_points
    angle_threshold = np.radians(angle_threshold_degrees)
    filtered = [raw_points[0], raw_points[1]]
    for i in range(2, len(raw_points)):
        p_prev = filtered[-1]
        p_prev2 = filtered[-2]
        p_new = raw_points[i]
        v1 = p_prev - p_prev2
        v2 = p_new - p_prev
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-12 or norm2 < 1e-12:
            filtered.append(p_new)
            continue
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle <= angle_threshold:
            filtered.append(p_new)
        else:
            print(f"[DEBUG] Skipping point {i} due to kink angle = {np.degrees(angle):.1f}°")
    return np.array(filtered)

def smooth_centerline(centerline_points, s=1.0, k=3, n_interp=200):
    if centerline_points.shape[0] < 3:
        return centerline_points.copy()
    x, y, z = centerline_points.T
    weights = np.ones(centerline_points.shape[0], dtype=float)
    weights[0] = 1000.0
    weights[-1] = 1000.0
    diffs = np.diff(centerline_points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    t = np.concatenate(([0], np.cumsum(seg_lengths)))
    total_length = t[-1]
    if total_length < 1e-12:
        return centerline_points.copy()
    t /= total_length
    tck, _ = splprep([x, y, z], u=t, w=weights, s=s, k=k)
    u_new = np.linspace(0, 1, n_interp)
    x_new, y_new, z_new = splev(u_new, tck)
    smoothed_points = np.vstack((x_new, y_new, z_new)).T
    return smoothed_points

##############################################################################
# 6. Loop Ordering and Refinement Helpers
##############################################################################
def order_loop_points_pca(points, debug=False):
    centroid = points.mean(axis=0)
    centered = points - centroid
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:2].T
    projected = centered.dot(basis)
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    sorted_indices = np.argsort(angles)
    ordered_points = points[sorted_indices]
    if not np.allclose(ordered_points[0], ordered_points[-1]):
        ordered_points = np.vstack([ordered_points, ordered_points[0]])
    if debug:
        plt.figure()
        plt.scatter(projected[:, 0], projected[:, 1], c='blue', label='Projected Points')
        projected_ordered = projected[sorted_indices]
        if not np.allclose(projected_ordered[0], projected_ordered[-1]):
            projected_ordered = np.vstack([projected_ordered, projected_ordered[0]])
        plt.plot(projected_ordered[:, 0], projected_ordered[:, 1], 'r-', label='Ordered')
        plt.title("PCA Projection Ordering")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.show()
    return ordered_points

def create_polyline(points):
    n_points = len(points)
    connectivity = np.hstack([[n_points], np.arange(n_points)])
    poly = pv.PolyData()
    poly.points = points
    poly.lines = connectivity
    return poly

def refine_loop(loop_poly, n_points=200, smoothing=0.0, spline_degree=3):
    pts = loop_poly.points.copy()
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_arc = np.concatenate(([0], np.cumsum(seg_lengths)))
    total_length = cum_arc[-1]
    u = cum_arc / total_length
    tck, _ = splprep([pts[:,0], pts[:,1], pts[:,2]], u=u, s=smoothing, k=spline_degree, per=True)
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

def visualize_original_and_refined(loop_poly, n_refined_points=200, smoothing=0.0, spline_degree=3, debug_order=False):
    original_points = loop_poly.points.copy()
    ordered_original = order_loop_points_pca(original_points, debug=debug_order)
    original_poly = create_polyline(ordered_original)
    temp_poly = pv.PolyData(ordered_original)
    refined_points = refine_loop(temp_poly, n_points=n_refined_points, smoothing=smoothing, spline_degree=spline_degree)
    refined_poly = create_polyline(refined_points)
    return refined_poly

##############################################################################
# 7. Generate and Visualize Intermediate Contours
##############################################################################
def generate_intermediate_contours(refined_points, centerpoint, n_contours=5):
    alphas = np.linspace(0, 1, n_contours + 2)[1:-1]
    contours = []
    for a in alphas:
        contour = centerpoint + a*(refined_points - centerpoint)
        if not np.allclose(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])
        contours.append(contour)
    return contours

def visualize_contours(contours, plotter=None):
    if plotter is None:
        plotter = pv.Plotter()
    colors = ['yellow', 'orange', 'green', 'cyan', 'blue', 'purple']
    for i, pts in enumerate(contours):
        poly = create_polyline(pts)
        color = colors[i % len(colors)]
        plotter.add_mesh(poly, color=color, line_width=2, label=f"Contour {i}")
    return plotter

##############################################################################
# Helper: Average Two Centerlines Indexwise
##############################################################################
def average_centerlines(centerline1, centerline2):
    return (centerline1 + centerline2) / 2.0

##############################################################################
# MAIN Script
##############################################################################
def main():
    step_file = "coil_taper.stp"
    msh_file = "coil_taper.msh"

    # Remove old mesh file if exists.
    if os.path.exists(msh_file):
        print(f"[INFO] Removing old mesh file '{msh_file}'...")
        os.remove(msh_file)
    print("[INFO] Meshing STEP file with Gmsh...")
    mesh_step_file(step_file, msh_file, element_size=0.25)

    print("[INFO] Loading mesh into PyVista...")
    pv_grid = load_msh_as_pv_mesh(msh_file)

    print("[INFO] Extracting surface mesh...")
    surf_poly = extract_surface_mesh_from_volume(pv_grid)

    print("[INFO] Extracting coil end loops...")
    loops = extract_coil_end_loops(surf_poly, feature_angle=75.0)
    loopA, loopB = loops[0], loops[1]

    print("[INFO] Computing forward centerline (loopA as moving, loopB as reference)...")
    raw_centerline_forward, marching_record = compute_centerline_3d_mce(surf_poly, loopA, loopB)
    if raw_centerline_forward is None or len(raw_centerline_forward) == 0:
        print("[WARNING] No forward centerline points found.")
        return
    print(f"[RESULT] Forward centerline has {len(raw_centerline_forward)} points.")

    print("[INFO] Computing reverse centerline (loopB as moving, loopA as reference)...")
    raw_centerline_reverse, _ = compute_centerline_3d_mce(surf_poly, loopB, loopA)
    if raw_centerline_reverse is None or len(raw_centerline_reverse) == 0:
        print("[WARNING] No reverse centerline points found.")
        return
    print(f"[RESULT] Reverse centerline has {len(raw_centerline_reverse)} points.")

    print("[INFO] Filtering centerlines to remove kinked regions...")
    filtered_centerline_forward = filter_centerline_by_angle(raw_centerline_forward, angle_threshold_degrees=45.0)
    filtered_centerline_reverse = filter_centerline_by_angle(raw_centerline_reverse, angle_threshold_degrees=45.0)
    print(f"[RESULT] Forward centerline filtered to {len(filtered_centerline_forward)} points.")
    print(f"[RESULT] Reverse centerline filtered to {len(filtered_centerline_reverse)} points.")

    print("[INFO] Smoothing forward centerline...")
    smoothed_centerline_forward = smooth_centerline(filtered_centerline_forward, s=0.01, k=3, n_interp=200)
    print("[INFO] Smoothing reverse centerline...")
    smoothed_centerline_reverse = smooth_centerline(filtered_centerline_reverse, s=0.01, k=3, n_interp=200)

    smoothed_centerline_reverse = smoothed_centerline_reverse[::-1]

    final_centerline = average_centerlines(smoothed_centerline_forward, smoothed_centerline_reverse)
    final_centerline_poly = create_polyline(final_centerline)

    print("[INFO] Refining end loop A with PCA ordering...")
    refined_loopA_poly = create_polyline(refine_loop(pv.PolyData(order_loop_points_pca(loopA.points)), n_points=50, smoothing=0.0, spline_degree=3))
    print("[INFO] Refining end loop B with PCA ordering...")
    refined_loopB_poly = create_polyline(refine_loop(pv.PolyData(order_loop_points_pca(loopB.points)), n_points=50, smoothing=0.0, spline_degree=3))

    centerpoint_A = final_centerline[0]
    centerpoint_B = final_centerline[-1]
    refined_loopA_points = refined_loopA_poly.points
    refined_loopB_points = refined_loopB_poly.points
    contours_A = generate_intermediate_contours(refined_loopA_points, centerpoint_A, n_contours=5)
    contours_B = generate_intermediate_contours(refined_loopB_points, centerpoint_B, n_contours=5)

    # (Optional) Visualize the full model.
    plotter = pv.Plotter()
    plotter.add_mesh(surf_poly, color="lightblue", opacity=0.5, label="Surface Mesh")
    plotter.add_mesh(final_centerline_poly, color="magenta", line_width=3, label="Averaged Centerline")
    plotter.add_mesh(refined_loopA_poly, color="red", line_width=4, label="Refined Loop A")
    plotter.add_mesh(refined_loopB_poly, color="green", line_width=4, label="Refined Loop B")
    plotter = visualize_contours(contours_A, plotter=plotter)
    plotter = visualize_contours(contours_B, plotter=plotter)
    plotter.add_legend(bcolor="white")
    plotter.show(title="Combined Visualization: Surface Mesh, Averaged Centerline, Refined End Loops & Intermediate Contours")

    # Save the centerline and surface mesh for later use.
    np.save("averaged_centerline.npy", final_centerline)
    print("[INFO] Averaged centerline saved to 'averaged_centerline.npy'.")
    surf_poly.save("surface_mesh.vtk")
    print("[INFO] Surface mesh saved to 'surface_mesh.vtk'.")

if __name__ == "__main__":
    main()
