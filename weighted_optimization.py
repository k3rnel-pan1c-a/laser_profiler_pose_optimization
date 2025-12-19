"""
Weighted CMA-ES optimization for laser profiler placement.

This module assigns higher importance to points that are inside specific submeshes
(e.g., blade/cutter regions) of the drill bit.
"""

import open3d as o3d
import numpy as np
import trimesh
import cma


class Profiler:
    """Laser profiler sensor configuration."""

    def __init__(self, C, forward, n_plane, ang_min, ang_max, rmin, rmax):
        self.C = C
        self.forward = forward
        self.n_plane = n_plane
        self.ang_min = ang_min
        self.ang_max = ang_max
        self.rmin = rmin
        self.rmax = rmax


def rotz(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rot_axis_angle(axis, angle):
    """Rodrigues' rotation formula as a matrix"""
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def laser_plane_filter(points, C, n_plane, eps=1e-3):
    d = np.abs((points - C) @ n_plane)
    return d < eps


def vertical_angle_filter(points, C, forward, ang_min, ang_max):
    v = points - C
    v_norm = np.linalg.norm(v, axis=1) + 1e-9
    forward_norm = np.linalg.norm(forward) + 1e-9
    cos_angles = (v @ forward) / (v_norm * forward_norm)
    angles = np.arccos(np.clip(cos_angles, -1, 1))
    return angles <= ang_max


def range_filter(points, C, rmin=0, rmax=100):
    dist = np.linalg.norm(points - C, axis=1)
    return (dist >= rmin) & (dist <= rmax)


def apply_occlusion_and_collect_rays(scene, points, C, eps=1e-3):
    v = points - C
    dist = np.linalg.norm(v, axis=1)
    v_unit = v / (dist[:, None] + 1e-9)
    rays = np.hstack([C[None, :].repeat(len(points), axis=0), v_unit]).astype(
        np.float32
    )
    ans = scene.cast_rays(o3d.core.Tensor(rays, o3d.core.Dtype.Float32))
    t_hit = ans["t_hit"].numpy()
    visible_mask = np.abs(t_hit - dist) < eps
    return visible_mask, v_unit, t_hit


def simulate_single_profiler(mesh, base_points, profiler, rotation_angles):
    C = profiler.C
    f = profiler.forward
    n_plane = profiler.n_plane
    ang_min = profiler.ang_min
    ang_max = profiler.ang_max
    rmin = profiler.rmin
    rmax = profiler.rmax

    covered = np.zeros(len(base_points), dtype=bool)

    for phi in rotation_angles:
        R = rotz(phi)
        points = (R @ base_points.T).T

        rotated_mesh = o3d.geometry.TriangleMesh(mesh)
        rotated_mesh.rotate(R, center=[0, 0, 0])

        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(rotated_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tmesh)

        idx = np.where(laser_plane_filter(points, C, n_plane))[0]
        if len(idx) == 0:
            continue

        idx = idx[vertical_angle_filter(points[idx], C, f, ang_min, ang_max)]
        if len(idx) == 0:
            continue

        idx = idx[range_filter(points[idx], C, rmin, rmax)]
        if len(idx) == 0:
            continue

        visible, dirs, t_hit = apply_occlusion_and_collect_rays(scene, points[idx], C)
        idx_visible = idx[visible]
        covered[idx_visible] = True

    return covered


def simulate_all_profilers(mesh, points, profilers, rotation_angles):
    covered = np.zeros(len(points), dtype=bool)
    for p in profilers:
        c = simulate_single_profiler(mesh, points, p, rotation_angles)
        covered |= c
    return covered


def decode_theta_to_profilers(theta, N, fixed_params):
    """Decode optimization vector into N Profiler objects."""
    profilers = []
    for i in range(N):
        base = 6 * i
        rho = theta[base + 0]
        azimuth = theta[base + 1]
        z = theta[base + 2]
        yaw = theta[base + 3]
        pitch = theta[base + 4]
        roll = theta[base + 5]

        C = np.array([rho * np.cos(azimuth), rho * np.sin(azimuth), z])

        base_forward = -C.copy()
        base_forward[2] = 0
        if np.linalg.norm(base_forward) < 1e-6:
            base_forward = np.array([1, 0, 0])
        else:
            base_forward = base_forward / np.linalg.norm(base_forward)

        up_world = np.array([0, 0, 1])
        right = np.cross(up_world, base_forward)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([0, 1, 0])
        else:
            right = right / np.linalg.norm(right)

        local_up = np.cross(base_forward, right)
        local_up = local_up / np.linalg.norm(local_up)

        R_yaw = rot_axis_angle(local_up, yaw)
        forward = R_yaw @ base_forward
        right = R_yaw @ right

        R_pitch = rot_axis_angle(right, pitch)
        forward = R_pitch @ forward
        local_up = R_pitch @ local_up

        forward = forward / np.linalg.norm(forward)

        R_roll = rot_axis_angle(forward, roll)
        local_up = R_roll @ local_up
        right = R_roll @ right

        local_up = local_up / np.linalg.norm(local_up)
        right = right / np.linalg.norm(right)

        n_plane = right

        profilers.append(
            Profiler(
                C=C,
                forward=forward,
                n_plane=n_plane,
                ang_min=fixed_params["ang_min"],
                ang_max=fixed_params["ang_max"],
                rmin=fixed_params["rmin"],
                rmax=fixed_params["rmax"],
            )
        )

    return profilers


def assign_weights_from_submeshes(
    points, submesh_file, high_weight=10.0, low_weight=1.0, submesh_indices=None
):
    """
    Assign higher weights to points that are inside submeshes (e.g., blade regions).

    Uses trimesh's contains() method to check if points are inside each submesh.

    Args:
        points: Nx3 array of point coordinates
        submesh_file: Path to mesh file with submeshes (e.g., merged_blade_mapping_big.obj)
        high_weight: Weight for points inside any submesh
        low_weight: Weight for points outside all submeshes
        submesh_indices: Optional list of submesh indices to use. If None, uses all submeshes.

    Returns:
        weights: N-length array of weights
        inside_mask: Boolean mask of points inside any submesh
        submesh_labels: N-length array with submesh index (-1 if not in any submesh)
    """
    print(f"Loading submesh file: {submesh_file}")
    mesh = trimesh.load_mesh(submesh_file)
    submeshes = mesh.split()

    print(f"Found {len(submeshes)} submeshes")

    # Determine which submeshes to use
    if submesh_indices is None:
        submesh_indices = list(range(len(submeshes)))

    print(f"Using submesh indices: {submesh_indices}")

    # Initialize
    weights = np.ones(len(points)) * low_weight
    inside_any = np.zeros(len(points), dtype=bool)
    submesh_labels = np.full(len(points), -1, dtype=np.int32)

    # Check each submesh
    for idx in submesh_indices:
        if idx >= len(submeshes):
            print(f"  Warning: submesh index {idx} out of range, skipping")
            continue

        submesh = submeshes[idx]
        print(
            f"  Checking submesh {idx} ({len(submesh.vertices)} vertices, {len(submesh.faces)} faces)..."
        )

        try:
            inside_mask = submesh.contains(points)
            n_inside = inside_mask.sum()
            print(
                f"    Found {n_inside} points inside ({100 * n_inside / len(points):.1f}%)"
            )

            # Update labels (first submesh that contains the point wins)
            new_inside = inside_mask & (submesh_labels == -1)
            submesh_labels[new_inside] = idx

            inside_any |= inside_mask

        except Exception as e:
            print(f"    Error checking submesh {idx}: {e}")

    # Assign weights
    weights[inside_any] = high_weight

    total_inside = inside_any.sum()
    print(
        f"\nTotal points inside submeshes: {total_inside} ({100 * total_inside / len(points):.1f}%)"
    )
    print(f"Points outside all submeshes: {len(points) - total_inside}")

    return weights, inside_any, submesh_labels


# =============================================================================
# WEIGHTED OBJECTIVE FUNCTION
# =============================================================================


def weighted_objective(
    theta,
    mesh,
    points,
    rotation_angles,
    N,
    fixed_params,
    weights,
):
    """
    Weighted objective function for CMA-ES optimization.

    Points with higher weights contribute more to the coverage score.

    Score = sum(covered * weights) / sum(weights)

    Args:
        theta: Optimization parameters
        mesh: Triangle mesh (Open3D)
        points: Point cloud (Nx3 array)
        rotation_angles: Drill bit rotation angles
        N: Number of profilers
        fixed_params: Fixed profiler parameters
        weights: Per-point importance weights (N-length array)

    Returns:
        Negative score (for minimization)
    """
    profilers = decode_theta_to_profilers(theta, N, fixed_params)
    covered = simulate_all_profilers(mesh, points, profilers, rotation_angles)

    # Weighted coverage (normalized by total weight)
    weighted_coverage = (covered * weights).sum() / weights.sum()

    return -weighted_coverage  # Negative because CMA-ES minimizes


def compute_detailed_coverage(covered, weights):
    """Compute detailed coverage statistics."""
    weighted_cov = (covered * weights).sum() / weights.sum()
    unweighted_cov = covered.mean()

    # Coverage by weight category
    unique_weights = np.unique(weights)
    coverage_by_weight = {}
    for w in unique_weights:
        mask = weights == w
        if mask.sum() > 0:
            coverage_by_weight[w] = covered[mask].mean()

    return {
        "weighted": weighted_cov,
        "unweighted": unweighted_cov,
        "by_weight": coverage_by_weight,
    }


def main():
    MESH_FILE = "dataset/cropped_mesh.ply"
    SUBMESH_FILE = "dataset/cutter_mesh.ply"

    N_PROFILERS = 4

    HIGH_WEIGHT = 10.0
    LOW_WEIGHT = 1.0

    print("Loading main mesh...")
    mesh = o3d.io.read_triangle_mesh(MESH_FILE)
    if not mesh.has_triangles():
        print("  Note: File may be a point cloud, attempting to load...")
    mesh.compute_vertex_normals()

    # mesh.translate(-mesh.get_center())

    print("Loading/sampling points...")
    pcd = o3d.io.read_point_cloud(MESH_FILE)
    points = np.asarray(pcd.points)

    # Center points same as mesh
    # points = points - points.mean(axis=0)

    # Subsample for optimization (faster)
    n_opt_points = 5000
    if len(points) > n_opt_points:
        indices = np.random.choice(len(points), n_opt_points, replace=False)
        points = points[indices]
    print(f"  Using {len(points)} points for optimization")

    rotation_angles = np.deg2rad(np.arange(0, 360, 5))

    fixed_params = {
        "ang_min": np.deg2rad(0),
        "ang_max": np.deg2rad(15),
        "rmin": 0.15,
        "rmax": 0.3,
    }

    print("\n" + "=" * 60)
    print("ASSIGNING WEIGHTS FROM SUBMESHES")
    print("=" * 60)

    weights, _, _ = assign_weights_from_submeshes(
        points,
        SUBMESH_FILE,
        high_weight=HIGH_WEIGHT,
        low_weight=LOW_WEIGHT,
        submesh_indices=None,  # Use all submeshes, or specify e.g., [0, 1, 2, 13]
    )

    x0 = []
    for i in range(N_PROFILERS):
        x0.extend([0.3, 0.0, 0.1, 0.0, 0.0, 0.0])
    x0 = np.array(x0)

    # Bounds
    lower_bounds = [0.3, -np.pi, 0.0, -np.pi / 4, -np.pi / 3, -np.pi / 2] * N_PROFILERS
    upper_bounds = [1.0, np.pi, 1.0, np.pi / 4, np.pi / 3, np.pi / 2] * N_PROFILERS

    print(f"Starting weighted CMA-ES optimization for {N_PROFILERS} profiler(s)...")
    print(f"Points: {len(points)}, Rotation angles: {len(rotation_angles)}")
    print()

    es = cma.CMAEvolutionStrategy(
        x0,
        0.2,
        {
            "bounds": [lower_bounds, upper_bounds],
            "maxiter": 50,
            "popsize": 16,
            "verbose": 1,
        },
    )

    iteration = 0
    while not es.stop():
        solutions = es.ask()
        fitness = [
            weighted_objective(
                x,
                mesh,
                points,
                rotation_angles,
                N_PROFILERS,
                fixed_params,
                weights,
            )
            for x in solutions
        ]
        es.tell(solutions, fitness)

        iteration += 1
        best_score = -min(fitness) * 100
        print(f"Iteration {iteration}: Best weighted score = {best_score:.2f}%")

    best_theta = es.result.xbest
    profilers = decode_theta_to_profilers(best_theta, N_PROFILERS, fixed_params)

    print("\nFinal evaluation with full resolution...")

    # Load full point cloud
    pcd_full = o3d.io.read_point_cloud(MESH_FILE)
    points_full = np.asarray(pcd_full.points)
    # points_full = points_full - points_full.mean(axis=0)

    rotation_angles_full = np.deg2rad(np.arange(0, 360, 1))

    # Recompute weights for full point cloud
    print("Recomputing weights for full point cloud...")
    weights_full, inside_mask_full, _ = assign_weights_from_submeshes(
        points_full,
        SUBMESH_FILE,
        high_weight=HIGH_WEIGHT,
        low_weight=LOW_WEIGHT,
    )

    covered = simulate_all_profilers(mesh, points_full, profilers, rotation_angles_full)

    # Detailed statistics
    stats = compute_detailed_coverage(covered, weights_full)

    print(f"\n{'=' * 60}")
    print("WEIGHTED OPTIMIZATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Weighted Coverage:   {stats['weighted'] * 100:.2f}%")
    print(f"Unweighted Coverage: {stats['unweighted'] * 100:.2f}%")
    print("\nCoverage by weight category:")
    for w, cov in sorted(stats["by_weight"].items(), reverse=True):
        count = (weights_full == w).sum()
        label = "BLADE REGIONS" if w == HIGH_WEIGHT else "Other"
        print(f"  {label} (weight={w:.1f}): {cov * 100:6.2f}% ({count:5d} points)")
    print(f"\nTotal covered points: {covered.sum()} / {len(points_full)}")

    # Print profiler parameters
    for i, p in enumerate(profilers):
        base = 6 * i
        rho, azimuth, z_pos, yaw, pitch, roll = best_theta[base : base + 6]
        print(f"\nProfiler {i + 1}:")
        print(f"  rho={rho:.3f}m, azimuth={np.rad2deg(azimuth):.1f}°, z={z_pos:.3f}m")
        print(
            f"  yaw={np.rad2deg(yaw):.1f}°, pitch={np.rad2deg(pitch):.1f}°, roll={np.rad2deg(roll):.1f}°"
        )

    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    vis_objects = []

    # Mesh (if available)
    if mesh.has_triangles():
        mesh_vis = o3d.geometry.TriangleMesh(mesh)
        mesh_vis.paint_uniform_color([0.5, 0.5, 0.5])
        mesh_vis.compute_vertex_normals()
        vis_objects.append(mesh_vis)

    # Color points by coverage status AND whether they're in blade regions
    # Blade region + covered = bright green
    # Blade region + uncovered = bright red (problem!)
    # Non-blade + covered = dark green
    # Non-blade + uncovered = dark gray (less important)

    colors = np.zeros((len(points_full), 3))

    for i in range(len(points_full)):
        is_blade = inside_mask_full[i]
        is_covered = covered[i]

        if is_blade:
            if is_covered:
                colors[i] = [0, 1, 0]  # Bright green - blade covered ✓
            else:
                colors[i] = [1, 0, 0]  # Bright red - blade NOT covered ⚠
        else:
            if is_covered:
                colors[i] = [0, 0.4, 0]  # Dark green - other covered
            else:
                colors[i] = [0.3, 0.3, 0.3]  # Gray - other not covered (less important)

    coverage_pcd = o3d.geometry.PointCloud()
    coverage_pcd.points = o3d.utility.Vector3dVector(points_full)
    coverage_pcd.colors = o3d.utility.Vector3dVector(colors)
    vis_objects.append(coverage_pcd)

    # Profiler markers
    profiler_colors = [[0, 0, 1], [1, 0.5, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    for i, p in enumerate(profilers):
        color = profiler_colors[i % len(profiler_colors)]

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        sphere.translate(p.C)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        vis_objects.append(sphere)

        arrow_end = p.C + p.forward * 0.1
        arrow = o3d.geometry.LineSet()
        arrow.points = o3d.utility.Vector3dVector([p.C.tolist(), arrow_end.tolist()])
        arrow.lines = o3d.utility.Vector2iVector([[0, 1]])
        arrow.colors = o3d.utility.Vector3dVector([color])
        vis_objects.append(arrow)

    # Coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis_objects.append(coord_frame)

    o3d.visualization.draw_geometries(
        vis_objects,
        window_name=f"Blade Coverage: {stats['by_weight'].get(HIGH_WEIGHT, 0) * 100:.1f}%",
        width=1200,
        height=800,
    )


if __name__ == "__main__":
    main()
