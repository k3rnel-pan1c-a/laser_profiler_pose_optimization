import open3d as o3d
import numpy as np
import cma


class Profiler:
    """Laser profiler sensor configuration."""

    def __init__(self, C, forward, n_plane, ang_min, ang_max, rmin, rmax):
        """
        Args:
            C: Sensor position (3D point)
            forward: Forward direction vector (unit vector)
            n_plane: Normal vector of the laser plane
            ang_min: Minimum angle from forward direction (radians)
            ang_max: Maximum angle from forward direction (radians)
            rmin: Minimum sensing range
            rmax: Maximum sensing range
        """
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
    angles = np.arccos(cos_angles)

    return (angles >= ang_min) & (angles <= ang_max)


def range_filter(points, C, rmin=0, rmax=100):
    dist = np.linalg.norm(points - C, axis=1)
    return (dist >= rmin) & (dist <= rmax)


def apply_occlusion_and_collect_rays(scene, points, C, eps=1e-3):
    v = points - C
    dist = np.linalg.norm(v, axis=1)
    v_unit = v / (dist[:, None] + 1e-9)

    rays = np.hstack([C[None, :].repeat(len(points), axis=0), v_unit]).astype(
        np.float32
    )  # None adds an extra dim, TODO: check the dim for C.

    ans = scene.cast_rays(o3d.core.Tensor(rays, o3d.core.Dtype.Float32))
    t_hit = ans["t_hit"].numpy()  # distance to intersection...

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

        # Rotate the mesh for raycasting (must match point rotation)
        rotated_mesh = o3d.geometry.TriangleMesh(mesh)
        rotated_mesh.rotate(R, center=[0, 0, 0])

        # Create raycasting scene with rotated mesh
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(rotated_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tmesh)

        # returns indices for points that can be considered part of the laser profiler's plane...
        idx = np.where(laser_plane_filter(points, C, n_plane))[0]

        if len(idx) == 0:
            continue

        # Use rotated points for angle, range, and occlusion checks...
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
    """
    Decode CMA-ES optimization vector into N Profiler objects.

    Each profiler is parameterized by:
        - rho: radial distance from origin (cylindrical coords)
        - azimuth: angle around Z-axis (cylindrical coords)
        - z: height along Z-axis
        - yaw: rotation around sensor's local Z-axis (horizontal aiming offset)
        - pitch: rotation around sensor's local X-axis (tilt up/down)
        - roll: rotation around sensor's local Y-axis (laser plane rotation)

    Args:
        theta: Optimization vector of length 6*N
        N: Number of profilers
        fixed_params: Dict with fixed profiler parameters (ang_min, ang_max, rmin, rmax)

    Returns:
        List of Profiler objects
    """
    profilers = []
    for i in range(N):
        base = 6 * i
        rho = theta[base + 0]  # Radial distance from Z-axis
        azimuth = theta[base + 1]  # Angle around Z-axis (position)
        z = theta[base + 2]  # Height
        yaw = theta[base + 3]  # Yaw: horizontal rotation offset from pointing at origin
        pitch = theta[base + 4]  # Pitch: tilt up/down (positive = looking down)
        roll = theta[base + 5]  # Roll: rotation of laser plane around forward axis

        # Compute position in Cartesian coordinates
        C = np.array([rho * np.cos(azimuth), rho * np.sin(azimuth), z])

        # Start with base direction pointing toward origin (horizontally)
        base_forward = -C.copy()
        base_forward[2] = 0  # Project to XY plane
        if np.linalg.norm(base_forward) < 1e-6:
            base_forward = np.array([1, 0, 0])
        else:
            base_forward = base_forward / np.linalg.norm(base_forward)

        # Define local coordinate frame (before rotations)
        # forward: toward origin, up: world Z, right: cross product
        up_world = np.array([0, 0, 1])
        right = np.cross(up_world, base_forward)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([0, 1, 0])
        else:
            right = right / np.linalg.norm(right)

        local_up = np.cross(base_forward, right)
        local_up = local_up / np.linalg.norm(local_up)

        # Apply yaw (around local_up)
        R_yaw = rot_axis_angle(local_up, yaw)
        forward = R_yaw @ base_forward
        right = R_yaw @ right

        # Apply pitch (around right axis) - positive = looking down
        R_pitch = rot_axis_angle(right, pitch)
        forward = R_pitch @ forward
        local_up = R_pitch @ local_up

        # Normalize
        forward = forward / np.linalg.norm(forward)

        # Apply roll (around forward axis) - rotates the laser plane
        R_roll = rot_axis_angle(forward, roll)
        local_up = R_roll @ local_up
        right = R_roll @ right

        # Normalize
        local_up = local_up / np.linalg.norm(local_up)
        right = right / np.linalg.norm(right)

        # Laser plane normal (perpendicular to forward, in the plane defined by roll)
        # The laser plane contains forward and local_up, so its normal is 'right'
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


def objective(theta, mesh, points, rotation_angles, N, fixed_params):
    """
    Objective function for CMA-ES optimization.
    Returns negative coverage (since CMA-ES minimizes).
    """
    profilers = decode_theta_to_profilers(theta, N, fixed_params)
    covered = simulate_all_profilers(mesh, points, profilers, rotation_angles)
    coverage = covered.mean()
    return -coverage  # Negative because CMA-ES minimizes


def main():
    MESH_FILE = "dataset/med_scaled_edited.ply"
    N_PROFILERS = 3  # Number of profilers to optimize

    # Load and prepare mesh
    mesh = o3d.io.read_triangle_mesh(MESH_FILE)
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())  # Center the mesh at origin

    # Sample points from mesh
    pcd = mesh.sample_points_poisson_disk(5000)  # Reduced for faster optimization
    points = np.asarray(pcd.points)

    # Rotation angles for drill bit (coarser for faster optimization)
    rotation_angles = np.deg2rad(np.arange(0, 360, 5))

    # Fixed profiler parameters (sensor specs that don't change)
    fixed_params = {
        "ang_min": np.deg2rad(-11),  # Minimum angle from forward
        "ang_max": np.deg2rad(11),  # Maximum angle from forward (cone half-angle)
        "rmin": 0.15,  # Minimum range
        "rmax": 0.3,  # Maximum range
    }

    # Parameters per profiler: [rho, azimuth, z, yaw, pitch, roll]
    dim = 6 * N_PROFILERS

    # Initial guess: profiler pointing at origin
    x0 = []
    for i in range(N_PROFILERS):
        rho = 0.3  # 30 cm from center
        azimuth = 0.0  # On X-axis
        z = 0.1  # 10 cm height
        yaw = 0.0  # No horizontal rotation offset
        pitch = 0.0  # Horizontal (no tilt)
        roll = 0.0  # No roll
        x0.extend([rho, azimuth, z, yaw, pitch, roll])
    x0 = np.array(x0)

    # Bounds: [lower bounds], [upper bounds]
    # rho: 0.2 to 0.6 m
    # azimuth: -pi to pi
    # z: -0.2 to 0.4 m
    # yaw: -pi/4 to pi/4 (±45° horizontal aim offset)
    # pitch: -pi/3 to pi/3 (±60° tilt)
    # roll: -pi/2 to pi/2 (±90° plane rotation)

    lower_bounds = [0.3, -np.pi, 0.0, -np.pi / 4, -np.pi / 3, -np.pi / 2] * N_PROFILERS
    upper_bounds = [1.0, np.pi, 1.0, np.pi / 4, np.pi / 3, np.pi / 2] * N_PROFILERS

    print(f"Starting CMA-ES optimization for {N_PROFILERS} profiler(s)...")
    print(f"Optimizing {dim} parameters: [rho, azimuth, z, yaw, pitch, roll]")
    print(f"Points: {len(points)}, Rotation angles: {len(rotation_angles)}")

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
            objective(x, mesh, points, rotation_angles, N_PROFILERS, fixed_params)
            for x in solutions
        ]
        es.tell(solutions, fitness)

        iteration += 1
        best_coverage = -min(fitness) * 100
        print(f"Iteration {iteration}: Best coverage = {best_coverage:.2f}%")

    best_theta = es.result.xbest
    profilers = decode_theta_to_profilers(best_theta, N_PROFILERS, fixed_params)

    print("\nFinal evaluation with full resolution...")
    pcd_full = mesh.sample_points_poisson_disk(20000)
    points_full = np.asarray(pcd_full.points)
    rotation_angles_full = np.deg2rad(np.arange(0, 360, 1))

    covered = simulate_all_profilers(mesh, points_full, profilers, rotation_angles_full)

    coverage_percentage = covered.mean() * 100
    print(f"\n{'=' * 50}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'=' * 50}")
    print(f"Final Coverage: {coverage_percentage:.2f}%")
    print(f"Covered points: {covered.sum()} / {len(points_full)}")

    for i, p in enumerate(profilers):
        base = 6 * i
        rho, azimuth, z_pos, yaw, pitch, roll = best_theta[base : base + 6]

        print(f"\nProfiler {i + 1}:")
        print(f"  Optimized parameters:")
        print(f"    rho = {rho:.3f} m")
        print(f"    azimuth = {np.rad2deg(azimuth):.1f}°")
        print(f"    z = {z_pos:.3f} m")
        print(f"    yaw = {np.rad2deg(yaw):.1f}°")
        print(f"    pitch = {np.rad2deg(pitch):.1f}°")
        print(f"    roll = {np.rad2deg(roll):.1f}°")
        print(f"  Position (C): [{p.C[0]:.3f}, {p.C[1]:.3f}, {p.C[2]:.3f}]")
        print(
            f"  Forward: [{p.forward[0]:.3f}, {p.forward[1]:.3f}, {p.forward[2]:.3f}]"
        )
        print(
            f"  Plane normal: [{p.n_plane[0]:.3f}, {p.n_plane[1]:.3f}, {p.n_plane[2]:.3f}]"
        )

    vis_objects = []

    # Add mesh
    mesh_vis = o3d.geometry.TriangleMesh(mesh)
    mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])
    mesh_vis.compute_vertex_normals()
    vis_objects.append(mesh_vis)

    if covered.sum() > 0:
        covered_pcd = o3d.geometry.PointCloud()
        covered_pcd.points = o3d.utility.Vector3dVector(points_full[covered])
        covered_pcd.paint_uniform_color([0, 1, 0])  # Green
        vis_objects.append(covered_pcd)

    if (~covered).sum() > 0:
        uncovered_pcd = o3d.geometry.PointCloud()
        uncovered_pcd.points = o3d.utility.Vector3dVector(points_full[~covered])
        uncovered_pcd.paint_uniform_color([1, 0, 0])  # Red
        vis_objects.append(uncovered_pcd)

    colors = [
        [0, 0, 1],
        [1, 0.5, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]

    for i, p in enumerate(profilers):
        color = colors[i % len(colors)]

        # Profiler position (sphere)
        profiler_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        profiler_sphere.translate(p.C)
        profiler_sphere.paint_uniform_color(color)
        profiler_sphere.compute_vertex_normals()
        vis_objects.append(profiler_sphere)

        # Forward direction (arrow/line)
        arrow_end = p.C + p.forward * 0.1
        arrow_points = [p.C.tolist(), arrow_end.tolist()]
        arrow_lines = [[0, 1]]
        arrow = o3d.geometry.LineSet()
        arrow.points = o3d.utility.Vector3dVector(arrow_points)
        arrow.lines = o3d.utility.Vector2iVector(arrow_lines)
        arrow.colors = o3d.utility.Vector3dVector([color])
        vis_objects.append(arrow)

    o3d.visualization.draw_geometries(
        vis_objects,
        window_name=f"Optimized {N_PROFILERS} Profiler(s) - Coverage: {coverage_percentage:.1f}%",
        width=1200,
        height=800,
    )


if __name__ == "__main__":
    main()
