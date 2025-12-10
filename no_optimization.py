import open3d as o3d
import numpy as np


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


def main():
    MESH_FILE = "dataset/med_scaled_edited.ply"

    # Load and prepare mesh
    mesh = o3d.io.read_triangle_mesh(MESH_FILE)
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())  # Center the mesh at origin

    # Sample points from mesh
    pcd = mesh.sample_points_poisson_disk(20000)
    points = np.asarray(pcd.points)

    # Rotation angles for drill bit
    rotation_angles = np.deg2rad(np.arange(0, 360, 1))

    # Create profiler pointing toward the origin (where drill bit is centered)
    C = np.array([0.6, 0.0, 0.3])  # Position: 60cm on X-axis, 30cm up on Z
    forward = -C / np.linalg.norm(C)  # Point toward origin
    up = np.array([0, 0, 1])  # Z-axis is up

    # Laser plane normal (perpendicular to forward and up)
    n_plane = np.cross(forward, up)
    if np.linalg.norm(n_plane) < 1e-6:
        n_plane = np.array([1, 0, 0])
    else:
        n_plane = n_plane / np.linalg.norm(n_plane)

    profiler = Profiler(
        C=C,
        forward=forward,
        n_plane=n_plane,
        ang_min=np.deg2rad(-10.7),  # Minimum angle from forward (0 degrees)
        ang_max=np.deg2rad(10.7),  # Maximum angle from forward (60 degrees cone)
        rmin=0.19,  # Minimum range: 19 cm
        rmax=0.29,  # Maximum range: 29 cm
    )

    # Run simulation
    covered = simulate_single_profiler(mesh, points, profiler, rotation_angles)

    coverage_percentage = covered.mean() * 100
    print(f"Coverage: {coverage_percentage:.2f}%")
    print(f"Covered points: {covered.sum()} / {len(points)}")

    # Visualization
    vis_objects = []

    # Add mesh (semi-transparent)
    mesh_vis = o3d.geometry.TriangleMesh(mesh)
    mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])
    mesh_vis.compute_vertex_normals()
    vis_objects.append(mesh_vis)

    # Add covered points (green)
    if covered.sum() > 0:
        covered_pcd = o3d.geometry.PointCloud()
        covered_pcd.points = o3d.utility.Vector3dVector(points[covered])
        covered_pcd.paint_uniform_color([0, 1, 0])  # Green
        vis_objects.append(covered_pcd)

    # Add uncovered points (red)
    if (~covered).sum() > 0:
        uncovered_pcd = o3d.geometry.PointCloud()
        uncovered_pcd.points = o3d.utility.Vector3dVector(points[~covered])
        uncovered_pcd.paint_uniform_color([1, 0, 0])  # Red
        vis_objects.append(uncovered_pcd)

    # Add profiler position marker (blue sphere)
    profiler_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    profiler_sphere.translate(profiler.C)
    profiler_sphere.paint_uniform_color([0, 0, 1])  # Blue
    profiler_sphere.compute_vertex_normals()
    vis_objects.append(profiler_sphere)

    # Visualize
    o3d.visualization.draw_geometries(
        vis_objects,
        window_name="Laser Profiler Coverage Analysis",
        width=1200,
        height=800,
    )


if __name__ == "__main__":
    main()
