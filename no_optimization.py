import open3d as o3d
import numpy as np


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


# def simulate_single_profiler(scene, base_points, profiler, rotation_angles):
#     C = profiler["C"]
#     f = profiler["forward"]
#     n_plane = profiler["n_plane"]
#     ang_min = profiler["ang_min"]
#     ang_max = profiler["ang_max"]
#     rmin = profiler["rmin"]
#     rmax = profiler["rmax"]

#     covered = np.zeros(len(base_points), dtype=bool)

#     for phi in rotation_angles:
#         R = rotz(phi)
#         points = (R @ base_points.T).T

#         # Check laser plane against base_points so all rotations see the same slice
#         idx = np.where(laser_plane_filter(base_points, C, n_plane))[0]
#         if len(idx) == 0:
#             continue

#         # Use rotated points for angle, range, and occlusion checks
#         idx = idx[vertical_angle_filter(points[idx], C, f, up, ang_min, ang_max)]
#         if len(idx) == 0:
#             continue

#         idx = idx[range_filter(points[idx], C, rmin, rmax)]
#         if len(idx) == 0:
#             continue

#         visible, dirs, t_hit = apply_occlusion_and_collect_rays(scene, points[idx], C)
#         idx_visible = idx[visible]

#         # Save rays for visualization
#         debug_rays.append(
#             {
#                 "C": C.copy(),
#                 "dirs": dirs[visible].copy(),
#                 "t_hit": t_hit[visible].copy(),
#             }
#         )

#         covered[idx_visible] = True

#     return covered, debug_rays


def main():
    MESH_FILE = "dataset/med_scaled_edited.ply"
    N_PROFILERS = 1

    mesh = o3d.io.read_triangle_mesh(MESH_FILE)
    mesh.compute_vertex_normals()

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    pcd = mesh.sample_points_uniformly(20000)
    points = np.asarray(pcd.points)
    print(points.shape)
    # rotation_angles = np.deg2rad(np.arange(0, 360, 1))


if __name__ == "__main__":
    main()
