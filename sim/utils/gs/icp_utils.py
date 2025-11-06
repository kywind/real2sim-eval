from pathlib import Path
import numpy as np
import open3d as o3d

# ------------------------------- Utilities ----------------------------------

def deepcopy_pcd(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    q = o3d.geometry.PointCloud()
    q.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    if pcd.has_colors():
        q.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    if pcd.has_normals():
        q.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals))
    return q


def transform_copy(pcd: o3d.geometry.PointCloud, T: np.ndarray) -> o3d.geometry.PointCloud:
    q = deepcopy_pcd(pcd)
    q.transform(T)
    return q


def paint_copy(pcd: o3d.geometry.PointCloud, color: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    q = deepcopy_pcd(pcd)
    q.paint_uniform_color(color)
    return q


def color_by_normals(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    q = deepcopy_pcd(pcd)
    if not q.has_normals():
        raise ValueError("Point cloud has no normals; compute normals first.")
    n = np.asarray(q.normals)
    # Map normals from [-1,1] to [0,1] for RGB
    c = (0.5 * (n + 1.0)).clip(0.0, 1.0)
    q.colors = o3d.utility.Vector3dVector(c)
    return q


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Loaded empty point cloud: {path}")
    return pcd


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int = 30) -> None:
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.normalize_normals()


def preprocess_for_features(pcd: o3d.geometry.PointCloud, voxel_size: float) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """Downsample, estimate normals, and compute FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    if pcd_down.is_empty():
        raise ValueError("Downsampled point cloud is empty; try a smaller voxel size.")

    radius_normal = 2.0 * voxel_size
    estimate_normals(pcd_down, radius=radius_normal)

    radius_feature = 5.0 * voxel_size
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, fpfh


def global_registration_ransac(
    src_down: o3d.geometry.PointCloud,
    tgt_down: o3d.geometry.PointCloud,
    src_fpfh: o3d.pipelines.registration.Feature,
    tgt_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float,
    ransac_n: int = 4,
    max_iterations: int = 100000,
    confidence: float = 0.999,
) -> o3d.pipelines.registration.RegistrationResult:
    distance_threshold = 1.5 * voxel_size

    checker_distance = o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    checker_edge = o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        True,  # mutual filter
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n,
        [checker_distance, checker_edge],
        o3d.pipelines.registration.RANSACConvergenceCriteria(max_iterations, confidence),
    )
    return result


def refine_with_icp(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    init_T: np.ndarray,
    voxel_size: float,
    icp_mode: str = "point_to_plane",
) -> tuple[o3d.pipelines.registration.RegistrationResult, o3d.pipelines.registration.RegistrationResult]:
    """Refine alignment using a two-stage ICP. Returns (coarse, fine)."""
    coarse_dist = 1.5 * voxel_size
    fine_dist = 0.5 * voxel_size

    if icp_mode == "point_to_plane":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    elif icp_mode == "point_to_point":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:
        raise ValueError("icp_mode must be 'point_to_plane' or 'point_to_point'")

    if icp_mode == "point_to_plane":
        radius_normals = max(2.0 * voxel_size, 0.01)
        estimate_normals(src, radius=radius_normals)
        estimate_normals(tgt, radius=radius_normals)

    result_coarse = o3d.pipelines.registration.registration_icp(
        src, tgt, coarse_dist, init_T, estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
    )

    result_fine = o3d.pipelines.registration.registration_icp(
        src, tgt, fine_dist, result_coarse.transformation, estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
    )
    return result_coarse, result_fine


def compute_ransac_inliers(src_down_T: o3d.geometry.PointCloud, tgt_down: o3d.geometry.PointCloud, thresh: float) -> np.ndarray:
    """Compute inlier mask after RANSAC by nearest neighbor distance to target."""
    tgt_kd = o3d.geometry.KDTreeFlann(tgt_down)
    src_pts = np.asarray(src_down_T.points)
    inlier_mask = np.zeros(len(src_pts), dtype=bool)
    for i, p in enumerate(src_pts):
        _, idx, d2 = tgt_kd.search_knn_vector_3d(p, 1)
        if len(idx) > 0 and d2[0] <= thresh * thresh:
            inlier_mask[i] = True
    return inlier_mask


# ------------------------------ Visualization --------------------------------

def show_geometries(geoms, title: str):
    o3d.visualization.draw_geometries(geoms, window_name=title)  # type: ignore


def save_png_of_geometries(geoms, out_png: Path, width: int = 1280, height: int = 960) -> bool:
    """Best-effort screenshot saver. Returns True on success."""
    vis = o3d.visualization.Visualizer()  # type: ignore
    try:
        vis.create_window(visible=False, width=width, height=height)
    except Exception:
        # Fallback: create visible window if headless fails
        vis.create_window(visible=True, width=width, height=height)
    for g in geoms:
        vis.add_geometry(g)
    # Optional: add a coordinate frame for orientation
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    vis.poll_events(); vis.update_renderer()
    ok = vis.capture_screen_image(str(out_png), do_render=True)
    vis.destroy_window()
    return ok


def save_scene_ply(geoms, out_ply: Path) -> None:
    # Merge all point clouds into one PLY for easy viewing
    pcs = [g for g in geoms if isinstance(g, o3d.geometry.PointCloud)]
    if not pcs:
        return
    merged = o3d.geometry.PointCloud()
    for p in pcs:
        merged += p
    o3d.io.write_point_cloud(str(out_ply), merged)

