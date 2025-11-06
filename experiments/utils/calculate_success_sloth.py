import os
import argparse
from glob import glob
import pickle as pkl
import numpy as np
from pathlib import Path
import open3d as o3d


def _parse_bbox(bbox):
    """
    Accepts:
      - (min_xyz, max_xyz), each shape (3,)
      - (center_xyz, half_size), center shape (3,), half_size scalar
    Returns:
      - (min_xyz, max_xyz), each shape (3,)
    """
    if len(bbox) != 2:
        raise ValueError("bbox must be a 2-tuple.")
    a, b = bbox
    a = np.asarray(a, dtype=float)
    if np.ndim(b) == 0:  # half_size
        center = a
        half = float(b)
        if center.shape != (3,):
            raise ValueError("center must have shape (3,).")
        min_xyz = center - half
        max_xyz = center + half
    else:
        min_xyz = a
        max_xyz = np.asarray(b, dtype=float)
        if min_xyz.shape != (3,) or max_xyz.shape != (3,):
            raise ValueError("min_xyz/max_xyz must have shape (3,).")
    # Ensure min <= max
    if np.any(min_xyz > max_xyz):
        raise ValueError("bbox min must be <= max component-wise.")
    return min_xyz, max_xyz

def _segment_plane_intersections_xz(
    p0, p1, y_plane, x_min, x_max, z_min, z_max, eps=1e-12
):
    """
    Vectorized test for intersections of segments p0->p1 with plane y = y_plane,
    additionally requiring the intersection point to lie within the xz-rectangle
    [x_min,x_max] x [z_min,z_max].

    Returns a boolean array of shape (M,) for M segments.
    """
    y0 = p0[:, 1]
    y1 = p1[:, 1]
    dy = y1 - y0

    # Proper crossings (not parallel): solve t = (y_plane - y0) / dy, 0<=t<=1
    parallel = np.isclose(dy, 0.0, atol=eps)
    t = np.zeros_like(dy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t[~parallel] = (y_plane - y0[~parallel]) / dy[~parallel]
    on_segment = (~parallel) & (t >= -eps) & (t <= 1.0 + eps)

    # Intersection points for non-parallel cases
    xi = p0[:, 0] + t * (p1[:, 0] - p0[:, 0])
    zi = p0[:, 2] + t * (p1[:, 2] - p0[:, 2])

    inside_rect = (xi >= x_min - eps) & (xi <= x_max + eps) & (zi >= z_min - eps) & (zi <= z_max + eps)
    hits_crossing = on_segment & inside_rect

    # Parallel coplanar segments (segment lies on plane): y0 == y_plane and y1 == y_plane
    coplanar = parallel & np.isclose(y0 - y_plane, 0.0, atol=eps)
    # A simple, conservative rule: count as intersecting if ANY endpoint lies within the rectangle.
    # (If you need exact segment-rectangle intersection in the plane, replace with a 2D segment/AABB test.)
    end0_in = (p0[:, 0] >= x_min - eps) & (p0[:, 0] <= x_max + eps) & (p0[:, 2] >= z_min - eps) & (p0[:, 2] <= z_max + eps)
    end1_in = (p1[:, 0] >= x_min - eps) & (p1[:, 0] <= x_max + eps) & (p1[:, 2] >= z_min - eps) & (p1[:, 2] <= z_max + eps)
    hits_coplanar = coplanar & (end0_in | end1_in)

    return hits_crossing | hits_coplanar

def count_xz_plane_intersections(
    vertices, springs, bbox, eps=1e-12,
):
    """
    Count how many spring segments intersect the bottom (y_min) and top (y_max) x-z planes
    of an axis-aligned cubic bounding box.

    Args:
      vertices: (N,3) float array of vertex positions.
      springs: (M,2) int array of index pairs into vertices.
      bbox: either (min_xyz, max_xyz) or (center_xyz, half_size).
      threshold_total: if provided, return 'meets_total_threshold' comparing (bot+top) vs this.
      threshold_each: if provided, return 'meets_each_threshold' where both planes individually meet this.
      eps: numeric tolerance.

    Returns:
      {
        'y_min_count': int,
        'y_max_count': int,
        'total_count': int,
        'meets_total_threshold': bool (if threshold_total given),
        'meets_each_threshold': bool (if threshold_each given),
      }
    """
    V = np.asarray(vertices, dtype=float)
    E = np.asarray(springs, dtype=int)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("vertices must have shape (N,3).")
    if E.ndim != 2 or E.shape[1] != 2:
        raise ValueError("springs must have shape (M,2).")
    if np.any(E < 0) or np.any(E >= V.shape[0]):
        raise ValueError("springs contain out-of-range vertex indices.")

    min_xyz, max_xyz = _parse_bbox(bbox)
    x_min, y_min, z_min = min_xyz.tolist()
    x_max, y_max, z_max = max_xyz.tolist()

    # Gather segment endpoints
    p0 = V[E[:, 0]]  # (M,3)
    p1 = V[E[:, 1]]  # (M,3)

    # Intersections with y = y_min and y = y_max planes
    hits_min = _segment_plane_intersections_xz(p0, p1, y_min, x_min, x_max, z_min, z_max, eps=eps)
    hits_max = _segment_plane_intersections_xz(p0, p1, y_max, x_min, x_max, z_min, z_max, eps=eps)

    c_min = int(np.count_nonzero(hits_min))
    c_max = int(np.count_nonzero(hits_max))
    total = c_min + c_max

    result = {
        "y_min_count": c_min,
        "y_max_count": c_max,
        "total_count": total,
    }
    return result


def find_episode_dirs(root):
    eps = [d for d in glob(os.path.join(root, "episode_*")) if os.path.isdir(d)]
    eps = sorted(set(eps))
    return eps


def is_sloth_success(state, state_init):
    packed = False
    meshes = state_init['physics']['static_meshes']
    assert len(meshes) == 1
    vertices = meshes[0]['vertices']
    faces = meshes[0]['faces']

    springs = state_init['physics']['init_springs']  # (N_springs, 2)
    x = state['renderer']['x']  # (N_vertices, 3)
    springs_np = springs.cpu().numpy()
    x_np = x.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    # Compute oriented bounding box (minimal OBB)
    obb = pcd.get_minimal_oriented_bounding_box(robust=True)
    extent = np.array(obb.extent)
    if np.abs(extent[0] * extent[1] * extent[2] - 0.2 * 0.13 * 0.27) > 1e-6:
        import ipdb; ipdb.set_trace()
    obb = obb.scale(1.05, obb.get_center())  # slightly enlarge

    pcd_test = o3d.geometry.PointCloud()
    pcd_test.points = o3d.utility.Vector3dVector(x_np)
    idx = obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(x_np))
    num_points_in_obb = len(idx)

    if num_points_in_obb >= 3050:
        packed = True

    return packed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory containing episode subdirectories.')
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"Processing data directory: {data_dir}")

    episode_dirs = find_episode_dirs(data_dir)
    if not episode_dirs:
        raise SystemExit(f"No episodes under: {data_dir}")

    packed_success_list = []
    for episode_dir in episode_dirs:
        state_files = sorted(glob(os.path.join(episode_dir, 'state/*.pkl')))
        print(f"Episode: {episode_dir}, Number of state files: {len(state_files)}")
        packed_count = 0
        packed_success = False
        state_init = None
        for state_file in state_files:
            if '000000.pkl' in state_file:
                with open(state_file, 'rb') as f:
                    state_init = pkl.load(f)
            if int(state_file.split('/')[-1].split('.')[0]) < 350:  # last 100 frames, total 450
                continue
            with open(state_file, 'rb') as f:
                state = pkl.load(f)
            packed = is_sloth_success(state, state_init)
            packed_count += packed * 1.0
            if packed_count >= 30:
                packed_success = True
        packed_success_list.append(packed_success)

    print("pack_sloth success list:", packed_success_list)

    success = np.zeros((len(episode_dirs) + 2), dtype=int)
    success[:-2] = np.array(packed_success_list, dtype=int)
    success[-2] = success[:-2].sum()
    success[-1] = success[:-2].mean() * 100

    np.savetxt(Path(data_dir) / 'success.txt', success, fmt='%d')
    print(f'pack_sloth success rate: {success[-2]} / {len(episode_dirs)} = {success[-1]:.1f}%')
