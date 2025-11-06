import os
import argparse
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import sys
sys.path.append(str(Path(__file__).parents[2]))

from sim.utils.gs.gs_processor import GSProcessor
from sim.utils.gs.colormap import colormap
from sim.utils.gs.icp_utils import *
from sim.utils.robot.robot_pc_sampler import RobotPcSampler
from sim.utils.robot.robot_pc_transformations import transform_gs_xarm_pusher


def visualize(src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud) -> None:
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([src, tgt, coordinate])  # type: ignore


def visualize_list(pcd_list) -> None:
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(pcd_list + [coordinate])  # type: ignore


def control_robot_with_pusher(params_path, total_mask_path, qpos=[0, -45, 0, 30, 0, 75, 0]):
    sp = GSProcessor()
    params_full = sp.load(params_path)
    total_mask_full = np.load(total_mask_path)
    total_mask_full = torch.from_numpy(total_mask_full).to(params_full['means3D'].device).to(params_full['means3D'].dtype)

    qpos = np.array(qpos) * np.pi / 180
    params_full = transform_gs_xarm_pusher(qpos, params_full, total_mask_full, sample_robot=None)
    
    sp.save(params_full, 'log/gs/temp_save/full.splat')
    save_dir_full = 'log/gs/temp_save/full.splat'
    sp.visualize_gs([save_dir_full], transform=True, merged=False, axis_on=True)


def ransac_icp(scene_gs_input_path, robot_bbox, do_visualization=True):
    voxel_size = 0.05
    icp_mode = "point_to_plane"
    max_ransac_iters = 100000
    ransac_n = 4
    confidence = 0.999

    print("[1/6] Loading point clouds…")
    urdf_path = 'assets/robots/xarm/xarm7_with_pusher.urdf'
    link_names = [
        'link1',
        'link2',
        'link3',
        'link4',
        'link5',
        'link6',
        'link7',
        'pusher_base_link',
    ]
    assert os.path.exists(urdf_path), f"URDF path does not exist: {urdf_path}"
    sample_robot = RobotPcSampler(urdf_path)

    base_qpos = np.array([0, -45, 0, 30, 0, 75, 0]) * np.pi / 180
    points = sample_robot.compute_robot_pcd(base_qpos, link_names=link_names, num_pts=2000)

    src_raw = o3d.geometry.PointCloud()
    src_raw.points = o3d.utility.Vector3dVector(points)

    sp = GSProcessor()
    params = sp.load(scene_gs_input_path)
    params = sp.crop(params, robot_bbox)

    pts = params['means3D']
    tgt_raw = o3d.geometry.PointCloud()
    tgt_raw.points = o3d.utility.Vector3dVector(pts)

    src_raw.paint_uniform_color([0, 1, 0])
    tgt_raw.paint_uniform_color([1, 0, 0])

    if do_visualization:
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([src_raw, tgt_raw, coordinate])  # type: ignore

    print("\n[2/6] Preprocessing (downsample + normals + FPFH)…")
    src_down, src_fpfh = preprocess_for_features(src_raw, voxel_size)
    tgt_down, tgt_fpfh = preprocess_for_features(tgt_raw, voxel_size)
    print(f"   Downsampled sizes: src={np.asarray(src_down.points).shape[0]}, tgt={np.asarray(tgt_down.points).shape[0]}")

    if do_visualization:
        visualize(src_down, tgt_down)

    print("\n[3/6] Global alignment via RANSAC…")
    ransac_result = global_registration_ransac(
        src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size,
        ransac_n=ransac_n, max_iterations=max_ransac_iters, confidence=confidence,
    )
    print("   RANSAC fitness=%.4f, inlier_rmse=%.6f" % (ransac_result.fitness, ransac_result.inlier_rmse))
    print("   RANSAC transformation:\n", ransac_result.transformation)

    # post-RANSAC overlay on raw clouds
    src_ransac_raw = transform_copy(src_raw, ransac_result.transformation)

    # RANSAC inliers visualization on downsampled clouds
    dist_thresh = 1.5 * voxel_size
    src_down_T = transform_copy(src_down, ransac_result.transformation)
    inliers = compute_ransac_inliers(src_down_T, tgt_down, dist_thresh)
    src_in = src_down_T.select_by_index(np.where(inliers)[0].tolist())
    src_out = src_down_T.select_by_index(np.where(~inliers)[0].tolist())
    src_in.paint_uniform_color((0.2, 0.9, 0.2))
    src_out.paint_uniform_color((1.0, 0.2, 0.2))

    if do_visualization:
        visualize(src_in, tgt_down)
        visualize(src_ransac_raw, tgt_raw)

    print("\n[4/6] Refinement via two-stage ICP (%s)…" % icp_mode)
    icp_coarse, icp_fine = refine_with_icp(src_raw, tgt_raw, ransac_result.transformation, voxel_size, icp_mode=icp_mode)
    print("   ICP (coarse)  fitness=%.4f, inlier_rmse=%.6f" % (icp_coarse.fitness, icp_coarse.inlier_rmse))
    print("   ICP (fine)    fitness=%.4f, inlier_rmse=%.6f" % (icp_fine.fitness, icp_fine.inlier_rmse))
    print("   Final ICP transformation:\n", icp_fine.transformation)
    print("   Final ICP transformation (inverted):\n", np.linalg.inv(icp_fine.transformation))

    if do_visualization:
        # post-ICP coarse
        src_icp_coarse = transform_copy(src_raw, icp_coarse.transformation)
        visualize(src_icp_coarse, tgt_raw)

        # post-ICP fine (final)
        src_icp_final = transform_copy(src_raw, icp_fine.transformation)
        visualize(src_icp_final, tgt_raw)

    return np.linalg.inv(icp_fine.transformation)


def segment_robot(scene_gs_input_path, scene_gs_save_path, scene_mask_save_path, gs_to_robo, do_visualization=True):
    urdf_path = 'assets/robots/xarm/xarm7_with_pusher.urdf'
    link_names = [
        'link1',
        'link2',
        'link3',
        'link4',
        'link5',
        'link6',
        'link7',
        'pusher_base_link',
    ]
    assert os.path.exists(urdf_path), f"URDF path does not exist: {urdf_path}"
    sample_robot = RobotPcSampler(urdf_path)

    base_qpos = np.array([0, -45, 0, 30, 0, 75, 0]) * np.pi / 180
    points = sample_robot.compute_robot_pcd(base_qpos, link_names=link_names, num_pts=2000)

    src_raw = o3d.geometry.PointCloud()
    src_raw.points = o3d.utility.Vector3dVector(points)

    sp = GSProcessor()
    params = sp.load(scene_gs_input_path)
    params = sp.rotate(params, gs_to_robo[:3, :3])
    params = sp.translate(params, gs_to_robo[:3, 3])
    pts = params['means3D'].cpu().numpy()

    tgt_raw = o3d.geometry.PointCloud()
    tgt_raw.points = o3d.utility.Vector3dVector(pts)

    src_raw.paint_uniform_color([0, 1, 0])
    tgt_raw.paint_uniform_color([1, 0, 0])

    if do_visualization:
        visualize(src_raw, tgt_raw)

    robot_bbox = np.array([
        [np.min(points[:, 0]) - 0.10, np.max(points[:, 0]) + 0.10],
        [np.min(points[:, 1]) - 0.10, np.max(points[:, 1]) + 0.10],
        [np.min(points[:, 2]), np.max(points[:, 2]) + 0.10],  # hard stop at z min to leave robot base points to table params
    ])

    pts_is_robot_mask = (pts[:, 0] > robot_bbox[0, 0]) & (pts[:, 0] < robot_bbox[0, 1]) & \
                        (pts[:, 1] > robot_bbox[1, 0]) & (pts[:, 1] < robot_bbox[1, 1]) & \
                        (pts[:, 2] > robot_bbox[2, 0]) & (pts[:, 2] < robot_bbox[2, 1])

    pts_robot = pts[pts_is_robot_mask]
    pts_scene = pts[~pts_is_robot_mask]

    robot_pcd = o3d.geometry.PointCloud()
    robot_pcd.points = o3d.utility.Vector3dVector(pts_robot)
    robot_pcd.paint_uniform_color([0, 0, 1])
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(pts_scene)
    scene_pcd.paint_uniform_color([1, 0, 0])
    if do_visualization:
        visualize_list([robot_pcd, scene_pcd, src_raw])

    knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(points)
    _, indices = knn.kneighbors(pts_robot)  # (n_scan_points, 1)
    indices_link = (indices / 2000).astype(np.int32).reshape(-1)

    scan_colors = np.asarray(robot_pcd.colors)
    robot_mask = np.zeros(pts_robot.shape[0], dtype=np.int32)

    for i in range(indices_link.max() + 1):
        mask = indices_link == i
        robot_mask[mask] = i + 2  # skip 0=empty and 1=link_base
        scan_colors[mask] = colormap[i]

    # offset pusher mask ids by 1
    robot_mask[robot_mask >= 9] += 1  # skip 9=link_eef

    if do_visualization:
        robot_pcd.colors = o3d.utility.Vector3dVector(scan_colors)
        visualize_list([robot_pcd, scene_pcd])

    total_mask_full = np.zeros(pts.shape[0], dtype=np.int32) - 1
    total_mask_full[pts_is_robot_mask] = robot_mask
    total_mask_full = total_mask_full.astype(np.int32)

    np.save(scene_mask_save_path, total_mask_full)
    sp.save(params, scene_gs_save_path)
    sp.visualize_gs([scene_gs_save_path], transform=False, merged=False, axis_on=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--construct', action='store_true', help='Whether to construct the pusher scene gs')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize the pusher scene')
    args = parser.parse_args()

    if args.construct:
        # constructing pusher scene gs
        scene_gs_input_path = 'log/gs/scans/scene_pusher/scene_pusher_raw.ply'
        scene_gs_save_path = 'log/gs/scans/scene_pusher/scene_pusher_new.ply'
        scene_mask_save_path = 'log/gs/scans/scene_pusher/scene_pusher_mask_new.npy'
        robot_bbox = np.array([
            [-0.5, 0.5],
            [-0.3, 0.3],
            [0.1, 0.8],
        ])
        gs_to_robo = ransac_icp(scene_gs_input_path, robot_bbox, do_visualization=False)
        segment_robot(scene_gs_input_path, scene_gs_save_path, scene_mask_save_path, gs_to_robo, do_visualization=False)

    if args.visualize:
        # visualize
        control_robot_with_pusher(
            params_path='log/gs/scans/scene_pusher/scene_pusher.ply', 
            total_mask_path='log/gs/scans/scene_pusher/scene_pusher_mask.npy',
            qpos=[10, -20, 30, 15, 4, 54, 20],
        )
