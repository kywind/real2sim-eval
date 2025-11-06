import os
import numpy as np
import open3d as o3d
import trimesh
import torch
import yaml
import pickle as pkl
from pathlib import Path
from omegaconf import OmegaConf
import argparse
import sys
sys.path.append(str(Path(__file__).parents[2]))

from utils.dir_utils import mkdir


def _init_start(
    cfg,
    object_points,
    controller_points,
    object_radius=0.02,
    object_max_neighbours=30,
    controller_radius=0.04,
    controller_max_neighbours=50,
    mask=None,
):
    object_points = object_points.cpu().numpy()
    if controller_points is not None:
        controller_points = controller_points.cpu().numpy()
    if mask is None:
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_points)
        pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

        # Connect the springs of the objects first
        points = np.asarray(object_pcd.points)
        spring_flags = np.zeros((len(points), len(points)))
        springs = []
        rest_lengths = []
        for i in range(len(points)):
            [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                points[i], object_radius, object_max_neighbours
            )
            idx = idx[1:]
            for j in idx:
                rest_length = np.linalg.norm(points[i] - points[j])
                if (
                    spring_flags[i, j] == 0
                    and spring_flags[j, i] == 0
                    and rest_length > 1e-4
                ):
                    spring_flags[i, j] = 1
                    spring_flags[j, i] = 1
                    springs.append([i, j])
                    rest_lengths.append(np.linalg.norm(points[i] - points[j]))

        num_object_springs = len(springs)

        if controller_points is not None:
            # Connect the springs between the controller points and the object points
            num_object_points = len(points)
            points = np.concatenate([points, controller_points], axis=0)
            for i in range(len(controller_points)):
                [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                    controller_points[i],
                    controller_radius,
                    controller_max_neighbours,
                )
                for j in idx:
                    springs.append([num_object_points + i, j])
                    rest_lengths.append(
                        np.linalg.norm(controller_points[i] - points[j])
                    )

        springs = np.array(springs)
        rest_lengths = np.array(rest_lengths)
        masses = np.ones(len(points))
        return (
            torch.tensor(points, dtype=torch.float32, device=cfg.device),
            torch.tensor(springs, dtype=torch.int32, device=cfg.device),
            torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
            torch.tensor(masses, dtype=torch.float32, device=cfg.device),
            num_object_springs,
        )
    else:
        mask = mask.cpu().numpy()
        # Get the unique value in masks
        unique_values = np.unique(mask)
        vertices = []
        springs = []
        rest_lengths = []
        index = 0
        # Loop different objects to connect the springs separately
        for value in unique_values:
            temp_points = object_points[mask == value]
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(temp_points)
            temp_tree = o3d.geometry.KDTreeFlann(temp_pcd)
            temp_spring_flags = np.zeros((len(temp_points), len(temp_points)))
            temp_springs = []
            temp_rest_lengths = []
            for i in range(len(temp_points)):
                [k, idx, _] = temp_tree.search_hybrid_vector_3d(
                    temp_points[i], object_radius, object_max_neighbours
                )
                idx = idx[1:]
                for j in idx:
                    rest_length = np.linalg.norm(temp_points[i] - temp_points[j])
                    if (
                        temp_spring_flags[i, j] == 0
                        and temp_spring_flags[j, i] == 0
                        and rest_length > 1e-4
                    ):
                        temp_spring_flags[i, j] = 1
                        temp_spring_flags[j, i] = 1
                        temp_springs.append([i + index, j + index])
                        temp_rest_lengths.append(rest_length)
            vertices += temp_points.tolist()
            springs += temp_springs
            rest_lengths += temp_rest_lengths
            index += len(temp_points)

        num_object_springs = len(springs)

        vertices = np.array(vertices)
        springs = np.array(springs)
        rest_lengths = np.array(rest_lengths)
        masses = np.ones(len(vertices))

        return (
            torch.tensor(vertices, dtype=torch.float32, device=cfg.device),
            torch.tensor(springs, dtype=torch.int32, device=cfg.device),
            torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
            torch.tensor(masses, dtype=torch.float32, device=cfg.device),
            num_object_springs,
        )


if __name__ == '__main__':
    # set numpy seed for trimesh
    np.random.seed(0)

    parser = argparse.ArgumentParser(description='Create rigid phystwin data and checkpoints.')
    parser.add_argument('--rigid_mesh_path', type=str, default='experiments/log/gs/scans/T/T_mesh.ply',
                        help='Path to the rigid mesh file (PLY format).')
    parser.add_argument('--ckpt_path', type=str, default='experiments/log/phystwin/T',
                        help='Path to save the phystwin checkpoints.')
    parser.add_argument('--case_name', type=str, default='T_0001',
                        help='Name of the phystwin case to create.')
    parser.add_argument('--cfg_path', type=str, default='experiments/cfg/physics/default.yaml',
                        help='Path to the phystwin configuration YAML file.')
    args = parser.parse_args()

    rigid_mesh_path = args.rigid_mesh_path
    ckpt_path = args.ckpt_path
    case_name = args.case_name
    cfg_path = args.cfg_path

    with open(cfg_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg_dict)

    assert os.path.exists(rigid_mesh_path), f"Rigid mesh path {rigid_mesh_path} does not exist."
    rigid_mesh = o3d.io.read_triangle_mesh(rigid_mesh_path)
    assert rigid_mesh is not None, f"Failed to load mesh from {rigid_mesh_path}."

    data_path_overwrite = f"{ckpt_path}/data"
    zeroth_order_ckpt_path_overwrite = f"{ckpt_path}/experiments_optimization"
    first_order_ckpt_path_overwrite = f"{ckpt_path}/experiments"

    mkdir(Path(data_path_overwrite) / case_name, resume=False, overwrite=False)
    mkdir(Path(zeroth_order_ckpt_path_overwrite) / case_name, resume=False, overwrite=False)
    mkdir(Path(first_order_ckpt_path_overwrite) / case_name / 'train', resume=False, overwrite=False)

    trimesh_mesh = trimesh.Trimesh(vertices=rigid_mesh.vertices, faces=rigid_mesh.triangles)
    # Sample the surface points
    surface_points, _ = trimesh.sample.sample_surface(trimesh_mesh, 1024)
    # Sample the interior points
    interior_points = trimesh.sample.volume_mesh(trimesh_mesh, 10000)
    all_points = np.concatenate([surface_points, interior_points], axis=0)
    
    min_bound = np.min(all_points, axis=0)
    volume_sample_size = 0.005
    index = []
    grid_flag = {}
    final_surface_points = []
    for i in range(surface_points.shape[0]):
        grid_index = tuple(
            np.floor((surface_points[i] - min_bound) / volume_sample_size).astype(
                int
            )
        )
        if grid_index not in grid_flag:
            grid_flag[grid_index] = 1
            final_surface_points.append(surface_points[i])
    final_interior_points = []
    for i in range(interior_points.shape[0]):
        grid_index = tuple(
            np.floor((interior_points[i] - min_bound) / volume_sample_size).astype(
                int
            )
        )
        if grid_index not in grid_flag:
            grid_flag[grid_index] = 1
            final_interior_points.append(interior_points[i])
    all_points = np.concatenate(
        [final_surface_points, final_interior_points],
        axis=0,
    )
    object_points = torch.tensor(
        all_points, dtype=torch.float32, device=cfg.device
    )

    # Connect the springs
    object_radius = 0.5  # hardcoded hyperparameter
    object_max_neighbours = 50  # hardcoded hyperparameter
    _, _, _, _, num_object_springs = _init_start(
        cfg,
        object_points,
        None,
        object_radius=object_radius,
        object_max_neighbours=object_max_neighbours,
    )

    # save data
    object_points = object_points.cpu().numpy()
    object_colors = np.ones_like(object_points)  # all white
    object_points = object_points[None]  # T == 1
    object_colors = object_colors[None]
    other_surface_points = np.zeros((0, 3))
    interior_points = np.zeros((0, 3))
    data = {
        "object_points": object_points,  # (T, N, 3)
        "object_colors": object_colors,  # (T, N, 3)
        "surface_points": other_surface_points,  # (n_surface, 3)
        "interior_points": interior_points  # (n_interior, 3)
    }
    with open(f"{data_path_overwrite}/{case_name}/final_data.pkl", "wb") as f:
        pkl.dump(data, f)

    # save zeroth order results
    optimal_params = {
        'global_spring_Y': 0.0,
        'object_radius': object_radius,
        'object_max_neighbours': object_max_neighbours
    }
    with open(f"{zeroth_order_ckpt_path_overwrite}/{case_name}/optimal_params.pkl", "wb") as f:
        pkl.dump(optimal_params, f)

    # save first order results
    spring_Y = torch.ones(
        num_object_springs, dtype=torch.float32, device=cfg.device
    ) * 3e4
    collide_elas = torch.tensor([0.2], dtype=torch.float32, device=cfg.device)
    collide_fric = torch.tensor([0.5], dtype=torch.float32, device=cfg.device)

    collide_self_elas = torch.tensor([0.2], dtype=torch.float32, device=cfg.device)  # self collision
    collide_self_fric = torch.tensor([0.5], dtype=torch.float32, device=cfg.device)  # self collision

    checkpoint = {
        'spring_Y': spring_Y,  # (n_springs_total,)
        'collide_elas': collide_elas,  # (1,)
        'collide_fric': collide_fric,  # (1,)
        'collide_object_elas': collide_self_elas,  # (1,)
        'collide_object_fric': collide_self_fric,  # (1,)
        'num_object_springs': num_object_springs
    }
    assert spring_Y.shape[0] == num_object_springs  # no controller springs

    best_model_path = f"{first_order_ckpt_path_overwrite}/{case_name}/train/best_0.pth"
    torch.save(checkpoint, best_model_path)
