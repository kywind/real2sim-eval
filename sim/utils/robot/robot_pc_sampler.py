import numpy as np
import torch
import copy
import open3d as o3d
import kornia
from urdfpy import URDF
import sapien.core as sapien


def trimesh_to_open3d(trimesh_mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    return o3d_mesh


def quat_mult_torch(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def rot_mat_to_quat(rot_mat):
    w = torch.sqrt(1 + rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2]) / 2
    x = (rot_mat[:, 2, 1] - rot_mat[:, 1, 2]) / (4 * w)
    y = (rot_mat[:, 0, 2] - rot_mat[:, 2, 0]) / (4 * w)
    z = (rot_mat[:, 1, 0] - rot_mat[:, 0, 1]) / (4 * w)
    return torch.stack([w, x, y, z], dim=-1)


class RobotPcSampler:
    def __init__(self, urdf_path, link_names=None, sapien_env_tuple=None):
        if sapien_env_tuple is not None:
            engine, scene, loader = sapien_env_tuple
            self.engine = engine
            self.scene = scene
        else:
            self.engine = sapien.Engine()
            self.scene = self.engine.create_scene()
            loader = self.scene.create_urdf_loader()
        self.sapien_robot = loader.load(urdf_path)
        self.robot_model = self.sapien_robot.create_pinocchio_model()
        self.urdf_robot = URDF.load(urdf_path)

        # load meshes and offsets from urdf_robot
        self.meshes = {}
        self.scales = {}
        self.offsets = {}
        prev_offset = np.eye(4)
        for link in self.urdf_robot.links:
            if link_names is not None and link.name not in link_names:
                continue
            if len(link.collisions) > 0:
                collision = link.collisions[0]
                prev_offset = collision.origin
                if collision.geometry.mesh != None:
                    if len(collision.geometry.mesh.meshes) > 0:
                        mesh = collision.geometry.mesh.meshes[0]
                        self.meshes[link.name] = trimesh_to_open3d(mesh)
                        self.scales[link.name] = collision.geometry.mesh.scale[0] if collision.geometry.mesh.scale is not None else 1.0
            self.offsets[link.name] = prev_offset
        self.pcd_dict = {}

    def compute_mesh_poses(self, qpos, link_names=None):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        if link_names is None:
            link_names = self.meshes.keys()
        link_idx_ls = []
        for link_name in link_names:
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    link_idx_ls.append(link_idx)
                    break
        link_pose_ls = np.stack([np.asarray(self.robot_model.get_link_pose(link_idx).to_transformation_matrix()) for link_idx in link_idx_ls])
        meshes_ls = [self.meshes[link_name] for link_name in link_names]
        offsets_ls = [self.offsets[link_name] for link_name in link_names]
        scales_ls = [self.scales[link_name] for link_name in link_names]
        poses = self.get_mesh_poses(poses=link_pose_ls, offsets=offsets_ls, scales=scales_ls)
        return poses
    
    def get_mesh_poses(self, poses, offsets, scales):
        try:
            assert poses.shape[0] == len(offsets)
        except:
            raise RuntimeError('poses and meshes must have the same length')

        N = poses.shape[0]
        all_mats = []
        for index in range(N):
            mat = poses[index]
            tf_obj_to_link = offsets[index]
            # print('offsets_ls[index]',offsets[index])
            mat = mat @ tf_obj_to_link
            all_mats.append(mat)
        return np.stack(all_mats)

    def sample_pc(self, link_names, num_pts):
        if link_names is None:
            link_names = self.meshes.keys()
        if num_pts is None:
            num_pts = [200] * len(link_names)
        cloud_list = {}
        meshes_ls = [self.meshes[link_name] for link_name in link_names]
        offsets_ls = [self.offsets[link_name] for link_name in link_names]
        scales_ls = [self.scales[link_name] for link_name in link_names]
        for index, link_name in enumerate(link_names):
            mesh = meshes_ls[index]
            mesh.scale(scales_ls[index], center=np.array([0, 0, 0]))
            sampled_cloud = mesh.sample_points_poisson_disk(number_of_points=num_pts[index])
            cloud_points = np.asarray(sampled_cloud.points)
            cloud_list[link_name] = cloud_points

        return cloud_list
    
    def transform_gs_torch(self, cloud_list, quat_list, qpos, base_qpos=None):
        device = cloud_list[list(cloud_list.keys())[0]].device
        dtype = cloud_list[list(cloud_list.keys())[0]].dtype
        link_base_pose_ls = None
        link_idx_ls = []
        link_names = cloud_list.keys()
        for link_name in link_names:
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    link_idx_ls.append(link_idx)
                    break
        if base_qpos is not None:
            fk = self.robot_model.compute_forward_kinematics(base_qpos)
            link_base_pose_ls = np.stack([np.asarray(self.robot_model.get_link_pose(link_idx).to_transformation_matrix()) for link_idx in link_idx_ls])
            link_base_pose_ls = torch.from_numpy(link_base_pose_ls).to(device).to(dtype)
        fk = self.robot_model.compute_forward_kinematics(qpos)
        link_pose_ls = np.stack([np.asarray(self.robot_model.get_link_pose(link_idx).to_transformation_matrix()) for link_idx in link_idx_ls])
        link_pose_ls = torch.from_numpy(link_pose_ls).to(device).to(dtype)
        all_pc = []
        all_quats = []
        for index, link_name in enumerate(link_names):
            tf_obj_to_link = torch.from_numpy(self.offsets[link_name]).to(device).to(dtype)
            mat = link_pose_ls[index] @ tf_obj_to_link
            cloud_points = cloud_list[link_name]
            if quat_list is not None:
                cloud_quats = quat_list[link_name]
            else:
                cloud_quats = None
            if base_qpos is not None:
                assert link_base_pose_ls is not None
                mat_base = link_base_pose_ls[index] @ tf_obj_to_link
                mat = mat @ torch.linalg.inv(mat_base)
            transformed_points = cloud_points @ mat[:3, :3].T + mat[:3, 3]
            quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(mat[:3, :3][None])  # (1, 4)
            if quat_list is not None:
                transformed_quats = quat_mult_torch(quat, cloud_quats)
            else:
                transformed_quats = torch.zeros((1, 4), device=device, dtype=dtype)
            all_pc.append(transformed_points)
            all_quats.append(transformed_quats)
        all_pc = torch.cat(all_pc, dim=0)
        all_quats = torch.cat(all_quats, dim=0)
        return all_pc, all_quats

    # compute robot pcd given qpos
    def compute_robot_pcd(self, qpos, link_names=None, num_pts=None, pcd_name=None):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        if link_names is None:
            link_names = self.meshes.keys()
        if num_pts is None:
            num_pts = [1000] * len(link_names)
        elif isinstance(num_pts, int):
            num_pts = [num_pts] * len(link_names)
        link_idx_ls = []
        for link_name in link_names:
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    link_idx_ls.append(link_idx)
                    break
        link_pose_ls = np.stack([np.asarray(self.robot_model.get_link_pose(link_idx).to_transformation_matrix()) for link_idx in link_idx_ls])
        meshes_ls = [self.meshes[link_name] for link_name in link_names]
        offsets_ls = [self.offsets[link_name] for link_name in link_names]
        scales_ls = [self.scales[link_name] for link_name in link_names]
        pcd = self.mesh_poses_to_pc(poses=link_pose_ls, meshes=meshes_ls, offsets=offsets_ls, num_pts=num_pts, scales=scales_ls, pcd_name=pcd_name)
        return pcd

    def mesh_poses_to_pc(self, poses, meshes, offsets, num_pts, scales, pcd_name=None):
        try:
            assert poses.shape[0] == len(meshes)
            assert poses.shape[0] == len(offsets)
            assert poses.shape[0] == len(num_pts)
            assert poses.shape[0] == len(scales)
        except:
            raise RuntimeError('poses and meshes must have the same length')

        N = poses.shape[0]
        all_pc = []
        for index in range(N):
            mat = poses[index]
            if pcd_name is None or pcd_name not in self.pcd_dict or len(self.pcd_dict[pcd_name]) <= index:
                mesh = copy.deepcopy(meshes[index])
                mesh.scale(scales[index], center=np.array([0, 0, 0]))
                sampled_cloud = mesh.sample_points_poisson_disk(number_of_points=num_pts[index])
                cloud_points = np.asarray(sampled_cloud.points)
                if pcd_name not in self.pcd_dict:
                    self.pcd_dict[pcd_name] = []
                self.pcd_dict[pcd_name].append(cloud_points)
            else:
                cloud_points = self.pcd_dict[pcd_name][index]
            
            tf_obj_to_link = offsets[index]
            mat = mat @ tf_obj_to_link
            transformed_points = cloud_points @ mat[:3, :3].T + mat[:3, 3]
            all_pc.append(transformed_points)
        all_pc = np.concatenate(all_pc, axis=0)
        return all_pc

    def compute_sensor_pc(self, qpos, sensor_names, pcd_name=None):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        sensor_idx_ls = []
        for link_name in sensor_names:
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    sensor_idx_ls.append(link_idx)
                    break
        sensor_pose = np.stack([np.asarray(self.robot_model.get_link_pose(link_idx).to_transformation_matrix()) for link_idx in sensor_idx_ls])
        offsets_ls = [self.offsets[link_name] for link_name in sensor_names]
        all_sensor_pc = []
        N = sensor_pose.shape[0]
        for index in range(N):
            sensor_pc = sensor_pose[index] @ offsets_ls[index]
            sensor_pc = sensor_pc[:3, 3]
            all_sensor_pc.append(sensor_pc)
        all_sensor_pc = np.stack(all_sensor_pc)
        return all_sensor_pc

    def compute_fk_links(self, qpos, link_idx):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        link_pose_ls = []
        for i in link_idx:
            link_pose_ls.append(self.robot_model.get_link_pose(i))
        return link_pose_ls

    def get_xarm_gripper_meshes(self, gripper_openness=1.0):  # 0: closed, 1: open
        g = 800 * gripper_openness  # gripper openness
        g = (800 - g) * 180 / np.pi
        base_qpos = np.array([0, -45, 0, 30, 0, 75, 0, 
                            g*0.001, g*0.001, g*0.001, g*0.001, g*0.001, g*0.001]) * np.pi / 180
        
        link_names = list(self.meshes.keys())
        meshes = [copy.deepcopy(self.meshes[link_name]) for link_name in link_names]
        poses = self.compute_mesh_poses(base_qpos, link_names=link_names)

        for i, mesh in enumerate(meshes):
            vertices = np.asarray(mesh.vertices)
            vertices = vertices @ poses[i][:3, :3].T + poses[i][:3, 3]
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.compute_vertex_normals()
        return meshes

    def get_xarm_pusher_meshes(self):
        base_qpos = np.array([0, -45, 0, 30, 0, 75, 0]) * np.pi / 180
        
        link_names = list(self.meshes.keys())
        meshes = [copy.deepcopy(self.meshes[link_name]) for link_name in link_names]
        poses = self.compute_mesh_poses(base_qpos, link_names=link_names)

        for i, mesh in enumerate(meshes):
            vertices = np.asarray(mesh.vertices)
            vertices = vertices @ poses[i][:3, :3].T + poses[i][:3, 3]
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.compute_vertex_normals()
        return meshes
