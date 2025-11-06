from pathlib import Path
import numpy as np
import torch
import copy
import random
import time
import open3d as o3d
import math
import transforms3d
import sapien.core as sapien
import kornia
import cv2
from sklearn.neighbors import NearestNeighbors

from ..utils.gs.transform_utils import setup_camera, Rt_to_w2c, interpolate_motions
from ..utils.gs.sh_utils import C0
from ..utils.gs.gs_processor import GSProcessor
from ..utils.robot.kinematics_utils import KinHelper
from ..utils.robot.robot_pc_sampler import RobotPcSampler
from ..utils.robot.robot_pc_transformations import get_eef_pts_xarm_gripper, get_eef_pts_xarm_pusher, \
    transform_gs_xarm_gripper, transform_eef_pts_xarm_gripper, transform_gs_xarm_pusher, transform_eef_pts_xarm_pusher

from diff_gaussian_rasterization import GaussianRasterizer


class GSRenderer:
    
    def __init__(self, cfg, local_rank=0):

        super().__init__()
        self.online = cfg.online

        self.cfg = cfg
        self.k_rel = 8  # knn for relations
        self.k_wgt = 16  # knn for weights

        self.device = f'cuda:{local_rank}'

        if self.online:
            import multiprocessing as mp
            from ..utils.gs.viser_gui import ViserViewer

            self._alive = mp.Value('b', False)
            self.viser_viewer = ViserViewer(self.device, cfg.viser_port)
            self.online_start_time = time.perf_counter()
            self.online_end_time = time.perf_counter()

        seed = cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.autograd.set_detect_anomaly(True)
        torch.backends.cudnn.benchmark = True

        self.metadata = {}
        self.metadata_wrist = {}
        self.state = {
            'x': None,  # world frame
            'v': None,  # world frame
            'x_his': None,  # world frame, does not include x
            'v_his': None,  # world frame, does not include v
            'color': None,  # for color ICP in phystwin
        }
        self.rendervar = {}
        self.rendervar_full = {}
        self.table_rendervar = {}
        self.gripper_rendervar = {}

        self.grippers = torch.empty(0)
        
        self.qpos_curr_xarm = np.array([0, -45, 0, 30, 0, 75, 0]) * np.pi / 180
        self.gripper_openness_curr_xarm = 800

        self.cameras = []
        self.wrist_cameras = []

        self.sp = GSProcessor()

        self.kin_helper = None
        self.relations = None
        self.weights = None
        self.k_rel_simple = 16

        self.visualize_mesh_points = cfg.physics.visualize_mesh_points
        self.visualize_phystwin_points = cfg.physics.visualize_phystwin_points
        self.visualize_eef_points = cfg.physics.visualize_eef_points

        self.params_meshes = {}
        self.meshes = {}

        engine = sapien.Engine()
        scene = engine.create_scene()
        loader = scene.create_urdf_loader()
        # for IK
        self.sample_robot = RobotPcSampler(self.cfg.env['urdf']['ik_urdf_path'], 
                                                   sapien_env_tuple=(engine, scene, loader))
        # for collision
        self.robot = RobotPcSampler(self.cfg.env['urdf']['collision_urdf_path'], 
                                    link_names=self.cfg.env['urdf']['collision_link_names'],
                                    sapien_env_tuple=(engine, scene, loader))
        self.init_gripper_openness_xarm = self.cfg.env['robot']['init_gripper_openness'] \
                    if 'init_gripper_openness' in self.cfg.env['robot'] else 0.0

        self.random_variables = []
        self.kin_helper = KinHelper('xarm7', sapien_env_tuple=(engine, scene, loader))

    def set_all_cameras(self):
        cfg = self.cfg
        camera_cfgs = cfg.env.cameras
        for camera_cfg in camera_cfgs:
            if camera_cfg.type == 'side':
                h = camera_cfg.h
                w = camera_cfg.w
                intr = np.array(camera_cfg.intr).reshape(3, 3).astype(np.float32)  # (3, 3)
                if 'c2w' in camera_cfg:
                    extrs = np.array(camera_cfg.c2w).reshape(4, 4).astype(np.float32)
                    extrs = np.linalg.inv(extrs)  # w2c
                else:
                    assert 'w2c' in camera_cfg
                    extrs = np.array(camera_cfg.w2c).reshape(4, 4).astype(np.float32)
                self.cameras.append([w, h, intr, extrs])
            else:
                assert camera_cfg.type == 'wrist'
                h_wr = camera_cfg.h
                w_wr = camera_cfg.w
                intr_wr = np.array(camera_cfg.intr).reshape(3, 3).astype(np.float32)  # (3, 3)
                if 'c2w' in camera_cfg:
                    extrs_wr = np.array(camera_cfg.c2w).reshape(4, 4).astype(np.float32)
                    extrs_wr = np.linalg.inv(extrs_wr)  # w2c
                else:
                    assert 'w2c' in camera_cfg
                    extrs_wr = np.array(camera_cfg.w2c).reshape(4, 4).astype(np.float32)
                self.wrist_cameras.append([w_wr, h_wr, intr_wr, extrs_wr])

        self.set_camera_custom(
            self.cfg.renderer.gs_center,
            self.cfg.renderer.gs_distance,
            self.cfg.renderer.gs_elevation,
            self.cfg.renderer.gs_azimuth
        )
        if self.wrist_cameras != []:
            w, h, intr, extrs = self.wrist_cameras[0]
            self.set_wrist_camera(w, h, intr, extrs)

    def set_camera_custom(self, center=(0, 0, 0), distance=0.8, elevation=20, azimuth=160.0, near=0.01, far=100.0):
        target = np.array(center)
        theta = 90 + azimuth
        z = distance * math.sin(math.radians(elevation))
        y = math.cos(math.radians(theta)) * distance * math.cos(math.radians(elevation))
        x = math.sin(math.radians(theta)) * distance * math.cos(math.radians(elevation))
        origin = target + np.array([x, y, z])
        
        look_at = target - origin
        look_at /= np.linalg.norm(look_at)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(look_at, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, look_at)
        w2c = np.eye(4)
        w2c[:3, 0] = right
        w2c[:3, 1] = -up
        w2c[:3, 2] = look_at
        w2c[:3, 3] = origin
        w2c = np.linalg.inv(w2c)
        w = 848
        h = 480
        intr = np.array(
            [[w / 2 * 1.0, 0., w / 2],
            [0., w / 2 * 1.0, h / 2],
            [0., 0., 1.]],
        )
        self.metadata = {
            'w': w,
            'h': h,
            'k': intr,
            'w2c': w2c,
            'near': near,
            'far': far,
        }
    
    def set_wrist_camera(self, w, h, intr, eef2c=None, R=None, t=None, near=0.01, far=100.0):
        assert not self.online  # viser viewer will automatically update camera
        if eef2c is None:
            assert R is not None and t is not None
            eef2c = Rt_to_w2c(R, t)
        self.metadata_wrist = {
            'w': w,
            'h': h,
            'k': intr,
            'eef2c': eef2c,
            'near': near,
            'far': far,
        }

    def knn_relations(self, bones):
        k = self.k_rel
        knn = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(bones.detach().cpu().numpy())
        _, indices = knn.kneighbors(bones.detach().cpu().numpy())  # (N, k)
        indices = indices[:, 1:]  # exclude self
        return indices
    
    def knn_weights(self, bones, pts):
        k = self.k_wgt
        dist = torch.norm(pts[:, None] - bones, dim=-1)  # (n_pts, n_bones)
        _, indices = torch.topk(dist, k, dim=-1, largest=False)
        bones_selected = bones[indices]  # (N, k, 3)
        dist = torch.norm(bones_selected - pts[:, None], dim=-1)  # (N, k)
        weights = 1 / (dist + 1e-6)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # (N, k)
        torch.cuda.empty_cache()  # free memory
        return weights, indices
    
    def update_camera(self, k, w2c, w=None, h=None, near=0.01, far=100.0):
        self.metadata['k'] = k
        self.metadata['w2c'] = w2c
        if w is not None:
            self.metadata['w'] = w
        if h is not None:
            self.metadata['h'] = h
        self.metadata['near'] = near
        self.metadata['far'] = far

    def reset_state(self, visualize_image=False):
        if self.online:
            self.viser_viewer.update()
            time.sleep(1)
            visualizer_metadata = self.viser_viewer.get_metadata()
            print('waiting for metadata', end='')
            while visualizer_metadata == {}:
                print('.', end='')
                self.viser_viewer.update()
                time.sleep(0.1)
                visualizer_metadata = self.viser_viewer.get_metadata()

        rendervar = self.rendervar
        xyz_0 = rendervar['means3D']
        color_0 = rendervar['shs'][:, 0] * C0 + 0.5

        downsample_indices = np.arange(1000)
        p_x = xyz_0[downsample_indices]
        p_color = color_0[downsample_indices]

        self.state['x'] = p_x.clone()
        self.state['v'] = torch.zeros_like(p_x)
        self.state['color'] = p_color.clone()

        self.update_rendervar()

        if self.online:
            self.render_online()

        if visualize_image:
            im, depth = self.render()
            im_vis = (im.permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)[:, :, ::-1].copy()
            cv2.imwrite('test.png', im_vis)

            depth_mask = depth[0].detach().cpu().numpy() < 15
            depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth[0].detach().cpu().numpy(), alpha=255 / depth[depth < 15].max().item()), cv2.COLORMAP_JET)
            depth_vis[~depth_mask] = 0
            cv2.imwrite('test_depth.png', depth_vis)

    def get_state(self):
        if self.grippers.shape[-1] == 14:
            return {
                'x': self.state['x'].clone(),
                'v': self.state['v'].clone(),
                'eef_xyz': self.grippers[:, :3].clone(),  # (n_grippers, 3)
                'eef_vel': self.grippers[:, 3:6].clone(),
                'eef_quat': self.grippers[:, 6:10].clone(),
                'eef_quat_vel': self.grippers[:, 10:13].clone(),
                'eef_gripper': self.grippers[:, 13:].clone(),
                'color': self.state['color'].clone(),
            }
        else:
            return {
                'x': self.state['x'].clone(),
                'v': self.state['v'].clone(),
                'eef_xyz': self.grippers[:, :3].clone(),  # (n_grippers, 3)
                'eef_vel': self.grippers[:, 3:6].clone(),
                'eef_quat': None,
                'eef_quat_vel': None,
                'eef_gripper': self.grippers[:, 7:].clone(),
                'color': self.state['color'].clone(),
            }

    def knn_relations_simple(self, bones, pts):
        k = self.k_rel_simple
        knn = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(bones.detach().cpu().numpy())
        _, indices = knn.kneighbors(pts.detach().cpu().numpy())
        return indices

    def knn_weights_simple(self, relations, bones, pts):  # relations: (N, k), bones: (n, 3), pts: (N, 3)
        k = self.k_rel_simple
        bones_rel = bones[relations]  # (N, k, 3)
        dist = torch.norm(pts[:, None, :] - bones_rel, dim=-1)  # (N, k)
        weights = 1 / (dist + 1e-6)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # (N, k)
        weights_all = torch.zeros((pts.shape[0], bones.shape[0]), device=pts.device)
        weights_all[torch.arange(pts.shape[0])[:, None], relations] = weights
        return weights_all  # (N, n)

    def set_eef(self, eef_xyz, eef_quat, eef_gripper, eef_xyz_next=None, eef_vel=None, eef_quat_next=None, eef_quat_vel=None):
        # eef_xyz: (n_grippers, 3)
        # eef_quat: (n_grippers, 4)
        # eef_gripper: (n_grippers, 1)
        cfg = self.cfg

        if eef_xyz_next is not None:
            eef_vel = (eef_xyz_next - eef_xyz) * cfg.physics.fps
        else:
            assert eef_vel is not None
        
        # construct rotational velocity
        if eef_quat_next is not None:
            eef_rot_this = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat.reshape(-1, 4))  # (n_grippers, 3, 3)
            eef_rot_next = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat_next.reshape(-1, 4))  # (n_grippers, 3, 3)
            eef_rot_delta = eef_rot_this.bmm(eef_rot_next.inverse())
            eef_aa = kornia.geometry.conversions.rotation_matrix_to_axis_angle(eef_rot_delta)  # (n_grippers, 3)

            eef_quat_vel = torch.zeros((cfg.env.robot.n_grippers, 3), dtype=eef_quat.dtype, device=self.device)
            eef_quat_vel = eef_aa * cfg.physics.fps  # (n_grippers, 3), radian per second
        else:
            assert eef_quat_vel is not None

        grippers = torch.zeros((cfg.env.robot.n_grippers, 14), device=self.device)
        grippers[:, :3] = eef_xyz
        grippers[:, 3:6] = eef_vel
        grippers[:, 6:10] = eef_quat
        grippers[:, 10:13] = eef_quat_vel
        grippers[:, 13:] = eef_gripper
        self.grippers = grippers.to(self.device).to(torch.float32)

    def load_scaniverse(self, randomize=False, index=None):

        ### load splat params
        params_obj = self.sp.load(self.cfg.gs['object']['path'])
        params_table_robot = self.sp.load(self.cfg.gs['scene']['table_splat_path'])

        ### load meshes
        true_index = None
        true_index_mesh = None
        if randomize and self.cfg.gs.use_grid_randomization:
            obj_grid_cfg = self.cfg.gs.object.grid_randomization
            n_object_rand = len(obj_grid_cfg.xy) * len(obj_grid_cfg.theta) \
                    if not obj_grid_cfg.one_to_one else len(obj_grid_cfg.xy)
            assert index is not None
            true_index_mesh = index // n_object_rand
            true_index = index % n_object_rand
        else:
            true_index = index

        params_meshes = {}
        meshes = {}
        for mesh_obj in self.cfg.gs['meshes']:
            mesh_name = mesh_obj['name']
            mesh = o3d.io.read_triangle_mesh(mesh_obj['mesh_path'])  # read mesh
            pose_mesh = np.array(mesh_obj['pose']).reshape(4, 4)  # 3d transformation

            if randomize and self.cfg.gs.use_grid_randomization and \
                    'grid_randomization' in mesh_obj and mesh_obj.grid_randomization:
                mesh_grid_cfg = mesh_obj.grid_randomization
                xy_list = mesh_grid_cfg.xy
                theta_list = mesh_grid_cfg.theta
                one_to_one = mesh_grid_cfg.one_to_one
                n_this_mesh_rand = len(xy_list) * len(theta_list) \
                    if not one_to_one else len(xy_list)

                assert true_index_mesh is not None
                true_index_this_mesh = true_index_mesh % n_this_mesh_rand
                true_index_mesh = true_index_mesh // n_this_mesh_rand  # recursive

                if one_to_one:
                    rand_x = xy_list[true_index_this_mesh][0]
                    rand_y = xy_list[true_index_this_mesh][1]
                    rand_z = 0.0
                    rand_a = theta_list[true_index_this_mesh] * np.pi / 180.
                else:
                    xy_index = true_index_this_mesh // len(theta_list)
                    theta_index = true_index_this_mesh % len(theta_list)
                    rand_x = xy_list[xy_index][0]
                    rand_y = xy_list[xy_index][1]
                    rand_z = 0.0
                    rand_a = theta_list[theta_index] * np.pi / 180.

                rand_trans = np.array([rand_x, rand_y, rand_z], dtype=np.float32)
                pose_mesh[:3, 3] += rand_trans
                rot_z = np.array([[np.cos(rand_a), -np.sin(rand_a), 0], [np.sin(rand_a), np.cos(rand_a), 0], [0, 0, 1]], dtype=np.float32)
                pose_mesh[:3, :3] = rot_z @ pose_mesh[:3, :3]

                self.random_variables.append([rand_x, rand_y, rand_z, rand_a])

            elif randomize:
                translation_range = np.array(mesh_obj['translation_range'])  # (6,)
                azimuth_range = np.array(mesh_obj['azimuth_range'])  # (6,)
                rand_x = np.random.uniform(translation_range[0], translation_range[1])
                rand_y = np.random.uniform(translation_range[2], translation_range[3])
                rand_z = np.random.uniform(translation_range[4], translation_range[5])
                rand_trans = np.array([rand_x, rand_y, rand_z], dtype=np.float32)
                pose_mesh[:3, 3] += rand_trans

                rand_a = np.random.uniform(azimuth_range[0], azimuth_range[1]) * np.pi / 180.
                rot_z = np.array([[np.cos(rand_a), -np.sin(rand_a), 0], [np.sin(rand_a), np.cos(rand_a), 0], [0, 0, 1]], dtype=np.float32)
                pose_mesh[:3, :3] = rot_z @ pose_mesh[:3, :3]

                self.random_variables.append([rand_x, rand_y, rand_z, rand_a])

            params_mesh = self.sp.load(mesh_obj["splat_path"])
            pts = params_mesh['means3D'].cpu().numpy()
            shs = params_mesh['sh_colors'].cpu().numpy()
            scales = torch.exp(params_mesh['log_scales']).cpu().numpy()
            quats = params_mesh['unnorm_rotations'].cpu().numpy()
            opacities = torch.sigmoid(params_mesh['logit_opacities']).cpu().numpy()

            n_gs = shs.shape[0]
            shs_dc = shs[:, :3][:, None]  # (n, 1, 3)
            shs_extra = shs[:, 3:]  # (n, 45)
            shs_extra = shs_extra.reshape(n_gs, 3, -1).transpose((0, 2, 1))  # shs_extra: (n, 15, 3)
            shs = np.concatenate([shs_dc, shs_extra], axis=1)  # (n, 16, 3)

            if 'color_A' in mesh_obj:
                A = np.array(mesh_obj['color_A'], dtype=np.float32).reshape(3, -1)
                b = np.array(mesh_obj['color_b'], dtype=np.float32).reshape(3)
                max_sh_degrees = np.sqrt(shs.shape[1]) - 1
                shs_new = []
                if A.shape[1] == 3:  # linear color correction
                    for si in range(int(max_sh_degrees) + 1):  # si: order
                        shs_temp = shs[:, si ** 2:(si + 1) ** 2, :]
                        if si == 0:
                            shs_temp_flat = np.squeeze(shs_temp, axis=1)  # (n, 1, 3) -> (n, 3)
                            shs_temp_flat_corrected = shs_temp_flat @ A.T
                            sh_offset = np.ones(3) * 0.5
                            bias = (1. / C0) * (sh_offset.reshape(1, 3) @ A.T + b - sh_offset)  # (1, 3)
                            shs_temp_flat_corrected_biased = shs_temp_flat_corrected + bias  # (n, 3)
                            shs_new.append(shs_temp_flat_corrected_biased[:, None])
                        else:
                            shs_temp_corrected = shs_temp @ A.T  # (n, 3)
                            shs_new.append(shs_temp_corrected)

                elif A.shape[1] == 6:  # quadratic color correction
                    assert not self.cfg.gs.use_shs
                    for si in range(int(max_sh_degrees) + 1):  # si: order
                        shs_temp = shs[:, si ** 2:(si + 1) ** 2, :]
                        if si == 0:
                            shs_temp_flat = np.squeeze(shs_temp, axis=1)  # (n, 1, 3) -> (n, 3)
                            A_2 = A[:, :3]  # (3, 3)
                            A_1 = A[:, 3:]  # (3, 3)
                            shs_corrected_1 = shs_temp_flat @ A_1.T  # (n, 3)
                            shs_corrected_2 = (shs_temp_flat + C0 * shs_temp_flat ** 2) @ A_2.T  # (n, 3)
                            sh_offset_1 = np.ones(3) * 0.5
                            sh_offset_2 = np.ones(3) * 0.25
                            bias = (1. / C0) * (sh_offset_2.reshape(1, 3) @ A_2.T + \
                                    sh_offset_1.reshape(1, 3) @ A_1.T + b - sh_offset_1)  # (1, 3)
                            shs_corrected_biased = shs_corrected_1 + shs_corrected_2 + bias  # (n, 3)
                            shs_new.append(shs_corrected_biased[:, None])
                        else:
                            A_1 = A[:, 3:]  # (3, 3)
                            shs_new.append(shs_temp @ A_1.T)  # (n, 3)

                shs = np.concatenate(shs_new, axis=1)

            # transform mesh and splat
            mesh.transform(pose_mesh)
            pts = pts @ pose_mesh[:3, :3].T + pose_mesh[:3, 3]  # (n, 3)

            vertices = np.asarray(mesh.vertices)

            # visualize mesh points as GS
            if self.visualize_mesh_points:
                new_pts = vertices.copy()
                new_colors = np.array([
                    [1, 0, 0],
                ]).repeat(new_pts.shape[0], axis=0)
                new_shs = (new_colors - 0.5) / C0
                new_shs = np.concatenate([
                    new_shs,
                    np.zeros((new_shs.shape[0], 45), dtype=new_shs.dtype)
                ], axis=1)
                new_shs = new_shs.reshape(new_shs.shape[0], 16, 3)
                new_scales = np.array([
                    [0.002, 0.002, 0.002],
                ]).repeat(new_pts.shape[0], axis=0)
                new_quats = np.array([
                    [1, 0, 0, 0],
                ]).repeat(new_pts.shape[0], axis=0)
                new_opacities = np.array([
                    [1],
                ]).repeat(new_pts.shape[0], axis=0)
                pts = np.concatenate([pts, new_pts], axis=0)
                shs = np.concatenate([shs, new_shs], axis=0)
                scales = np.concatenate([scales, new_scales], axis=0)
                quats = np.concatenate([quats, new_quats], axis=0)
                opacities = np.concatenate([opacities, new_opacities], axis=0)

            params_meshes[mesh_name] = {
                'means3D': torch.from_numpy(pts).to(torch.float32).to(self.device),
                'shs': torch.from_numpy(shs).to(torch.float32).to(self.device),
                'scales': torch.from_numpy(scales).to(torch.float32).to(self.device),
                'rotations': torch.nn.functional.normalize(torch.from_numpy(quats).to(torch.float32).to(self.device), dim=-1),
                'opacities': torch.from_numpy(opacities).to(torch.float32).to(self.device),
            }
            meshes[mesh_name] = mesh
        self.params_meshes = params_meshes
        self.meshes = meshes
        
        total_mask_full = np.load(self.cfg.gs['scene']['total_mask_path'])
        total_mask_full = torch.from_numpy(total_mask_full).to(self.device).to(torch.float32)
        self.total_mask_full = total_mask_full

        ### load robot points ###
        init_quat = self.cfg.env['robot']['init_quat'] if 'init_quat' in self.cfg.env['robot'] else [0, 1, 0, 0]
        init_gripper = self.cfg.env['robot']['init_gripper'] if 'init_gripper' in self.cfg.env['robot'] else [1.0]

        eef_xyz = torch.tensor(self.cfg.env['robot']['init_eef_xyz'], dtype=torch.float32).to(self.device).reshape(-1, 3)  # (n_grippers, 3)
        eef_quat = torch.tensor(init_quat, dtype=torch.float32).to(self.device).reshape(-1, 4)  # (n_grippers, 4)
        eef_gripper = torch.tensor(init_gripper, dtype=torch.float32).to(self.device).reshape(-1, 1)  # (n_grippers, 1)
        self.set_eef(eef_xyz, eef_quat, eef_gripper, eef_xyz_next=eef_xyz, eef_quat_next=eef_quat)
        self.init_eef_xyz = eef_xyz.clone()
        self.init_eef_quat = eef_quat.clone()

        if self.cfg.env['robot']['use_pusher']:
            self.eef_pts, self.eef_pts_func = get_eef_pts_xarm_pusher(
                eef_xyz[0], eef_quat[0], self.robot, self.sample_robot, self.kin_helper, self.qpos_curr_xarm, self.device)
        else:
            self.eef_pts, self.eef_pts_func = get_eef_pts_xarm_gripper(
                eef_xyz[0], eef_quat[0], eef_gripper[0], self.robot, self.sample_robot, self.kin_helper, self.qpos_curr_xarm, self.device)

        ### load robot points end ###

        pts = params_obj['means3D'].cpu().numpy()
        shs = params_obj['sh_colors'].cpu().numpy()  # shs: (n, 48)
        scales = torch.exp(params_obj['log_scales']).cpu().numpy()
        quats = params_obj['unnorm_rotations'].cpu().numpy()
        opacities = torch.sigmoid(params_obj['logit_opacities']).cpu().numpy()

        n_gs = shs.shape[0]
        shs_dc = shs[:, :3][:, None]  # (n, 1, 3)
        shs_extra = shs[:, 3:]  # (n, 45)
        shs_extra = shs_extra.reshape(n_gs, 3, -1).transpose((0, 2, 1))  # shs_extra: (n, 15, 3)
        shs = np.concatenate([shs_dc, shs_extra], axis=1)  # (n, 16, 3)

        if 'color_A' in self.cfg.gs['object']:
            A = np.array(self.cfg.gs['object']['color_A'], dtype=np.float32).reshape(3, -1)
            b = np.array(self.cfg.gs['object']['color_b'], dtype=np.float32).reshape(3)
            max_sh_degrees = np.sqrt(shs.shape[1]) - 1
            shs_new = []
            if A.shape[1] == 3:  # linear color correction
                for si in range(int(max_sh_degrees) + 1):  # si: order
                    shs_temp = shs[:, si ** 2:(si + 1) ** 2, :]
                    if si == 0:
                        shs_temp_flat = np.squeeze(shs_temp, axis=1)  # (n, 1, 3) -> (n, 3)
                        shs_temp_flat_corrected = shs_temp_flat @ A.T
                        sh_offset = np.ones(3) * 0.5
                        bias = (1. / C0) * (sh_offset.reshape(1, 3) @ A.T + b - sh_offset)  # (1, 3)
                        shs_temp_flat_corrected_biased = shs_temp_flat_corrected + bias  # (n, 3)
                        shs_new.append(shs_temp_flat_corrected_biased[:, None])
                    else:
                        shs_temp_corrected = shs_temp @ A.T  # (n, 3)
                        shs_new.append(shs_temp_corrected)

            elif A.shape[1] == 6:  # quadratic color correction
                assert not self.cfg.gs.use_shs
                for si in range(int(max_sh_degrees) + 1):  # si: order
                    shs_temp = shs[:, si ** 2:(si + 1) ** 2, :]
                    if si == 0:
                        shs_temp_flat = np.squeeze(shs_temp, axis=1)  # (n, 1, 3) -> (n, 3)
                        A_2 = A[:, :3]  # (3, 3)
                        A_1 = A[:, 3:]  # (3, 3)
                        shs_corrected_1 = shs_temp_flat @ A_1.T  # (n, 3)
                        shs_corrected_2 = (shs_temp_flat + C0 * shs_temp_flat ** 2) @ A_2.T  # (n, 3)
                        sh_offset_1 = np.ones(3) * 0.5
                        sh_offset_2 = np.ones(3) * 0.25
                        bias = (1. / C0) * (sh_offset_2.reshape(1, 3) @ A_2.T + \
                                sh_offset_1.reshape(1, 3) @ A_1.T + b - sh_offset_1)  # (1, 3)
                        shs_corrected_biased = shs_corrected_1 + shs_corrected_2 + bias  # (n, 3)
                        shs_new.append(shs_corrected_biased[:, None])
                    else:
                        A_1 = A[:, 3:]  # (3, 3)
                        shs_new.append(shs_temp @ A_1.T)  # (n, 3)

            shs = np.concatenate(shs_new, axis=1)

        self.rendervar = {
            'means3D': torch.from_numpy(pts).to(torch.float32).to(self.device),
            'shs': torch.from_numpy(shs).to(torch.float32).to(self.device),
            'scales': torch.from_numpy(scales).to(torch.float32).to(self.device),
            'rotations': torch.nn.functional.normalize(torch.from_numpy(quats).to(torch.float32).to(self.device), dim=-1),
            'opacities': torch.from_numpy(opacities).to(torch.float32).to(self.device),
        }
        self.rendervar['means2D'] = torch.zeros_like(self.rendervar['means3D'])  # (N, 3)

        # transform splat
        pose_obj = np.array(self.cfg.gs['object']['pose']).reshape(4, 4)

        # apply grid randomization for the phystwin object
        if randomize and self.cfg.gs.use_grid_randomization:
            obj_grid_cfg = self.cfg.gs.object.grid_randomization
            xy_list = obj_grid_cfg.xy
            theta_list = obj_grid_cfg.theta
            one_to_one = obj_grid_cfg.one_to_one
            assert true_index is not None
            if one_to_one:
                rand_x = xy_list[true_index][0]
                rand_y = xy_list[true_index][1]
                rand_z = 0.0
                rand_a = theta_list[true_index] * np.pi / 180.
            else:
                xy_index = true_index // len(theta_list)
                theta_index = true_index % len(theta_list)
                rand_x = xy_list[xy_index][0]
                rand_y = xy_list[xy_index][1]
                rand_z = 0.0
                rand_a = theta_list[theta_index] * np.pi / 180.

            rand_trans = np.array([rand_x, rand_y, rand_z], dtype=np.float32)
            pose_obj[:3, 3] += rand_trans
            rot_z = np.array([[np.cos(rand_a), -np.sin(rand_a), 0], [np.sin(rand_a), np.cos(rand_a), 0], [0, 0, 1]], dtype=np.float32)
            pose_obj[:3, :3] = rot_z @ pose_obj[:3, :3]

            self.random_variables.append([rand_x, rand_y, rand_z, rand_a])

        elif randomize and not self.cfg.gs.use_grid_randomization:
            translation_range = np.array(self.cfg.gs['object']['translation_range'])  # (6,)
            azimuth_range = np.array(self.cfg.gs['object']['azimuth_range'])  # (6,)
            rand_x = np.random.uniform(translation_range[0], translation_range[1])
            rand_y = np.random.uniform(translation_range[2], translation_range[3])
            rand_z = np.random.uniform(translation_range[4], translation_range[5])
            rand_trans = np.array([rand_x, rand_y, rand_z], dtype=np.float32)
            pose_obj[:3, 3] += rand_trans

            rand_a = np.random.uniform(azimuth_range[0], azimuth_range[1]) * np.pi / 180.
            rot_z = np.array([[np.cos(rand_a), -np.sin(rand_a), 0], [np.sin(rand_a), np.cos(rand_a), 0], [0, 0, 1]], dtype=np.float32)
            pose_obj[:3, :3] = rot_z @ pose_obj[:3, :3]

            self.random_variables.append([rand_x, rand_y, rand_z, rand_a])

        pose_obj = torch.from_numpy(pose_obj).to(self.device).to(torch.float32)  # (4, 4)
        self.pose_obj = pose_obj

        xyz = self.rendervar['means3D']
        quat = self.rendervar['rotations']
        rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(quat)  # (n, 3, 3)
        xyz = xyz @ pose_obj[:3, :3].T + pose_obj[:3, 3]  # (n, 3)
        rot = pose_obj[:3, :3] @ rot  # (n, 3, 3)
        quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(rot)  # (n, 4)
        self.rendervar['means3D'] = xyz
        self.rendervar['rotations'] = torch.nn.functional.normalize(quat, dim=-1)  # (n, 4)

        pts = params_table_robot['means3D'].cpu().numpy()
        shs = params_table_robot['sh_colors'].cpu().numpy()  # shs: (n, 48)
        scales = torch.exp(params_table_robot['log_scales']).cpu().numpy()
        quats = params_table_robot['unnorm_rotations'].cpu().numpy()
        opacities = torch.sigmoid(params_table_robot['logit_opacities']).cpu().numpy()

        n_gs = shs.shape[0]
        shs_dc = shs[:, :3][:, None]  # (n, 1, 3)
        shs_extra = shs[:, 3:]  # (n, 45)
        shs_extra = shs_extra.reshape(n_gs, 3, -1).transpose((0, 2, 1))  # shs_extra: (n, 15, 3)
        shs = np.concatenate([shs_dc, shs_extra], axis=1)  # (n, 16, 3)

        if 'color_A' in self.cfg.gs['scene']:
            A = np.array(self.cfg.gs['scene']['color_A'], dtype=np.float32).reshape(3, -1)
            b = np.array(self.cfg.gs['scene']['color_b'], dtype=np.float32).reshape(3)
            max_sh_degrees = np.sqrt(shs.shape[1]) - 1
            shs_new = []
            if A.shape[1] == 3:  # linear color correction
                for si in range(int(max_sh_degrees) + 1):  # si: order
                    shs_temp = shs[:, si ** 2:(si + 1) ** 2, :]
                    if si == 0:
                        shs_temp_flat = np.squeeze(shs_temp, axis=1)  # (n, 1, 3) -> (n, 3)
                        shs_temp_flat_corrected = shs_temp_flat @ A.T
                        sh_offset = np.ones(3) * 0.5
                        bias = (1. / C0) * (sh_offset.reshape(1, 3) @ A.T + b - sh_offset)  # (1, 3)
                        shs_temp_flat_corrected_biased = shs_temp_flat_corrected + bias  # (n, 3)
                        shs_new.append(shs_temp_flat_corrected_biased[:, None])
                    else:
                        shs_temp_corrected = shs_temp @ A.T  # (n, 3)
                        shs_new.append(shs_temp_corrected)

            elif A.shape[1] == 6:  # quadratic color correction
                assert not self.cfg.gs.use_shs
                for si in range(int(max_sh_degrees) + 1):  # si: order
                    shs_temp = shs[:, si ** 2:(si + 1) ** 2, :]
                    if si == 0:
                        shs_temp_flat = np.squeeze(shs_temp, axis=1)  # (n, 1, 3) -> (n, 3)
                        A_2 = A[:, :3]  # (3, 3)
                        A_1 = A[:, 3:]  # (3, 3)
                        shs_corrected_1 = shs_temp_flat @ A_1.T  # (n, 3)
                        shs_corrected_2 = (shs_temp_flat + C0 * shs_temp_flat ** 2) @ A_2.T  # (n, 3)
                        sh_offset_1 = np.ones(3) * 0.5
                        sh_offset_2 = np.ones(3) * 0.25
                        bias = (1. / C0) * (sh_offset_2.reshape(1, 3) @ A_2.T + \
                                sh_offset_1.reshape(1, 3) @ A_1.T + b - sh_offset_1)  # (1, 3)
                        shs_corrected_biased = shs_corrected_1 + shs_corrected_2 + bias  # (n, 3)
                        shs_new.append(shs_corrected_biased[:, None])
                    else:
                        A_1 = A[:, 3:]  # (3, 3)
                        shs_new.append(shs_temp @ A_1.T)  # (n, 3)

            shs = np.concatenate(shs_new, axis=1)

        pts = torch.tensor(pts).to(torch.float32).to(self.device)
        shs = torch.tensor(shs).to(torch.float32).to(self.device)
        scales = torch.tensor(scales).to(torch.float32).to(self.device)
        quats = torch.tensor(quats).to(torch.float32).to(self.device)
        opacities = torch.tensor(opacities).to(torch.float32).to(self.device)

        self.table_rendervar = {
            'means3D': pts,
            'shs': shs,
            'scales': scales,
            'rotations': quats,
            'opacities': opacities,
        }
        self.table_rendervar['means2D'] = torch.zeros_like(self.table_rendervar['means3D'])  # (N, 3)


    def update_rendervar(self, x_pred=None, gripper_now=None, qpos_now=None):
        rendervar = self.rendervar

        # world frame
        xyz = rendervar['means3D'].clone()
        shs = rendervar['shs'].clone()
        quat = rendervar['rotations'].clone()
        opa = rendervar['opacities'].clone()
        scales = rendervar['scales'].clone()

        p_x = self.state['x'].clone()

        if x_pred is not None:
            p_x_pred = x_pred

            if self.cfg.physics.use_lbs:
                if (not self.cfg.physics.precompute_relations) or self.relations is None:
                    if self.relations is None:
                        assert self.weights is None
                    self.relations = self.knn_relations(p_x)
                    self.weights = self.knn_weights(p_x, xyz)  # a tuple
                relations = self.relations
                weights, weights_indices = self.weights
                xyz, _, _ = interpolate_motions(
                    bones=p_x,
                    motions=p_x_pred - p_x,
                    relations=relations,
                    weights=weights,
                    weights_indices=weights_indices,
                    xyz=xyz,
                    quat=None,
                    device=self.device,
                )
            else:
                if self.relations is None:
                    assert self.weights is None
                    self.relations = self.knn_relations_simple(p_x, xyz)
                    self.weights = self.knn_weights_simple(self.relations, p_x, xyz)
                relations = self.relations  # (N, k)
                weights = self.weights  # (N, n)
                xyz = weights @ p_x_pred  # (N, 3)

        # normalize
        quat = torch.nn.functional.normalize(quat, dim=-1)
        
        self.rendervar = {
            'means3D': xyz,
            'shs': shs,
            'rotations': quat,
            'opacities': opa,
            'scales': scales,
            'means2D': torch.zeros_like(xyz),
        }

        # visualize object dynamics points
        if self.visualize_phystwin_points and x_pred is not None:
            new_xyz = x_pred.clone()
            new_rgb = torch.tensor([
                [0, 1, 0],
            ], dtype=torch.float32, device=self.device).repeat(new_xyz.shape[0], 1)
            new_shs = (new_rgb - 0.5) / C0
            new_shs = torch.cat([
                new_shs,
                torch.zeros((new_shs.shape[0], 45), device=self.device, dtype=new_shs.dtype)
            ], dim=1)
            new_shs = new_shs.reshape(new_shs.shape[0], 16, 3)
            new_scales = torch.tensor([
                [0.001, 0.001, 0.001],
            ], dtype=torch.float32, device=self.device).repeat(new_xyz.shape[0], 1)
            new_quat = torch.tensor([
                [1, 0, 0, 0],
            ], dtype=torch.float32, device=self.device).repeat(new_xyz.shape[0], 1)
            new_opa = torch.tensor([
                [1],
            ], dtype=torch.float32, device=self.device).repeat(new_xyz.shape[0], 1)
            xyz = torch.cat([xyz, new_xyz], dim=0)
            shs = torch.cat([shs, new_shs], dim=0)
            scales = torch.cat([scales, new_scales], dim=0)
            quat = torch.cat([quat, new_quat], dim=0)
            opa = torch.cat([opa, new_opa], dim=0)

        assert self.params_meshes is not None
        for mesh_obj in self.params_meshes.keys():
            mesh = self.meshes[mesh_obj]
            params_mesh = self.params_meshes[mesh_obj]
            m_pts = params_mesh['means3D'].clone()
            m_shs = params_mesh['shs'].clone()
            m_quats = params_mesh['rotations'].clone()
            m_opacities = params_mesh['opacities'].clone()
            m_scales = params_mesh['scales'].clone()

            # merge
            xyz = torch.cat([xyz, m_pts], dim=0)
            shs = torch.cat([shs, m_shs], dim=0)
            quat = torch.cat([quat, m_quats], dim=0)
            opa = torch.cat([opa, m_opacities], dim=0)
            scales = torch.cat([scales, m_scales], dim=0)

        if qpos_now is None:
            # add gripper pos
            if gripper_now is None:
                gripper_now = self.grippers.clone()
            
            eef_xyz = gripper_now[:, :3]
            eef_quat = gripper_now[:, 6:10]
            eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat)  # (n_grippers, 3, 3)

            # this will only be used for xarm
            gripper_openness_xarm = gripper_now[:, 13].item() * 800.0

            qpos = np.zeros(
                (self.cfg.env['robot']['n_grippers'], self.cfg.env['robot']['n_qpos']), 
                dtype=np.float32
            )  # (1, 7) for xarm
            for gi in range(self.cfg.env.robot.n_grippers):
                eef_xyz_base = eef_xyz[gi]  # (3,)
                eef_rot_base = eef_rot[gi]  # (3, 3)

                e2b = torch.eye(4, device=self.device)
                e2b[:3, :3] = eef_rot_base
                e2b[:3, 3:4] = eef_xyz_base[:, None]  # (3, 1)

                fk_trans_mat = e2b.cpu().numpy()

                cur_xyzrpy = np.zeros(6)
                cur_xyzrpy[:3] = fk_trans_mat[:3, 3]
                cur_xyzrpy[3:] = transforms3d.euler.mat2euler(fk_trans_mat[:3, :3])
                
                assert self.kin_helper is not None
                qpos[gi] = self.kin_helper.compute_ik_sapien(self.qpos_curr_xarm, cur_xyzrpy)

        else:
            qpos = qpos_now[:, :-1].cpu().numpy()  # need to remove gripper openness only for xarm
            if gripper_now is None:
                gripper_now = self.grippers.clone()
            gripper_openness_xarm = gripper_now[:, 13].item() * 800.0

        if self.visualize_eef_points:
            ## get eef_pts directly
            assert qpos.shape[0] == 1
            if self.cfg.env['robot']['use_pusher']:
                eef_pts = transform_eef_pts_xarm_pusher(self.robot, qpos[0], device=self.device, sample_robot=self.sample_robot)
            else:
                eef_pts = transform_eef_pts_xarm_gripper(self.robot, qpos[0], gripper_openness_xarm, device=self.device, sample_robot=self.sample_robot)

            eef_colors = torch.tensor([
                [1, 0, 0],
            ], dtype=eef_pts.dtype, device=eef_pts.device).repeat(eef_pts.shape[0], 1)
            eef_shs = (eef_colors - 0.5) / C0
            eef_shs = torch.cat([
                eef_shs,
                torch.zeros((eef_shs.shape[0], 45), device=self.device, dtype=eef_shs.dtype)
            ], dim=1)
            eef_shs = eef_shs.reshape(eef_shs.shape[0], 16, 3)
            eef_scales = torch.tensor([
                [0.001, 0.001, 0.001],
            ], dtype=eef_pts.dtype, device=eef_pts.device).repeat(eef_pts.shape[0], 1)
            eef_quats = torch.tensor([
                [1, 0, 0, 0],
            ], dtype=eef_pts.dtype, device=eef_pts.device).repeat(eef_pts.shape[0], 1)
            eef_opacities = torch.tensor([
                [1],
            ], dtype=eef_pts.dtype, device=eef_pts.device).repeat(eef_pts.shape[0], 1)

            xyz = torch.cat([xyz, eef_pts], dim=0)
            shs = torch.cat([shs, eef_shs], dim=0)
            quat = torch.cat([quat, eef_quats], dim=0)
            opa = torch.cat([opa, eef_opacities], dim=0)
            scales = torch.cat([scales, eef_scales], dim=0)

        table_params = {k: v.clone() for k, v in self.table_rendervar.items()}
        assert qpos.shape[0] == 1
        if self.cfg.env['robot']['use_pusher']:
            table_params = transform_gs_xarm_pusher(qpos[0], table_params, total_mask=self.total_mask_full, sample_robot=self.sample_robot)
        else:
            table_params = transform_gs_xarm_gripper(qpos[0], gripper_openness_xarm, table_params, init_gripper=self.init_gripper_openness_xarm, 
                                        total_mask=self.total_mask_full, sample_robot=self.sample_robot)
        
        t_pts = table_params['means3D']
        t_shs = table_params['shs']
        t_quats = table_params['rotations']
        t_opacities = table_params['opacities']
        t_scales = table_params['scales']

        xyz = torch.cat([xyz, t_pts], dim=0)
        shs = torch.cat([shs, t_shs], dim=0)
        quat = torch.cat([quat, t_quats], dim=0)
        opa = torch.cat([opa, t_opacities], dim=0)
        scales = torch.cat([scales, t_scales], dim=0)

        # normalize
        quat = torch.nn.functional.normalize(quat, dim=-1)

        self.rendervar_full = {
            'means3D': xyz,
            'shs': shs,
            'rotations': quat,
            'opacities': opa,
            'scales': scales,
            'means2D': torch.zeros_like(xyz),
        }

        assert len(qpos) == 1
        self.qpos_curr_xarm = qpos[0]  # update current qpos
        self.gripper_openness_curr_xarm = gripper_openness_xarm  # update current gripper openness

    @torch.no_grad
    def render(self, render_data=None, bg=[0.0, 0.0, 0.0], camera=None):
        assert self.metadata != {}
        if render_data is None:
            assert self.rendervar_full != {}
            render_data = self.rendervar_full
        render_data = {k: v.to(self.device) for k, v in render_data.items()}

        if camera is not None:
            w, h, k, w2c = camera
        else:
            w, h = self.metadata['w'], self.metadata['h']
            k, w2c = self.metadata['k'], self.metadata['w2c']

        max_sh_degrees = int(np.sqrt(render_data['shs'].shape[1]) - 1)
        if self.cfg.gs.use_shs:
            cam = setup_camera(w, h, k, w2c, self.metadata['near'], self.metadata['far'], bg, 
                    z_threshold=0.05, sh_degree=max_sh_degrees, device=self.device)
            im, _, depth, = GaussianRasterizer(raster_settings=cam)(**render_data)
        else:
            cam = setup_camera(w, h, k, w2c, self.metadata['near'], self.metadata['far'], bg, 
                    z_threshold=0.05, sh_degree=0, device=self.device)
            shs = render_data['shs'].clone()
            render_data['shs'] = shs[:, 0:1].clone()
            im, _, depth, = GaussianRasterizer(raster_settings=cam)(**render_data)
            render_data['shs'] = shs
        im = torch.clamp(im, 0., 1.)
        return im, depth

    @torch.no_grad
    def render_wrist(self, render_data=None, bg=[0.0, 0.0, 0.0], camera=None):
        assert self.metadata_wrist != {}
        if render_data is None:
            assert self.rendervar_full != {}
            render_data = self.rendervar_full
        render_data = {k: v.to(self.device) for k, v in render_data.items()}

        if camera is not None:
            w, h, k, eef2c = camera
        else:
            w, h = self.metadata_wrist['w'], self.metadata_wrist['h']
            k, eef2c = self.metadata_wrist['k'], self.metadata_wrist['eef2c']
        
        # eef2c to w2c
        eef_xyz = self.grippers[:, :3].clone()
        eef_quat = self.grippers[:, 6:10].clone()
        eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat)  # (n_grippers, 3, 3)

        world_to_base = np.eye(4, dtype=np.float32)

        # only for gripper 0
        eef_xyz_base = eef_xyz[0]  # (3,)
        eef_rot_base = eef_rot[0]  # (3, 3)

        e2b = torch.eye(4, device=self.device)
        e2b[:3, :3] = eef_rot_base
        e2b[:3, 3:4] = eef_xyz_base[:, None]

        eef2b = e2b.cpu().numpy()  # (4, 4)
        b2eef = np.linalg.inv(eef2b)  # (4, 4)
        b2c = eef2c @ b2eef

        w2c = b2c @ world_to_base  # new: from the current robot base frame to the world frame

        max_sh_degrees = int(np.sqrt(render_data['shs'].shape[1]) - 1)
        if self.cfg.gs.use_shs:
            cam = setup_camera(w, h, k, w2c, self.metadata_wrist['near'], self.metadata_wrist['far'], bg, 
                    z_threshold=0.05, sh_degree=max_sh_degrees, device=self.device)
            im, _, depth, = GaussianRasterizer(raster_settings=cam)(**render_data)
        else:
            cam = setup_camera(w, h, k, w2c, self.metadata_wrist['near'], self.metadata_wrist['far'], bg, 
                    z_threshold=0.05, sh_degree=0, device=self.device)
            shs = render_data['shs'].clone()
            render_data['shs'] = shs[:, 0:1].clone()
            im, _, depth, = GaussianRasterizer(raster_settings=cam)(**render_data)
            render_data['shs'] = shs
        im = torch.clamp(im, 0., 1.)
        return im, depth
    
    def render_fixed_cameras(self):
        im_list, depth_list = [], []
        for camera in self.cameras:
            im, depth = self.render(camera=camera)
            im_list.append(im)
            depth_list.append(depth)
        return im_list, depth_list

    def render_wrist_cameras(self):
        im_list, depth_list = [], []
        for camera in self.wrist_cameras:
            im, depth = self.render_wrist(camera=camera)
            im_list.append(im)
            depth_list.append(depth)
        return im_list, depth_list

    @torch.no_grad
    def render_online(self, render_data=None, bg=[0.0, 0.0, 0.0]):
        metadata = self.viser_viewer.get_metadata()

        if render_data is None:
            assert self.rendervar_full != {}
            render_data = copy.deepcopy(self.rendervar_full)

        render_data = {k: v.to(self.device) for k, v in render_data.items()}
        w, h = metadata['w'], metadata['h']
        k, w2c = metadata['k'], metadata['w2c']

        max_sh_degrees = int(np.sqrt(render_data['shs'].shape[1]) - 1)
        if self.cfg.gs.use_shs:
            cam = setup_camera(w, h, k, w2c, self.metadata['near'], self.metadata['far'], bg, 
                    z_threshold=0.05, sh_degree=max_sh_degrees, device=self.device)
            im, _, depth, = GaussianRasterizer(raster_settings=cam)(**render_data)
        else:
            cam = setup_camera(w, h, k, w2c, self.metadata['near'], self.metadata['far'], bg, 
                    z_threshold=0.05, sh_degree=0, device=self.device)
            shs = render_data['shs'].clone()
            render_data['shs'] = shs[:, 0:1].clone()
            render_data['shs'] = torch.clamp(render_data['shs'], -0.5 / C0, 0.5 / C0)
            im, _, depth, = GaussianRasterizer(raster_settings=cam)(**render_data)
            render_data['shs'] = shs

        self.viser_viewer.set_output({'image': (im.permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8)})
        self.viser_viewer.update()
        self.online_end_time = time.perf_counter()
        self.viser_viewer.set_fps(1.0 / (self.online_end_time - self.online_start_time))
        self.online_start_time = self.online_end_time

    def update_phystwin_pts(self, phystwin_pts):
        assert self.state['x'].shape != phystwin_pts.shape
        self.state['x'] = phystwin_pts.clone()

    def update_state(self, state):
        assert self.state['x'].shape == state['x'].shape

        if 'qpos' in state:
            eef_xyz, eef_quat = self.compute_fk(state['qpos'].cpu().numpy())
            eef_gripper = 1 - state['qpos'][:, -1:]

            eef_quat_prev = self.grippers[:, 6:10].clone()  # before update
            eef_rot_prev = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat_prev.reshape(-1, 4))  # (n_grippers, 3, 3)
            eef_rot_this = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat.reshape(-1, 4))  # (n_grippers, 3, 3)
            eef_rot_delta = eef_rot_prev.bmm(eef_rot_this.inverse())
            eef_aa = kornia.geometry.conversions.rotation_matrix_to_axis_angle(eef_rot_delta)  # (n_grippers, 3)
            eef_quat_vel = torch.zeros((self.cfg.env.robot.n_grippers, 3), dtype=eef_quat.dtype, device=self.device)
            eef_quat_vel = eef_aa * self.cfg.physics.fps  # (n_grippers, 3), radian per second

            eef_xyz_prev = self.grippers[:, :3].clone()  # before update
            eef_xyz_this = eef_xyz.clone()  # after update
            eef_xyz_delta = eef_xyz_this - eef_xyz_prev  # (n_grippers, 3)
            eef_vel = eef_xyz_delta * self.cfg.physics.fps  # (n_grippers, 3), meter per second

            self.set_eef(eef_xyz, eef_quat, eef_gripper, eef_vel=eef_vel, eef_quat_vel=eef_quat_vel)
            if 'current_openness' in state:
                self.grippers[:, 13:] = state['current_openness'].reshape(-1, 1).clone()
            self.update_rendervar(state['x'].clone(), qpos_now=state['qpos'].clone() if 'qpos' in state else None)

        else:
            self.grippers[:, :3] = state['eef_xyz'].clone()
            if 'eef_vel' in state:
                self.grippers[:, 3:6] = state['eef_vel'].clone()
            if state['eef_quat'] is not None:
                self.grippers[:, 6:10] = state['eef_quat'].clone()
                if 'eef_quat_vel' in state:
                    self.grippers[:, 10:13] = state['eef_quat_vel'].clone()
                self.grippers[:, 13:] = state['eef_gripper'].clone()
                if 'current_openness' in state:
                    self.grippers[:, 13:] = state['current_openness'].reshape(-1, 1).clone()
            else:
                self.grippers[:, 7:] = state['eef_gripper'].clone()
                if 'current_openness' in state:
                    self.grippers[:, 7:] = state['current_openness'].reshape(-1, 1).clone()
            self.update_rendervar(state['x'].clone())

        self.state['x'] = state['x'].clone()
        self.state['v'] = state['v'].clone()

    def compute_fk(self, joint_commands):
        eef_xyz = []
        eef_rot = []
        assert self.kin_helper is not None
        for i in range(joint_commands.shape[0]):
            e2b = self.kin_helper.compute_fk_sapien_links(joint_commands[i][:7], [self.kin_helper.sapien_eef_idx])[0]  # (4, 4)
            eef_xyz_base = e2b[:3, 3]  # (3,)
            eef_rot_base = e2b[:3, :3]  # (3, 3)
            eef_xyz.append(eef_xyz_base)
            eef_rot.append(eef_rot_base)
        
        eef_xyz = torch.from_numpy(np.array(eef_xyz)).to(torch.float32).to(self.device)
        eef_rot = torch.from_numpy(np.array(eef_rot)).to(torch.float32).to(self.device)
        eef_quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(eef_rot)
        return eef_xyz, eef_quat

    def mimic_velocity_control(self, action):
        # the xarm in the paper's experiments is controlled with a position command -> joint velocity transformation
        # to ensure smoothness, which we call velocity control. This function mimics this process.

        assert action.shape == (1, 13),  "the function only supports single gripper for now"
        target_xyz = action[0, 0:3]
        target_rot = action[0, 3:12].reshape(3, 3)
        target_gripper = action[0, 12].item()

        target_e2b = torch.eye(4, device=target_xyz.device)
        target_e2b[:3, :3] = target_rot
        target_e2b[:3, 3:4] = target_xyz[:, None]

        fk_trans_mat = target_e2b.cpu().numpy()
        target_xyzrpy = np.zeros(6)
        target_xyzrpy[:3] = fk_trans_mat[:3, 3]
        target_xyzrpy[3:] = transforms3d.euler.mat2euler(fk_trans_mat[:3, :3])
        
        assert self.kin_helper is not None
        qpos = self.kin_helper.compute_ik_sapien(self.qpos_curr_xarm, target_xyzrpy)

        delta = qpos - self.qpos_curr_xarm
        joint_delta_norm = np.linalg.norm(delta)

        max_delta_norm = 0.10
        if joint_delta_norm > max_delta_norm:
            delta = delta / joint_delta_norm * max_delta_norm

        dt = 1. / 30  # hardcoded: the frame rate we use on the actual robot
        COMMAND_CHECK_INTERVAL = 0.02
        v = delta / COMMAND_CHECK_INTERVAL * 0.15
        delta_qpos = v * dt

        new_qpos = self.qpos_curr_xarm + delta_qpos
        fk_trans_mat = self.kin_helper.compute_fk_sapien_links(new_qpos, [self.kin_helper.sapien_eef_idx])[0]
        fk_trans_mat = torch.from_numpy(fk_trans_mat).to(target_xyz.device).to(torch.float32)
        target_xyz_new = fk_trans_mat[:3, 3]
        target_rot_new = fk_trans_mat[:3, :3]

        action[0, 0:3] = target_xyz_new.reshape(1, 3)
        action[0, 3:12] = target_rot_new.reshape(1, 9)

        # gripper speed limiting
        current_gripper = self.gripper_openness_curr_xarm / 800.0
        delta_gripper = target_gripper - current_gripper
        if delta_gripper > 0:
            delta_gripper = min(delta_gripper, 2. / 30.)  # gripper max speed empirically measured
        else:
            delta_gripper = max(delta_gripper, -2. / 30.)
        action[0, 12] = delta_gripper + current_gripper

        return action
