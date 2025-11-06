import numpy as np
from typing import Callable
import torch
import warp as wp
import kornia
import open3d as o3d
import os
import pickle as pkl
import glob
from typing import Optional
import copy

from .spring_mass_warp import SpringMassSystemWarp
from ..utils.robot.kinematics_utils import KinHelper
from ..utils.robot.robot_pc_sampler import RobotPcSampler


class PhysTwinDynamics:

    def __init__(
        self,
        cfg,
        exp_root,
        ckpt_path,
        case_name,
        local_rank: int = 0,
    ):
        wp.init()
        wp.ScopedTimer.enabled = False
        wp.set_module_options({'fast_math': False})

        self.cfg = cfg
        self.exp_root = exp_root
        self.ckpt_path = ckpt_path
        self.case_name = case_name

        self.device = f'cuda:{local_rank}'

    def reset(
        self, 
        state,
        init_meshes_dict: Optional[dict] = None,
        robot: Optional[object] = None,
        eef_pts_func: Optional[Callable] = None,
        kin_helper: Optional[KinHelper] = None,
        init_eef_xyz: Optional[torch.Tensor] = None,
        pose_obj: Optional[torch.Tensor] = None
    ):
        pts = state['x']  # (n_particles, 3)
        if isinstance(pts, torch.Tensor):
            p_x = pts.to(torch.float32).to(self.device)
        else:
            p_x = torch.tensor(pts).to(torch.float32).to(self.device)

        global_translation = torch.tensor([0.0, 0.0, -self.cfg.physics.table_height], dtype=p_x.dtype, device=p_x.device)
        self.reset_metadata = {
            'global_translation': global_translation,
        }

        if init_meshes_dict is not None:
            init_meshes = []
            for key, mesh in init_meshes_dict.items():
                mesh_new = copy.deepcopy(mesh)
                vertices = np.asarray(mesh_new.vertices)
                vertices = torch.tensor(vertices, dtype=torch.float32).to(self.device)
                vertices = vertices + global_translation
                mesh_new.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
                init_meshes.append(mesh_new)
        else:
            init_meshes = None
        self.init_meshes = init_meshes

        # robot pc sampler
        self.robot = robot
        self.kin_helper = kin_helper
        
        # robot gripper pts interpolation function
        self.eef_pts_func = eef_pts_func
        self.init_eef_xyz = init_eef_xyz

        # reset model
        init_pts = p_x.clone()
        init_pts = init_pts + global_translation  # data frame to model frame
        self.dynamics_module = SpringMassDynamicsModule(
            phystwin_cfg=self.cfg.physics,
            device=self.device,
            wp_device=self.device,
            case_name=self.case_name,
            data_path=f"{self.ckpt_path}/data",
            zeroth_order_ckpt_path=f"{self.ckpt_path}/experiments_optimization",
            first_order_ckpt_path=f"{self.ckpt_path}/experiments",
            init_pts=init_pts,
            init_pose=pose_obj,
            static_meshes=self.init_meshes,
            robot=robot,
            robot_type=self.cfg.env['robot']['type'],
            use_pusher=self.cfg.env['robot']['use_pusher'],
        )
        assert self.dynamics_module.init_pts_aligned is not None
        init_pts_aligned = self.dynamics_module.init_pts_aligned.clone()
        init_pts_aligned = init_pts_aligned - global_translation
        return init_pts_aligned

    def step(self, state, action):
        assert self.dynamics_module is not None
        cfg = self.cfg

        eef_xyz = state['eef_xyz']  # (n_grippers, 3)
        eef_quat = state['eef_quat']  # (n_grippers, 4)
        eef_gripper = state['eef_gripper']  # (n_grippers, 1)
        eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat)  # (n_grippers, 3, 3)

        if action.shape[-1] == 13:
            mode = 'xyz_rot'
            eef_xyz_next = action[..., :3]  # (n_grippers, 3)
            eef_rot_next = action[..., 3:12].reshape(-1, 3, 3)  # (n_grippers, 3, 3)
            eef_gripper_next = action[..., 12:]  # (n_grippers, 1)
            eef_quat_next = kornia.geometry.conversions.rotation_matrix_to_quaternion(eef_rot_next)  # (n_grippers, 4)

        elif action.shape[-1] == 8:  # joint mode
            mode = 'joint'
            eef_xyz_next, eef_quat_next = self.compute_fk(action[:, :-1].cpu().numpy())  # (n_gripper, 7)
            eef_gripper_next = 1 - action[:, -1:]  # (n_gripper, 1)
            eef_rot_next = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat_next)  # (n_grippers, 3, 3)

        else:
            raise NotImplementedError

        global_translation = self.reset_metadata['global_translation']
        eef_xyz += global_translation
        eef_xyz_next += global_translation

        eef_vel = torch.zeros_like(eef_xyz)  # (n_grippers, 3)
        eef_vel = (eef_xyz_next - eef_xyz) * cfg.physics.fps

        eef_rot_delta = eef_rot.bmm(eef_rot_next.inverse())
        eef_aa = kornia.geometry.conversions.rotation_matrix_to_axis_angle(eef_rot_delta)  # (n_grippers, 3)
        eef_rot_vel = eef_aa.reshape(-1, 3) * cfg.physics.fps  # (n_gripper, 3)

        x0 = self.dynamics_module.current_points
        x_pred = self.dynamics_module.step(
            eef_xyz=eef_xyz,
            eef_vel=eef_vel,
            eef_rot=eef_rot,
            eef_rot_vel=eef_rot_vel,
            gripper_openness=eef_gripper_next,
            eef_pts_func=self.eef_pts_func,
            init_eef_xyz=self.init_eef_xyz,
        )
        v_pred = (x_pred - x0) * cfg.physics.fps

        # inverse preprocess
        x_pred = x_pred - global_translation
        eef_xyz = eef_xyz - global_translation
        eef_xyz_next = eef_xyz_next - global_translation

        current_openness = torch.tensor([self.dynamics_module.current_openness], dtype=torch.float32).to(self.device)

        next_state = {
            'current_openness': current_openness,
            'x': x_pred,
            'v': v_pred,
            'eef_xyz': eef_xyz_next,
            'eef_vel': eef_vel,
            'eef_quat': eef_quat_next,
            'eef_quat_vel': eef_rot_vel,
            'eef_gripper': eef_gripper_next,
        }
        if mode == 'joint':
            next_state['qpos'] = action  # (n_grippers, 8)
        return next_state
    
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

    def get_state(self):
        init_springs = self.dynamics_module.init_springs.clone()
        init_rest_lengths = self.dynamics_module.init_rest_lengths.clone()
        init_spring_Y = self.dynamics_module.init_spring_Y.clone()
        static_meshes = [{
            'vertices': np.asarray(mesh.vertices),
            'faces': np.asarray(mesh.triangles),
        } for mesh in self.init_meshes] if self.init_meshes is not None else []
        state = {
            'init_springs': init_springs,
            'init_rest_lengths': init_rest_lengths,
            'init_spring_Y': init_spring_Y,
            'static_meshes': static_meshes,
        }
        return state


class SpringMassDynamicsModule:
    def __init__(
        self,
        phystwin_cfg,
        device,
        wp_device,
        case_name,
        data_path,
        zeroth_order_ckpt_path,
        first_order_ckpt_path,
        init_pts,  # (n_points, 3)
        init_pose,
        static_meshes,
        robot,
        robot_type,
        use_pusher,
    ):
        # set the parameters
        phystwin_cfg.num_substeps = round(1.0 / phystwin_cfg.fps / phystwin_cfg.dt)
        self.device = device
        self.wp_device = wp_device
        self.phystwin_cfg = phystwin_cfg
        self.robot_type = robot_type
        self.use_pusher = use_pusher

        # load object points
        with open(f"{data_path}/{case_name}/final_data.pkl", "rb") as f:
            data = pkl.load(f)
        object_points = data["object_points"]  # (T, N, 3)
        object_colors = data["object_colors"]  # (T, N, 3)
        other_surface_points = data["surface_points"]  # (n_surface, 3)
        interior_points = data["interior_points"]  # (n_interior, 3)

        # align input points with object points
        self.init_pts = init_pts.to(torch.float32).to(self.device)
        object_pts = np.concatenate([object_points[0], other_surface_points, interior_points], axis=0)
        init_pose_np = init_pose.cpu().numpy()
        init_pts_aligned = object_pts @ init_pose_np[:3, :3].T + init_pose_np[:3, 3]

        # load the zeroth order checkpoint
        optimal_path = f"{zeroth_order_ckpt_path}/{case_name}/optimal_params.pkl"
        assert os.path.exists(optimal_path), f"{case_name}: Optimal parameters not found: {optimal_path}"
        with open(optimal_path, "rb") as f:
            optimal_params = pkl.load(f)
        optimal_params["init_spring_Y"] = optimal_params.pop("global_spring_Y")
        if "collide_object_elas" in optimal_params:
            optimal_params["collide_self_elas"] = optimal_params.pop("collide_object_elas")
        if "collide_object_fric" in optimal_params:
            optimal_params["collide_self_fric"] = optimal_params.pop("collide_object_fric")
        for key, value in optimal_params.items():
            assert hasattr(phystwin_cfg, key)
            current_value = getattr(phystwin_cfg, key)
            if isinstance(current_value, int):
                value = int(value)
            elif isinstance(current_value, float):
                value = float(value)
            setattr(phystwin_cfg, key, value)

        # connect springs among aligned object points
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_pts)
        pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)
        spring_flags = np.zeros((len(object_pts), len(object_pts)))
        init_springs = []
        for i in range(len(object_pts)):
            [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                object_pts[i], phystwin_cfg.object_radius, phystwin_cfg.object_max_neighbours
            )
            idx = idx[1:]
            for j in idx:
                rest_length = np.linalg.norm(init_pts_aligned[i] - init_pts_aligned[j])
                if (
                    spring_flags[i, j] == 0
                    and spring_flags[j, i] == 0
                    and rest_length > 1e-4
                ):
                    spring_flags[i, j] = 1
                    spring_flags[j, i] = 1
                    init_springs.append([i, j])
        init_springs = torch.tensor(init_springs, dtype=torch.int32, device=self.device)  # (n_springs, 2)
        init_pts_aligned = torch.tensor(init_pts_aligned, dtype=torch.float32, device=self.device)
        init_rest_lengths = torch.linalg.norm(init_pts_aligned[init_springs[:, 0]] - init_pts_aligned[init_springs[:, 1]], dim=1)
        
        # load the first order checkpoint
        best_model_path = glob.glob(f"{first_order_ckpt_path}/{case_name}/train/best_*.pth")[0]
        checkpoint = torch.load(best_model_path, map_location=self.device)
        init_spring_Y = checkpoint["spring_Y"]  # (n_springs_total,)
        collide_elas = checkpoint["collide_elas"]  # (1,)
        collide_fric = checkpoint["collide_fric"]  # (1,)
        collide_self_elas = checkpoint["collide_object_elas"]  # (1,)
        collide_self_fric = checkpoint["collide_object_fric"]  # (1,)
        num_object_springs = checkpoint["num_object_springs"]  # (n_springs_object,)
        assert init_springs.shape[0] == num_object_springs
        init_spring_Y = init_spring_Y[:num_object_springs]  # throw away the object-control springs

        self.collide_elas = collide_elas
        self.collide_fric = collide_fric
        self.collide_self_elas = collide_self_elas
        self.collide_self_fric = collide_self_fric

        if use_pusher:  # hardcoded: we use smaller eef frictions for pusher
            phystwin_cfg.collide_eef_fric = 0.2

        collide_eef_elas = torch.tensor([phystwin_cfg.collide_eef_elas], dtype=torch.float32, device=self.device)
        collide_eef_fric = torch.tensor([phystwin_cfg.collide_eef_fric], dtype=torch.float32, device=self.device)

        # store object-only parameters
        self.init_pts_aligned = init_pts_aligned
        self.init_springs = init_springs.clone()
        self.init_rest_lengths = init_rest_lengths.clone()
        self.init_spring_Y = init_spring_Y.clone()

        # load robot mesh
        if robot is not None:
            assert isinstance(robot, RobotPcSampler)
            if self.use_pusher:
                dynamic_meshes = robot.get_xarm_pusher_meshes()
            else:
                dynamic_meshes = robot.get_xarm_gripper_meshes(gripper_openness=1.0)
        else:
            dynamic_meshes = []
        assert isinstance(dynamic_meshes, list) and isinstance(static_meshes, list)
        dynamic_vertices = np.concatenate([
            np.asarray(mesh.vertices) for mesh in dynamic_meshes
        ], axis=0)
        dynamic_vertices = torch.tensor(
            dynamic_vertices, dtype=torch.float32, device=self.device
        )

        # initialize simulator with no control points
        init_masses = torch.ones(self.init_pts_aligned.shape[0], dtype=torch.float32, device=self.device)
        self.simulator = SpringMassSystemWarp(
            phystwin_cfg=phystwin_cfg,
            device=self.wp_device,
            init_vertices=self.init_pts_aligned,
            init_springs=self.init_springs,
            init_rest_lengths=self.init_rest_lengths,
            init_masses=init_masses,
            num_object_points=self.init_pts_aligned.shape[0],
            init_spring_Y=torch.log(self.init_spring_Y).detach().clone(),
            collide_elas=collide_elas.detach().clone(),
            collide_fric=collide_fric.detach().clone(),
            collide_eef_elas=collide_eef_elas.detach().clone(),
            collide_eef_fric=collide_eef_fric.detach().clone(),
            collide_self_elas=collide_self_elas.detach().clone(),
            collide_self_fric=collide_self_fric.detach().clone(),
            init_collision_mask=None,
            init_velocities=None,
            dynamic_meshes=dynamic_meshes,
            static_meshes=static_meshes,
            dynamic_points=dynamic_vertices,
            use_pusher=self.use_pusher,
        )
        self.current_openness = None
        self.grasped = False


    def step(self, eef_xyz, eef_vel, eef_rot, eef_rot_vel, gripper_openness, eef_pts_func, init_eef_xyz):
        n_grippers, _ = eef_xyz.shape

        if self.phystwin_cfg.self_collision:
            self.simulator.update_collision_graph()

        if not self.use_pusher:
            # read gripper openness
            openness = gripper_openness.item()
            if self.current_openness is None:
                self.current_openness = openness

            n_substeps = self.phystwin_cfg.num_substeps
            dts = torch.linspace(1, n_substeps, n_substeps, device=self.device) * self.phystwin_cfg.dt

            eef_xyz_next = eef_xyz[None] + eef_vel[None] * dts[:, None, None]  # (n_substeps, n_grippers, 3)
            eef_aa_delta = eef_rot_vel[None] * dts[:, None, None]  # (n_substeps, n_grippers, 3)
            eef_rot_delta = kornia.geometry.conversions.axis_angle_to_rotation_matrix(eef_aa_delta.reshape(-1, 3)).reshape(n_substeps, n_grippers, 3, 3)
            eef_rot_next = eef_rot_delta.permute(0, 1, 3, 2) @ eef_rot

            # calculate force
            mesh_map = self.simulator.mesh_map.numpy()
            gripper_left_face_mask = mesh_map == 0  # first two meshes are the gripper meshes
            gripper_right_face_mask = mesh_map == 1
            force = self.simulator.collision_forces.numpy()  # first two meshes are the gripper meshes
            gripper_left_force = force[gripper_left_face_mask]
            gripper_right_force = force[gripper_right_face_mask]

            gripper_left_force_filtered = gripper_left_force[18] + gripper_left_force[19] + gripper_left_force[1]
            gripper_right_force_filtered = gripper_right_force[18] + gripper_right_force[19] + gripper_right_force[1]

            force_filtered = np.stack([gripper_left_force_filtered, gripper_right_force_filtered], axis=0)  # (2, 3)
            force_filtered_norm = np.linalg.norm(force_filtered, axis=1)
            force_threshold = self.phystwin_cfg.grasp_force_threshold

            openness_before = self.current_openness
            if np.all(force_filtered_norm < 100):  # both fingers forces are small
                self.grasped = False  # release grasp
            if openness < self.current_openness:
                if np.all(force_filtered_norm > force_threshold):  # both fingers forces are large
                    assert self.current_openness is not None
                    openness = self.current_openness
                    self.grasped = True  # establish grasp
                elif self.grasped:
                    self.current_openness = max(openness, self.current_openness - 0.05)
                    openness = self.current_openness
                else:
                    self.current_openness = openness
            else:
                self.current_openness = openness
            assert self.current_openness == openness
            # print("Final openness:", openness)

            openness = np.clip(openness, 0.0, 1.0)
            eef_pts = eef_pts_func(openness)
            eef_pts = torch.from_numpy(eef_pts).to(torch.float32).to(self.device)

            openness_before = np.clip(openness_before, 0.0, 1.0)
            eef_pts_before = eef_pts_func(openness_before)
            eef_pts_before = torch.from_numpy(eef_pts_before).to(torch.float32).to(self.device)
            eef_pts_delta = eef_pts - eef_pts_before  # (n_points, 3)
            eef_pts_delta[:, 1] *= -1
            eef_pts_delta[:, 2] *= -1  # flip y, z
            relative_eef_pts = eef_pts_before - init_eef_xyz
            relative_eef_pts[:, 1] *= -1
            relative_eef_pts[:, 2] *= -1  # flip y, z
            relative_eef_pts = relative_eef_pts[None, None, :, :]  # (1, 1, n_points, 3)
            relative_eef_pts = relative_eef_pts + eef_pts_delta[None, None] / (self.phystwin_cfg.dt * n_substeps) * dts[:, None, None, None]  # (num_substeps, 1, n_points, 3)

            # (num_substeps, n_points, 3)
            interpolated_dynamic_points = eef_xyz_next[:, :, None] + relative_eef_pts @ eef_rot_next.permute(0, 1, 3, 2)
            interpolated_dynamic_points = interpolated_dynamic_points[:, 0]  # take the first gripper

            # (num_substeps, 3)
            interpolated_center = eef_xyz_next[:, 0]

            # (3,)
            dynamic_velocity = eef_vel[0] * 0.5

            eef_pts_delta = eef_pts_delta @ eef_rot[0].permute(1, 0)  # (n_points, 3)
            closing_velocity = eef_pts_delta / (2 * self.phystwin_cfg.dt * n_substeps)  # average velocity
            left_closing_velocity = closing_velocity[:len(closing_velocity) // 2]
            right_closing_velocity = closing_velocity[len(closing_velocity) // 2:]
            left_closing_velocity = left_closing_velocity.mean(0)
            right_closing_velocity = right_closing_velocity.mean(0)
            closing_velocity = torch.stack([left_closing_velocity, right_closing_velocity], dim=0)  # (2, 3)
            dynamic_velocity = dynamic_velocity + closing_velocity  # add the gripper closing velocity

            # (3,)
            dynamic_omega = -eef_rot_vel[0] * 0.5
            dynamic_omega = dynamic_omega[None]  # (1, 3)

            # Update the simulator with the gripper changes
            self.simulator.set_mesh_interactive(
                interpolated_dynamic_points,
                interpolated_center,
                dynamic_velocity,
                dynamic_omega,
            )
        
        elif self.use_pusher:
            n_substeps = self.phystwin_cfg.num_substeps
            dts = torch.linspace(1, n_substeps, n_substeps, device=self.device) * self.phystwin_cfg.dt

            self.current_openness = 1.0  # just for placeholding

            eef_xyz_next = eef_xyz[None] + eef_vel[None] * dts[:, None, None]  # (n_substeps, n_grippers, 3)
            eef_aa_delta = eef_rot_vel[None] * dts[:, None, None]  # (n_substeps, n_grippers, 3)
            eef_rot_delta = kornia.geometry.conversions.axis_angle_to_rotation_matrix(eef_aa_delta.reshape(-1, 3)).reshape(n_substeps, n_grippers, 3, 3)
            eef_rot_next = eef_rot_delta.permute(0, 1, 3, 2) @ eef_rot

            # modified from renderer_phystwin.py
            eef_pts = eef_pts_func(1.0)
            eef_pts = torch.from_numpy(eef_pts).to(torch.float32).to(self.device)

            eef_pts_before = eef_pts_func(1.0)
            eef_pts_before = torch.from_numpy(eef_pts_before).to(torch.float32).to(self.device)
            eef_pts_delta = eef_pts - eef_pts_before  # (n_points, 3)
            eef_pts_delta[:, 1] *= -1
            eef_pts_delta[:, 2] *= -1  # flip y, z

            # multi-steps
            relative_eef_pts = eef_pts_before - init_eef_xyz
            relative_eef_pts[:, 1] *= -1
            relative_eef_pts[:, 2] *= -1  # flip y, z
            relative_eef_pts = relative_eef_pts[None, None, :, :]  # (1, 1, n_points, 3)
            relative_eef_pts = relative_eef_pts + eef_pts_delta[None, None] / (self.phystwin_cfg.dt * n_substeps) * dts[:, None, None, None]  # (num_substeps, 1, n_points, 3)

            # (num_substeps, n_points, 3)
            interpolated_dynamic_points = eef_xyz_next[:, :, None] + relative_eef_pts @ eef_rot_next.permute(0, 1, 3, 2)
            interpolated_dynamic_points = interpolated_dynamic_points[:, 0]  # take the first gripper

            # (num_substeps, 3)
            interpolated_center = eef_xyz_next[:, 0]

            # (3,)
            dynamic_velocity = eef_vel[0] * 0.5
            dynamic_velocity = dynamic_velocity[None]  # (1, 3)

            # (3,)
            dynamic_omega = -eef_rot_vel[0] * 0.5
            dynamic_omega = dynamic_omega[None]  # (1, 3)

            self.simulator.set_mesh_interactive(
                interpolated_dynamic_points,
                interpolated_center,
                dynamic_velocity,
                dynamic_omega
            )
        
        else:
            raise NotImplementedError("Only xarm robot is supported for now.")

        if self.phystwin_cfg.use_graph:
            assert self.simulator.graph is not None
            wp.capture_launch(self.simulator.graph)
        else:
            self.simulator.step()

        return self.current_points

    @property
    def current_points(self):
        x = wp.to_torch(self.simulator.wp_state.wp_x)
        return x

    @property
    def current_velocities(self):
        v = wp.to_torch(self.simulator.wp_state.wp_v)
        return v
