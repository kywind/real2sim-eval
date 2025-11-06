import torch
import numpy as np
import transforms3d
import kornia
import scipy
import scipy.interpolate
import sapien.core as sapien

from .robot_pc_sampler import RobotPcSampler


def transform_gs_xarm_gripper(qpos, gripper_openness, params, total_mask, 
        init_qpos=[0, -45, 0, 30, 0, 75, 0], init_gripper=750, sample_robot=None):
    if sample_robot is None:
        engine = sapien.Engine()
        scene = engine.create_scene()
        loader = scene.create_urdf_loader()
        sample_robot = RobotPcSampler("assets/robots/xarm/xarm7_with_gripper.urdf", 
                                                sapien_env_tuple=(engine, scene, loader))
    
    if 'unnorm_rotations' in params:
        rotation_name = 'unnorm_rotations'
    elif 'rotations' in params:
        rotation_name = 'rotations'
    else:
        raise RuntimeError('rotation name not found in params')

    scan_points = params['means3D']
    scan_quats = torch.nn.functional.normalize(params[rotation_name], dim=-1)  # （N, 4）

    links = sample_robot.sapien_robot.get_links()
    assert len(links) == 18
    link_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
    # 1: link_base, 2-8: link1-link7, 9: link_eef, 10-16: fingers (0: world, 17: link_tcp)

    init_qpos = np.array(init_qpos) * np.pi / 180
    g = 800 - init_gripper
    base_qpos = np.array(list(init_qpos) + [g*0.001] * 6)
    
    g_curr = 800 - gripper_openness
    qpos = np.array(list(qpos) + [g_curr*0.001] * 6)

    points_links = {links[i].name: scan_points[total_mask == i] for i in link_id_list}
    quats_links = {links[i].name: scan_quats[total_mask == i] for i in link_id_list}
    new_points_links, new_quats_links = sample_robot.transform_gs_torch(points_links, quats_links, qpos, base_qpos=base_qpos)

    n_point_count = 0
    for i in link_id_list:
        scan_points[total_mask == i] = new_points_links[n_point_count:n_point_count + len(points_links[links[i].name])]
        scan_quats[total_mask == i] = new_quats_links[n_point_count:n_point_count + len(points_links[links[i].name])]
        n_point_count += len(points_links[links[i].name])

    params['means3D'] = scan_points
    params[rotation_name] = scan_quats
    return params


def transform_eef_pts_xarm_gripper(robot, qpos, gripper_openness, device, 
        init_qpos=[0, -45, 0, 30, 0, 75, 0], init_gripper=750, sample_robot=None):
    if sample_robot is None:
        engine = sapien.Engine()
        scene = engine.create_scene()
        loader = scene.create_urdf_loader()
        sample_robot = RobotPcSampler("assets/robots/xarm/xarm7_with_gripper.urdf", 
                                                sapien_env_tuple=(engine, scene, loader))

    meshes = robot.get_xarm_gripper_meshes(gripper_openness=init_gripper / 800.0)
    scan_points = np.concatenate([np.asarray(m.vertices) for m in meshes], axis=0)
    scan_points = torch.from_numpy(scan_points).to(torch.float32).to(device)

    init_qpos = np.array(init_qpos) * np.pi / 180
    g = 800 - init_gripper
    base_qpos = np.array(list(init_qpos) + [g*0.001] * 6)
    
    g_curr = 800 - gripper_openness
    qpos = np.array(list(qpos) + [g_curr*0.001] * 6)

    points_links_gripper = {
        'left_finger': scan_points[:len(scan_points) // 2],
        'right_finger': scan_points[len(scan_points) // 2:],
    }
    new_points_links_gripper, new_quats_links_gripper = robot.transform_gs_torch(points_links_gripper, None, qpos, base_qpos=base_qpos)

    scan_points[:len(scan_points) // 2] = new_points_links_gripper[:len(points_links_gripper['left_finger'])]
    scan_points[len(scan_points) // 2:] = new_points_links_gripper[len(points_links_gripper['left_finger']):len(points_links_gripper['left_finger']) + len(points_links_gripper['right_finger'])]

    return scan_points


def transform_gs_xarm_pusher(qpos, params, total_mask, init_qpos=[0, -45, 0, 30, 0, 75, 0], sample_robot=None):
    if sample_robot is None:
        engine = sapien.Engine()
        scene = engine.create_scene()
        loader = scene.create_urdf_loader()
        sample_robot = RobotPcSampler("assets/robots/xarm/xarm7_with_pusher.urdf", 
                                                sapien_env_tuple=(engine, scene, loader))
    
    if 'unnorm_rotations' in params:
        rotation_name = 'unnorm_rotations'
    elif 'rotations' in params:
        rotation_name = 'rotations'
    else:
        raise RuntimeError('rotation name not found in params')

    ### transform ###

    scan_points = params['means3D']
    scan_quats = torch.nn.functional.normalize(params[rotation_name], dim=-1)  # （N, 4）

    links = sample_robot.sapien_robot.get_links()
    assert len(links) == 11
    link_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 10]
    # 1: link_base, 2-8: link1-link7, 9: link_eef, 10: pusher

    base_qpos = np.array(init_qpos) * np.pi / 180
    qpos = np.array([qpos[0], qpos[1], qpos[2], qpos[3], qpos[4], qpos[5], qpos[6]])

    points_links = {links[i].name: scan_points[total_mask == i] for i in link_id_list}
    quats_links = {links[i].name: scan_quats[total_mask == i] for i in link_id_list}
    new_points_links, new_quats_links = sample_robot.transform_gs_torch(points_links, quats_links, qpos, base_qpos=base_qpos)

    n_point_count = 0
    for i in link_id_list:
        scan_points[total_mask == i] = new_points_links[n_point_count:n_point_count + len(points_links[links[i].name])]
        scan_quats[total_mask == i] = new_quats_links[n_point_count:n_point_count + len(points_links[links[i].name])]
        n_point_count += len(points_links[links[i].name])

    ### end ###

    params['means3D'] = scan_points
    params[rotation_name] = scan_quats
    return params


def transform_eef_pts_xarm_pusher(robot, qpos, device, init_qpos=[0, -45, 0, 30, 0, 75, 0], sample_robot=None):
    if sample_robot is None:
        engine = sapien.Engine()
        scene = engine.create_scene()
        loader = scene.create_urdf_loader()
        sample_robot = RobotPcSampler("assets/robots/xarm/xarm7_with_pusher.urdf", 
                                                sapien_env_tuple=(engine, scene, loader))

    meshes = robot.get_xarm_pusher_meshes()
    scan_points = np.concatenate([np.asarray(m.vertices) for m in meshes], axis=0)
    scan_points = torch.from_numpy(scan_points).to(torch.float32).to(device)

    base_qpos = np.array(init_qpos) * np.pi / 180

    qpos = np.array([qpos[0], qpos[1], qpos[2], qpos[3], qpos[4], qpos[5], qpos[6]])

    points_links_gripper = {
        'pusher_base_link': scan_points,
    }
    new_points_links_gripper, new_quats_links_gripper = robot.transform_gs_torch(points_links_gripper, None, qpos, base_qpos=base_qpos)
    return new_points_links_gripper


def get_eef_pts_xarm_gripper(eef_xyz, eef_quat, eef_gripper, robot, sample_robot, kin_helper, qpos_curr_xarm, device):
    assert eef_xyz.shape[0] == 3
    assert eef_quat.shape[0] == 4
    assert qpos_curr_xarm.shape[0] == 7

    eef_points = torch.tensor([[0., 0., 0.]], device=device)  # the eef point in the gripper frame

    eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat[None])[0]  # (3, 3)

    e2b = torch.eye(4, device=device)
    e2b[:3, :3] = eef_rot
    e2b[:3, 3:4] = eef_xyz[:, None] - eef_rot @ eef_points.T

    fk_trans_mat = e2b.cpu().numpy()

    cur_xyzrpy = np.zeros(6)
    cur_xyzrpy[:3] = fk_trans_mat[:3, 3]
    cur_xyzrpy[3:] = transforms3d.euler.mat2euler(fk_trans_mat[:3, :3])
    
    assert kin_helper is not None
    qpos = kin_helper.compute_ik_sapien(qpos_curr_xarm, cur_xyzrpy)

    gripper_openness_temp = eef_gripper.item() * 800.0
    eef_pts = transform_eef_pts_xarm_gripper(robot, qpos, gripper_openness_temp, device=device, sample_robot=sample_robot)  # (n, 3)
    eef_pts = eef_pts.clone()

    eef_pts_list = []
    for gi in range(101):
        gripper_openness_temp = gi * 0.01 * 800.0
        eef_pts = transform_eef_pts_xarm_gripper(robot, qpos, gripper_openness_temp, device=device, sample_robot=sample_robot)  # (n, 3)
        eef_pts_list.append(eef_pts.cpu().numpy())
    eef_pts_list = np.stack(eef_pts_list, axis=0)  # (100, n, 3)
    eef_pts_func = scipy.interpolate.interp1d(np.arange(101) / 100.0, eef_pts_list, axis=0)

    return eef_pts, eef_pts_func


def get_eef_pts_xarm_pusher(eef_xyz, eef_quat, robot, sample_robot, kin_helper, qpos_curr_xarm, device):
    assert eef_xyz.shape[0] == 3
    assert eef_quat.shape[0] == 4
    assert qpos_curr_xarm.shape[0] == 7

    eef_points = torch.tensor([[0., 0., 0.]], device=device)  # the eef point in the gripper frame

    eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat[None])[0]  # (3, 3)

    e2b = torch.eye(4, device=device)
    e2b[:3, :3] = eef_rot
    e2b[:3, 3:4] = eef_xyz[:, None] - eef_rot @ eef_points.T

    fk_trans_mat = e2b.cpu().numpy()

    cur_xyzrpy = np.zeros(6)
    cur_xyzrpy[:3] = fk_trans_mat[:3, 3]
    cur_xyzrpy[3:] = transforms3d.euler.mat2euler(fk_trans_mat[:3, :3])
    
    assert kin_helper is not None
    qpos = kin_helper.compute_ik_sapien(qpos_curr_xarm, cur_xyzrpy)

    eef_pts = transform_eef_pts_xarm_pusher(robot, qpos, device=device, sample_robot=sample_robot)  # (n, 3)
    eef_pts = eef_pts.clone()

    eef_pts_list = []
    for gi in range(101):
        eef_pts = transform_eef_pts_xarm_pusher(robot, qpos, device=device, sample_robot=sample_robot)  # (n, 3)
        eef_pts_list.append(eef_pts.cpu().numpy())
    eef_pts_list = np.stack(eef_pts_list, axis=0)  # (100, n, 3)
    eef_pts_func = scipy.interpolate.interp1d(np.arange(101) / 100.0, eef_pts_list, axis=0)

    return eef_pts, eef_pts_func
