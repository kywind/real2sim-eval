from pathlib import Path
import hydra
from omegaconf import OmegaConf
import os
import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime
import json
import kornia
import glob
import time
import sys
sys.path.append(str(Path(__file__).parents[1]))
OmegaConf.register_new_resolver("eval", eval, replace=True)

from experiments.utils.dir_utils import mkdir
from experiments.utils.ffmpeg import make_video
import sim.envs
from sim.utils.robot.kinematics_utils import KinHelper

kin_helper = KinHelper('xarm7')

# utils for transforming qpos to cartesian
def compute_fk(qpos):                
    eef_xyz = []
    eef_rot = []
    assert kin_helper is not None
    for i in range(qpos.shape[0]):
        e2b = kin_helper.compute_fk_sapien_links(qpos[i][:7], [kin_helper.sapien_eef_idx])[0]  # (4, 4)
        eef_xyz_base = e2b[:3, 3]  # (3,)
        eef_rot_base = e2b[:3, :3]  # (3, 3)
        eef_xyz.append(eef_xyz_base)
        eef_rot.append(eef_rot_base)
    eef_xyz = np.array(eef_xyz).astype(np.float32).reshape(-1, 3)
    eef_rot = np.array(eef_rot).astype(np.float32).reshape(-1, 3, 3)
    return eef_xyz, eef_rot

# utils for loading robot json
def load_robot_json(path, use_qpos=True, prefix='action'):
    with open(path, 'r') as f:
        robot = json.load(f)

    if f'{prefix}.xy' in robot.keys():  # planar pushing
        if use_qpos:
            robot_trans, robot_rot = compute_fk(np.array(robot[f'{prefix}.qpos']).reshape(1, -1))
        else:
            xy = np.array(robot[f'{prefix}.xy']).reshape(-1, 2)  # (1, 2)
            robot_trans = np.zeros((1, 3), dtype=np.float32)
            robot_trans[:, :2] = xy
            robot_trans[:, 2] = 0.22  # fixed height
            robot_rot = np.eye(3, dtype=np.float32)
            robot_rot[1, 1] *= -1
            robot_rot[2, 2] *= -1
            robot_rot = robot_rot[None]  # (1, 3, 3)
        gripper = np.array([1.0], dtype=np.float32).reshape(-1, 1)

    else:  # full 6-DoF
        if use_qpos:
            robot_trans, robot_rot = compute_fk(np.array(robot[f'{prefix}.qpos']).reshape(1, -1))
        else:
            if f'{prefix}.cartesian' in robot:
                e2b = np.array(robot[f'{prefix}.cartesian']).reshape(4, 4)
                robot_rot = e2b[:3, :3][None]  # (1, 3, 3)
                robot_trans = e2b[:3, 3]  # (1, 3)
            else:
                assert f'{prefix}.ee_pos' in robot and f'{prefix}.ee_quat' in robot
                eef_xyz = np.array(robot[f'{prefix}.ee_pos']).reshape(1, 3)  # (1, 3)
                eef_quat = np.array(robot[f'{prefix}.ee_quat']).reshape(1, 4)  # (1, 4) wxyz
                robot_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(
                    torch.from_numpy(eef_quat).to(torch.float32)
                ).numpy()  # (1, 3, 3)
                robot_trans = eef_xyz  # (1, 3)
        gripper = 1.0 - np.array(robot[f'{prefix}.gripper_qpos']).reshape(-1)  # (1,)

    return robot_trans, robot_rot, gripper


@hydra.main(version_base='1.2', config_path='../cfg', config_name="replay")
def main(cfg):
    OmegaConf.resolve(cfg)

    # robot traj to replay
    gt_dir = Path(cfg.gt_dir)
    assert gt_dir.exists(), f"GT directory {cfg.gt_dir} does not exist"

    if (gt_dir / 'episode_0000').exists():  # there are multiple episodes
        use_episodes = True
        n_episodes = len(sorted(glob.glob(str(gt_dir / 'episode_*'))))
    else:  # there is only one episode
        use_episodes = False
        n_episodes = 1

    # unique run name
    if cfg.timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        timestamp = cfg.timestamp
    run_name = f"{timestamp}"

    out_path = str(Path(cfg.exp_root) / 'output_replay')
    mkdir(Path(f'{out_path}/{run_name}'), resume=False, overwrite=False)
    OmegaConf.save(cfg, f'{out_path}/{run_name}/hydra.yaml', resolve=True)

    for episode_id in range(n_episodes):
        os.makedirs(f'{out_path}/{run_name}/episode_{episode_id:04d}', exist_ok=True)
        if use_episodes:
            episode_gt_dir = gt_dir / f'episode_{episode_id:04d}'
        else:
            episode_gt_dir = gt_dir
        if not (episode_gt_dir / "robot").exists():
            print(f"Episode directory {episode_gt_dir} does not exist")
            continue

        robot_paths = sorted(glob.glob(str(episode_gt_dir / "robot" / "*.json")))
        n_frames = len(robot_paths)

        robot_traj_list = []
        robot_rot_list = []
        robot_gripper_list = []
        for frame_id in range(n_frames):
            robot_traj, robot_rot, robot_gripper = load_robot_json(robot_paths[frame_id], use_qpos=cfg.use_qpos)
            robot_traj_list.append(robot_traj)
            robot_rot_list.append(robot_rot)
            robot_gripper_list.append(robot_gripper)

        robot_traj_list = np.stack(robot_traj_list)  # (n, n_grippers, 3)
        robot_rot_list = np.stack(robot_rot_list)  # (n, n_grippers, 3, 3)
        robot_gripper_list = np.stack(robot_gripper_list)  # (n, n_grippers)

        n_steps = len(robot_traj_list)

        frame_rate = cfg.physics.fps
        duration = n_steps // frame_rate  # seconds
        print(f"Replaying {n_steps} steps, duration {duration}s")

        # random reset
        env = gym.make(cfg.env_name, max_episode_steps=frame_rate * duration, cfg=cfg, 
            obs_mode=cfg.obs_mode, exp_root=cfg.exp_root, local_rank=0, randomize=True)
        obs, reset_info = env.reset(seed=episode_id)

        os.makedirs(f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_0/rgb', exist_ok=True)
        os.makedirs(f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_1/rgb', exist_ok=True)
        os.makedirs(f'{out_path}/{run_name}/episode_{episode_id:04d}/calibration', exist_ok=True)
        os.makedirs(f'{out_path}/{run_name}/episode_{episode_id:04d}/robot', exist_ok=True)
        os.makedirs(f'{out_path}/{run_name}/start_images', exist_ok=True)
        os.makedirs(f'{out_path}/{run_name}/final_images', exist_ok=True)

        # save calibration data
        rvecs = []
        tvecs = []
        for cam_id in range(len(cfg.env.cameras)):
            camera = cfg.env.cameras[cam_id]
            if 'c2w' in camera:
                trans_mat = np.array(camera['c2w']).reshape(4, 4).astype(np.float32)
                trans_mat = np.linalg.inv(trans_mat)  # w2c
            else:
                assert 'w2c' in camera
                trans_mat = np.array(camera['w2c']).reshape(4, 4).astype(np.float32)
            rvec = R.from_matrix(trans_mat[:3, :3]).as_rotvec()
            tvec = trans_mat[:3, 3]
            rvecs.append(rvec)
            tvecs.append(tvec)
        rvecs_save_npy = np.stack(rvecs).reshape(-1, 3, 1)
        tvecs_save_npy = np.stack(tvecs).reshape(-1, 3, 1)

        np.save(f'{out_path}/{run_name}/episode_{episode_id:04d}/calibration/rvecs.npy', rvecs_save_npy)
        np.save(f'{out_path}/{run_name}/episode_{episode_id:04d}/calibration/tvecs.npy', tvecs_save_npy)

        intrs = []
        for cam_id in range(len(cfg.env.cameras)):
            camera = cfg.env.cameras[cam_id]
            intr_mat = np.array(camera['intr']).reshape(3, 3).astype(np.float32)
            intrs.append(intr_mat)
        intrs_save = np.stack(intrs).reshape(-1, 3, 3)
        np.save(f'{out_path}/{run_name}/episode_{episode_id:04d}/calibration/intrinsics.npy', intrs_save)

        print("Resetting robot initial state")
        eef_xyz = obs['robot']['eef_xyz']  # (n_grippers, 3)
        eef_quat = obs['robot']['eef_quat']  # (n_grippers, 4)
        eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat)  # (n_grippers, 3, 3)
        eef_gripper = obs['robot']['eef_gripper']  # (n_grippers, 1)
        action = torch.cat([
            eef_xyz,
            eef_rot.reshape(eef_rot.shape[0], -1),
            eef_gripper
        ], dim=1)
        for _ in range(30):  # stabilize for 1s
            env.step({'action': action, 'do_velocity_control': False})   # just one step to set the initial state
        obs = env.unwrapped.get_obs()

        for cnt in range(n_steps):
            torch.cuda.synchronize()
            tt0 = time.perf_counter()

            image_list = obs['image_list']  # list of (3, H, W) tensors
            image_list_wrist = obs['image_wrist_list']  # list of (3, H, W) tensors

            index_side = 0
            index_wrist = 0
            for cam_id in range(len(cfg.env.cameras)):
                camera = cfg.env.cameras[cam_id]
                if camera['type'] == 'side':
                    image = image_list[index_side]
                    index_side += 1
                elif camera['type'] == 'wrist':
                    image = image_list_wrist[index_wrist]
                    index_wrist += 1
                else:
                    raise ValueError(f"Unknown camera type {camera['type']}")

                image = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                cv2.imwrite(f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_{cam_id}/rgb/{cnt:06d}.jpg', image)

                if cnt == 0:
                    cv2.imwrite(f'{out_path}/{run_name}/start_images/episode_{episode_id:04d}_camera_{cam_id}.jpg', image)

            # get observations
            pos = obs['robot']['eef_xyz']  # (n_grippers, 3)
            quat_wxyz = obs['robot']['eef_quat']  # (n_grippers, 4)
            gripper_qpos = 1.0 - obs['robot']['eef_gripper']  # (n_grippers, 1); in policy space, 1 is closed, 0 is open

            # get action
            n_grippers = robot_traj_list.shape[1]
            assert n_grippers == pos.shape[0] and n_grippers == quat_wxyz.shape[0] and n_grippers == gripper_qpos.shape[0]
            assert n_grippers == cfg.env.robot.n_grippers

            eef_xyz = robot_traj_list[cnt].reshape(n_grippers, 3)
            eef_rot = robot_rot_list[cnt].reshape(n_grippers, -1)
            eef_gripper = robot_gripper_list[cnt].reshape(n_grippers, 1)  # eef_gripper is in sim space

            eef_xyz = torch.from_numpy(eef_xyz).to(torch.float32).to(env.unwrapped.physics.device)
            eef_rot = torch.from_numpy(eef_rot).to(torch.float32).to(env.unwrapped.physics.device).reshape(n_grippers, 3, 3)
            eef_gripper = torch.from_numpy(eef_gripper).to(torch.float32).to(env.unwrapped.physics.device)
            eef_quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(eef_rot)

            eef_gripper = 1.0 - eef_gripper  # convert to policy space (0 open, 1 close)

            # save robot data
            robot_save = {
                "obs.ee_pos": pos[0].cpu().numpy().tolist(),  # [3] 
                "obs.ee_quat": quat_wxyz[0].cpu().numpy().tolist(),  # [4]
                "obs.gripper_qpos": gripper_qpos[0].cpu().numpy().tolist(),  # [1] (0 open, 1 close)
                "action.ee_pos": eef_xyz[0].cpu().numpy().tolist(),  # [3]
                "action.ee_quat": eef_quat[0].cpu().numpy().tolist(),  # [4]
                "action.gripper_qpos": eef_gripper[0].cpu().numpy().tolist(),  # [1] (0 open, 1 close)
            }

            with open(f'{out_path}/{run_name}/episode_{episode_id:04d}/robot/{cnt:06d}.json', 'w') as f:
                json.dump(robot_save, f, indent=4)

            eef_gripper = 1.0 - eef_gripper  # convert to sim tradition (1 open, 0 close)

            action = torch.cat([
                eef_xyz,
                eef_rot.reshape(n_grippers, -1),
                eef_gripper
            ], dim=1)
            _, _, done, truncated, _ = env.step({
                'action': action, 
                'do_velocity_control': cfg.env.robot.do_velocity_control
            })
            obs = env.unwrapped.get_obs()

            if cnt == n_steps - 1:
                image_list = obs['image_list']  # list of (3, H, W) tensors
                image_list_wrist = obs['image_wrist_list']  # list of (3, H, W) tensors

                index_side = 0
                index_wrist = 0
                for cam_id in range(len(cfg.env.cameras)):
                    camera = cfg.env.cameras[cam_id]
                    if camera['type'] == 'side':
                        image = image_list[index_side]
                        index_side += 1
                    elif camera['type'] == 'wrist':
                        image = image_list_wrist[index_wrist]
                        index_wrist += 1
                    else:
                        raise ValueError(f"Unknown camera type {camera['type']}")
                    
                    image = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_{cam_id}/rgb/{cnt + 1:06d}.jpg', image)
                    cv2.imwrite(f'{out_path}/{run_name}/final_images/episode_{episode_id:04d}_camera_{cam_id}.jpg', image)

            torch.cuda.synchronize()
            tt1 = time.perf_counter()
            print(f"Episode: {episode_id}, step: {cnt - 1}, time: {tt1 - tt0:.4f}, fps: {1 / (tt1 - tt0):.2f}")

        for cam_id in range(len(cfg.env.cameras)):
            make_video(
                Path(f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_{cam_id}/rgb'),
                Path(f'{out_path}/{run_name}/episode_{episode_id:04d}_camera_{cam_id}.mp4'),
                '%06d.jpg',
                frame_rate=frame_rate,
            )


if __name__ == '__main__':
    main()
