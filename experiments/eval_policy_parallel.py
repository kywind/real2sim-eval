from pathlib import Path
import hydra
from omegaconf import OmegaConf
import os
import numpy as np
import cv2
import torch
import time
from datetime import datetime
import pickle as pkl
import multiprocessing
from scipy.spatial.transform import Rotation as R
import kornia
import gymnasium as gym
import json
import sys
sys.path.append(str(Path(__file__).parents[1]))
OmegaConf.register_new_resolver("eval", eval, replace=True)

from experiments.utils.dir_utils import mkdir
from experiments.utils.ffmpeg import make_video
import sim.envs
from policy.inference.inference_wrapper import PolicyInferenceWrapper


def main(cfg, episode_list, local_rank=0, run_name='100000-000000'):
    frame_rate = cfg.physics.fps
    duration = cfg.env.sim.duration

    out_path = os.path.join(cfg.exp_root, 'output_eval_policy')
    mkdir(Path(f'{out_path}/{run_name}'), resume=True, overwrite=False)
    OmegaConf.save(cfg, f'{out_path}/{run_name}/hydra.yaml', resolve=True)

    print(f"Process on local rank {local_rank} handling {len(episode_list)} episodes: {episode_list}")
    for episode_id in episode_list:
        episode_id = int(episode_id)

        policy = PolicyInferenceWrapper(
            inference_cfg_path=cfg.policy.inference_cfg_path,
            checkpoint_path=cfg.policy.checkpoint_path,
            local_rank=local_rank,
        )

        # random reset
        env = gym.make(cfg.env_name, max_episode_steps=frame_rate * duration + 30, cfg=cfg, 
            obs_mode=cfg.obs_mode, exp_root=cfg.exp_root, local_rank=local_rank, randomize=True)
        obs, reset_info = env.reset(seed=episode_id)

        print("Reset info", reset_info)

        os.makedirs(f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_0/rgb', exist_ok=True)
        os.makedirs(f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_1/rgb', exist_ok=True)
        os.makedirs(f'{out_path}/{run_name}/episode_{episode_id:04d}/calibration', exist_ok=True)
        os.makedirs(f'{out_path}/{run_name}/episode_{episode_id:04d}/robot', exist_ok=True)
        os.makedirs(f'{out_path}/{run_name}/episode_{episode_id:04d}/state', exist_ok=True)
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

        if cfg.env.robot.use_pusher:
            action[:, 2] = 0.22
            eef_rot = torch.eye(3).to(action.device)
            eef_rot[:, 1] *= -1
            eef_rot[:, 2] *= -1
            action[:, 3:12] = eef_rot.reshape(-1)  # keep level

        for _ in range(30):  # stabilize for 1s
            env.step({'action': action, 'do_velocity_control': False})   # just one step to set the initial state
        obs = env.unwrapped.get_obs()

        random_variables = {
            'value': env.unwrapped.renderer.random_variables
        }
        with open(f'{out_path}/{run_name}/episode_{episode_id:04d}/random_variables.json', 'w') as f:
            json.dump(random_variables, f, indent=4)

        done, truncated = False, False
        cnt = 0
        while not (done or truncated):
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
                    image = policy.visualize_overlay(image)
                    index_side += 1
                elif camera['type'] == 'wrist':
                    image = image_list_wrist[index_wrist]
                    index_wrist += 1
                else:
                    raise ValueError(f"Unknown camera type: {camera['type']}")

                image = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                cv2.imwrite(f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_{cam_id}/rgb/{cnt:06d}.jpg', image)

                if cnt == 0:
                    cv2.imwrite(f'{out_path}/{run_name}/start_images/episode_{episode_id:04d}_camera_{cam_id}.jpg', image)

            pos = obs['robot']['eef_xyz']  # (n_grippers, 3)
            quat_wxyz = obs['robot']['eef_quat']  # (n_grippers, 4)
            gripper_qpos = 1.0 - obs['robot']['eef_gripper']  # (n_grippers, 1); in policy space, 1 is closed, 0 is open

            if cfg.env.robot.use_pusher:
                state = pos[:, :2]  # (n_grippers, 2)
            else:
                # gripper_qpos is now in policy space
                state = torch.cat([pos, quat_wxyz, gripper_qpos], dim=1)  # (n_grippers, 8)

            obs_dict = {}
            obs_dict["observation.state"] = state
            obs_dict["observation.images.front"] = image_list[0][None]
            obs_dict["observation.images.wrist"] = image_list_wrist[0][None]

            with torch.no_grad():
                cartesian_action = policy.inference(obs_dict)
            eef_xyz = cartesian_action[:, :3]
            eef_quat = cartesian_action[:, 3:7]
            eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat)
            eef_gripper = cartesian_action[:, 7:8]

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

            state_save = env.unwrapped.get_state()
            if cnt != 0:
                state_save.pop('physics')  # save space
            with open(f'{out_path}/{run_name}/episode_{episode_id:04d}/state/{cnt:06d}.pkl', 'wb') as f:
                pkl.dump(state_save, f)

            eef_gripper = 1.0 - eef_gripper  # convert to sim space (1 open, 0 close)

            action = torch.cat([
                eef_xyz,
                eef_rot.reshape(eef_rot.shape[0], -1),
                eef_gripper
            ], dim=1)  # (n_gripper, 8)

            cnt += 1

            _, _, done, truncated, _ = env.step({
                'action': action, 
                'do_velocity_control': cfg.env.robot.do_velocity_control
            })
            obs = env.unwrapped.get_obs()

            if done or truncated:
                image_list = obs['image_list']  # list of (3, H, W) tensors
                image_list_wrist = obs['image_wrist_list']  # list of (3, H, W) tensors

                index_side = 0
                index_wrist = 0
                for cam_id in range(len(cfg.env.cameras)):
                    camera = cfg.env.cameras[cam_id]
                    if camera['type'] == 'side':
                        image = image_list[index_side]
                        image = policy.visualize_overlay(image)
                        index_side += 1
                    elif camera['type'] == 'wrist':
                        image = image_list_wrist[index_wrist]
                        index_wrist += 1
                    else:
                        raise ValueError(f"Unknown camera type: {camera['type']}")
                    
                    image = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_{cam_id}/rgb/{cnt:06d}.jpg', image)
                    cv2.imwrite(f'{out_path}/{run_name}/final_images/episode_{episode_id:04d}_camera_{cam_id}.jpg', image)
                
                policy.reset()
            
            torch.cuda.synchronize()
            tt1 = time.perf_counter()
            print(f"Episode: {episode_id}, step: {cnt - 1}, time: {tt1 - tt0:.4f}, fps: {1 / (tt1 - tt0):.2f}")

        for cam_id in range(len(cfg.env.cameras)):
            make_video(
                Path(f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_{cam_id}/rgb'),
                Path(f'{out_path}/{run_name}/episode_{episode_id:04d}/vis_camera_{cam_id}.mp4'),
                '%06d.jpg',
                frame_rate=frame_rate,
            )


@hydra.main(version_base='1.2', config_path='../cfg', config_name="eval_policy")
def main_parallel(cfg):
    OmegaConf.resolve(cfg)

    if cfg.gs.use_grid_randomization:
        assert 'grid_randomization' in cfg.gs.object, "Object grid randomization config not found."
        obj_grid_cfg = cfg.gs.object.grid_randomization
        len_grid = len(obj_grid_cfg.xy) * len(obj_grid_cfg.theta) if not obj_grid_cfg.one_to_one else len(obj_grid_cfg.xy)
        len_mesh = 1
        for mesh_cfg in cfg.gs.meshes:
            if 'grid_randomization' in mesh_cfg and mesh_cfg.grid_randomization:
                mesh_grid_cfg = mesh_cfg.grid_randomization
                len_mesh *= len(mesh_grid_cfg.xy) * len(mesh_grid_cfg.theta) if not mesh_grid_cfg.one_to_one else len(mesh_grid_cfg.xy)
        cfg.policy.n_episodes = len_grid * len_mesh
    print("Total episodes:", cfg.policy.n_episodes)

    # unique run name
    if cfg.timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        timestamp = cfg.timestamp

    run_name = f"{timestamp}"

    n_processes = torch.cuda.device_count()
    episodes = np.arange(cfg.policy.n_episodes)
    local_rank_list = episodes % n_processes
    episodes_per_process = []
    for rank in range(n_processes):
        episodes_per_process.append(list(episodes[local_rank_list == rank]))
    print("Episode ids for each process:", [len(eps) for eps in episodes_per_process])

    with multiprocessing.Pool(processes=n_processes) as pool:
        try:
            async_res = pool.starmap_async(
                main, 
                [(cfg, episode_list, local_rank, run_name) for local_rank, episode_list in enumerate(episodes_per_process)]
            )
            results = async_res.get()
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected, terminating processes...")
            pool.terminate()
        else:
            pool.close()
        finally:
            pool.join()

    print("All episodes completed.")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main_parallel()
