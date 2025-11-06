import os
import cv2
import os
import cv2
import numpy as np
import torch
import gymnasium as gym
from omegaconf import OmegaConf
import hydra
import kornia
from datetime import datetime
import pickle as pkl
from pathlib import Path
try:
    from pynput import keyboard
except ImportError:
    print("pynput not installed, keyboard controls will not work.")
import sys
sys.path.append(str(Path(__file__).parents[1]))
OmegaConf.register_new_resolver("eval", eval, replace=True)

from experiments.utils.dir_utils import mkdir
import sim.envs


class InteractivePlayground:

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.save_state:
            self.cnt = 0
            self.episode_id = 0
            self.out_path = os.path.join(cfg.exp_root, 'output_keyboard_teleop')
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_name = f"{timestamp}"
            mkdir(Path(f'{self.out_path}/{self.run_name}'), resume=False, overwrite=False)
            OmegaConf.save(cfg, f'{self.out_path}/{self.run_name}/hydra.yaml', resolve=True)

    def on_press(self, key):
        try:
            self.pressed_keys.add(key.char)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.pressed_keys.remove(key.char)
        except (KeyError, AttributeError):
            try:
                self.pressed_keys.remove(str(key))
            except KeyError:
                pass

    def get_trans_change(self):
        trans_change = np.zeros((1, 3))
        for key in self.pressed_keys:
            print(f"Pressed key: {key}")
            if key in self.key_mappings:
                if (
                    key == "n"
                    or key == "m"
                    or key == "z"
                    or key == "x"
                    or key == "c"
                    or key == "v"
                ):
                    pass
                else:
                    idx, change = self.key_mappings[key]
                    trans_change[idx] += change
        return trans_change

    def get_finger_change(self):
        for key in self.pressed_keys:
            if key in self.key_mappings:
                if key == "n" or key == "m":
                    return self.key_mappings[key]
        return 0.0

    def get_rot_change(self):
        for key in self.pressed_keys:
            if key in self.key_mappings:
                if key == "z" or key == "x" or key == "c" or key == "v":
                    return np.array(self.key_mappings[key])
        return np.zeros(3)

    def init_control_ui(self):
        self.interm_size = 512
        self.rotations = {
            "w": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1
            ),  # Forward
            "a": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 90, 1
            ),  # Left
            "s": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180, 1
            ),  # Backward
            "d": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 270, 1
            ),  # Right
            "r": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1
            ),  # Up
            "f": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180, 1
            ),  # Down
            "i": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1
            ),  # Forward
            "j": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 90, 1
            ),  # Left
            "k": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180, 1
            ),  # Backward
            "l": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 270, 1
            ),  # Right
            "u": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1
            ),  # Up
            "o": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180, 1
            ),  # Down
        }

    def run(self):
        cfg = self.cfg

        frame_rate = cfg.physics.fps
        duration = 600000  # infinite duration

        env = gym.make(cfg.env_name, max_episode_steps=frame_rate * duration, cfg=cfg, 
            obs_mode=cfg.obs_mode, exp_root=cfg.exp_root, local_rank=0, randomize=False)
        obs, reset_info = env.reset(seed=0)
        
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
        for _ in range(1):  # stabilize
            env.step({'action': action, 'do_velocity_control': False})   # just one step to set the initial state
        obs = env.unwrapped.get_obs()

        print("UI Controls:")
        print("- Set 1: WASD (XY movement), QE (Z movement)")
        self.key_mappings = {
            # Set 1 controls
            "w": (0, np.array([-0.005, 0, 0])),
            "s": (0, np.array([0.005, 0, 0])),
            "a": (0, np.array([0, -0.005, 0])),
            "d": (0, np.array([0, 0.005, 0])),
            "f": (0, np.array([0, 0, -0.005])),
            "r": (0, np.array([0, 0, 0.005])),

            "i": (0, np.array([-0.001, 0, 0])),
            "k": (0, np.array([0.001, 0, 0])),
            "j": (0, np.array([0, -0.001, 0])),
            "l": (0, np.array([0, 0.001, 0])),

            # Set the finger
            "n": 0.05,
            "m": -0.05,

            # Set the rotation
            "z": [0, 0, 2.0 / 180 * np.pi],
            "x": [0, 0, -2.0 / 180 * np.pi],
            "c": [2.0 / 180 * np.pi, 0, 0],
            "v": [-2.0 / 180 * np.pi, 0, 0],
        }
        self.pressed_keys = set()
        self.init_control_ui()

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        accumulate_trans = np.array([cfg.env.robot.init_eef_xyz])
        accumulate_rot = torch.eye(3, dtype=torch.float32, device='cuda:0')
        accumulate_rot[1, 1] = -1.
        accumulate_rot[2, 2] = -1.
        current_finger = 1.0  # hardcoded: start with closed gripper

        while True:
            trans_change = self.get_trans_change()
            finger_change = self.get_finger_change()
            rot_change = self.get_rot_change()
            
            current_finger += finger_change
            current_finger = max(0.0, min(1.0, current_finger))

            accumulate_trans += trans_change

            new_rot = torch.tensor(rot_change, dtype=torch.float32, device='cuda:0')
            new_rot = kornia.geometry.conversions.axis_angle_to_rotation_matrix(new_rot[None])[0]
            accumulate_rot = torch.matmul(accumulate_rot, new_rot)

            action = np.concatenate([
                accumulate_trans.reshape(1, 3),
                accumulate_rot.reshape(1, -1).cpu().numpy(),
                np.array([current_finger]).reshape(1, 1)
            ], axis=1)
            action = torch.from_numpy(action).to(torch.float32).to('cuda:0')

            if cfg.env.robot.use_pusher:
                pos_z = 0.22

                eef_xyz = torch.cat([
                    action[:, :2], 
                    pos_z * torch.ones((cfg.env.robot.n_grippers, 1), 
                    device=action.device)
                ], dim=1)  # (n_grippers, 3)
                eef_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=eef_quat.device)
                eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat)
                eef_gripper = torch.tensor([[0.0]], device=eef_quat.device)

                action = torch.cat([
                    eef_xyz,
                    eef_rot.reshape(eef_rot.shape[0], -1),
                    eef_gripper
                ], dim=1)  # (n_gripper, 8)

            _, _, done, truncated, _ = env.step({
                'action': action, 
                'do_velocity_control': cfg.env.robot.do_velocity_control
            })
            obs = env.unwrapped.get_obs()

            im = obs['image_list'][0]  # camera 0
            image = im.permute(1, 2, 0)

            image = image.clamp(0, 1)
            frame = (image[..., :3] * 255).clone().cpu().numpy().astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            im = obs['image_wrist_list'][0]  # camera 1
            image = im.permute(1, 2, 0)

            image = image.clamp(0, 1)
            frame_1 = (image[..., :3] * 255).clone().cpu().numpy().astype(np.uint8)
            frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_RGB2BGR)

            frame = cv2.hconcat([frame, frame_1])

            cv2.imshow("Interactive Playground", frame)
            cv2.waitKey(1)

            if cfg.save_state:
                state_save = env.unwrapped.get_state()
                with open(f'{self.out_path}/{self.run_name}/episode_{self.episode_id:04d}/state/{self.cnt:06d}.pkl', 'wb') as f:
                    pkl.dump(state_save, f)
                self.cnt += 1


@hydra.main(version_base='1.2', config_path='../cfg', config_name="keyboard_teleop")
def main(cfg):
    playground = InteractivePlayground(cfg)
    playground.run()


if __name__ == "__main__":
    main()
