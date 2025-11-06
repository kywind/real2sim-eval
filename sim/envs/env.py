from pathlib import Path
from omegaconf import DictConfig
import gymnasium as gym
import numpy as np
import time

from ..physics import PhysTwinDynamics
from ..renderer import GSRenderer
from ..utils.env.registration import register_env


@register_env("BaseEnv-v0", max_episode_steps=2000)
class BaseEnv(gym.Env):

    def __init__(self, 
            cfg: DictConfig,
            exp_root: str | Path,
            randomize: bool = False,
            local_rank: int = 0,
            **kwargs
        ):

        self.renderer = GSRenderer(cfg, local_rank)
        self.physics = PhysTwinDynamics(cfg, exp_root, cfg.physics.ckpt_path, cfg.physics.case_name, local_rank)
        self.cfg = cfg

        # randomize initial configuration of objects
        self.randomize = randomize

    def reset(self, *, seed=None, options=None):
        super().reset(seed=None, options=options)
        np.random.seed(seed)
        reset_info = {}
        self.renderer.load_scaniverse(randomize=self.randomize, index=seed)  # overload seed as randomization index
        self.renderer.set_all_cameras()
        self.renderer.reset_state(visualize_image=False)

        state = self.renderer.get_state()  # world frame
        phystwin_pts = self.physics.reset(
            state, 
            init_meshes_dict=self.renderer.meshes, 
            robot=self.renderer.robot,
            eef_pts_func=self.renderer.eef_pts_func,
            kin_helper=self.renderer.kin_helper,
            init_eef_xyz=self.renderer.init_eef_xyz,
            pose_obj=self.renderer.pose_obj
        )
        self.renderer.update_phystwin_pts(phystwin_pts)

        obs = self.get_obs()
        return obs, reset_info
    
    def get_obs(self, render_extra=False):
        state = self.renderer.get_state()
        im_list, depth_list = self.renderer.render_fixed_cameras()
        im_wrist_list, depth_wrist_list = self.renderer.render_wrist_cameras()
        im_extra, depth_extra = None, None
        if render_extra:  # extra viewpoints specified by cfg.render
            im_extra, depth_extra = self.renderer.render()

        robot = {
            'eef_xyz': state['eef_xyz'],
            'eef_quat': state['eef_quat'],
            'eef_gripper': state['eef_gripper'],
        }
        return {
            'image_list': im_list,
            'depth_list': depth_list,
            'image_wrist_list': im_wrist_list,
            'depth_wrist_list': depth_wrist_list,
            'image_extra': im_extra,
            'depth_extra': depth_extra,
            'robot': robot,
        }

    def get_language_instruction(self):
        return None
    
    def render(self):
        im, depth = self.renderer.render()
        return im, depth

    def close(self):
        return None

    def step(self, action_dict):
        state = self.renderer.get_state()
        action = action_dict['action']
        do_velocity_control = action_dict.get('do_velocity_control', True)
        if do_velocity_control:
            action = self.renderer.mimic_velocity_control(action)
        state = self.physics.step(state, action)  # world frame
        self.renderer.update_state(state)  # world frame
        return None, None, None, None, None

    def get_state(self):
        renderer_state = self.renderer.get_state()
        physics_state = self.physics.get_state()
        state = {
            'renderer': {
                'x': renderer_state['x'],
            },
            'physics': {
                'static_meshes': physics_state['static_meshes'],
                'init_springs': physics_state['init_springs']
            }
        }
        return state
