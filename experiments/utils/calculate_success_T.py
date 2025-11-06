import argparse
import os
from glob import glob
from typing import List
import pickle as pkl
import numpy as np
from pathlib import Path


def find_episode_dirs(root: str) -> List[str]:
    eps = [d for d in glob(os.path.join(root, "episode_*")) if os.path.isdir(d)]
    eps = sorted(set(eps))
    return eps


def is_pusht_success(state: dict, x_target: np.ndarray, state_init: dict) -> bool:

    meshes = state_init['physics']['static_meshes']
    assert len(meshes) == 0

    x = state['renderer']['x']  # (N_vertices, 3)
    x_np = x.cpu().numpy()

    assert x_np.shape[0] == x_target.shape[0]
    mse = ((x_np - x_target) ** 2).sum(1).mean()

    return mse < 0.002


import json
def load_rand_xy_theta(episode_dir: str):
    """Return (rand_x, rand_y, rand_theta) from random_variables.json (units = meters)."""
    jpath = Path(episode_dir) / "random_variables.json"
    if not jpath.exists():
        raise FileNotFoundError(f"Missing random_variables.json at {jpath}")
    with open(jpath, "r") as f:
        data = json.load(f)
    v = data["value"]
    return float(v[0][0]), float(v[0][1]), float(v[0][3])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory containing episode subdirectories.')
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"Processing data directory: {data_dir}")

    with open('experiments/utils/T_final_state.pkl', 'rb') as f:
        target_data = pkl.load(f)
    x_target = target_data['renderer']['x'].cpu().numpy()

    episode_dirs = find_episode_dirs(data_dir)
    if not episode_dirs:
        raise SystemExit(f"No episodes under: {data_dir}")
    
    # return
    pusht_success_list = []
    for episode_dir in episode_dirs:
        state_files = sorted(glob(os.path.join(episode_dir, 'state/*.pkl')))
        pusht_count = 0
        pusht_success = False
        state_init = None
        for state_file in state_files:
            if '000000.pkl' in state_file:
                with open(state_file, 'rb') as f:
                    state_init = pkl.load(f)
            if int(state_file.split('/')[-1].split('.')[0]) < 1700:
                continue
            with open(state_file, 'rb') as f:
                state = pkl.load(f)
            pusht = is_pusht_success(state, x_target, state_init)
            pusht_count += pusht * 1.0
            if pusht_count >= 30:
                pusht_success = True
                break

        pusht_success_list.append(pusht_success)

    print(f'pusht success rate: {sum(pusht_success_list)} / {len(pusht_success_list)} = {sum(pusht_success_list) / len(pusht_success_list) * 100:.1f}%')
