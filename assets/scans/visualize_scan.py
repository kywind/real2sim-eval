from pathlib import Path
import argparse
import sys
sys.path.append(str(Path(__file__).parents[2]))

from sim.utils.gs.gs_processor import GSProcessor


def visualize_gs(gs_path):
    sp = GSProcessor()
    sp.visualize_gs([gs_path], axis_on=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gs_path', type=str, required=True, help='Path to the gs file to visualize')
    args = parser.parse_args()
    gs_path = args.gs_path
    visualize_gs(gs_path)
