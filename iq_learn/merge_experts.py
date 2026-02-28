"""
Merge multiple expert demonstration pkl files into a single combined file.

Usage:
    python merge_experts.py \
        --inputs experts/Breakout_a2c_15.pkl experts/Breakout_dqn_15.pkl \
        --output experts/Breakout_multi_30.pkl \
        --shuffle --seed 42
"""

import argparse
import pickle
import random
import numpy as np
from collections import defaultdict


def load_expert_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def print_file_stats(path, data):
    n_trajs = len(data["lengths"])
    lengths = np.array(data["lengths"])
    rewards = np.array([sum(r) for r in data["rewards"]])
    print(f"  {path}")
    print(f"    trajectories: {n_trajs}")
    print(f"    rewards: {rewards.mean():.2f} +/- {rewards.std():.2f}  "
          f"(min={rewards.min():.2f}, max={rewards.max():.2f})")
    print(f"    lengths: {lengths.mean():.1f} +/- {lengths.std():.1f}  "
          f"(min={lengths.min()}, max={lengths.max()})")


def merge_experts(input_files, output_file, shuffle=False, seed=None):
    all_data = []
    for path in input_files:
        data = load_expert_file(path)
        print_file_stats(path, data)
        all_data.append(data)

    # Determine all keys from the first file
    keys = list(all_data[0].keys())

    # Build per-trajectory index: list of (file_idx, traj_idx) pairs
    indices = []
    for file_idx, data in enumerate(all_data):
        n_trajs = len(data["lengths"])
        for traj_idx in range(n_trajs):
            indices.append((file_idx, traj_idx))

    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)

    # Merge trajectories in order
    merged = defaultdict(list)
    for file_idx, traj_idx in indices:
        data = all_data[file_idx]
        for key in keys:
            merged[key].append(data[key][traj_idx])

    # Print combined stats
    n_total = len(merged["lengths"])
    lengths = np.array(merged["lengths"])
    rewards = np.array([sum(r) for r in merged["rewards"]])
    print(f"\nCombined:")
    print(f"  trajectories: {n_total}")
    print(f"  rewards: {rewards.mean():.2f} +/- {rewards.std():.2f}  "
          f"(min={rewards.min():.2f}, max={rewards.max():.2f})")
    print(f"  lengths: {lengths.mean():.1f} +/- {lengths.std():.1f}  "
          f"(min={lengths.min()}, max={lengths.max()})")

    with open(output_file, 'wb') as f:
        pickle.dump(dict(merged), f)
    print(f"\nSaved merged file to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple expert pkl files")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Input pkl files to merge")
    parser.add_argument("--output", required=True,
                        help="Output merged pkl file")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle trajectory order across files")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for shuffling")
    args = parser.parse_args()

    print(f"Merging {len(args.inputs)} expert files:\n")
    merge_experts(args.inputs, args.output, shuffle=args.shuffle, seed=args.seed)


if __name__ == "__main__":
    main()
