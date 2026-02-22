"""
Inspect a Breakout expert pickle file and print per-episode rewards.
Usage:
    python inspect_expert.py experts/BreakoutNoFrameskip-v4_20.pkl
    python inspect_expert.py experts/BreakoutNoFrameskip-v4_20.pkl --sort
"""

import argparse
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl_file', help='Path to expert .pkl file')
    parser.add_argument('--sort', action='store_true',
                        help='Sort episodes by reward (descending)')
    args = parser.parse_args()

    with open(args.pkl_file, 'rb') as f:
        data = pickle.load(f)

    rewards_per_ep = [sum(r) for r in data['rewards']]
    lengths_per_ep = data['lengths']

    episodes = list(enumerate(zip(rewards_per_ep, lengths_per_ep)))
    if args.sort:
        episodes = sorted(episodes, key=lambda x: x[1][0], reverse=True)

    print(f"\n{'Ep':>4}  {'Reward':>10}  {'Steps':>8}")
    print('-' * 28)
    for ep_idx, (reward, length) in episodes:
        print(f"{ep_idx:>4}  {reward:>10.1f}  {length:>8}")

    arr = np.array(rewards_per_ep)
    print('-' * 28)
    print(f"{'Total eps:':>14} {len(arr)}")
    print(f"{'Mean reward:':>14} {arr.mean():.1f}")
    print(f"{'Std reward:':>14} {arr.std():.1f}")
    print(f"{'Min reward:':>14} {arr.min():.1f}")
    print(f"{'Max reward:':>14} {arr.max():.1f}")


if __name__ == '__main__':
    main()
