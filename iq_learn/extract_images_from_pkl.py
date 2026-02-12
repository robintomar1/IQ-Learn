#!/usr/bin/env python3
"""
Extract and save images from an expert trajectory .pkl file (e.g. CartPole_Vision.pkl).
States can be raw arrays (H, W, C) or dicts with an 'image' key.
"""

import argparse
import os
import pickle
import numpy as np
from pathlib import Path


class NumpyCompatUnpickler(pickle.Unpickler):
    """Handle pickles saved with different numpy (e.g. numpy._core vs numpy.core)."""

    def find_class(self, module, name):
        if module == "numpy._core.multiarray":
            module = "numpy.core.multiarray"
        elif module == "numpy._core.umath":
            module = "numpy.core.umath"
        elif module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def get_image_from_state(state):
    """Get a single image array from a state (array or dict with 'image' key)."""
    if isinstance(state, np.ndarray):
        if state.ndim >= 2:
            return state
        return None
    if isinstance(state, dict):
        if "image" in state:
            return np.asarray(state["image"])
        # use first value that looks like an image (3D array)
        for v in state.values():
            arr = np.asarray(v)
            if arr.ndim >= 2:
                return arr
    return None


def _describe_value(v, depth=0):
    """Return a short string describing a value (list, array, dict, etc.)."""
    if isinstance(v, np.ndarray):
        return f"ndarray shape={v.shape}, dtype={v.dtype}"
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return "[]"
        first = v[0]
        if isinstance(first, np.ndarray):
            return f"list of {len(v)} arrays, first shape={first.shape}"
        if isinstance(first, (list, tuple)):
            return f"list of {len(v)} sequences (lengths ~{[len(x) for x in v[:3]]}{'...' if len(v) > 3 else ''})"
        if isinstance(first, dict):
            return f"list of {len(v)} dicts, keys={list(first.keys())}"
        return f"list of {len(v)} items"
    if isinstance(v, dict):
        return f"dict keys={list(v.keys())}"
    return type(v).__name__


def print_data_summary(data):
    """Print summary of pkl contents: keys, total images, per-episode counts, other data."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Top-level keys: {list(data.keys())}")
    print()

    # States = images per episode and total
    if "states" in data:
        states_list = data["states"]
        num_episodes = len(states_list)
        per_episode = [len(traj) for traj in states_list]
        total_images = sum(per_episode)
        print("Images (from 'states'):")
        print(f"  Total images:     {total_images}")
        print(f"  Num episodes:      {num_episodes}")
        print(f"  Per episode:       min={min(per_episode)}, max={max(per_episode)}, mean={np.mean(per_episode):.1f}")
        if num_episodes <= 20:
            print(f"  Per-episode list: {per_episode}")
        else:
            print(f"  First 10 lengths: {per_episode[:10]} ... last 5: {per_episode[-5:]}")
        if total_images > 0:
            first_state = states_list[0][0]
            img = get_image_from_state(first_state)
            if img is not None:
                print(f"  Sample state shape: {img.shape} dtype={img.dtype}")
        print()

    # Other keys
    print("Other data:")
    for key in data.keys():
        if key == "states":
            continue
        v = data[key]
        desc = _describe_value(v)
        if key == "lengths":
            if isinstance(v, np.ndarray):
                print(f"  {key}: {desc} (per-episode frame counts)")
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, np.integer)):
                print(f"  {key}: list of {len(v)} ints (per-episode frame counts)")
            else:
                print(f"  {key}: {desc}")
        else:
            print(f"  {key}: {desc}")
    print("=" * 60 + "\n")


def extract_images(pkl_path, output_dir, max_per_traj=50, max_trajs=None, prefix="img"):
    """
    Load pkl, find all image-like states, and save them as PNGs.

    Args:
        pkl_path: Path to the .pkl file.
        output_dir: Directory to write images (e.g. extracted_images/CartPole_Vision).
        max_per_traj: Max images to save per trajectory (None = all).
        max_trajs: Max trajectories to process (None = all).
        prefix: Filename prefix for saved images.
    """
    pkl_path = Path(pkl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, "rb") as f:
        try:
            data = pickle.load(f)
        except ModuleNotFoundError as e:
            if "numpy" in str(e):
                f.seek(0)
                data = NumpyCompatUnpickler(f).load()
            else:
                raise

    if "states" not in data:
        raise KeyError(f"Expected 'states' in pkl. Keys found: {list(data.keys())}")

    print_data_summary(data)

    states_list = data["states"]
    n_trajs = len(states_list)
    if max_trajs is not None:
        n_trajs = min(n_trajs, max_trajs)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    total_saved = 0
    for traj_idx in range(n_trajs):
        traj_states = states_list[traj_idx]
        n_steps = len(traj_states)
        to_take = n_steps if max_per_traj is None else min(n_steps, max_per_traj)
        step_indices = np.linspace(0, n_steps - 1, to_take, dtype=int) if n_steps else []

        for step_idx in step_indices:
            state = traj_states[step_idx]
            img = get_image_from_state(state)
            if img is None:
                continue

            # Normalize to [0,1] if needed (e.g. float already in [0,1])
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.max() > 1.0 and np.issubdtype(img.dtype, np.floating):
                img = img / 255.0
            img = np.clip(img, 0, 1)

            # Normalize layout to (H, W) or (H, W, C) for saving/display
            # Handle (C, H, W) -> (H, W, C) or (H, W)
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            while img.ndim > 3 and img.shape[0] == 1:
                img = img.squeeze(0)
            if img.ndim == 2:
                img = np.expand_dims(img, -1)

            out_name = f"{prefix}_traj{traj_idx:04d}_step{step_idx:05d}.png"
            out_path = output_dir / out_name

            if plt is not None:
                plt.figure(figsize=(4, 4))
                if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
                    plt.imshow(img.squeeze(), cmap="gray")
                else:
                    plt.imshow(img)
                plt.axis("off")
                plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
                plt.close()
            else:
                # Save with imageio or raw numpy
                try:
                    import imageio
                    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                    if img_uint8.shape[-1] == 1:
                        img_uint8 = img_uint8.squeeze(-1)
                    imageio.imwrite(out_path, img_uint8)
                except Exception:
                    np.save(out_path.with_suffix(".npy"), img)

            total_saved += 1

    print(f"Saved {total_saved} images to {output_dir}")
    return total_saved


def main():
    parser = argparse.ArgumentParser(description="Extract images from expert trajectory pkl")
    parser.add_argument(
        "pkl_file",
        nargs="?",
        default="experts/CartPole_Vision.pkl",
        help="Path to .pkl file (default: experts/CartPole_Vision.pkl)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory (default: extracted_images/<pkl_stem>)",
    )
    parser.add_argument(
        "--max-per-traj",
        type=int,
        default=50,
        help="Max images per trajectory (default: 50, use 0 for no limit)",
    )
    parser.add_argument(
        "--max-trajs",
        type=int,
        default=None,
        help="Max trajectories to process (default: all)",
    )
    parser.add_argument(
        "--prefix",
        default="img",
        help="Filename prefix for images (default: img)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print data summary (total images, per episode, other keys); do not extract images",
    )
    args = parser.parse_args()

    pkl_path = Path(args.pkl_file)
    if not pkl_path.is_absolute():
        # Try relative to script dir
        script_dir = Path(__file__).resolve().parent
        pkl_path = script_dir / pkl_path
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pkl file not found: {pkl_path}")

    if args.summary_only:
        with open(pkl_path, "rb") as f:
            try:
                data = pickle.load(f)
            except ModuleNotFoundError as e:
                if "numpy" in str(e):
                    f.seek(0)
                    data = NumpyCompatUnpickler(f).load()
                else:
                    raise
        print_data_summary(data)
        return

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "extracted_images" / pkl_path.stem
    output_dir = Path(output_dir)

    max_per_traj = args.max_per_traj if args.max_per_traj > 0 else None
    extract_images(
        pkl_path,
        output_dir,
        max_per_traj=max_per_traj,
        max_trajs=args.max_trajs,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
