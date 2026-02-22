# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
conda create -n iq_learn python=3.10 -y
conda activate iq_learn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r updated_requirements.txt
pip install autorom[accept-rom-license] && AutoROM --accept-license
```

Use `updated_requirements.txt` (not `requirements.txt`) — it includes envpool, trackio, and other additions.

## Common Commands

All commands run from `iq_learn/` with `iq_learn` conda env active.

**Train IQ-Learn on Breakout (primary use case):**
```bash
python train_iq.py env=breakout
```

**Override config values on the command line (Hydra syntax):**
```bash
python train_iq.py env=breakout num_envs=8 train.batch=128
python train_iq.py env=breakout env.learn_steps=500000 train.updates_per_step=4
```

**Resume an interrupted run (full state including optimizer):**
```bash
python train_iq.py env=breakout resume=outputs/YYYY-MM-DD/HH-MM-SS/checkpoint/checkpoint.pt
```

**Warm-start from best checkpoint (weights only, no optimizer state):**
```bash
python train_iq.py env=breakout pretrain=outputs/YYYY-MM-DD/HH-MM-SS/results_best/softq_iq_BreakoutNoFrameskip-v4
```

**Generate expert data from pretrained SB3 A2C agent:**
```bash
python expert_generation.py env=breakout expert.demos=20
```

**Inspect expert data:**
```bash
python inspect_expert.py
```

**Monitor training:**
```bash
tensorboard --logdir outputs/
```

**Other environments (from README examples):**
```bash
# CartPole offline IL
python train_iq.py agent=softq method=iq env=cartpole expert.demos=1 expert.subsample_freq=20 agent.init_temp=0.001 method.chi=True method.loss=value_expert

# MuJoCo Humanoid
python train_iq.py env=humanoid agent=sac expert.demos=1 method.loss=v0 method.regularize=True
```

## Architecture Overview

### Algorithm

IQ-Learn is a non-adversarial imitation learning method. The core idea is to replace the standard RL critic loss with `iq_loss` (in `iq.py`). The critic learns an implicit reward by minimizing a Bellman consistency objective over both expert and policy transitions. No discriminator or reward network is needed.

**Key insight:** The Q-function implicitly encodes the reward — `r(s,a) = Q(s,a) - γV(s')` — so the recovered reward is free after learning.

### Training Flow

`train_iq.py` is the entry point. It:
1. Loads config via Hydra (`conf/config.yaml` + env/agent/method overlays)
2. Creates environment(s) via `make_envs.py`
3. Instantiates agent (`SoftQ` for discrete, `SAC` for continuous) via `agent/__init__.py`
4. Loads expert demonstrations from `experts/*.pkl` into a replay buffer
5. Runs a training loop with two paths:
   - **Single-env** (`num_envs=1`): episode-based loop using gymnasium env
   - **Multi-env** (`num_envs>1`): vectorized loop using envpool (C++ backend, no subprocess IPC)
6. At each step, calls `iq_update` → `iq_update_critic` → `iq_loss` from `iq.py`
7. Saves checkpoints to `outputs/YYYY-MM-DD/HH-MM-SS/` (Hydra-managed)

The `iq_update` and `iq_update_critic` functions are defined in `train_iq.py` and monkey-patched onto the agent at runtime via `types.MethodType`.

### Key Files

| File | Purpose |
|------|---------|
| `iq.py` | Standalone IQ-Learn loss — the core algorithm. Import this to add IQ to other projects. |
| `train_iq.py` | Main training script; also defines `iq_update`, `iq_update_critic` (monkey-patched onto agent) |
| `agent/softq.py` | SoftQ agent for discrete action spaces (Atari, CartPole) |
| `agent/sac.py` | SAC agent for continuous action spaces (MuJoCo) |
| `agent/softq_models.py` | Q-network architectures: `SimpleQNetwork`, `AtariQNetwork`, `DoubleQNetwork` |
| `agent/sac_models.py` | SAC models including multimodal `DoubleQCritic` with image+state inputs |
| `dataset/memory.py` | Two replay buffers: `Memory` (deque-based, single-env) and `NumpyReplayBuffer` (preallocated arrays, multi-env) |
| `make_envs.py` | Environment factory: handles Atari, DMControl, Dusty (Isaac), custom envs; envpool for vectorized Atari |
| `expert_generation.py` | Generates expert `.pkl` files from a trained policy (SB3 or custom) |
| `baselines_zoo/baselines_expert.py` | Loads pretrained SB3 agents for use as experts |

### Configuration System (Hydra)

Config is composed from `conf/`:
- `config.yaml` — global defaults (gamma, seed, num_envs, train params)
- `conf/env/breakout.yaml` — primary file to edit for Breakout runs
- `conf/method/iq.yaml` — IQ-Learn loss type (`value_expert`/`value`/`v0`), chi divergence, grad penalty
- `conf/agent/softq.yaml` — SoftQ hyperparams (lr, tau, target update frequency)
- `conf/agent/sac.yaml` — SAC hyperparams

Hydra saves a copy of the resolved config in each `outputs/` run directory. All paths in the training script are resolved relative to the Hydra output dir, so use `hydra.utils.to_absolute_path()` for any path that should be absolute (expert files, checkpoints).

### Replay Buffers

- `Memory`: Python deque, used for single-env training. Stores `LazyFrames` (lazy frame-stacked Atari observations).
- `NumpyReplayBuffer`: Preallocated numpy ring buffer, used for multi-env (envpool) training. Stores uint8 observations and converts to float32 on the GPU during sampling. Drop-in replacement for `Memory`.

The buffer class is selected automatically: `NumpyReplayBuffer if num_envs > 1 else Memory`.

### Multi-env Scaling Rules

When using `num_envs > 1`, scale these hyperparameters proportionally:
```
critic_target_update_frequency: 100 / num_envs
updates_per_step: 8 / num_envs
eval_interval: 25000 / num_envs
```

### Supported Environments

- **Atari**: `BreakoutNoFrameskip-v4`, `PongNoFrameskip-v4`, `SpaceInvadersNoFrameskip-v4`, `BeamRiderNoFrameskip-v4`, `QbertNoFrameskip-v4`, `SeaquestNoFrameskip-v4`
  - Wrapped with `AtariWrapper` → `PyTorchFrame` → `FrameStack(4)` in single-env mode
  - envpool handles all preprocessing in multi-env mode
- **MuJoCo**: Hopper, HalfCheetah, Ant, Walker2d, Humanoid — uses SAC agent
- **DMControl**: prefix `dmc_` (e.g. `dmc_cheetah_run`)
- **Custom**: `PointMazeRight-v0`, `PointMazeLeft-v0`, `dusty` (Isaac Gym wrapper)

### Multimodal Observations

`sac_models.py` supports dict observations with `image` and `state` keys. `iq.py` has a `slice_obs` helper that handles both tensor and dict observations for gradient penalty and expert masking operations.

### Output Directory Structure

```
outputs/YYYY-MM-DD/HH-MM-SS/
├── checkpoint/checkpoint.pt   # Full resume checkpoint (Q-net, target, optimizer, counters)
├── results/                   # Periodic model saves (every save_interval epochs)
├── results_best/              # Best eval checkpoint
└── logs/                      # TensorBoard events, train.csv, eval.csv
```
