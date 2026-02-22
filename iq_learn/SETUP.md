# IQ-Learn Breakout — Setup & Operations Guide

IQ-Learn (Inverse soft-Q Learning) for imitation learning on Atari Breakout.
This guide covers environment setup, expert data generation, training, monitoring,
and resuming interrupted runs.

---

## 1. Prerequisites

- Linux (tested on Ubuntu 22.04)
- NVIDIA GPU with CUDA 12.x
- [Anaconda or Miniconda](https://docs.anaconda.com/miniconda/)
- ~15 GB free disk space (replay buffer + model checkpoints)
- ~16 GB RAM recommended

---

## 2. Environment Setup

### 2.1 Create the conda environment

```bash
conda create -n iq_learn python=3.10 -y
conda activate iq_learn
```

### 2.2 Install PyTorch (CUDA 12.x)

Check https://pytorch.org/get-started/locally/ for the exact command matching your
CUDA driver. For CUDA 12.x:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2.3 Install remaining dependencies

```bash
cd /path/to/IQ-Learn-cb/iq_learn
pip install -r updated_requirements.txt
```

### 2.4 Install Atari ROMs

```bash
pip install autorom[accept-rom-license]
AutoROM --accept-license
```

### 2.5 Verify the setup

```bash
python -c "
import torch, gymnasium, stable_baselines3, ale_py
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Gymnasium:', gymnasium.__version__)
print('SB3:', stable_baselines3.__version__)
env = gymnasium.make('BreakoutNoFrameskip-v4')
print('Breakout env OK:', env.observation_space)
env.close()
"
```

Expected output:
```
PyTorch: 2.10.0+cu128
CUDA available: True
Gymnasium: 1.x.x
SB3: 2.x.x
Breakout env OK: Box(0, 255, (210, 160, 3), uint8)
```

---

## 3. Repository Layout

```
iq_learn/
├── conf/                    # Hydra config files
│   ├── config.yaml          # Global defaults
│   ├── env/breakout.yaml    # Breakout-specific hyperparameters  ← edit this
│   └── method/iq.yaml       # IQ-Learn loss settings
├── agent/
│   └── softq.py             # SoftQ agent (Q-network + policy)
├── experts/
│   └── BreakoutNoFrameskip-v4_20.pkl   # Expert demo data (20 trajectories)
├── baselines_zoo/
│   └── baselines_expert.py  # Loads pretrained SB3 agents
├── rl-baselines3-zoo/       # Pretrained SB3 model zoo
├── outputs/                 # All training runs saved here (auto-created by Hydra)
│   └── YYYY-MM-DD/HH-MM-SS/
│       ├── checkpoint/checkpoint.pt   # Full resume checkpoint
│       ├── results/                   # Periodic model saves
│       ├── results_best/              # Best eval checkpoint
│       └── logs/                      # TensorBoard events + CSVs
├── train_iq.py              # Main training script
└── expert_generation.py     # Generate expert data from a pretrained agent
```

---

## 4. Key Configuration — `conf/env/breakout.yaml`

This is the primary file you'll edit between runs.

```yaml
env:
  name: BreakoutNoFrameskip-v4
  demo: BreakoutNoFrameskip-v4_20.pkl   # Expert data file in experts/
  replay_mem: 200000      # Online replay buffer capacity
  initial_mem: 5000       # Steps to collect before learning starts
  learn_steps: 1e6        # Total gradient update steps
  eval_interval: 25000    # Evaluate policy every N learn_steps

expert:
  demos: 20               # Number of expert trajectories to load
  subsample_freq: 1       # Load every Nth frame (1 = all frames)

agent:
  critic_target_update_frequency: 100   # Sync target network every N learn_steps
                                        # Scale DOWN if using many parallel envs

train:
  batch: 64               # Batch size per gradient update
  use_target: True        # Use separate frozen target network (must be True)
  soft_update: True       # Blend target toward Q-net (True) vs hard copy (False)
  updates_per_step: 8     # Gradient updates per environment step
                          # Scale DOWN if using many parallel envs (num_envs > 1)
```

**Global settings in `conf/config.yaml`:**

```yaml
gamma: 0.99              # Discount factor (future reward weight)
num_envs: 1              # Parallel environments (1 = single env, stable)
                         # If > 1, also adjust critic_target_update_frequency
                         #   and updates_per_step proportionally
```

**IQ-Learn loss in `conf/method/iq.yaml`:**

```yaml
method:
  loss: value_expert     # Use only expert states for Bellman consistency (stable)
  chi: False             # χ² regularisation (keep False for Atari)
```

---

## 5. Hyperparameter Quick Reference

| Parameter | Effect | Typical range |
|-----------|--------|---------------|
| `batch` | Samples per gradient update. Larger = smoother gradients but slower if CPU-bound (LazyFrames). | 64–256 |
| `updates_per_step` | Gradient updates per env step. Higher = more sample efficient but risks overfitting to stale buffer. Divide by `num_envs` when scaling up envs. | 1–8 |
| `critic_target_update_frequency` | Steps between target network syncs. Lower = more stable targets. Divide by `num_envs` when scaling. | 10–1000 |
| `critic_tau` | Soft update blend: `target = τ×Q + (1-τ)×target`. Smaller = smoother but slower target tracking. | 0.005–0.1 |
| `replay_mem` | Buffer size. Must hold enough diverse experience to decorrelate batches. | 100k–1M |
| `num_envs` | Parallel envs. Multiplies data throughput but requires scaling other params. | 1–128 |
| `gamma` | Discount factor. 0.99 = standard for Atari. | 0.95–0.999 |
| `init_temp` (α) | Policy softmax temperature. 0.01 = near-greedy. | 0.01–0.1 |
| `eval_interval` | How often to run evaluation. Higher = less overhead. With `num_envs>1`, this counts learn-steps not env interactions. | 5000–25000 |

**When using `num_envs > 1`, scale these together:**
```
critic_target_update_frequency: 100 / num_envs   (e.g. 100 → 10 for 10 envs)
updates_per_step: 8 / num_envs                    (e.g. 8 → 1 for 10 envs)
eval_interval: 5000 / num_envs                    (e.g. 5000 → 500 for 10 envs)
```

---

## 6. Common Operations

All commands must be run from the `iq_learn/` directory with the conda env active:

```bash
cd /path/to/IQ-Learn-cb/iq_learn
conda activate iq_learn
```

---

### 6.1 Generate Expert Data

Uses the pretrained A2C agent from rl-baselines3-zoo to generate expert trajectories.
Only trajectories with reward > 0 are saved.

```bash
python expert_generation.py \
    --env BreakoutNoFrameskip-v4 \
    --episodes 20 \
    --eval.threshold 100
```

Output: `experts/BreakoutNoFrameskip-v4_20.pkl`

To inspect the generated data:

```bash
python inspect_expert.py
```

---

### 6.2 Start Training (Fresh Run)

```bash
python train_iq.py env=breakout
```

Hydra creates a timestamped output directory automatically:
```
outputs/YYYY-MM-DD/HH-MM-SS/
```

**Override any config value on the command line:**

```bash
# Change number of parallel envs and batch size
python train_iq.py env=breakout num_envs=8 train.batch=128

# Change learn steps
python train_iq.py env=breakout env.learn_steps=500000

# Change updates per step
python train_iq.py env=breakout train.updates_per_step=4
```

---

### 6.3 Resume an Interrupted Run

A full checkpoint (`checkpoint.pt`) is saved automatically every few episodes,
storing: Q-network, target network, optimizer state, learn_steps, epoch, best eval score.

```bash
python train_iq.py env=breakout \
    resume=outputs/YYYY-MM-DD/HH-MM-SS/checkpoint/checkpoint.pt
```

This picks up exactly where training stopped — same learn_steps, same optimizer momentum.

**Note:** The online replay buffer is not saved (too large). It refills in ~30 seconds
before learning resumes.

---

### 6.4 Warm-Start from Best Checkpoint (weights only)

Use this to initialise training from a previously learned policy without restoring
full training state. Useful when changing hyperparameters between runs.

```bash
python train_iq.py env=breakout \
    pretrain=outputs/YYYY-MM-DD/HH-MM-SS/results_best/softq_iq_BreakoutNoFrameskip-v4
```

---

### 6.5 Run in Background (long training sessions)

```bash
nohup python train_iq.py env=breakout > train.log 2>&1 &
echo "PID: $!"
```

Monitor output:
```bash
tail -f train.log
```

Check if still running:
```bash
ps aux | grep train_iq | grep -v grep
```

Check GPU utilisation:
```bash
watch -n 2 nvidia-smi
```

---

### 6.6 Monitor Training with TensorBoard

View all runs together (recommended — lets you compare):
```bash
tensorboard --logdir outputs/
```

View a single run:
```bash
tensorboard --logdir outputs/YYYY-MM-DD/HH-MM-SS/logs/
```

Open browser at: **http://localhost:6006**

**Key metrics to watch:**

| Metric | Healthy sign | Warning sign |
|--------|-------------|--------------|
| `eval/episode_reward` | Trending upward | Flat or collapsing |
| `train/critic_loss` | Small and stable (~0.0001) | Sudden 10–100× spike = divergence |
| `train/episode_reward` | Gradually increasing | Stuck at 0 for >50k steps |
| `train/duration` | Episodes getting longer | Getting shorter = policy dying faster |

---

### 6.7 Read Training Logs Directly

Training CSV (one row per episode):
```bash
# Header
head -1 outputs/YYYY-MM-DD/HH-MM-SS/logs/train.csv

# Latest 20 episodes
tail -20 outputs/YYYY-MM-DD/HH-MM-SS/logs/train.csv

# Columns: critic_loss, duration, episode, episode_reward, step
```

Eval CSV (one row per evaluation):
```bash
cat outputs/YYYY-MM-DD/HH-MM-SS/logs/eval.csv

# Columns: episode, episode_reward, step
```

---

### 6.8 Kill Training and Restart

```bash
# Find the process
ps aux | grep train_iq | grep -v grep

# Kill it
kill <PID>

# Restart (fresh)
python train_iq.py env=breakout

# Restart (resume)
python train_iq.py env=breakout resume=outputs/.../checkpoint/checkpoint.pt
```

---

## 7. Recommended Configurations by Hardware

### Single GPU, ~8 GB VRAM (e.g. RTX 3070 / 4060)

```yaml
# conf/env/breakout.yaml
train:
  batch: 64
  updates_per_step: 8
  use_target: True
  soft_update: True
agent:
  critic_target_update_frequency: 100

# conf/config.yaml
num_envs: 1
```

### Single GPU, 16+ GB VRAM (e.g. A100, H100, RTX 4090)

```yaml
# conf/env/breakout.yaml
train:
  batch: 256
  updates_per_step: 2
  use_target: True
  soft_update: True
agent:
  critic_target_update_frequency: 15   # 100 / num_envs (approx)

# conf/config.yaml
num_envs: 8
```

---

## 8. Diagnosing Common Problems

### Training diverges (CLOSS spikes suddenly)

**Symptoms:** `critic_loss` jumps 10–100× in one eval interval, `eval/episode_reward` collapses.

**Fixes:**
- Lower `critic_target_update_frequency` (e.g. 100 → 50)
- Lower `updates_per_step` (e.g. 8 → 4)
- If using `num_envs > 1`, scale the above parameters down proportionally

### Training is too slow (low GPU utilisation)

**Symptoms:** GPU utilisation <40%, CPU at 100%.

**Cause:** LazyFrames (frame-stacked Atari observations) are materialised on CPU
per batch. Large batch sizes cause CPU bottleneck, not GPU.

**Fixes:**
- Keep `batch: 64` (CPU data prep scales linearly with batch size)
- Increase `num_envs` instead of batch size to improve throughput
- Increase `updates_per_step` moderately (e.g. 4 → 8)

### Policy learns then forgets (plateau after initial improvement)

**Symptoms:** `eval/episode_reward` improves for ~50k steps then flatlines or drops.

**Fixes:**
- Reduce learning rate (`critic_lr: 0.00005`)
- Reduce `critic_tau` (e.g. 0.1 → 0.05) for smoother target updates
- Warm-start the next run from `results_best/` checkpoint

### `Learn begins!` never prints / steps stuck at 1

**Cause:** Online replay buffer hasn't filled to `initial_mem` yet.
With EpisodicLifeEnv and an untrained policy, episodes are very short.
This is normal — wait 1–2 minutes.

---

## 9. File Checklist for a New Machine

Copy these files/folders to the new machine:

```
IQ-Learn-cb/
├── iq_learn/
│   ├── experts/BreakoutNoFrameskip-v4_20.pkl      # Expert data
│   ├── rl-baselines3-zoo/                          # Pretrained SB3 agents
│   │   └── rl-trained-agents/a2c/
│   │       └── BreakoutNoFrameskip-v4_1/
│   ├── conf/                                       # All config files
│   ├── updated_requirements.txt
│   └── [all .py source files]
└── (optional) outputs/YYYY-MM-DD/.../              # Previous checkpoints
```

To transfer a checkpoint for warm-starting on the new machine:
```bash
scp outputs/YYYY-MM-DD/HH-MM-SS/results_best/softq_iq_BreakoutNoFrameskip-v4 \
    user@newmachine:/path/to/iq_learn/trained_policies/
```
