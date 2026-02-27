# IQ-Learn Breakout Investigation Summary

Date: 2026-02-23

## What I Looked At

- Training runs in `outputs/2026-02-21`, `outputs/2026-02-22`, `outputs/2026-02-23`
- Hydra configs saved per run under `.hydra/config.yaml` and `.hydra/overrides.yaml`
- Expert datasets in `experts/`
- Core training code: `train_iq.py`, `agent/softq.py`, `make_envs.py`, `iq.py`, `dataset/memory.py`, `dataset/expert_dataset.py`
- Repo docs: `SETUP.md` and related references
- IQ-Learn paper PDF at `../docs/NeurIPS-2021-iq-learn-inverse-soft-q-learning-for-imitation-Paper.pdf` (text extraction via `pdftotext` showed high-level guidance, no detailed hyperparameter appendix visible in extracted text)

## What the Recent Runs Showed (Feb 21–23)

Summary of best rewards from CSV logs:

- 2026-02-21: best train max ≈ 12, best eval max ≈ 8.7
- 2026-02-22: best eval max ≈ 9.1, most runs far below
- 2026-02-23: best eval max ≈ 4.1

The rewards plateau well below 10 in most runs.

## What the Expert Data Shows

Using `inspect_expert.py`:

- `experts/BreakoutNoFrameskip-v4_20.pkl`
  - Mean reward ≈ 47.1, max 69
  - **Exactly one `done` per trajectory** (episode ends at game over)
- `experts/BreakoutNoFrameskip-v4_20_trunc1000.pkl`
  - Mean reward ≈ 33.1, max 41
- `experts/BreakoutNoFrameskip-v4_6good.pkl`
  - Mean reward = 26 (all episodes 26)
- `experts/BreakoutNoFrameskip-v4_30_filtered.pkl`
  - Mean reward ≈ 63.2, max 68

## Key Code Findings

### 1) Episode termination mismatch (most important)

- **Training/eval environment** uses `AtariWrapper` from SB3 in `make_envs.py`.
- Default `AtariWrapper` behavior is `terminal_on_life_loss=True` (episodic life).
- **Expert demos** end only at game over (one `done` per trajectory).

**Conclusion:** the expert data and training/eval have different episode termination definitions. This creates a Bellman target mismatch (done flags) between expert and policy data, which can severely degrade learning.

### 2) Evaluation was stochastic

`SoftQ.choose_action(..., sample=False)` still sampled from the policy distribution. This makes eval noisy and artificially low.

### 3) Config drift from recommended Atari settings

Recent runs had:

- `method.chi=True` and `method.loss=v0` or `value`
- Low-quality / truncated expert demos
- Large `num_envs` with low updates per step

Repo docs (`SETUP.md`) recommend for Atari:

- `method.loss=value_expert`
- `method.chi=False`
- `num_envs=1` (or scale updates/targets down carefully)

## Conclusions

1. **Episode termination mismatch (life-loss vs game-over) is the most likely root cause** of low rewards, because IQ-Learn is sensitive to `done` signals when computing Bellman targets.
2. Stochastic evaluation masks progress and makes rewards look worse.
3. The recent configs deviate from the repo’s own Atari recommendations, and many runs used lower-quality expert data.

## Changes Implemented (Code + Config)

### A) Align episode termination between expert and training

**Reasoning:** Ensure expert demos and training share the same definition of episode end.

Changes:

- Added config flags:
  - `env.atari_terminal_on_life_loss` (default False)
  - `env.atari_clip_reward` (default False)
- Updated `make_envs.py`:
  - Pass flags to `AtariWrapper`.
  - Pass flags to `envpool` creation.
- Updated `train_iq.py`:
  - Pass these flags when creating envpool.
- Updated `expert_generation.py`:
  - Only override `done` by lives if `atari_terminal_on_life_loss=True`.

### B) Deterministic evaluation

**Reasoning:** `sample=False` should mean greedy argmax, not stochastic.

Change:

- `agent/softq.py`: `choose_action(..., sample=False)` now returns `argmax`.

### C) Explicit Breakout config

**Reasoning:** Make the new Atari flags explicit for Breakout.

Change:

- `conf/env/breakout.yaml` now includes:
  - `atari_terminal_on_life_loss: False`
  - `atari_clip_reward: False`

### D) (Operational) Disable TrackIO/TensorBoard for local smoke runs

**Reasoning:** TrackIO launched a Gradio server and failed to find ports. TensorBoardX failed due to permission issues with multiprocessing semaphores.

Changes:

- `train_iq.py`:
  - If `DISABLE_TRACKIO=1`, skip `wandb.init(...)`.
  - If `DISABLE_TB=1`, skip `SummaryWriter` and guard `writer.add_scalar` calls.

This keeps the run from crashing in constrained environments.

## Notes on the Paper

The paper mentions that IQ-Learn on Atari uses Soft DQN-style defaults and is sensitive to entropy temperature. The extracted PDF text did **not** show full hyperparameter tables or Appendix D details (likely a limitation of text extraction). The repo’s `SETUP.md` provides practical hyperparameter guidance aligned with Atari experiments.

## Smoke Run Status

- First attempts failed due to TrackIO (Gradio port) and TensorBoardX (multiprocessing permission) issues.
- After disabling TrackIO/TB, training started successfully but was CPU-only in this environment (CUDA not available). The run was stopped per request.

## Command to Run on GPU Box

```bash
DISABLE_TRACKIO=1 DISABLE_TB=1 python train_iq.py env=breakout env.learn_steps=20000 env.eval_interval=2000
```

## File Changes Summary

Modified:

- `make_envs.py`
- `train_iq.py`
- `expert_generation.py`
- `agent/softq.py`
- `conf/config.yaml`
- `conf/env/breakout.yaml`

## Next Steps

1. Run the smoke test on the GPU machine using the command above.
2. Confirm episode lengths are longer (full-game episodes, not life-loss).
3. If rewards still stagnate below ~10, revisit expert demo quality or re-generate experts with the updated wrapper.

