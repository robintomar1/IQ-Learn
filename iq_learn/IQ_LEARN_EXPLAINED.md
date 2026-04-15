# IQ-Learn: How It Works

## The Problem

You have recordings of an expert (e.g. a human playing Breakout) but **no reward function**. You want an agent that imitates the expert.

- **Behavior Cloning** just copies actions ("do what the expert did") but fails when the agent drifts to states the expert never visited.
- **GAIL** trains a discriminator to tell expert vs agent apart and uses that as a fake reward — but this is unstable adversarial training (like GANs).
- **IQ-Learn** learns a single Q-function that implicitly contains both the reward AND the policy. No adversarial training needed.

## What does a Q-function do?

Think of Q as a **scoring system**. For every situation (state) and every possible move (action), Q gives a score: "how good is it to do this move in this situation?"

```
State: Ball heading toward paddle        Q-values:

  ┌──────────────┐                        NOOP  → -2.0  (bad, you'll miss)
  │  ●            │                        LEFT  → +5.0  (good, move toward ball)
  │       ──▶     │                        RIGHT → -3.0  (terrible, wrong way)
  │          ▄▄▄  │                        FIRE  → -1.0  (useless right now)
  └──────────────┘
                                          Policy: pick LEFT (highest Q)
```

## How does normal RL learn Q?

The environment tells you the reward after every action. You use that to train Q:

```
You do action LEFT, get reward +1, land in new state s'

Target:  r + γ·V(s')  =  1.0 + 0.99 × 4.2  =  5.158
                          ───   ─────────────
                          env     future value
                          told    (from Q-network)
                          you
                          this

Loss: make Q(s, LEFT) close to 5.158
```

**The environment reward `r` is the teacher.** Without it, you can't train Q.

## The key insight: reward lives inside Q

In any RL method, the Bellman equation says:

```
Q(s, a) = r(s, a) + γ·V(s')
```

Rearranging: `r(s, a) = Q(s, a) - γ·V(s')`. The reward is hiding inside Q. If we learn the right Q-function, we get the reward for free — we never need the environment to tell us.

But how do we learn the "right" Q? We use the expert demos as a signal: **whatever the expert did should have HIGH implied reward, whatever random/bad actions have should have LOW implied reward.**

## The three loss terms

IQ-Learn trains Q with three terms that work like a physical system:

**Term 1 — "Fill the expert pipes"**: Push `Q(s,a) - γV(s')` UP for expert transitions. This makes Q assign high value to what the expert did.

**Term 2 — "Don't overflow everywhere"**: Without Term 1 being checked, Q would go to infinity. This term enforces `V(s) ≈ γ·V(s')` across all data (expert + agent) — a Bellman consistency constraint that acts as a drain keeping values physically consistent.

**Term 3 — "Pressure relief valve"**: Penalizes `(implied_reward)²` to prevent extreme values. Controlled by α (alpha). Like a relief valve: if pressure gets too high, this pushes back.

The three terms find an **equilibrium** — a Q-function where expert actions are valued highest, but everything stays bounded and self-consistent.

## The ~15 lines of code

Starting from a working Soft-Q agent, IQ-Learn only changes the loss computation. In `train_iq.py` (lines 376-395):

```python
# Standard RL loss (what already exists in agent/softq.py):
#   loss = (Q(s,a) - [r + γ·V_target(s')])²     ← needs environment reward r
#
# IQ-Learn replaces it with:

current_Q = self.critic(obs, action)                          # Q(s,a) for all data
y = (1 - done) * self.gamma * self.get_targetV(next_obs)      # γ·V(s') — NO reward added

reward = (current_Q - y)[is_expert]       # implied reward, expert transitions only
loss  = -(reward).mean()                  # TERM 1: maximize expert reward

value_loss = (self.getV(obs) - y).mean()
loss += value_loss                        # TERM 2: Bellman regularizer (all data)

chi2_loss = 1/(4 * alpha) * (reward**2).mean()
loss += chi2_loss                         # TERM 3: χ² divergence penalty
```

Everything else stays identical — same Q-network, same target network, same action selection, same replay buffer. The full version with multiple divergence options is in `iq.py`.

## Works with any RL method

IQ-Learn only touches the **critic loss**. The rest of the RL method is untouched:

| Component | Standard RL | IQ-Learn |
|---|---|---|
| Q-network | same | same |
| Target network | same | same |
| Action selection | same | same |
| Actor (SAC only) | same | same |
| Replay buffer | same | same + expert buffer |
| **Critic loss** | `(Q - [r+γV'])²` | `-expert_reward + regularizers` |

For **Soft-Q** (discrete actions, e.g. Breakout): no actor, policy comes from `softmax(Q/α)`. Replace critic loss, done.

For **SAC** (continuous actions, e.g. Humanoid): separate actor network. Replace critic loss. Actor loss (`maximize Q(s, π(s))`) stays untouched — it follows whatever Q says is good, and Q now encodes expert-like behavior.

## Data flow in this codebase

```
train_iq.py main()
     │
     │  env.step(action)  ──────────▶  online_memory_replay  (agent's own experience)
     │
     │  (expert data pre-loaded)  ──▶  expert_memory_replay  (loaded from .pkl at startup)
     │
     │  Every step:
     ▼
iq_update()                                        [train_iq.py]
     │
     ├── Sample from online_memory_replay  ──▶ policy_batch
     ├── Sample from expert_memory_replay  ──▶ expert_batch
     │
     ▼
iq_update_critic()                                 [train_iq.py]
     │
     ├── get_concat_samples()                      [utils/utils.py]
     │   Merge batches + create is_expert flag
     │
     ├── Compute current_V, next_V
     │
     ▼
iq_loss()                                          [iq.py]
     │
     ├── Term 1: -mean(implied_reward[expert])
     ├── Term 2: value_loss (Bellman regularizer)
     ├── Term 3: χ² regularizer (if enabled)
     │
     ▼
critic_optimizer.step()                            Updates Q-network
     │
     ▼
soft_update(Q, Q_target)                           Slowly update target
```

## Training Modes: Offline vs Online

There are three ways to run IQ-Learn training, depending on whether the agent interacts with the environment during learning:

| | Offline | Online (1 env) | Online (4 envs) |
|---|---|---|---|
| **What it does** | Learns purely from expert demos | Learns from demos + collects its own experience | Same, but 4 parallel envs via envpool |
| **Env interaction** | None | Yes | Yes |
| **Command** | `python train_iq.py env=breakout offline=True` | `python train_iq.py env=breakout` | `python train_iq.py env=breakout num_envs=4` |

### How offline training works

In offline mode, there is **no environment at all**. Every training step:

1. Sample batch A from expert buffer
2. Sample batch B from expert buffer (independently)
3. Concatenate them — half gets labeled "policy", half gets labeled "expert"
4. Compute IQ loss and update Q-network
5. Repeat

This works because the `value_expert` loss (set in `conf/method/iq.yaml`) computes all three loss terms using **only expert data**:

- Term 1: implied reward on expert transitions → expert-only
- Term 2: Bellman regularizer on expert states → expert-only
- Term 3: chi-squared penalty on expert transitions → expert-only

No online data needed.

### How online training works

In online mode, the agent also plays the game during training. Every step:

1. Agent takes an action in the environment (softmax sampling = exploration)
2. The transition goes into the online replay buffer
3. Sample batch A from **online buffer**, batch B from **expert buffer**
4. Concatenate them, compute IQ loss, update Q-network

The online data gives the value function V(s) a wider range of states to be defined over. But with `value_expert` loss, the actual reward-learning terms still only use expert data.

### What does online data actually contribute?

Not much, when the expert dataset is large enough. Here's why:

- The loss terms that recover the reward all filter to `[is_expert]` — online data is masked out
- The online data only indirectly affects V(s) computation over the full concatenated batch
- With 60 expert trajectories (~93K transitions) in Breakout, the expert data already covers the important states

Think of it this way: the expert demos are the **textbook**. Online data is **extra practice problems**. If the textbook is thorough enough, the practice problems don't add much.

## Breakout Training Results (actual runs)

### Speed to learn

How many gradient steps to reach 250+ eval reward:

```
Online 4 envs:  45K steps   ← fastest (fills replay buffer 4x quicker, starts learning sooner)
Offline:        50K steps   ← no warmup delay, clean gradients from expert data
Online 1 env:   60K steps   ← slowest (wastes ~5K steps filling replay buffer before learning)
```

### Wall-clock time

```
Offline:        ~3.1 hours  ← 2x faster (no Atari frame processing overhead)
Online 1 env:   ~6.4 hours
Online 4 envs:  ~6.4 hours  (more steps done, but env overhead eats the speedup)
```

### Final quality (mean of last 10 evals)

```
Online 1 env:   326.2
Offline:        324.9
Online 4 envs:  319.3
```

### Peak eval reward

```
Online 4 envs:  352.6
Offline:        351.7
Online 1 env:   346.6
```

### Critic loss over time

All three follow the same curve:

```
Step     Offline    1 env      4 envs
500      0.117      0.120      0.117
10K      0.066      0.086      ~0.06
50K      0.010      0.010      ~0.007
100K     0.005      0.004      0.005
200K     0.003      0.003      0.002
```

### What this tells us

**All three converge to the same performance (~320-350 eval reward).** The Q-network learns the same thing regardless of training mode. The differences are purely about speed and efficiency.

## Why online rewards looked broken

During training, the online agent scored 1-4 points per Breakout game while eval showed 300+. This looks like a bug but it's not:

- **Online play**: uses softmax sampling (semi-random exploration). In Breakout, even slightly random play is terrible — you need precise paddle positioning
- **Eval play**: uses argmax (always picks the best action). The learned Q-values produce excellent play
- **It doesn't matter**: the online rewards are never used in the loss. The agent learns from expert data, not from its own score

The online agent's job is just to visit different states and fill the replay buffer. Playing well is irrelevant.

## When to use which mode

**Use offline when:**
- You have enough expert demos (10+ trajectories with good coverage)
- You want the fastest training
- You're using `value_expert` loss
- You don't need the agent to handle states outside the expert distribution

**Use online when:**
- You have very few expert demos (1-5) and need the agent to explore for coverage
- You're using `value` loss (needs policy state distribution)
- You want the agent to generalize beyond what the expert showed
- The environment might change over time

## Loss Variants: Why Are There So Many?

Looking at `iq.py`, you'll see a surprising number of knobs: `method.loss`, `method.div`, `method.chi`, `method.regularize`, `method.grad_pen`. This feels like overkill for what should be a simple imitation algorithm. Here's what each one actually controls and when to use which.

### The loss is always three terms

No matter how you configure it, the loss is always the same three-term structure from earlier:

```
loss = Term1 (make expert look valuable)
     + Term2 (Bellman regularizer - prevent Q from blowing up)
     + Term3 (chi-squared penalty - optional, extra stability)
```

The config knobs change **how each term is computed** and **which data it uses**.

### Knob 1: `method.loss` — how to compute Term 2

Term 2 is the Bellman regularizer `E[V(s) - gamma*V(s')]`. This averages over some distribution of states. There are three choices for what distribution:

| Value | Formula | Data needed | When to use |
|---|---|---|---|
| `value_expert` | `E_expert[V(s) - gamma*V(s')]` | Expert only | **Default for discrete (Soft-Q) and offline.** Most stable. |
| `value` | `E_all[V(s) - gamma*V(s')]` | Expert + policy | Continuous (SAC) with good online exploration |
| `v0` | `(1-gamma)*V(s_0)` on initial states | Just initial states | Continuous (SAC), when policy data is noisy |

**Why does this matter?** The Bellman regularizer tries to prevent Q-values from growing unbounded. The "distribution of states" you average over shapes the Q-function:

- `value_expert` → Q is well-calibrated on states the expert visits. Great for imitation, works offline.
- `value` → Q is well-calibrated everywhere the agent goes. Needs online data but handles distribution shift better.
- `v0` → Only constrains V at the start of episodes. Lower variance than `value` but weaker regularization.

**From the codebase scripts:**
- All Atari tasks (`run_atari.sh`): `value_expert`
- All offline discrete tasks (`run_offline.sh`): `value_expert`
- MuJoCo cheetah/ant (continuous, stable): `value`
- MuJoCo hopper/walker/humanoid (continuous, harder): `v0`

### Knob 2: `method.div` — how to shape Term 1

Term 1 pushes `Q(s,a) - gamma*V(s')` up for expert transitions. The `div` setting determines **how aggressively** to push up, by multiplying by a weight `phi_grad`:

```python
loss = -(phi_grad * reward).mean()
```

Different values of `phi_grad` correspond to different f-divergences between expert and policy distributions:

| `div` | phi_grad | Divergence | Notes |
|---|---|---|---|
| (none) or `chi` | 1 | Chi-squared | **Default.** Simple, stable. |
| `kl` | `exp(-reward-1)` | KL (original) | Biased dual form — theoretically suboptimal |
| `kl_fix` | `exp(-reward)` | KL (corrected) | Unbiased, from the paper |
| `kl2` | `softmax(-reward)*N` | KL (batch-normalized) | Numerically stabler than `kl_fix` |
| `js` | `exp(-reward)/(2-exp(-reward))` | Jensen-Shannon | GAN-like |
| `hellinger` | `1/(1+reward)^2` | Hellinger | Rarely used |

**Which to pick?** Chi-squared (the default, `phi_grad=1`) is boringly effective. The KL variants can converge faster in theory but are prone to numerical instability (exponentials of rewards can explode). **In practice, almost every working config in this repo uses chi-squared** — the other divergences are mostly research options.

### Knob 3: `method.chi` (or `div="chi"`) — add Term 3 on expert data

When `method.chi: True`, a third term gets added:

```python
chi2_loss = 1/(4 * alpha) * (Q - gamma*V')**2  on expert transitions only
```

This is a chi-squared divergence penalty on expert transitions. It:
- Bounds the implied reward (prevents it from growing without limit)
- Adds a smooth, convex penalty that stabilizes training
- **Works offline** (uses only expert data)

`alpha` controls strength — smaller `alpha` = stronger penalty. For Breakout we use `alpha=0.1`.

### Knob 4: `method.regularize` — add Term 3 on all data

Same formula as `chi`, but applied to **all transitions (expert + policy)** instead of just expert:

```python
chi2_loss = 1/(4 * alpha) * (Q - gamma*V')**2  on all transitions
```

This provides stronger regularization because it covers a wider distribution of states. But it **requires online data** — you can't use this offline.

**`chi` vs `regularize`:**
- `chi`: expert-only → works offline, weaker regularization
- `regularize`: all data → requires online, stronger regularization, better for continuous control

### Knob 5: `method.grad_pen` — Wasserstein gradient penalty

Optional. Adds a WGAN-GP-style gradient penalty that pushes the gradient norm of Q toward 1 when interpolating between expert and policy states:

```python
gp_loss = lambda_gp * (||grad Q(interpolated)||_2 - 1)**2
```

This is rarely used. It adds compute cost (extra gradient computation per step) and is mostly a research artifact. None of the default configs enable it.

## The Recipes That Actually Work

Looking at what the paper authors use in their scripts, there are really only three combinations that matter:

### Recipe 1: Discrete actions (Atari, Grid worlds)

```yaml
method:
  loss: value_expert
  chi: True
  alpha: 0.1  # or 0.5
```

Why: Discrete action spaces work well with soft Q-learning. The expert-only losses keep everything offline-compatible. Chi-squared is the stable default.

Used for: Breakout, Pong, Qbert, CartPole, Acrobot, LunarLander.

### Recipe 2: Continuous control, "easy" tasks (HalfCheetah, Ant)

```yaml
method:
  loss: value
  regularize: True
  alpha: 0.5
```

Why: Continuous control uses SAC (with an actor network). `value` loss uses both expert and policy states, which works when exploration is well-behaved. `regularize` provides strong stabilization.

Used for: Cheetah, Ant.

### Recipe 3: Continuous control, "hard" tasks (Hopper, Walker, Humanoid)

```yaml
method:
  loss: v0
  regularize: True
  alpha: 0.5
```

Why: In hard continuous tasks, the policy often goes off-distribution early, making `value` loss noisy. Using only initial-state values (`v0`) reduces variance in Term 2. `regularize` still keeps things bounded.

Used for: Hopper, Walker, Humanoid.

## Decision Tree: Which Config Should I Use?

```
Is your action space discrete? (Atari, grid worlds)
│
├── YES → Use Recipe 1 (value_expert + chi)
│         Works for both offline and online. Start here.
│
└── NO (continuous control)
    │
    ├── Can you do online training?
    │   │
    │   ├── NO → Use value_expert + chi
    │   │       (will work but may be suboptimal for continuous)
    │   │
    │   └── YES → Is the task "easy" (cheetah, ant)?
    │       │
    │       ├── YES → Use Recipe 2 (value + regularize)
    │       │
    │       └── NO  → Use Recipe 3 (v0 + regularize)
```

## Summary Table

| Setting | What it does | Offline-safe? | When |
|---|---|---|---|
| `loss=value_expert` | Term 2 on expert states | YES | Default, discrete tasks |
| `loss=value` | Term 2 on all states | No | Continuous, easy tasks |
| `loss=v0` | Term 2 on initial states only | YES | Continuous, hard tasks |
| `div=chi` (default) | Term 1 with phi_grad=1 | YES | Default, stable |
| `div=kl_fix` | Term 1 with KL weights | YES | Research, faster in theory |
| `chi=True` | Term 3 on expert data | YES | Discrete tasks |
| `regularize=True` | Term 3 on all data | No | Continuous tasks |
| `grad_pen=True` | Optional WGAN-GP penalty | No | Rarely used |

**TL;DR**: For Breakout (and most discrete imitation tasks), stick with `loss=value_expert, chi=True`. It's what every working Atari config in the paper uses, and it's the only combination that works both online and offline.
