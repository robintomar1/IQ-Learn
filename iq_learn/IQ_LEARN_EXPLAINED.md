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
