"""
Copyright 2022 Div Garg. All rights reserved.

Example training code for IQ-Learn which minimially modifies `train_rl.py`.
"""

import datetime
import os
import random
import time
from collections import deque
from itertools import count
import types

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import trackio as wandb
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from wrappers.atari_wrapper import LazyFrames
from make_envs import make_env, EnvFactory, make_envpool_atari
from dataset.memory import Memory, NumpyReplayBuffer
from agent import make_agent
from utils.utils import eval_mode, average_dicts, get_concat_samples, evaluate, soft_update, hard_update
from utils.logger import Logger
from iq import iq_loss

torch.set_num_threads(min(24, os.cpu_count() or 8))

CHECKPOINT_FILE = "checkpoint/checkpoint.pt"


def save_checkpoint(path, agent, scaler, epoch, steps, learn_steps, best_eval_returns):
    """Save full training state so training can be resumed exactly."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'q_net': agent.q_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.critic_optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'steps': steps,
        'learn_steps': learn_steps,
        'best_eval_returns': best_eval_returns,
    }, path)


def load_checkpoint(path, agent, scaler):
    """Load full training state from a checkpoint. Returns (epoch, steps, learn_steps, best_eval_returns)."""
    print(f'=> Resuming from checkpoint: {path}')
    ckpt = torch.load(path, map_location=agent.device, weights_only=False)
    agent.q_net.load_state_dict(ckpt['q_net'])
    agent.target_net.load_state_dict(ckpt['target_net'])
    agent.critic_optimizer.load_state_dict(ckpt['optimizer'])
    if scaler is not None and ckpt.get('scaler') is not None:
        scaler.load_state_dict(ckpt['scaler'])
    print(f'   epoch={ckpt["epoch"]}  steps={ckpt["steps"]}  '
          f'learn_steps={ckpt["learn_steps"]}  best_eval={ckpt["best_eval_returns"]:.2f}')
    return ckpt['epoch'], ckpt['steps'], ckpt['learn_steps'], ckpt['best_eval_returns']


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    if os.environ.get("DISABLE_TRACKIO", "0") != "1":
        wandb.init(project=args.project_name) #entity='iq-learn',
                  # sync_tensorboard=True, reinit=True, config=args)

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env_args = args.env
    env = make_env(args)
    eval_env = make_env(args)

    # Seed envs
    # env.seed(args.seed)
    # eval_env.seed(args.seed + 10)

    REPLAY_MEMORY = int(env_args.replay_mem)
    INITIAL_MEMORY = int(env_args.initial_mem)
    EPISODE_STEPS = int(env_args.eps_steps)
    EPISODE_WINDOW = int(env_args.eps_window)
    LEARN_STEPS = int(env_args.learn_steps)
    INITIAL_STATES = 128  # Num initial states to use to calculate value of initial state distribution s_0

    agent = make_agent(env, args)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    agent.scaler = scaler
    agent.use_amp = use_amp
    if use_amp:
        print('AMP (mixed precision) enabled')

    if args.pretrain:
        pretrain_path = hydra.utils.to_absolute_path(args.pretrain)
        if os.path.isfile(pretrain_path):
            print("=> loading pretrain '{}'".format(args.pretrain))
            agent.load(pretrain_path)
        else:
            print("[Attention]: Did not find checkpoint {}".format(args.pretrain))

    steps = 0
    learn_steps = 0
    begin_learn = False
    best_eval_returns = -np.inf
    start_epoch = 0

    if args.resume:
        resume_path = hydra.utils.to_absolute_path(args.resume)
        if os.path.isfile(resume_path):
            start_epoch, steps, learn_steps, best_eval_returns = load_checkpoint(
                resume_path, agent, scaler)
            begin_learn = learn_steps > 0
        else:
            print(f'[Warning] Resume checkpoint not found: {resume_path}')

    # Use NumpyReplayBuffer for multi-env path: vectorized sampling, no Python
    # deque iteration, uint8 storage + GPU-side float32 cast for Atari obs.
    # Both expert and online buffers use it to eliminate the np.array(list_of_arrays)
    # bottleneck in get_samples which was ~150ms for batch=2048 with a deque.
    num_envs = getattr(args, 'num_envs', 1)
    BufferCls = NumpyReplayBuffer if num_envs > 1 else Memory

    # Load expert data
    expert_memory_replay = BufferCls(REPLAY_MEMORY//2, args.seed)
    expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'),
                              num_trajs=args.expert.demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')

    online_memory_replay = BufferCls(REPLAY_MEMORY//2, args.seed+1)

    # Setup logging
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.env.name, args.exp_name, ts_str)
    writer = None
    if os.environ.get("DISABLE_TB", "0") != "1":
        writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')
    logger = Logger(args.log_dir,
                    log_frequency=args.log_interval,
                    writer=writer,
                    save_tb=writer is None,
                    agent=args.agent.name,
                    wandb=wandb)

    # track mean reward and scores
    scores_window = deque(maxlen=EPISODE_WINDOW)  # last N scores
    rewards_window = deque(maxlen=EPISODE_WINDOW)  # last N rewards

    # Patch iq_update methods once (shared by both training paths)
    agent.iq_update = types.MethodType(iq_update, agent)
    agent.iq_update_critic = types.MethodType(iq_update_critic, agent)
    n_updates = getattr(args.train, 'updates_per_step', 1) or 1

    if num_envs > 1:
        # ── Vectorized training loop ──────────────────────────────────────────
        vec_env = make_envpool_atari(
            args.env.name,
            num_envs,
            seed=args.seed,
            terminal_on_life_loss=getattr(args.env, "atari_terminal_on_life_loss", True),
            clip_reward=getattr(args.env, "atari_clip_reward", True),
        )
        print(f'Using {num_envs} parallel environments (envpool)')

        states, _ = vec_env.reset()
        ep_rewards = np.zeros(num_envs)
        epoch = start_epoch

        while True:
            if steps < args.num_seed_steps:
                actions = np.array([vec_env.single_action_space.sample()
                                    for _ in range(num_envs)])
            else:
                with eval_mode(agent):
                    actions = agent.choose_action_batch(states)

            next_states, rewards, terminated, truncated, infos = vec_env.step(actions.astype(np.int32))

            # Vectorised insert — one Python call for all num_envs transitions
            online_memory_replay.add_batch(
                states, next_states, actions, rewards, terminated)

            ep_rewards += rewards
            steps += num_envs

            # Track completed episodes — reward bookkeeping only, no I/O
            # Logging and checkpointing are rate-limited in the learn block below
            for i in range(num_envs):
                if terminated[i] or truncated[i]:
                    rewards_window.append(ep_rewards[i])
                    ep_rewards[i] = 0.0
                    epoch += 1

            states = next_states

            # Eval
            if learn_steps % args.env.eval_interval == 0:
                eval_returns, eval_timesteps = evaluate(
                    agent, eval_env, num_episodes=args.eval.eps)
                returns = np.mean(eval_returns)
                learn_steps += 1  # prevent repeated eval at step 0
                logger.log('eval/episode_reward', returns, learn_steps)
                logger.log('eval/episode', epoch, learn_steps)
                logger.dump(learn_steps, ty='eval')
                print(f'  [EVAL] learn_steps={learn_steps} mean_return={returns:.2f}')
                if returns > best_eval_returns:
                    best_eval_returns = returns
                    save(agent, epoch, args, output_dir='results_best')

            # Learn
            if online_memory_replay.size() > INITIAL_MEMORY:
                if not begin_learn:
                    print('Learn begins!')
                    begin_learn = True

                learn_steps += 1
                if learn_steps >= LEARN_STEPS:
                    print('Finished!')
                    wandb.finish()
                    vec_env.close()
                    break

                for _ in range(n_updates):
                    losses = agent.iq_update(online_memory_replay,
                                             expert_memory_replay, logger, learn_steps)

                if learn_steps % args.log_interval == 0:
                    if writer is not None:
                        for key, loss in losses.items():
                            writer.add_scalar(key, loss, global_step=learn_steps)
                    if rewards_window:
                        logger.log('train/episode', epoch, learn_steps)
                        logger.log('train/episode_reward', np.mean(rewards_window), learn_steps)
                        logger.log('train/duration', 0, learn_steps)
                        logger.dump(learn_steps, save=True)
                        print(f'  [Step {learn_steps}] mean_ep_reward='
                              f'{np.mean(rewards_window):.1f} epochs={epoch}')

                if learn_steps % args.checkpoint_interval == 0:
                    save(agent, 0, args, output_dir='results')
                    save_checkpoint(CHECKPOINT_FILE, agent, scaler, epoch,
                                    steps, learn_steps, best_eval_returns)

    else:
        # ── Single-env episode-based loop (original) ─────────────────────────
        episode_reward = 0
        for epoch in count(start_epoch):
            state, info = env.reset()
            episode_reward = 0
            done = False

            start_time = time.time()
            for episode_step in range(EPISODE_STEPS):
                if steps < args.num_seed_steps:
                    action = env.action_space.sample()
                else:
                    with eval_mode(agent):
                        action = agent.choose_action(state, sample=True)
                next_state, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated
                if done:
                    next_state, _ = env.reset()
                episode_reward += reward
                steps += 1

                if learn_steps % args.env.eval_interval == 0:
                    eval_returns, eval_timesteps = evaluate(
                        agent, eval_env, num_episodes=args.eval.eps)
                    returns = np.mean(eval_returns)
                    learn_steps += 1  # To prevent repeated eval at timestep 0
                    logger.log('eval/episode_reward', returns, learn_steps)
                    logger.log('eval/episode', epoch, learn_steps)
                    logger.dump(learn_steps, ty='eval')
                    print(f'  [EVAL] learn_steps={learn_steps} mean_return={returns:.2f}')
                    if returns > best_eval_returns:
                        best_eval_returns = returns
                        save(agent, epoch, args, output_dir='results_best')
                        save_checkpoint(CHECKPOINT_FILE, agent, scaler, epoch + 1,
                                        steps, learn_steps, best_eval_returns)

                # only store done true when episode finishes without hitting timelimit
                done_no_lim = done
                if (str(env.__class__.__name__).find('TimeLimit') >= 0
                        and episode_step + 1 == env._max_episode_steps):
                    done_no_lim = 0
                online_memory_replay.add((state, next_state, action, reward, done_no_lim))

                if online_memory_replay.size() > INITIAL_MEMORY:
                    if begin_learn is False:
                        print('Learn begins!')
                        begin_learn = True

                    learn_steps += 1
                    if learn_steps == LEARN_STEPS:
                        print('Finished!')
                        wandb.finish()
                        return

                    for _ in range(n_updates):
                        losses = agent.iq_update(online_memory_replay,
                                                 expert_memory_replay, logger, learn_steps)

                    if learn_steps % args.log_interval == 0:
                        if writer is not None:
                            for key, loss in losses.items():
                                writer.add_scalar(key, loss, global_step=learn_steps)

                if done:
                    break
                state = next_state

            rewards_window.append(episode_reward)
            logger.log('train/episode', epoch, learn_steps)
            logger.log('train/episode_reward', episode_reward, learn_steps)
            logger.log('train/duration', time.time() - start_time, learn_steps)
            logger.dump(learn_steps, save=True)
            if (epoch + 1) % 10 == 0 or epoch < 3:
                print(f'  [Ep {epoch}] reward={episode_reward:.1f} learn_steps={learn_steps}')
            save(agent, epoch, args, output_dir='results')
            save_checkpoint(CHECKPOINT_FILE, agent, scaler, epoch + 1,
                            steps, learn_steps, best_eval_returns)

    save_path = "q_network_final.pth"
    torch.save(agent.q_net.state_dict(), save_path)
    print(f"Q-function saved to {save_path}")

def save(agent, epoch, args, output_dir='results'):
    # Always save for results_best; use save_interval for periodic results
    if output_dir == 'results_best' or epoch % args.save_interval == 0:
        if args.method.type == "sqil":
            name = f'sqil_{args.env.name}'
        else:
            name = f'iq_{args.env.name}'

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        agent.save(f'{output_dir}/{args.agent.name}_{name}')


# Minimal IQ-Learn objective
def iq_learn_update(self, policy_batch, expert_batch, logger, step):
    args = self.args
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

    if args.only_expert_states:
        expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done

    obs, next_obs, action, reward, done, is_expert = get_concat_samples(
        policy_batch, expert_batch, args)

    loss_dict = {}

    ######
    # IQ-Learn minimal implementation with X^2 divergence (~15 lines)
    # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
    current_Q = self.critic(obs, action)
    y = (1 - done) * self.gamma * self.getV(next_obs)
    if args.train.use_target:
        with torch.no_grad():
            y = (1 - done) * self.gamma * self.get_targetV(next_obs)

    reward = (current_Q - y)[is_expert]
    loss = -(reward).mean()

    # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
    value_loss = (self.getV(obs) - y).mean()
    loss += value_loss

    # Use χ2 divergence (adds a extra term to the loss)
    chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
    loss += chi2_loss
    ######

    self.critic_optimizer.zero_grad()
    loss.backward()
    self.critic_optimizer.step()
    return loss


def iq_update_critic(self, policy_batch, expert_batch, logger, step):
    args = self.args
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

    if args.only_expert_states:
        # Use policy actions instead of experts actions for IL with only observations
        expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done

    batch = get_concat_samples(policy_batch, expert_batch, args)
    obs, next_obs, action = batch[0:3]

    agent = self
    use_amp = getattr(self, 'use_amp', False)
    scaler = getattr(self, 'scaler', None)

    with torch.amp.autocast('cuda', enabled=use_amp):
        current_V = self.getV(obs)
        if args.train.use_target:
            with torch.no_grad():
                next_V = self.get_targetV(next_obs)
        else:
            next_V = self.getV(next_obs)

        if "DoubleQ" in self.args.q_net._target_:
            current_Q1, current_Q2 = self.critic(obs, action, both=True)
            q1_loss, loss_dict1 = iq_loss(agent, current_Q1, current_V, next_V, batch)
            q2_loss, loss_dict2 = iq_loss(agent, current_Q2, current_V, next_V, batch)
            critic_loss = 1/2 * (q1_loss + q2_loss)
            # merge loss dicts
            loss_dict = average_dicts(loss_dict1, loss_dict2)
        else:
            current_Q = self.critic(obs, action)
            critic_loss, loss_dict = iq_loss(agent, current_Q, current_V, next_V, batch)

    logger.log('train/critic_loss', critic_loss, step)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(critic_loss).backward()
        scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        scaler.step(self.critic_optimizer)
        scaler.update()
    else:
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.critic_optimizer.step()
    return loss_dict


def iq_update(self, policy_buffer, expert_buffer, logger, step):
    policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

    losses = self.iq_update_critic(policy_batch, expert_batch, logger, step)

    if self.actor and step % self.actor_update_frequency == 0:
        if not self.args.agent.vdice_actor:

            if self.args.offline:
                obs = expert_batch[0]
            else:
                # Use both policy and expert observations
                obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)

            if self.args.num_actor_updates:
                for i in range(self.args.num_actor_updates):
                    actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)

            losses.update(actor_alpha_losses)

    if step % self.critic_target_update_frequency == 0:
        if self.args.train.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses


if __name__ == "__main__":
    main()
