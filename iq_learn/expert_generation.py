import random
from itertools import count
from collections import defaultdict

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from agent import make_agent
from make_envs import make_env, is_atari
from dataset.memory import Memory
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)

    env = make_env(args)
    if args.eval.use_baselines:
        from baselines_zoo.baselines_expert import BaselinesExpert
        # Use absolute path to the rl-baselines3-zoo folder
        baselines_folder = hydra.utils.to_absolute_path('rl-baselines3-zoo/rl-trained-agents')
        agent = BaselinesExpert(args.env.name, folder=baselines_folder, algorithm=args.eval.expert_algorithm)
        # For baselines experts, we don't need the expert_file path
        print(f'Loading expert from: {baselines_folder}')
        agent.load("", "")  # BaselinesExpert uses its own path logic
    else:
        agent = make_agent(env, args)
        expert_file = f'{args.method.type}.para'
        if args.eval.policy:
            expert_file = f'{args.eval.policy}'
        print(f'Loading expert from: {expert_file}')
        agent.load(hydra.utils.to_absolute_path(expert_file), f'_{args.env.name}')

    episode_reward = 0

    REPLAY_MEMORY = None
    REWARD_THRESHOLD = args.eval.threshold
    MAX_EPS = args.expert.demos
    EPS_STEPS = int(args.env.eps_steps)
    MAX_IDLE_STEPS = 50  # cut trajectory if no reward for this many consecutive steps
    MIN_REWARD_DENSITY = 0.01  # minimum reward/step ratio to save a trajectory

    memory_replay = Memory(REPLAY_MEMORY)
    total_steps = 0
    saved_eps = 0
    expert_trajs = defaultdict(list)
    expert_lengths = []
    expert_rewards = []

    for epoch in count():
        if saved_eps >= MAX_EPS:
            break

        state, info = env.reset()
        episode_reward = 0
        traj = []

        episode_infos = None
        idle_steps = 0
        use_success = False
        score = None
        for time_steps in range(EPS_STEPS):
            action = agent.choose_action(state, deterministic=args.eval.deterministic)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if is_atari(args.env.name) and isinstance(action, np.ndarray):
                action = action.item()

            traj.append((state, next_state, action, reward, done))

            episode_reward += reward
            if memory_replay.size() == REPLAY_MEMORY:
                print('expert replay saved...')
                memory_replay.save(f'experts/{args.env.name}_{args.expert.demos}')
                exit()

            # Track consecutive steps without reward and cut early if stuck
            if reward > 0:
                idle_steps = 0
            else:
                idle_steps += 1
            if idle_steps >= MAX_IDLE_STEPS:
                print(f'  Early termination at step {time_steps}: no reward for {MAX_IDLE_STEPS} consecutive steps')
                break

            state = next_state

            if is_atari(args.env.name):
                if (getattr(args.env, "atari_terminal_on_life_loss", True)
                        and 'ale.lives' in info):  # true for breakout, false for pong
                    done = info['ale.lives'] == 0
                episode_infos = info.get("episode")
                if episode_infos is not None:
                    episode_reward = episode_infos['r']
                    print("Atari Episode Length", episode_infos["l"])

            if done:
                use_success = 'is_success' in info.keys()
                score = info.get('is_success')
                break

        states, next_states, actions, rewards, dones = zip(*traj)
        traj_reward = sum(rewards)
        traj_density = traj_reward / len(traj) if len(traj) > 0 else 0
        if (not REWARD_THRESHOLD or traj_reward >= REWARD_THRESHOLD) and traj_reward > 0 and traj_density >= MIN_REWARD_DENSITY and (not use_success or score >= 1.):
            saved_eps += 1

            expert_trajs["states"].append(states)
            expert_trajs["next_states"].append(next_states)
            expert_trajs["actions"].append(actions)
            expert_trajs["rewards"].append(rewards)
            expert_trajs["dones"].append(dones)
            expert_trajs["lengths"].append(len(traj))
            expert_lengths.append(len(traj))
            expert_rewards.append(episode_reward)
            print('Ep {}\tSaving Episode reward: {:.2f}\t'.format(epoch, episode_reward))
        else:
            print('Ep {}\tSkipped episode with reward: {:.2f}\t'.format(epoch, episode_reward))

    # for k, v in expert_trajs.items():
    #     expert_trajs[k] = np.array(v)

    get_data_stats(expert_trajs, np.array(expert_rewards), np.array(expert_lengths))

    print('Final size of Replay Buffer: {}'.format(sum(expert_trajs["lengths"])))
    algorithm = args.eval.expert_algorithm
    with open(hydra.utils.to_absolute_path(f'experts/{args.env.name}_{algorithm}_{args.expert.demos}_filtered.pkl'), 'wb') as f:
        pickle.dump(expert_trajs, f)
    exit()


def get_data_stats(d, rewards, lengths):
    # lengths = d["lengths"]

    print("rewards: {:.2f} +/- {:.2f}".format(rewards.mean(), rewards.std()))
    print("len: {:.2f} +/- {:.2f}".format(lengths.mean(), lengths.std()))


def padded(a, target_length, axis=0):
    """Add padding at end of to make array fixed size"""
    x = np.array(a)
    pad_size = target_length - x.shape[axis]
    axis_nb = len(x.shape)

    if pad_size < 0:
        return a
    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)
    return np.pad(a, pad_width=npad, mode="constant", constant_values=0)


if __name__ == '__main__':
    main()
