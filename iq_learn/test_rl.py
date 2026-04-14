from itertools import count
import torch
import gymnasium as gym
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from make_envs import make_env, make_atari, is_atari
from agent import make_agent
from utils.utils import eval_mode

import ale_py
gym.register_envs(ale_py)


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(OmegaConf.to_yaml(cfg))
    return cfg


def make_render_env(args):
    """Create env with render_mode='human' for visual playback."""
    env = gym.make(args.env.name, render_mode="human")
    if is_atari(args.env.name):
        env = make_atari(env, args)
    return env


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)

    EPISODE_STEPS = int(args.env.eps_steps)

    # Use non-render env for agent creation (obs/action dims)
    env = make_env(args)
    agent = make_agent(env, args)

    policy_file = 'experts'
    if args.eval.policy:
        policy_file = f'{args.eval.policy}'
    print(f'Loading policy from: {policy_file}', f'_{args.env.name}')

    agent.load(hydra.utils.to_absolute_path(policy_file), f'_{args.env.name}')

    if args.eval_only:
        evaluate(agent, env, num_episodes=args.eval.eps)
        exit()

    # Close non-render env and create one with rendering
    env.close()
    env = make_render_env(args)

    for epoch in count():
        state, info = env.reset()
        episode_reward = 0
        for episode_step in range(EPISODE_STEPS):
            with eval_mode(agent):
                action = agent.choose_action(state, sample=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break
            state = next_state
        print('Ep {}\tScore: {:.2f}'.format(epoch, episode_reward))


def evaluate(actor, env, num_episodes=10):
    """Evaluates the policy."""
    total_timesteps = []
    total_returns = []

    for _ in range(num_episodes):
        eps_timesteps = 0
        eps_returns = 0

        state, info = env.reset()
        done = False
        while not done:
            with eval_mode(actor):
                action = actor.choose_action(state, sample=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if 'lives' in info:
                done = info['lives'] == 0

            eps_returns += reward
            eps_timesteps += 1
            state = next_state

        total_timesteps.append(eps_timesteps)
        total_returns.append(eps_returns)

    total_returns = np.array(total_returns)
    total_timesteps = np.array(total_timesteps)

    print("rewards: {:.2f} +/- {:.2f}".format(total_returns.mean(), total_returns.std()))
    print("len: {:.2f} +/- {:.2f}".format(total_timesteps.mean(), total_timesteps.std()))


if __name__ == "__main__":
    main()