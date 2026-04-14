import gymnasium as gym
from agent.sac import SAC
from agent.softq import SoftQ
from omegaconf import OmegaConf

def make_agent(env, args):
    obs_shape = env.observation_space.shape

    if isinstance(env.action_space, gym.spaces.Discrete):
        print('--> Using Soft-Q agent')
        action_dim = env.action_space.n

        # For SoftQ, we typically assume vector observations
        obs_dim = obs_shape[0]

        args.agent.obs_dim = obs_dim
        args.agent.action_dim = int(action_dim)

        agent = SoftQ(obs_dim, action_dim, args.train.batch, args)

    else:
        print('--> Using SAC agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]

        # Pass the full shape (e.g., [3, 84, 84]) so the PixelEncoder detects it's an image
        args.agent.obs_dim = obs_shape
        args.agent.action_dim = action_dim

        agent = SAC(obs_shape, action_dim, action_range, args.train.batch, args)

    return agent
