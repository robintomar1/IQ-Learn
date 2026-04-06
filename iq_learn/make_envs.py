import gymnasium as gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor

from wrappers.atari_wrapper import ScaledFloatFrame, FrameStack, FrameStackEager, PyTorchFrame
from wrappers.normalize_action_wrapper import check_and_normalize_box_actions

from envs.dusty_env import DustyEnv
import envs
import numpy as np
import os

# Register all custom envs
envs.register_custom_envs()

# Register ALE environments for Atari games
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass
try:
    from shimmy.registration import register_gymnasium_envs
    register_gymnasium_envs()
except ImportError:
    pass

def make_dcm(cfg):
    import dmc2gym
    """Helper function to create dm_control environment"""
    if cfg.env.name == 'dmc_ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    elif cfg.env.name == 'dmc_point_mass_easy':
        domain_name = 'point_mass'
        task_name = 'easy'
    else:
        domain_name = cfg.env.name.split('_')[1]
        task_name = '_'.join(cfg.env.name.split('_')[2:])
    
    if cfg.env.from_pixels:
        # Set env variables for Mujoco rendering
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["EGL_DEVICE_ID"] = os.environ["CUDA_VISIBLE_DEVICES"]

        # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
        camera_id = 2 if domain_name == 'quadruped' else 0

        env = dmc2gym.make(domain_name=domain_name,
                        task_name=task_name,
                        seed=cfg.seed,
                        visualize_reward=False,
                        from_pixels=True,
                        height=cfg.env.image_size,
                        width=cfg.env.image_size,
                        frame_skip=cfg.env.action_repeat,
                        camera_id=camera_id)

        print(env.observation_space.dtype)
        # env = FrameStack(env, k=cfg.env.frame_stack)
        env = FrameStackEager(env, k=cfg.env.frame_stack)
        
    else:
        env = dmc2gym.make(domain_name=domain_name,
                        task_name=task_name,
                        seed=cfg.seed,
                        visualize_reward=True)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env

def make_atari(env, args):
    terminal_on_life_loss = getattr(args.env, "atari_terminal_on_life_loss", True)
    clip_reward = getattr(args.env, "atari_clip_reward", True)
    env = AtariWrapper(env,
                       terminal_on_life_loss=terminal_on_life_loss,
                       clip_reward=clip_reward)
    env = PyTorchFrame(env)
    env = FrameStack(env, 4)
    return env

def is_atari(env_name):
    return env_name in ['PongNoFrameskip-v4', 
                        'BreakoutNoFrameskip-v4', 
                        'SpaceInvadersNoFrameskip-v4', 
                        'BeamRiderNoFrameskip-v4',
                        'QbertNoFrameskip-v4',
                        'SeaquestNoFrameskip-v4']


class EnvFactory:
    """Picklable env factory for AsyncVectorEnv subprocesses."""
    def __init__(self, args):
        self.args = args

    def __call__(self):
        return make_env(self.args, monitor=False)


# Mapping from gymnasium NoFrameskip env names to envpool v5 names
_ENVPOOL_NAME_MAP = {
    'BreakoutNoFrameskip-v4':       'Breakout-v5',
    'PongNoFrameskip-v4':           'Pong-v5',
    'SpaceInvadersNoFrameskip-v4':  'SpaceInvaders-v5',
    'BeamRiderNoFrameskip-v4':      'BeamRider-v5',
    'QbertNoFrameskip-v4':          'Qbert-v5',
    'SeaquestNoFrameskip-v4':       'Seaquest-v5',
}


def make_envpool_atari(env_name, num_envs, seed=0, terminal_on_life_loss=True, clip_reward=True):
    """Create a vectorised Atari env using envpool (C++ backend, no subprocess IPC).

    Applies the same preprocessing as the gymnasium pipeline:
    frame_skip=4, grayscale, 84x84 resize, 4-frame stack, episodic life, noop reset.
    Returns obs of shape (num_envs, 4, 84, 84) dtype uint8.
    Actions must be int32.
    """
    import envpool
    ep_name = _ENVPOOL_NAME_MAP.get(env_name)
    if ep_name is None:
        raise ValueError(f"No envpool mapping for env '{env_name}'. "
                         f"Supported: {list(_ENVPOOL_NAME_MAP)}")
    return envpool.make(
        ep_name,
        env_type='gymnasium',
        num_envs=num_envs,
        seed=seed,
        episodic_life=terminal_on_life_loss,
        reward_clip=clip_reward,
        stack_num=4,
        gray_scale=True,
        img_height=84,
        img_width=84,
        noop_max=30,
        frame_skip=4,
    )


def make_env(args, monitor=True):
    if 'dmc' in args.env.name:
        env = make_dcm(args)
    elif args.env.name == "dusty":
        print(f"🌟 Creating Custom Dusty Environment (Offline Mode)")
        env = DustyEnv()
        # Wrappers are usually not needed for a Mock Env, 
        # but if your agent expects normalized float inputs (0.0-1.0),
        # you might manually wrap it or handle it in the dataset loader.
        return env
    else:
        env = gym.make(args.env.name)
    
    if monitor:
        env = Monitor(env, "gym")

    if is_atari(args.env.name):
        env = make_atari(env, args)

    # --- ADD THIS BLOCK ---

    # Normalize box actions to [-1, 1]
    env = check_and_normalize_box_actions(env)
    return env
