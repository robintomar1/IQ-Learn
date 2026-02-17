"""
BaselinesExpert: A wrapper to load and use Stable Baselines3 trained models
as experts in the IQ-Learn framework.

This allows you to use any trained agent from rl-baselines3-zoo/rl-trained-agents
in your expert generation script.
"""

import os
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable_baselines3 not available. Install with: pip install stable-baselines3")

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False


class BaselinesExpert:
    """
    Expert agent that loads Stable Baselines3 models from rl-trained-agents directory.
    
    This class provides a unified interface to use pre-trained RL agents from the
    rl-baselines3-zoo as expert demonstrators for imitation learning.
    """
    
    def __init__(self, env_name, folder='rl-trained-agents', algorithm=None, run_id=1):
        """
        Initialize the BaselinesExpert.
        
        Args:
            env_name (str): Environment name (e.g., 'BreakoutNoFrameskip-v4')
            folder (str): Path to the trained agents folder
            algorithm (str, optional): Algorithm name. If None, will try to auto-detect
            run_id (int): Run ID for the specific trained model (default: 1)
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable_baselines3 is required. Install with: pip install stable-baselines3")
            
        # Mapping of algorithm names to their corresponding SB3 classes
        self.ALGORITHM_MAP = {
            'a2c': A2C,
            'ddpg': DDPG, 
            'dqn': DQN,
            'ppo': PPO,
            'sac': SAC,
            'td3': TD3,
            # Add more algorithms as needed
        }
        
        self.env_name = env_name
        self.folder = folder
        self.algorithm = algorithm
        self.run_id = run_id
        self.model = None
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        
    def load(self, path, suffix=""):
        """
        Load the Stable Baselines3 model.
        
        Args:
            path (str): Base path (usually ignored, we use self.folder)
            suffix (str): Suffix for the model file (usually ignored)
        """
        # If algorithm not specified, try to find it
        if self.algorithm is None:
            self.algorithm = self._find_algorithm()
            
        if self.algorithm is None:
            raise ValueError(f"Could not find trained model for environment {self.env_name}")
            
        # Construct path to the model
        model_dir = os.path.join(self.folder, self.algorithm, f"{self.env_name}_{self.run_id}")
        model_path = os.path.join(model_dir, f"{self.env_name}.zip")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        # Load the model
        algorithm_class = self.ALGORITHM_MAP.get(self.algorithm.lower())
        if algorithm_class is None:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
        print(f"Loading {self.algorithm.upper()} model from {model_path}")
        self.model = algorithm_class.load(model_path)
        
        # Set device (move model to device instead of setting attribute)
        if TORCH_AVAILABLE and hasattr(self.model, 'set_parameters'):
            try:
                # For SB3 models, use the built-in device handling
                self.model = self.model.to(self.device)
            except:
                # If that fails, try moving the policy
                try:
                    if hasattr(self.model, 'policy'):
                        self.model.policy = self.model.policy.to(self.device)
                except:
                    pass  # Device setting failed, but model should still work
            
    def _find_algorithm(self):
        """
        Auto-detect which algorithm was used to train the model for this environment.
        """
        for algo in self.ALGORITHM_MAP.keys():
            model_dir = os.path.join(self.folder, algo, f"{self.env_name}_{self.run_id}")
            model_path = os.path.join(model_dir, f"{self.env_name}.zip")
            if os.path.exists(model_path):
                return algo
        return None
        
    def choose_action(self, state, deterministic=True):
        """
        Choose an action given the current state.
        
        Args:
            state: Current observation/state
            deterministic (bool): Whether to use deterministic policy
            
        Returns:
            action: Selected action
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
            
        # Handle LazyFrames and ensure state is in the right format
        if hasattr(state, '__array__'):
            # LazyFrames and similar objects that can be converted to numpy
            state = np.array(state)
        elif isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        elif not isinstance(state, np.ndarray):
            state = np.array(state)
            
        # For image observations, ensure correct shape (add batch dimension if needed)
        if len(state.shape) == 3:  # (C, H, W) -> (1, C, H, W)
            state = state[np.newaxis, ...]
        elif len(state.shape) == 1:  # (obs_dim,) -> (1, obs_dim)
            state = state[np.newaxis, ...]
                
        # Get action from the model
        action, _ = self.model.predict(state, deterministic=deterministic)
        
        # Handle different action formats
        if isinstance(action, np.ndarray):
            if action.shape == (1,):  # Single discrete action
                return action[0]
            elif len(action.shape) == 1 and len(action) == 1:  # Single action
                return action[0] 
            else:
                return action
        else:
            return action
            
    def save(self, path, suffix=""):
        """
        Save method for compatibility (not typically used for pre-trained models).
        """
        if self.model is not None:
            save_path = f"{path}{suffix}_sb3_model"
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
        else:
            print("No model to save")
            
    def evaluate(self, env, n_episodes=10):
        """
        Evaluate the loaded model on the given environment.
        
        Args:
            env: Gymnasium environment
            n_episodes (int): Number of episodes to evaluate
            
        Returns:
            tuple: (mean_reward, std_reward)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
            
        # Wrap environment if needed
        if not isinstance(env, DummyVecEnv):
            env = DummyVecEnv([lambda: env])
            
        mean_reward, std_reward = evaluate_policy(
            self.model, env, n_eval_episodes=n_episodes, deterministic=True
        )
        
        return mean_reward, std_reward


def list_available_models(folder='rl-trained-agents'):
    """
    List all available trained models in the rl-trained-agents directory.
    
    Args:
        folder (str): Path to the trained agents folder
        
    Returns:
        dict: Dictionary mapping algorithm -> list of (env_name, run_id) tuples
    """
    available_models = {}
    
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist")
        return available_models
        
    for algo in os.listdir(folder):
        algo_path = os.path.join(folder, algo)
        if os.path.isdir(algo_path):
            available_models[algo] = []
            
            for model_dir in os.listdir(algo_path):
                if '_' in model_dir:
                    parts = model_dir.rsplit('_', 1)
                    if len(parts) == 2:
                        env_name, run_id = parts
                        model_path = os.path.join(algo_path, model_dir, f"{env_name}.zip")
                        if os.path.exists(model_path):
                            available_models[algo].append((env_name, int(run_id)))
                            
    return available_models


if __name__ == "__main__":
    # Example usage
    print("Available models:")
    models = list_available_models()
    for algo, envs in models.items():
        print(f"  {algo.upper()}:")
        for env_name, run_id in envs[:5]:  # Show first 5
            print(f"    - {env_name} (run {run_id})")
        if len(envs) > 5:
            print(f"    ... and {len(envs)-5} more")