from collections import deque
import numpy as np
import random
import torch

try:
    from wrappers.atari_wrapper import LazyFrames
except ImportError:
    class LazyFrames: pass

from dataset.expert_dataset import ExpertDataset


class NumpyReplayBuffer:
    """Preallocated numpy ring-buffer replay memory.

    Stores all transitions in contiguous numpy arrays enabling fully-vectorized
    sampling with a single fancy-index op rather than Python deque iteration.

    For uint8 observations (envpool Atari) stores as uint8 (4x memory savings)
    and transfers to GPU as uint8 before casting to float32 on the device
    (4x less PCIe bandwidth vs converting on CPU first).

    Drop-in replacement for Memory in the multi-env envpool training path.
    Arrays are lazily initialized on the first call to add() / add_batch().
    """

    def __init__(self, capacity: int, seed: int = 0) -> None:
        np.random.seed(seed)
        self.capacity = capacity
        self._ptr = 0
        self._size = 0

        self._states = None
        self._next_states = None
        self._actions = None
        self._rewards = None
        self._dones = None
        self.obs_dtype = None

    def _init_arrays(self, obs_shape, obs_dtype) -> None:
        self.obs_dtype = obs_dtype
        self._states = np.empty((self.capacity, *obs_shape), dtype=obs_dtype)
        self._next_states = np.empty((self.capacity, *obs_shape), dtype=obs_dtype)
        self._actions = np.empty(self.capacity, dtype=np.int64)
        self._rewards = np.empty(self.capacity, dtype=np.float32)
        self._dones = np.empty(self.capacity, dtype=np.bool_)

    def add(self, experience) -> None:
        state, next_state, action, reward, done = experience
        obs = np.asarray(state)
        if self._states is None:
            self._init_arrays(obs.shape, obs.dtype)
        idx = self._ptr
        self._states[idx] = obs
        self._next_states[idx] = np.asarray(next_state)
        self._actions[idx] = action
        self._rewards[idx] = float(reward)
        self._dones[idx] = bool(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(self, states, next_states, actions, rewards, dones) -> None:
        """Vectorized add for a full env step — one Python call for all envs."""
        n = len(states)
        if self._states is None:
            self._init_arrays(states.shape[1:], states.dtype)
        idxs = np.arange(self._ptr, self._ptr + n) % self.capacity
        self._states[idxs] = states
        self._next_states[idxs] = next_states
        self._actions[idxs] = actions
        self._rewards[idxs] = rewards
        self._dones[idxs] = dones
        self._ptr = (self._ptr + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self._ptr = 0
        self._size = 0

    def save(self, path) -> None:
        if self._states is None:
            return
        n = self._size
        np.save(path, {
            'states': self._states[:n],
            'next_states': self._next_states[:n],
            'actions': self._actions[:n],
            'rewards': self._rewards[:n],
            'dones': self._dones[:n],
        })

    def load(self, path, num_trajs, sample_freq, seed) -> None:
        if not path.endswith("pkl"):
            path += '.npy'
        data = ExpertDataset(path, num_trajs, sample_freq, seed)
        for i in range(len(data)):
            self.add(data[i])

    def get_samples(self, batch_size, device):
        n = min(batch_size, self._size)
        idxs = np.random.choice(self._size, size=n, replace=False)

        states = self._states[idxs]
        next_states = self._next_states[idxs]

        if self.obs_dtype == np.uint8:
            # Transfer uint8 to GPU (4x less PCIe), cast to float32 on device
            batch_state = torch.from_numpy(states).to(
                device=device, dtype=torch.float32).div_(255.0)
            batch_next_state = torch.from_numpy(next_states).to(
                device=device, dtype=torch.float32).div_(255.0)
        else:
            batch_state = torch.as_tensor(states, dtype=torch.float32, device=device)
            batch_next_state = torch.as_tensor(next_states, dtype=torch.float32, device=device)

        batch_action = torch.as_tensor(
            self._actions[idxs], dtype=torch.float32, device=device).unsqueeze(1)
        batch_reward = torch.as_tensor(
            self._rewards[idxs], dtype=torch.float32, device=device).unsqueeze(1)
        batch_done = torch.as_tensor(
            self._dones[idxs], dtype=torch.float32, device=device).unsqueeze(1)

        return batch_state, batch_next_state, batch_action, batch_reward, batch_done


class Memory(object):
    def __init__(self, memory_size: int, seed: int = 0) -> None:
        random.seed(seed)
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def add_batch(self, states, next_states, actions, rewards, dones) -> None:
        """Vectorised add for a full env step — one Python call for all envs.
        All inputs are numpy arrays of shape (num_envs, ...).
        """
        self.buffer.extend(zip(states, next_states,
                               actions.tolist(), rewards.tolist(),
                               [bool(d) for d in dones]))

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

    def save(self, path):
        b = np.asarray(self.buffer)
        print(b.shape)
        np.save(path, b)

    def load(self, path, num_trajs, sample_freq, seed):
        # If path has no extension add npy
        if not path.endswith("pkl"):
            path += '.npy'
        data = ExpertDataset(path, num_trajs, sample_freq, seed)
        # data = np.load(path, allow_pickle=True)
        for i in range(len(data)):
            self.add(data[i])

    def get_samples(self, batch_size, device):
        batch = self.sample(batch_size, False)

        batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

        # Handle multimodal dictionary observations
        if isinstance(batch_state[0], dict):
            state_dict = {}
            for key in batch_state[0].keys():
                stacked_val = np.stack([s[key] for s in batch_state])
                state_dict[key] = torch.as_tensor(stacked_val, dtype=torch.float, device=device)
            batch_state = state_dict

            next_state_dict = {}
            for key in batch_next_state[0].keys():
                stacked_val = np.stack([s[key] for s in batch_next_state])
                next_state_dict[key] = torch.as_tensor(stacked_val, dtype=torch.float, device=device)
            batch_next_state = next_state_dict

        else:
            # Handle Atari LazyFrames and uint8 scaling
            if isinstance(batch_state[0], LazyFrames):
                batch_state = np.array(batch_state) / 255.0
            elif isinstance(batch_state[0], np.ndarray) and batch_state[0].dtype == np.uint8:
                batch_state = np.array(batch_state, dtype=np.float32) / 255.0
            if isinstance(batch_next_state[0], LazyFrames):
                batch_next_state = np.array(batch_next_state) / 255.0
            elif isinstance(batch_next_state[0], np.ndarray) and batch_next_state[0].dtype == np.uint8:
                batch_next_state = np.array(batch_next_state, dtype=np.float32) / 255.0

            batch_state = torch.as_tensor(np.array(batch_state), dtype=torch.float, device=device)
            batch_next_state = torch.as_tensor(np.array(batch_next_state), dtype=torch.float, device=device)

        batch_action = torch.as_tensor(np.array(batch_action), dtype=torch.float, device=device)
        if batch_action.ndim == 1:
            batch_action = batch_action.unsqueeze(1)

        batch_reward = torch.as_tensor(batch_reward, dtype=torch.float, device=device).unsqueeze(1)
        batch_done = torch.as_tensor(batch_done, dtype=torch.float, device=device).unsqueeze(1)

        return batch_state, batch_next_state, batch_action, batch_reward, batch_done