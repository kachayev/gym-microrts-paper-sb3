import numpy as np
from typing import Dict

from stable_baselines3.common.vec_env import VecFrameStack, VecEnvWrapper
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations, StackedDictObservations

from gym import spaces


class VecKeyedFrameStack(VecFrameStack):
    """Work exactly as built-in VecFrameStack but also provide an option to exclude
    givens keys from stacking when working with Dict spaces (similar to how it's done
    for VecNorm)."""

    def __init__(self, venv, n_stack: int, channels_order=None, stacked_keys=None):
        self.venv = venv
        self.n_stack = n_stack
        self.stacked_keys = stacked_keys

        wrapped_obs_space = venv.observation_space

        if isinstance(wrapped_obs_space, spaces.Box):
            self.stackedobs = StackedObservations(venv.num_envs, n_stack, wrapped_obs_space, channels_order)
        elif isinstance(wrapped_obs_space, spaces.Dict):
            self.stackedobs = KeyedStackedDictObservations(venv.num_envs, n_stack, wrapped_obs_space, channels_order, stacked_keys)
        else:
            raise Exception("VecFrameStack only works with gym.spaces.Box and gym.spaces.Dict observation spaces")

        observation_space = self.stackedobs.stack_observation_space(wrapped_obs_space)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)


class KeyedStackedDictObservations(StackedObservations):

    def __init__(self, num_envs, n_stack, observation_space, channels_order=None, stacked_keys=None):
        self.stacked_keys = set(stacked_keys) or set(observation_space.spaces.keys())
        self.n_stack = n_stack
        self.channels_first = {}
        self.stack_dimension = {}
        self.stackedobs = {}
        self.repeat_axis = {}

        for key, subspace in observation_space.spaces.items():
            if key not in self.stacked_keys:
                self.stackedobs[key] = np.zeros((num_envs,) + subspace.low.shape, subspace.low.dtype)
            else:
                assert isinstance(subspace, spaces.Box), "StackedDictObservations only works with nested gym.spaces.Box"
                if isinstance(channels_order, str) or channels_order is None:
                    subspace_channel_order = channels_order
                else:
                    subspace_channel_order = channels_order[key]
                (
                    self.channels_first[key],
                    self.stack_dimension[key],
                    self.stackedobs[key],
                    self.repeat_axis[key],
                ) = self.compute_stacking(num_envs, n_stack, subspace, subspace_channel_order)

    def stack_observation_space(self, observation_space: spaces.Dict) -> spaces.Dict:
        spaces_dict = {}
        for key, subspace in observation_space.spaces.items():
            if key not in self.stacked_keys:
                spaces_dict[key] = subspace
            else:
                low = np.repeat(subspace.low, self.n_stack, axis=self.repeat_axis[key])
                high = np.repeat(subspace.high, self.n_stack, axis=self.repeat_axis[key])
                spaces_dict[key] = spaces.Box(low=low, high=high, dtype=subspace.dtype)
        return spaces.Dict(spaces=spaces_dict)

    def reset(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for key, obs in observation.items():
            if key not in self.stacked_keys:
                self.stackedobs[key] = obs
            else:
                self.stackedobs[key][...] = 0
                if self.channels_first[key]:
                    self.stackedobs[key][:, -obs.shape[self.stack_dimension[key]] :, ...] = obs
                else:
                    self.stackedobs[key][..., -obs.shape[self.stack_dimension[key]] :] = obs
        return self.stackedobs

    def update(self, observations, dones, infos):
        for key in self.stackedobs.keys():
            if key not in self.stacked_keys:
                self.stackedobs[key] = observations[key]
            else:
                stack_ax_size = observations[key].shape[self.stack_dimension[key]]
                self.stackedobs[key] = np.roll(
                    self.stackedobs[key],
                    shift=-stack_ax_size,
                    axis=self.stack_dimension[key],
                )

                for i, done in enumerate(dones):
                    if done:
                        if "terminal_observation" in infos[i]:
                            old_terminal = infos[i]["terminal_observation"][key]
                            if self.channels_first[key]:
                                new_terminal = np.vstack(
                                    (
                                        self.stackedobs[key][i, :-stack_ax_size, ...],
                                        old_terminal,
                                    )
                                )
                            else:
                                new_terminal = np.concatenate(
                                    (
                                        self.stackedobs[key][i, ..., :-stack_ax_size],
                                        old_terminal,
                                    ),
                                    axis=self.stack_dimension[key],
                                )
                            infos[i]["terminal_observation"][key] = new_terminal
                        self.stackedobs[key][i] = 0
                if self.channels_first[key]:
                    self.stackedobs[key][:, -stack_ax_size:, ...] = observations[key]
                else:
                    self.stackedobs[key][..., -stack_ax_size:] = observations[key]
        return self.stackedobs, infos

