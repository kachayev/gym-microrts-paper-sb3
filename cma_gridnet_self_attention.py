import multiprocessing as mp
import numpy as np
import time
import torch
from torch import nn

import cma

from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecMonitor

from gym_microrts import microrts_ai

from ppo_gridnet_diverse_encode_decode_sb3 import (
    CustomMicroRTSGridMode,
    HierachicalMultiCategoricalDistribution,
    MicroRTSStatsRecorder,
)

from ppo_gridnet_self_attention import (GlobalMultiHeadAttentionEncoder, Actor)

class CMA:

    def __init__(self, population_size, init_sigma, init_params, init_seed=0):
        self.algorithm = cma.CMAEvolutionStrategy(
            x0=init_params,
            sigma0=init_sigma,
            inopts=dict(popsize=population_size, seed=init_seed, randn=np.random.randn)
        )
        self.population = None

    def get_population(self):
        self.population = self.algorithm.ask()
        return self.population

    def evolve(self, fitness):
        self.algorithm.tell(self.population, -fitness)

    def get_current_parameters(self):
        return self.algorithm.result.xfavorite


class Solution:

    def get_l2_penalty(self):
        params = self.get_params()
        return self._l2_coefficient * np.sum(params ** 2)

    def add_noise(self, noise):
        self.set_params(self.get_params() + noise)

    def add_noise_to_layer(self, noise, layer_index):
        layer_params = self.get_params_from_layer(layer_index)
        self.set_params_to_layer(
            params=layer_params + noise, layer_index=layer_index)


class TorchSolution(Solution):

    def __init__(self):
        self._layers = []

    def get_output(self, inputs):
        # torch.set_num_threads(1)
        with torch.no_grad():
            return self._get_output(inputs)

    def get_params(self):
        params = []
        for layer in self._layers:
            weight_dict = layer.state_dict()
            for k in sorted(weight_dict.keys()):
                params.append(weight_dict[k].numpy().copy().ravel())
        return np.concatenate(params)

    def set_params(self, params):
        offset = 0
        for i, layer in enumerate(self._layers):
            weights_to_set = {}
            weight_dict = layer.state_dict()
            for k in sorted(weight_dict.keys()):
                weight = weight_dict[k].numpy()
                weight_size = weight.size
                weights_to_set[k] = torch.from_numpy(
                    params[offset:(offset + weight_size)].reshape(weight.shape))
                offset += weight_size
            self._layers[i].load_state_dict(state_dict=weights_to_set)

    def get_params_from_layer(self, layer_index):
        params = []
        layer = self._layers[layer_index]
        weight_dict = layer.state_dict()
        for k in sorted(weight_dict.keys()):
            params.append(weight_dict[k].numpy().copy().ravel())
        return np.concatenate(params)

    def set_params_to_layer(self, params, layer_index):
        weights_to_set = {}
        weight_dict = self._layers[layer_index].state_dict()
        offset = 0
        for k in sorted(weight_dict.keys()):
            weight = weight_dict[k].numpy()
            weight_size = weight.size
            weights_to_set[k] = torch.from_numpy(
                params[offset:(offset + weight_size)].reshape(weight.shape))
            offset += weight_size
        self._layers[layer_index].load_state_dict(state_dict=weights_to_set)

    def get_num_params_per_layer(self):
        num_params_per_layer = []
        for layer in self._layers:
            weight_dict = layer.state_dict()
            num_params = 0
            for k in sorted(weight_dict.keys()):
                weights = weight_dict[k].numpy()
                num_params += weights.size
            num_params_per_layer.append(num_params)
        return num_params_per_layer

    def _save_to_file(self, filename):
        params = self.get_params()
        np.savez(filename, params=params)

    def save(self, log_dir, iter_count, best_so_far):
        filename = os.path.join(log_dir, 'model_{}.npz'.format(iter_count))
        self._save_to_file(filename=filename)
        if best_so_far:
            filename = os.path.join(log_dir, 'best_model.npz')
            self._save_to_file(filename=filename)

    def load(self, filename):
        with np.load(filename) as data:
            params = data['params']
            self.set_params(params)

    @property
    def layers(self):
        return self._layers


class SelfAttentionGridnetSolution(TorchSolution):

    def __init__(self, obs_space, action_space):
        super().__init__()

        input_channels, output_channels, actor_hidden_dim = 27, 78, 16

        self.device = get_device("auto")
        self._mask_value = torch.tensor(-1e+8)

        self.action_space_size = action_space.nvec.sum()
        self.height, self.width, self.input_channels = obs_space['obs'].shape
        self.num_cells = self.height * self.width
        self.action_plane = action_space.nvec[:action_space.nvec.size // self.num_cells]
        self.action_dist = HierachicalMultiCategoricalDistribution(self.num_cells, self.action_plane)

        self.encoder = GlobalMultiHeadAttentionEncoder(
            input_channels,
            embed_dim=actor_hidden_dim,
            num_heads=1,
            seq_len=self.num_cells,
            bias=True,
            context_dropout=0.0,
            context_norm_eps=1e-5,
            embed_dropout=0.0,
            embed_norm_eps=1e-5,
            combine_inputs=True,
        ).to(self.device)
        self.actor = Actor(output_channels, hidden_dim=actor_hidden_dim).to(self.device)

    def _mask_action_logits(self, latent_pi, masks, mask_value=None):
        mask_value = mask_value or self._mask_value
        return torch.where(masks, latent_pi, mask_value)

    def _get_output(self, inputs):
        obs, masks = inputs["obs"], inputs["masks"]
        obs = torch.from_numpy(obs).float().to(self.device)
        masks = torch.from_numpy(masks).bool().to(self.device)
        latent_pi, _ = self.encoder(obs)
        logits = self._mask_action_logits(self.actor(latent_pi), masks)
        actions = self.action_dist.actions_from_params(action_logits=logits, deterministic=False)
        return actions.cpu().numpy()


class MicroRTSTask:

    def __init__(self, steps_per_rollout, **kwargs):
        envs = CustomMicroRTSGridMode(**kwargs)
        envs = MicroRTSStatsRecorder(envs)
        envs = VecMonitor(envs)
        self._env = envs
        self.steps_per_rollout = steps_per_rollout

    @property
    def env(self):
        return self._env

    def rollout(self, solution):
        obs = self.reset()
        if hasattr(solution, 'reset'):
            solution.reset()

        rewards = [None]*self.steps_per_rollout
        for idx in range(self.steps_per_rollout):
            action = solution.get_output(inputs=obs)
            obs, reward, done, info = self._env.step(action)
            rewards[idx] = reward
        return np.sum(np.array(rewards), axis=0)
    
    def reset(self):
        return self._env.reset()

solution = None
task = None

def worker_init(steps_per_rollout, num_envs):
    global solution, task
    task = MicroRTSTask(
        steps_per_rollout,
        num_selfplay_envs=0,
        max_steps=2_000,
        render_theme=2,
        ai2s=[microrts_ai.coacAI for _ in range(num_envs)],
        map_paths=["maps/16x16/basesWorkers16x16.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    )
    solution = SelfAttentionGridnetSolution(task.env.observation_space, task.env.action_space)

# xxx(okachaiev): add L2 or entropy penalty
def worker_run(req):
    global solution, task
    index, params = req
    solution.set_params(params)

    start_time = time.time()
    rewards = task.rollout(solution)
    time_cost = time.time() - start_time

    print(f"Roll-out index={index}, time={time_cost:.2f}s, reward={rewards.mean():.2f}")

    return rewards.mean()

if __name__ == "__main__":
    num_workers = 8
    steps_per_rollout = 2_000
    num_envs = 2

    init_params = np.zeros(4270)
    algo = CMA(population_size=256, init_params=init_params, init_sigma=0.01, init_seed=1047)

    def train(pool, algorithm, max_iter=5):
        best_eval_score = -float('inf')
        for iter_idx in range(max_iter):
            start_time = time.time()
            population = algorithm.get_population()
            fitness = pool.map(
                func=worker_run,
                iterable=enumerate(population)
            )
            fitness = np.array(fitness_scores).mean(axis=1)
            algorithm.evolve(fitness)
            time_cost = time.time() - start_time
            best_eval_score = max(best_eval_score, fitness.max())
            print(f"training time: {time_cost}s, iter: {iter_idx+1}, best: {best_eval_score}, scores: {fitness}")

    with mp.get_context('spawn').Pool(
        initializer=worker_init,
        initargs=(steps_per_rollout, num_envs),
        processes=num_workers,
    ) as pool:
        train(pool, algo, max_iter=16)