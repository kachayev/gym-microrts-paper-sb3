import abc
import numpy as np
import time
import torch
from torch import nn

import cma

from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.utils import get_device

from gym_microrts import microrts_ai

from ppo_gridnet_diverse_encode_decode_sb3 import (
    CustomMicroRTSGridMode,
    MicroRTSStatsRecorder,
    HierachicalMultiCategoricalDistribution,
)

class CMA:

    def __init__(self, population_size, init_sigma, init_params):
        self.algorithm = cma.CMAEvolutionStrategy(
            x0=init_params,
            sigma0=init_sigma,
            inopts=dict(popsize=population_size, seed=0, randn=np.random.randn)
        )
        self.population = None

    def get_population(self):
        self.population = self.algorithm.ask()
        return self.population

    def evolve(self, fitness):
        self.algorithm.tell(self.population, -fitness)

    def get_current_parameters(self):
        return self.algorithm.result.xfavorite


class BaseSolution(abc.ABC):
    @abc.abstractmethod
    def get_output(self, inputs, update_filter):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_params(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_params(self, params):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_params_from_layer(self, layer_index):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_params_to_layer(self, params, layer_index):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_num_params_per_layer(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, log_dir, iter_count, best_so_far):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, filename):
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()

    def get_l2_penalty(self):
        params = self.get_params()
        return self._l2_coefficient * np.sum(params ** 2)

    def add_noise(self, noise):
        self.set_params(self.get_params() + noise)

    def add_noise_to_layer(self, noise, layer_index):
        layer_params = self.get_params_from_layer(layer_index)
        self.set_params_to_layer(
            params=layer_params + noise, layer_index=layer_index)


class BaseTorchSolution(BaseSolution):
    def __init__(self):
        self._layers = []

    def get_output(self, inputs, update_filter=False):
        torch.set_num_threads(1)
        with torch.no_grad():
            return self._get_output(inputs, update_filter)

    @abc.abstractmethod
    def _get_output(self, inputs, update_filter):
        raise NotImplementedError()

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

    def reset(self):
        raise NotImplementedError()

    @property
    def layers(self):
        return self._layers


class GymTask:

    def __init__(self):
        self._env = None
        self._render = False
        self._logger = None

    def create_task(self, **kwargs):
        raise NotImplementedError()

    def seed(self, seed):
        if isinstance(self, TakeCoverTask):
            self._env.game.set_seed(seed)
        else:
            self._env.seed(seed)

    def reset(self):
        return self._env.reset()

    def step(self, action, evaluate):
        return self._env.step(action)

    def close(self):
        self._env.close()

    def _process_reward(self, reward, done, evaluate):
        return reward

    def _process_action(self, action):
        return action

    def _process_observation(self, observation):
        return observation

    def _overwrite_terminate_flag(self, reward, done, step_cnt, evaluate):
        return done

    def rollout(self, solution, evaluate):
        ob = self.reset()
        ob = self._process_observation(ob)
        if hasattr(solution, 'reset'):
            solution.reset()

        start_time = time.time()

        rewards = []
        done = False
        step_idx = 0
        while not done:
            action = solution.get_output(inputs=ob, update_filter=not evaluate)
            action = self._process_action(action)
            ob, r, done, _ = self.step(action, evaluate)
            ob = self._process_observation(ob)
            step_idx += 1
            done = self._overwrite_terminate_flag(r, done, step_idx, evaluate)
            step_reward = self._process_reward(r, done, evaluate)
            rewards.append(step_reward)

        time_cost = time.time() - start_time
        actual_reward = np.sum(rewards)
        
        print('Roll-out time={0:.2f}s, steps={1}, reward={2:.2f}'.format(
            time_cost, step_idx, actual_reward))

        return actual_reward

# xxx(okachaiev): make it work with vector envs to take advantage of GPU
# xxx(okachaiev): technically, we don't need Gym API here at all... if
#                 I can put the network module into JVM application,
#                 it's much cheaper to just let JVM run and accept requests
class MicroRTSTask(GymTask):

    def _overwrite_terminate_flag(self, reward, done, step_cnt, evaluate):
        return done[0]

    def _process_observation(self, observation):
        return observation['obs'], observation['masks']

    def create_task(self, **kwargs):
        envs = CustomMicroRTSGridMode(
            num_selfplay_envs=0,
            max_steps=2_000,
            render_theme=2,
            ai2s=[microrts_ai.coacAI],
            map_paths=["maps/16x16/basesWorkers16x16.xml"],
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
        )
        envs = MicroRTSStatsRecorder(envs)
        envs = VecMonitor(envs)

        self._env = envs

        return self

class Encoder(nn.Module):

    def __init__(self, input_channels: int):
        super(Encoder, self).__init__()

        self.input_channels = input_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

    def forward(self, x):
        x = x.permute((0,3,1,2))
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        return x


class Policy(nn.Module):
    
    def __init__(self, output_channels, hidden_dim=32, num_cells=256):
        super(Policy, self).__init__()

        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        self.num_cells = num_cells
        self.network = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_channels)
        )

    def forward(self, x):
        x = self.network(x.unsqueeze(-1))
        x = x.reshape((-1, self.output_channels*self.num_cells))
        return x


# xxx(okachaiev): I don't like the fact we need to manipulate with
# "_layers". seems like having "register_layer" as a public API would
# be much easier to deal with
class GridnetSolution(BaseTorchSolution):

    def __init__(self, observation_space, action_space, device: str = "auto"):
        super().__init__()

        self.device = get_device(device)

        self.action_space_size = action_space.nvec.sum()
        self.height, self.width, self.input_channels = observation_space['obs'].shape
        self.num_cells = self.height * self.width
        self.action_plane = action_space.nvec[:action_space.nvec.size // self.num_cells]
        self.action_dist = HierachicalMultiCategoricalDistribution(self.num_cells, self.action_plane)

        self.latent_net = Encoder(self.input_channels).to(self.device)
        self.register_layers(self.latent_net)
        self.policy_net = Policy(self.action_plane.sum()).to(self.device)
        self.register_layers(self.policy_net)

        self._mask_value = torch.tensor(-1e+8, device=self.device)

        print('Number of parameters: {}'.format(self.get_num_params_per_layer()))

    def register_layers(self, module: nn.Module):
        for layer in module.children():
            if list(layer.parameters()):
                self._layers.append(layer)

    def _mask_action_logits(self, latent_pi, masks, mask_value=None):
        mask_value = mask_value or self._mask_value
        return torch.where(masks, latent_pi, mask_value)

    def _get_output(self, inputs, update_filter=False):
        obs, masks = inputs
        obs = torch.from_numpy(obs).float().to(self.device)
        masks = torch.from_numpy(masks).bool().to(self.device)

        latent_pi = self.latent_net(obs)
        latent_pi = self._mask_action_logits(self.policy_net(latent_pi), masks)

        actions = self.action_dist.actions_from_params(action_logits=latent_pi, deterministic=False)
        return actions.cpu().numpy()

    def reset(self):
        pass


class CompressedParams:

    def __init__(self, init_seed):
        self.init_seed = init_seed
        self.rng = np.random.RandomState(seed=init_seed)
        self.generations = []

    def evolve(self, sigma):
        self.generations.append((self.rng.randint(1 << 31 - 1), sigma))
        return self

    def uncompress(self, params):
        N = len(params)
        for seed, sigma in self.generations:
            params += np.random.RandomState(seed).normal(0., sigma, N)
        return params

# xxx(okachaiev): ABC class to describe algorithm?
class GaussianNoiseGA:

    def __init__(self, population_size, init_seed, truncate=None, sigma=0.05):
        self.rng = np.random.RandomState(seed=init_seed)
        self.sigma = sigma
        self.truncate = truncate or population_size // 2
        self.population_size = population_size
        self.population = [
            CompressedParams(self.rng.randint(1 << 31 - 1))
            for _ in range(population_size)
        ]

    def get_population(self):
        return self.population

    def evolve(self, fitness):
        # choice best models to clone
        top_k = np.argpartition(fitness,-self.truncate)[-self.truncate:]
        ind = np.random.choice(top_k, self.population_size-1)
        elite = [self._evolve_single(np.argmax(fitness))] 
        self.population = elite + [self._evolve_single(i) for i in ind]
        return self

    def _evolve_single(self, ind: int) -> CompressedParams:
        new_params = CompressedParams(self.population[ind].init_seed)
        new_params.generations = self.population[ind].generations[:]
        new_params.evolve(self.sigma)
        return new_params

    def get_current_parameters(self):
        pass


if __name__ == "__main__":
    # xxx(okachaiev): this API is horrible :(
    task = MicroRTSTask().create_task()
    solution = GridnetSolution(task._env.observation_space, task._env.action_space)
    rng = np.random.RandomState(seed=0)

    task.rollout(solution, evaluate=False)

    algo = GaussianNoiseGA(population_size=24, truncate=16, init_seed=42)

    def collect_fitness(algorithm, n_repeat: int = 2, evaluate: bool = False):
        population = algorithm.get_population()
        fitness_scores = np.zeros((len(population), n_repeat))
        for idx, compressed_params in enumerate(population):
            solution = GridnetSolution(task._env.observation_space, task._env.action_space)
            # xxx(okachaiev): there should be much more performant way of doing this
            params = compressed_params.uncompress(solution.get_params())
            solution.set_params(params)
            # xxx(okachaiev): instead of seq. execution I can rely on vec env
            for n_iter in range(n_repeat):
                fitness_scores[idx][n_iter] = task.rollout(solution, evaluate=evaluate)
        return fitness_scores

    def train_once(algorithm, n_repeat: int = 5):
        fitness = collect_fitness(algorithm, n_repeat, evaluate=False).mean(axis=1)
        algorithm.evolve(fitness)
        return fitness

    def evaluate(algorithm):
        return collect_fitness(algorithm, evaluate=True)

    def train(algorithm, max_iter: int = 5, eval_every_n_iter: int = 10):
        # Evaluate before train.
        eval_scores = evaluate(algorithm)
        print(f"iter: {0}, scores: {eval_scores}")

        best_eval_score = -float('Inf')

        for iter_idx in range(max_iter):
            start_time = time.time()
            scores = train_once(algorithm)
            time_cost = time.time() - start_time
            
            print(f"training time: {time_cost}s, iter: {iter_idx+1}, scores: {scores}")
            
            if (iter_idx + 1) % eval_every_n_iter == 0:
                start_time = time.time()
                eval_scores = evaluate(algorithm)
                time_cost = time.time() - start_time
                
                print(f"evaluation time: {time_cost}s")

                mean_score = eval_scores.mean()
                if mean_score > best_eval_score:
                    best_eval_score = mean_score
                    best_so_far = True
                else:
                    best_so_far = False

                print(f"iter: {iter_idx + 1}, scores: {eval_scores}")

    def run_worker(solution, params: np.ndarray, env_seed: int = 0, evaluate: bool = False):
        solution.set_params(params)
        # task.seed(env_seed)
        score = task.rollout(solution, evaluate)
        penalty = 0 if evaluate else solution.get_l2_penalty()
        return score - penalty

    train(algo, max_iter=10)