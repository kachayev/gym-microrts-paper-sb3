import abc
from collections import OrderedDict
import multiprocessing as mp
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
    HierachicalMultiCategoricalDistribution,
    MicroRTSStatsRecorder,
    layer_init,
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
        # torch.set_num_threads(1)
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
            layer_init(nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

    def forward(self, x):
        x = x.permute((0,3,1,2))
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        return x


class CNNActor(nn.Module):

    def __init__(self, output_channels, num_cells=256):
        super(CNNActor, self).__init__()

        self.output_channels = output_channels
        self.num_cells = num_cells
        self.actor = nn.Sequential(
            layer_init(nn.ConvTranspose2d(num_cells, 128, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(32, output_channels, 3, stride=2, padding=1, output_padding=1)),
        )
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.reshape((1, self.num_cells, 1, 1))
        x = self.actor(x)
        x = x.permute((0, 2, 3, 1))
        x = x.reshape((-1, self.output_channels*self.num_cells))
        return x

class LinearActor(nn.Module):
    
    def __init__(self, output_channels, hidden_dim=32, num_cells=256):
        super(LinearActor, self).__init__()

        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        self.num_cells = num_cells
        self.network = nn.Sequential(
            layer_init(nn.Linear(1, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, self.output_channels))
        )

    def forward(self, x):
        x = self.network(x.unsqueeze(-1))
        x = x.reshape((-1, self.output_channels*self.num_cells))
        return x


# xxx(okachaiev): I don't like the fact we need to manipulate with
# "_layers". seems like having "register_layer" as a public API would
# be much easier to deal with
class GridnetSolution(BaseTorchSolution):

    def __init__(self, observation_space, action_space, actor="linear", device: str = "auto"):
        super().__init__()

        self.device = get_device(device)

        self.action_space_size = action_space.nvec.sum()
        self.height, self.width, self.input_channels = observation_space['obs'].shape
        self.num_cells = self.height * self.width
        self.action_plane = action_space.nvec[:action_space.nvec.size // self.num_cells]
        self.action_dist = HierachicalMultiCategoricalDistribution(self.num_cells, self.action_plane)

        self.latent_net = Encoder(self.input_channels).to(self.device)
        self.register_layers(self.latent_net)
        if actor == "linear":
            self.policy_net = LinearActor(self.action_plane.sum()).to(self.device)
        elif actor == "cnn":
            self.policy_net = CNNActor(self.action_plane.sum()).to(self.device)
        self.register_layers(self.policy_net)

        self._mask_value = torch.tensor(-1e+8, device=self.device)

        print('Number of parameters: {}'.format(self.get_num_params_per_layer()))

    def rename_layers(self, weights, layers_mapping):
        return OrderedDict([
            (layers_mapping[name], tensor)
            for name, tensor in weights.items()
            if name in layers_mapping
        ])

    # xxx(okachaiev): I believe there's should be an easier way of doing this...
    # most likely named modules is a way to go thought I'm not sure what to
    # do about indecies being messed up. maybe that's the reason why layers
    # like Reshape are considered to be a bad practice in torch community
    def load_from_pretrained(self, path: str):
        weights = torch.load(path, map_location=self.device)

        encoder_layers = {
            "encoder._encoder.1.weight": "encoder.0.weight",
            "encoder._encoder.1.bias": "encoder.0.bias",
            "encoder._encoder.4.weight": "encoder.3.weight",
            "encoder._encoder.4.bias": "encoder.3.bias",
            "encoder._encoder.7.weight": "encoder.6.weight",
            "encoder._encoder.7.bias": "encoder.6.bias",
            "encoder._encoder.10.weight": "encoder.9.weight",
            "encoder._encoder.10.bias": "encoder.9.bias",
        }
        self.latent_net.load_state_dict(self.rename_layers(weights, encoder_layers))

        actor_layers = {
            "actor.deconv.0.weight": "actor.0.weight",
            "actor.deconv.0.bias": "actor.0.bias",
            "actor.deconv.2.weight": "actor.2.weight",
            "actor.deconv.2.bias": "actor.2.bias",
            "actor.deconv.4.weight": "actor.4.weight",
            "actor.deconv.4.bias": "actor.4.bias",
            "actor.deconv.6.weight": "actor.6.weight",
            "actor.deconv.6.bias": "actor.6.bias",
        }
        self.policy_net.load_state_dict(self.rename_layers(weights, actor_layers))

    def register_layers(self, module: nn.Module, std=np.sqrt(2)):
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
            CompressedParams(self.rng.randint(1 << 31 - 1)).evolve(self.sigma)
            for _ in range(population_size)
        ]

    def get_population(self):
        return self.population

    def evolve(self, fitness):
        print(f"fitness best: {np.max(fitness)}, avg: {np.mean(fitness)}")

        top_k = np.argpartition(fitness,-self.truncate)[-self.truncate:]
        # xxx(okachaiev): i should also cache their fitness
        # so I don't need to re-run again :thinking:
        elite = [self.population[k] for k in top_k]

        print(f"top_k: {top_k}, fitness: {fitness[top_k]}")

        ind = np.random.choice(top_k, self.population_size-self.truncate)
        evolution = [self._evolve_single(i) for i in ind]
        self.population = elite + evolution
        return self

    def _evolve_single(self, ind: int) -> CompressedParams:
        params = CompressedParams(self.population[ind].init_seed)
        params.generations = self.population[ind].generations[:]
        params.evolve(self.sigma)
        return params

    def get_current_parameters(self):
        pass

solution = None
task = None

def worker_init(actor="cnn"):
    global solution, task

    task = MicroRTSTask().create_task()
    solution = GridnetSolution(task._env.observation_space, task._env.action_space, actor=actor)
    solution.load_from_pretrained("../gym-microrts-paper/trained_models/ppo_gridnet_selfplay_diverse_encode_decode/agent-4.pt")

def worker_run(req):
    global solution, task

    compressed_params, n_repeat, evaluate = req
    current_params = solution.get_params()
    std_prev = current_params.std()
    new_params = compressed_params.uncompress(current_params)
    std_new = new_params.std()
    solution.set_params(new_params)

    print(f"generations {len(compressed_params.generations)}, std prev: {std_prev}, std new: {std_new}")

    fitness_scores = np.zeros(n_repeat)
    for n_iter in range(n_repeat):
        fitness_scores[n_iter] = task.rollout(solution, evaluate=evaluate)

    return fitness_scores


if __name__ == "__main__":
    actor = "cnn"

    # xxx(okachaiev): this API is horrible :(
    # task = MicroRTSTask().create_task()
    # solution = GridnetSolution(task._env.observation_space, task._env.action_space, actor=actor)
    # rng = np.random.RandomState(seed=0)

    # task.rollout(solution, evaluate=False)

    num_workers = 10
    algo = GaussianNoiseGA(population_size=1024, truncate=20, init_seed=1048, sigma=0.002)

    def collect_fitness(pool, algorithm, n_repeat: int = 2, evaluate: bool = False):
        population = algorithm.get_population()
        fitness_scores = pool.map(
            func=worker_run,
            iterable=((params, n_repeat, evaluate) for params in population)
        )
        return np.array(fitness_scores)

    def train_once(pool, algorithm, n_repeat: int = 2):
        fitness = collect_fitness(pool, algorithm, n_repeat, evaluate=False).mean(axis=1)
        algorithm.evolve(fitness)
        return fitness

    def evaluate(pool, algorithm):
        return collect_fitness(pool, algorithm, evaluate=True)

    def train(pool, algorithm, max_iter: int = 5, eval_every_n_iter: int = 5, n_repeat = 2):
        eval_scores = evaluate(pool, algorithm)
        print(f"iter: {0}, scores: {eval_scores}")

        best_eval_score = -float('Inf')

        for iter_idx in range(max_iter):
            start_time = time.time()
            scores = train_once(pool, algorithm, n_repeat)
            time_cost = time.time() - start_time
            
            print(f"training time: {time_cost}s, iter: {iter_idx+1}, scores: {scores}")
            
            if (iter_idx + 1) % eval_every_n_iter == 0:
                start_time = time.time()
                eval_scores = evaluate(pool, algorithm)
                time_cost = time.time() - start_time
                
                print(f"evaluation time: {time_cost}s")

                mean_score = eval_scores.mean()
                if mean_score > best_eval_score:
                    best_eval_score = mean_score
                    best_so_far = True
                else:
                    best_so_far = False

                print(f"iter: {iter_idx + 1}, scores: {eval_scores}")

    with mp.get_context('spawn').Pool(
            initializer=worker_init,
            initargs=(actor,),
            processes=num_workers,
    ) as pool:
        train(pool, algo, max_iter=16, eval_every_n_iter=4, n_repeat=8)