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

    def __init__(
            self,
            population_size,
            init_sigma,
            init_params,
            seed=42
        ):
        self.algorithm = cma.CMAEvolutionStrategy(
            x0=init_params,
            sigma0=init_sigma,
            inopts={
                'popsize': population_size,
                'seed': seed, 
                'randn': np.random.randn,
            },
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
    """Base solution."""

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
        if not hasattr(self, '_l2_coefficient'):
            raise ValueError('l2_coefficient not specified.')
        params = self.get_params()
        return self._l2_coefficient * np.sum(params ** 2)

    def add_noise(self, noise):
        self.set_params(self.get_params() + noise)

    def add_noise_to_layer(self, noise, layer_index):
        layer_params = self.get_params_from_layer(layer_index)
        assert layer_params.size == noise.size, '#params={}, #noise={}'.format(
            layer_params.size, noise.size)
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

    def roll_out(self, solution, evaluate):
        ob = self.reset()
        ob = self._process_observation(ob)
        if hasattr(solution, 'reset'):
            solution.reset()

        start_time = time.time()

        rewards = []
        done = False
        step_cnt = 0
        while not done:
            action = solution.get_output(inputs=ob, update_filter=not evaluate)
            action = self._process_action(action)
            ob, r, done, _ = self.step(action, evaluate)
            ob = self._process_observation(ob)

            step_cnt += 1
            done = self._overwrite_terminate_flag(r, done, step_cnt, evaluate)
            step_reward = self._process_reward(r, done, evaluate)
            rewards.append(step_reward)

        time_cost = time.time() - start_time
        actual_reward = np.sum(rewards)
        
        print('Roll-out time={0:.2f}s, steps={1}, reward={2:.2f}'.format(
            time_cost, step_cnt, actual_reward))

        return actual_reward

class CMAMaster:

    def __init__(self,
                 logger,
                 log_dir,
                 workers,
                 bucket_name,
                 experiment_name,
                 credential_json,
                 seed,
                 n_repeat,
                 max_iter,
                 eval_every_n_iter,
                 n_eval_roll_outs):
        self._logger = logger
        self._log_dir = log_dir
        self._n_repeat = n_repeat
        self._max_iter = max_iter
        self._eval_every_n_iter = eval_every_n_iter
        self._n_eval_roll_outs = n_eval_roll_outs
        self._solution = misc.utility.create_solution()
        self._rnd = np.random.RandomState(seed=seed)
        self._algorithm = CMA(
            logger=logger, seed=seed, init_params=self._solution.get_params())

    def train(self):
        # Evaluate before train.
        eval_scores = self._evaluate()
        misc.utility.log_scores(
            logger=self._logger, iter_cnt=0, scores=eval_scores, evaluate=True)
        misc.utility.save_scores(
            log_dir=self._log_dir, n_iter=0, scores=eval_scores)
        best_eval_score = -float('Inf')

        self._logger.info(
            'Start training for {} iterations.'.format(self._max_iter))
        for iter_cnt in range(self._max_iter):
            # Training.
            start_time = time.time()
            scores = self._train_once()
            time_cost = time.time() - start_time
            self._logger.info('1-step training time: {}s'.format(time_cost))
            misc.utility.log_scores(
                logger=self._logger, iter_cnt=iter_cnt + 1, scores=scores)

            # Evaluate periodically.
            if (iter_cnt + 1) % self._eval_every_n_iter == 0:
                # Evaluate.
                start_time = time.time()
                eval_scores = self._evaluate()
                time_cost = time.time() - start_time
                self._logger.info('Evaluation time: {}s'.format(time_cost))

                # Record results and save the model.
                mean_score = eval_scores.mean()
                if mean_score > best_eval_score:
                    best_eval_score = mean_score
                    best_so_far = True
                else:
                    best_so_far = False
                misc.utility.log_scores(logger=self._logger,
                                        iter_cnt=iter_cnt + 1,
                                        scores=eval_scores,
                                        evaluate=True)
                misc.utility.save_scores(log_dir=self._log_dir,
                                         n_iter=iter_cnt + 1,
                                         scores=eval_scores)
                self._save_solution(iter_count=iter_cnt + 1,
                                    best_so_far=best_so_far)

    def _create_rpc_requests(self, evaluate):
        """Create gRPC requests."""

        if evaluate:
            n_repeat = 1
            num_roll_outs = self._n_eval_roll_outs
            params_list = [self._algorithm.get_current_parameters()]
        else:
            n_repeat = self._n_repeat
            params_list = self._algorithm.get_population()
            num_roll_outs = len(params_list) * n_repeat

        env_seed_list = self._rnd.randint(
            low=0, high=MAX_INT, size=num_roll_outs)

        requests = []
        for i, env_seed in enumerate(env_seed_list):
            ix = 0 if evaluate else i // n_repeat
            requests.append(self._communication_helper.create_cma_request(
                roll_out_index=i,
                env_seed=env_seed,
                parameters=params_list[ix],
                evaluate=evaluate,
            ))
        return requests

    def _evaluate(self):
        requests = self._create_rpc_requests(evaluate=True)
        fitness = self._communication_helper.collect_fitness_from_workers(
            requests=requests,
        )
        return fitness

    def _train_once(self):
        requests = self._create_rpc_requests(evaluate=False)

        fitness = self._communication_helper.collect_fitness_from_workers(
            requests=requests,
        )
        fitness = fitness.reshape([-1, self._n_repeat]).mean(axis=1)
        self._algorithm.evolve(fitness)

        return fitness

    def _save_solution(self, iter_count, best_so_far):
        self._update_solution()
        self._solution.save(self._log_dir, iter_count, best_so_far)

    def _update_solution(self):
        self._solution.set_params(self._algorithm.get_current_parameters())


class CMAWorker:

    def __init__(self, logger):
        self._logger = logger
        self._task = misc.utility.create_task(logger=logger)
        self._solution = misc.utility.create_solution()
        self._communication_helper = misc.communication.CommunicationHelper(
            logger=logger)

    def _handle_master_request(self, request):
        params = np.asarray(request.cma_parameters.parameters)
        self._solution.set_params(params)
        self._task.seed(request.env_seed)
        score = self._task.roll_out(self._solution, request.evaluate)
        penalty = 0 if request.evaluate else self._solution.get_l2_penalty()
        return score - penalty

    def performRollOut(self, request, context):
        fitness = self._handle_master_request(request)
        return self._communication_helper.report_fitness(
            roll_out_index=request.roll_out_index,
            fitness=fitness,
        )



# 
# todos:
# 1) "agent" with torch layers that can go from obs to action
# 2) connect this to get_params/set_params for layers
# 3) roll out env, get sum of rewards (to use as fitness)
# 4) "worker" to create instance of env & "agent" (solution), handle new params
# 5) CMA to handle population and cycle of calles to workers


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

    def __init__(self, input_channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.MaxPool2d(3, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.MaxPool2d(3, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.MaxPool2d(3, stride=2, padding=1)
        )

    def forward(self, x):
        x = x.permute((0,3,1,2))
        x = self.network(x)
        return x

class Policy(nn.Module):
    
    def __init__(self, output_channels, action_space_size):
        super().__init__()
        self.action_space_size = action_space_size
        self.network = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, 3, stride=2, padding=1, output_padding=1),
        )
    
    def forward(self, x):
        x = self.network(x)
        x = x.permute((0,2,3,1))
        x = x.reshape((-1, self.action_space_size))
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
        self.policy_net = Policy(self.action_plane.sum(), self.action_space_size).to(self.device)
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


if __name__ == "__main__":
    # xxx(okachaiev): this API is horrible :(
    task = MicroRTSTask().create_task()
    solution = GridnetSolution(task._env.observation_space, task._env.action_space)
    rng = np.random.RandomState(seed=0)

    task.roll_out(solution, evaluate=False)

    algo = CMA(population_size=4, init_sigma=0.1, init_params=solution.get_params())

    def collect_fitness_from_workers(algorithm, num_roll_outs: int = 2, n_repeat: int = 2, evaluate: bool = False):
        if evaluate:
            n_repeat = 1
            params_list = [algorithm.get_current_parameters()]
        else:
            params_list = algorithm.get_population()
            num_roll_outs = len(params_list) * n_repeat

        env_seed_list = rng.randint(
            low=0, high=MAX_INT, size=num_roll_outs)

        fitness_scores = np.zeros(num_roll_outs)
        for roll_out_ind in range(num_roll_outs):
            # xxx(okachaiev): cycle
            ix = 0 if evaluate else iroll_out_ind // n_repeat
            fitness = run_worker(params=params_list[ix], evaluate=evaluate)
            fitness_scores[roll_out_ind] = fitness
        return fitness_scores

    def train_once(algorithm, n_repeat: int = 5):
        fitness = collect_fitness_from_workers(algorithm, evaluate=False)
        fitness = fitness.reshape([-1, n_repeat]).mean(axis=1)
        algorithm.evolve(fitness)
        return fitness

    def evaluate(algorithm):
        return collect_fitness_from_workers(algorithm, evaluate=True)

    def train(algorithm, max_iter: int = 5, eval_every_n_iter: int = 10):
        # Evaluate before train.
        eval_scores = evaluate(algorithm)
        print(f"iter: {0}, scores: {eval_scores}")

        best_eval_score = -float('Inf')

        for iter_cnt in range(max_iter):
            # Training.
            start_time = time.time()
            scores = train_once(algorithm)
            time_cost = time.time() - start_time
            
            print(f"1-step training time: {time_cost}s, iter: {iter_cnt+1}, scores: {scores}")
            
            # Evaluate periodically.
            if (iter_cnt + 1) % eval_every_n_iter == 0:
                # Evaluate.
                start_time = time.time()
                eval_scores = evaluate(algorithm)
                time_cost = time.time() - start_time
                
                print(f"Evaluation time: {time_cost}s")

                # Record results and save the model.
                mean_score = eval_scores.mean()
                if mean_score > best_eval_score:
                    best_eval_score = mean_score
                    best_so_far = True
                else:
                    best_so_far = False

                print(f"iter: {iter_cnt + 1}, scores: {eval_scores}")

    def run_worker(params: np.ndarray, env_seed: int = 0, evaluate: bool = False):
        solution.set_params(params)
        task.seed(env_seed)
        score = task.roll_out(solution, evaluate)
        penalty = 0 if evaluate else solution.get_l2_penalty()
        return score - penalty

    ## train(algo)