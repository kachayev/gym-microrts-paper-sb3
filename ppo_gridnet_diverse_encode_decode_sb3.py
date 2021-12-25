import argparse
from distutils.util import strtobool
import numpy as np
import os
import time
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from typing import Callable, List, Tuple, Union

from stable_baselines3 import PPO
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy, register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecMonitor, VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback

import gym
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv


def _parse_bot_envs(values: Union[str, List[str]]) -> List[Callable]:
    if isinstance(values, str):
        values = values.split(' ')
    bots = []
    for value in values:
        key, value = value.split('=')
        bots.extend([getattr(microrts_ai, key) for _ in range(int(value))])
    return bots


class ParseBotEnvs(argparse.Action):
    def __call__(self, _parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, _parse_bot_envs(values))

# default argument values are defined to be as close to the paper implementation as possible
# https://github.com/vwxyzjn/gym-microrts-paper/blob/cf291b303c04e98be2f00acbbe6bbb2c23a8bac5/ppo_gridnet_diverse_encode_decode.py#L25
def parse_arguments():
    parser = argparse.ArgumentParser(description='PPO agent')

    # environment setup
    parser.add_argument('--exp-folder', type=str, default="agents",
                        help='folder to store experiments')
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--max-steps', type=int, default=2_000,
                        help='max number of steps per game environment')
    parser.add_argument('--bot-envs', nargs='*', action=ParseBotEnvs,
                        default=_parse_bot_envs('randomBiasedAI=2 lightRushAI=2 workerRushAI=2 coacAI=18'),
                        help='bot envs to setup following "bot_name=<num envs>" format')
    parser.add_argument('--num-selfplay-envs', type=int, default=0,
                        help='the number of self play envs; 16 self play envs means 8 games')

    # hyperparams
    parser.add_argument('--total-timesteps', type=int, default=100_000_000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--num-steps', type=int, default=256,
                        help='the number of steps per game environment')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='minibatch size')
    parser.add_argument('--target-kl', type=float, default=0.03,
                        help='the target-kl variable that is referred by --kl')
    parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggle learning rate annealing for policy and value networks")

    # xxx(okachaiev): I assume this one is called `clip_range` in SB3 and `clip_range` in the paper
    parser.add_argument('--clip-range', type=float, default=0.1,
                        help="the surrogate clipping coefficient")

    # xxx(okachaiev): SB3 is a bit more flexible and allows `clip_range_vf`
    # to be set independently
    parser.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')

    # xxx(okachaiev): I assume this one is called `n_epochs` in SB3 and `update_epochs` in the paper
    parser.add_argument('--n-epochs', type=int, default=4,
                        help="the K epochs to update the policy")

    # xxx(okachaiev): the code for the paper has advantages norm as a toggle (with True by default)
    # https://github.com/vwxyzjn/gym-microrts-paper/blob/cf291b303c04e98be2f00acbbe6bbb2c23a8bac5/ppo_gridnet_diverse_encode_decode.py#L81
    # SB3 does adv norm always, though I'm not sure it was mentioned in the origin paper
    # also, see this discussion:
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/issues/102

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    args.experiment_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.clip_range_vf = args.clip_range if args.clip_vloss else None
    if args.anneal_lr:
        lr = lambda f: f * args.learning_rate

    return args


class CustomMicroRTSGridMode(MicroRTSGridModeVecEnv):

    def __init__(self, *args, **kwargs):
        # xxx(okachaiev): seems like there's no need to provide
        # this parameter separately
        kwargs['num_bot_envs'] = len(kwargs.get('ai2s', []))
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Dict({
            "obs": self.observation_space,
            "masks": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.action_space.nvec.sum(),),
                dtype=np.int32
            ),
        })

    def get_action_mask(self):
        return super().get_action_mask().reshape(self.num_envs, -1)

    def step_wait(self):
        obs, rewards, dones, infos = super().step_wait()
        masks = self.get_action_mask()
        return {"obs": obs, "masks": masks}, rewards, dones, infos

    def reset(self):
        obs = super().reset()
        masks = self.get_action_mask()
        return {"obs": obs, "masks": masks}

    def seed(self, seed_value):
        # xxx(okachaiev): it would be nice if we could pass seed value into
        # the game env itself. just ignoring for now
        pass


class MicroRTSStatsRecorder(VecEnvWrapper):
    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        for i in range(len(dones)):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]
        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                raw_rewards = np.array(self.raw_rewards[i]).sum(0)
                raw_names = [str(rf) for rf in self.rfs]
                info["microrts_stats"] = dict(zip(raw_names, raw_rewards))
                self.raw_rewards[i] = []
                newinfos[i] = info
        return obs, rews, dones, newinfos


class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)

# xxx(okachaiev): using modules for functional non-gradient
# transformations seems like not a torch-like pattern
class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.reshape(self.shape)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class NoopFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 0):
        super(BaseFeaturesExtractor, self).__init__()
        self._observation_space = observation_space
        self._features_dim = features_dim

    def forward(self, observations):
        return observations

# xxx(okachaiev): should this go to "feature extractor" configuration?
class MicroRTSExtractor(nn.Module):

    def __init__(self, input_channels=27, output_channels=78, action_space_size=None, device = "auto"):
        super().__init__()

        # xxx(okachaiev): requires reading the documentation
        # to know about these properties. maybe ABC class
        # with proper exception if the method is not implemented
        # would be cleaner?
        self.latent_dim_pi = action_space_size
        self.latent_dim_vf = 1
    
        self.device = get_device(device)

        self.shared_net = nn.Sequential(
            Transpose((0, 3, 1, 2)),
            layer_init(nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1)
        ).to(self.device)

        self.policy_net = nn.Sequential(
            layer_init(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(32, output_channels, 3, stride=2, padding=1, output_padding=1)),
            Transpose((0, 2, 3, 1)),
            Reshape((-1, action_space_size))
        ).to(self.device)

        # xxx(okachaiev): hack (seems like)
        # not sure what is the correct approach in SB3
        # should it be here? or should this be a "value_net"
        # and "action_net" from the policy itself?
        self.value_net = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(256, 128), std=1),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1),
        ).to(self.device)

    def _mask_action_logits(self, latent_pi, masks, mask_value=None):
        mask_value = mask_value or torch.tensor(-1e+8, device=self.device)
        return torch.where(masks, latent_pi, mask_value)

    def forward(self, features):
        obs, masks = features
        shared_latent = self.shared_net(obs)
        return self._mask_action_logits(self.policy_net(shared_latent), masks), self.value_net(shared_latent)

    def forward_actor(self, features):
        obs, masks = features
        return self._mask_action_logits(self.policy_net(self.shared_net(obs)), masks)

    def forward_critic(self, features):
        obs, _ = features
        return self.value_net(self.shared_net(obs))


class HierachicalMultiCategoricalDistribution(Distribution):

    def __init__(self, split_level: int, action_dims: List[int]):
        super(HierachicalMultiCategoricalDistribution, self).__init__()
        self.num_envs = None
        self.split_level = split_level
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Identity()

    # xxx(okachaiev): hack (or seems like the one)
    # working with a mutable distribution object seems like the
    # safest operation... as we change dimentionality a few times
    # when switching between `forward` and `evaluate_actions` we
    # recreate array of Categorical distributions of a different
    # size
    def proba_distribution(self, action_logits: torch.Tensor) -> "HierachicalMultiCategoricalDistribution":
        action_logits = action_logits.reshape((-1,self.split_level,self.action_dims.sum()))
        self.num_envs = action_logits.shape[0]
        self.distribution = [Categorical(logits=split) for split in torch.split(action_logits, tuple(self.action_dims), dim=-1)]
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.reshape((-1,self.split_level,len(self.action_dims)))
        return torch.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, torch.unbind(actions, dim=2))], dim=2
        ).sum(dim=-1).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([dist.entropy() for dist in self.distribution], dim=2).sum(dim=-1).sum(dim=-1)

    def sample(self) -> torch.Tensor:
        return torch.stack([dist.sample() for dist in self.distribution], dim=2).reshape((self.num_envs, -1))

    def mode(self) -> torch.Tensor:
        return torch.stack([torch.argmax(dist.probs, dim=2) for dist in self.distribution], dim=2).respahe((self.num_envs, -1))

    def actions_from_params(self, action_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class MicroRTSGridActorCritic(ActorCriticPolicy):

    def __init__(self, observation_space, action_space, *args, **kwargs):
        self.height, self.width, self.input_channels = observation_space['obs'].shape
        self.num_cells = self.height * self.width
        self.action_plane = action_space.nvec[:action_space.nvec.size // self.num_cells]

        super().__init__(observation_space, action_space, *args, **kwargs)

        self.action_dist = HierachicalMultiCategoricalDistribution(self.num_cells, self.action_plane)

        # xxx(okachaiev): hack
        # not sure if turning nets & extractors into identity
        # layers is how it supposed to be done
        self.action_net = nn.Identity()
        # xxx(okachaiev): hack
        # it seems like we can avoid doing additional network here:
        # https://github.com/DLR-RM/stable-baselines3/blob/201fbffa8c40a628ecb2b30fd0973f3b171e6c4c/stable_baselines3/common/policies.py#L557
        # in case self.mlp_extractor.latent_dim_vf == 1
        self.value_net = nn.Identity()

    # xxx(okachaiev): feels like a hack
    # it would be much nicers if we can return distribution object from
    # extract call without passing latent values thought the policy object
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor, latent_sde=False) -> Distribution:
        return self.action_dist.proba_distribution(action_logits=latent_pi)

    # xxx(okachaiev): would be nice if SB3 provided configuration for
    # MlpExtractor class. in this case I wouldn't need to reload
    # "internal" function of the class
    # xxx(okachaiev): also, should it be called "latent extractor"?
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MicroRTSExtractor(
            input_channels=self.input_channels,
            output_channels=self.action_plane.sum(),
            action_space_size=self.action_space.nvec.sum(),
        )

    def extract_features(self, obs):
        return obs['obs'].float(), obs['masks'].bool()



class MicroRTSStatsCallback(BaseCallback):
    """
    Edit _on_step for plotting in tensorboard every 10000 steps.
    """
    def __init__(self, verbose=0):
        super(MicroRTSStatsCallback, self).__init__(verbose)
    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            if "episode" in info.keys():
                self.logger.record("charts/episodic_return", info["episode"]["r"])
                for key in info["microrts_stats"]:
                    self.logger.record(f"charts/episodic_return/{key}", info["microrts_stats"][key])
                self.logger.dump(self.num_timesteps)
                break


if __name__ == "__main__":
    register_policy('MicroRTSGridActorCritic', MicroRTSGridActorCritic)

    args = parse_arguments()

    envs = CustomMicroRTSGridMode(
        num_selfplay_envs=args.num_selfplay_envs,
        max_steps=args.max_steps,
        render_theme=2,
        ai2s=args.bot_envs,
        map_paths=["maps/16x16/basesWorkers16x16.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    )
    envs = MicroRTSStatsRecorder(envs)
    envs = VecMonitor(envs)

    model = PPO(
        'MicroRTSGridActorCritic',
        envs,
        verbose=1,
        policy_kwargs=dict(
            ortho_init=False,
            features_extractor_class=NoopFeaturesExtractor,
        ),
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        n_steps=args.num_steps,
        batch_size=args.batch_size,
        target_kl=args.target_kl,
        clip_range=args.clip_range,
        n_epochs=args.n_epochs,
        seed=args.seed,
        device='auto',
        tensorboard_log=f"runs/test",
    )
    model.learn(total_timesteps=args.total_timesteps, callback=MicroRTSStatsCallback())
    model.save(f"{args.exp_folder}/{args.experiment_name}")