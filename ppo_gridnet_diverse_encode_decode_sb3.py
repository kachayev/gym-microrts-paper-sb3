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
from stable_baselines3.common.vec_env import VecMonitor

import gym
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

# import for custom action dist parameter
import torch as th
from typing import Any, Dict, Optional, Type
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.torch_layers import (
    FlattenExtractor,
    NatureCNN
)
from stable_baselines3.common.type_aliases import Schedule


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

    def __init__(self, action_space, num_cells=256):
        super(HierachicalMultiCategoricalDistribution, self).__init__()
        self.num_envs = None
        self.num_cells = num_cells
        self.action_dims = action_space.nvec[:action_space.nvec.size // self.num_cells]
        self.action_dims_total = self.action_dims.sum()
        self.action_dims_size = len(self.action_dims)

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Identity()

    # xxx(okachaiev): hack (or seems like the one)
    # working with a mutable distribution object seems like the
    # safest operation... as we change dimentionality a few times
    # when switching between `forward` and `evaluate_actions` we
    # recreate array of Categorical distributions of a different
    # size
    def proba_distribution(self, action_logits: torch.Tensor) -> "HierachicalMultiCategoricalDistribution":
        action_logits = action_logits.reshape((-1,self.num_cells,self.action_dims_total))
        self.num_envs = action_logits.shape[0]
        self.distribution = [Categorical(logits=split) for split in torch.split(action_logits, tuple(self.action_dims), dim=-1)]
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.reshape((-1,self.num_cells,self.action_dims_size))
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


class CustomActionDistActorCriticPolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        action_dist_class: Optional[Type[Distribution]] = None,
        action_dist_kwargs: Optional[Dict[str, Any]] = None,
    ):

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(ActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = {}
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs.update({
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": sde_net_arch is not None,
            })

        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        self.custom_action_dist_class = action_dist_class
        self.custom_action_dist_kwargs = action_dist_kwargs or {}
        # Action distribution
        if action_dist_class is None:
            self.action_dist = make_proba_distribution(
                action_space,
                use_sde=use_sde,
                dist_kwargs=dist_kwargs,
            )
        else:
            # xxx(okachaiev): merge with `dist_kwargs`?
            self.action_dist = action_dist_class(action_space, **self.custom_action_dist_kwargs)

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                sde_net_arch=self.sde_net_arch,
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                # xxx(okachaiev): simply add custom action dist parameters 
                action_dist_class=self.custom_action_dist_class,
                action_dist_kwargs=self.custom_action_dist_kwargs,
            )
        )
        return data

    # xxx(okachaiev): what if my distribution also needs this?
    # we can use trait class "WeightsSampler" that declares as single
    # method `sample_weights`. and use isinstance check. we can also directly
    # check if method is in there, which is quite common in Python,
    # but might be confusing for users of the API
    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)


    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate features extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
            )
        # xxx(okachaiev): simplified code a little bit
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif self.custom_action_dist_class is not None:
            # xxx(okachaiev): same comment here as with `_get_action_dist_from_latent`
            # seems like a hack. Technically, there's no need to check 3 dist above + this
            # "custom" use case. we should rely on the fact that dist is a subclass of `Distribution`
            # and just call `proba_distribution_net` with appropriate set of inputs (how to know
            # which of them are appropriate is harder question though)
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        elif self.custom_action_dist_class is not None:
            # xxx(okachaeiv): if log_std or latent_sde are needed, have to subclass from the
            # corresponding distribution. not sure this is going to be obvious for users :(
            # maybe their's a better way to declare what information is neccessary for each call?
            # also, why `action_net` is applied here and not in the distribution?
            # it seems if distribution class maintain `action_net` internally and applies it when
            # necessary, the flow is more straightfoward. e.g. no need to deal with `proba_distribution_net`
            # at all, -- most likely it's done this way to incorportate layer into nn.Module :(
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")



class MicroRTSGridActorCritic(CustomActionDistActorCriticPolicy):

    def __init__(self, observation_space, action_space, *args, **kwargs):
        self.height, self.width, self.input_channels = observation_space['obs'].shape
        self.num_cells = self.height * self.width
        self.action_plane = action_space.nvec[:action_space.nvec.size // self.num_cells]

        super().__init__(observation_space, action_space, *args, **kwargs)

        # xxx(okachaiev): hack
        # not sure if turning nets & extractors into identity
        # layers is how it supposed to be done
        self.action_net = nn.Identity()
        # xxx(okachaiev): hack
        # it seems like we can avoid doing additional network here:
        # https://github.com/DLR-RM/stable-baselines3/blob/201fbffa8c40a628ecb2b30fd0973f3b171e6c4c/stable_baselines3/common/policies.py#L557
        # in case self.mlp_extractor.latent_dim_vf == 1
        self.value_net = nn.Identity()

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
    envs = VecMonitor(envs, info_keywords=("microrts_stats",))

    model = PPO(
        'MicroRTSGridActorCritic',
        envs,
        verbose=1,
        policy_kwargs=dict(
            ortho_init=False,
            features_extractor_class=NoopFeaturesExtractor,
            action_dist_class=HierachicalMultiCategoricalDistribution,
            action_dist_kwargs=dict(num_cells=256),
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
    )
    model.learn(total_timesteps=args.total_timesteps)
    model.save(f"{args.exp_folder}/{args.experiment_name}")