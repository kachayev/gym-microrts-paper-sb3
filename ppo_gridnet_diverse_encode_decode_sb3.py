import numpy as np
import time
import torch
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy, register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecMonitor

import gym
import gym_microrts
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai


class CustomMicroRTSGridMode(MicroRTSGridModeVecEnv):

    def __init__(self, *args, **kwargs):
        # xxx(okachaiev): seems like there's no need to provide
        # this parameter separately
        kwargs['num_bot_envs'] = len(kwargs.get('ai2s', []))
        super().__init__(*args, **kwargs)
        self.num_cells = self.height*self.width
        # self.action_space = gym.spaces.MultiDiscrete(np.array([
        #     [6, 4, 4, 4, 4, len(self.utt['unitTypes']), 7 * 7]
        # ] * self.height * self.width).flatten())
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

    def step_async(self, action):
        action = action.reshape(self.num_envs, self.num_cells, -1)
        return super().step_async(action)

    def step(self, action):
        action = action.reshape(self.num_envs, self.num_cells, -1)
        return super().step(action)

    def step_wait(self):
        obs, rewards, dones, infos = super().step_wait()
        masks = self.get_action_mask()
        return {"obs": obs, "masks": masks}, rewards, dones, infos

    def reset(self):
        obs = super().reset()
        masks = self.get_action_mask()
        return {"obs": obs, "masks": masks}


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

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations):
        observations

# xxx(okachaiev): should this go to "feature extractor" configuration?
class MicroRTSExtractor(nn.Module):

    def __init__(self, input_channels=27, output_channels=78, action_space_size=None, device = "auto"):
        super().__init__()

        # xxx(okachaiev): requires reading the documentation
        # to know about these properties. maybe ABC class
        # with proper exception if the method is not implemented
        # would be cleaner?
        self.latent_dim_pi = output_channels
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
            Reshape((-1,action_space_size))
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


class MicroRTSGridActorCritic(ActorCriticPolicy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # xxx(okachaiev): hack
        # not sure if turning nets & extractors into identity
        # layers is how it supposed to be done
        self.action_net = nn.Identity()
        # xxx(okachaiev): hack
        # it seems like we can avoid doing additional network here:
        # https://github.com/DLR-RM/stable-baselines3/blob/201fbffa8c40a628ecb2b30fd0973f3b171e6c4c/stable_baselines3/common/policies.py#L557
        # in case self.mlp_extractor.latent_dim_vf == 1
        self.value_net = nn.Identity()

    def _build_mlp_extractor(self) -> None:
        # xxx(okachaiev): would be nice if SB3 provided configuration for
        # MlpExtractor class. in this case I wouldn't need to reload
        # "internal" function of the class
        self.mlp_extractor = MicroRTSExtractor(
            input_channels=27,
            # output_channels=self.action_space.nvec[1:].sum(),
            # xxx(okachaiev): need to find a way to propagate parameters
            output_channels=78,
            action_space_size=self.action_space.nvec.sum(),
        )

    def extract_features(self, obs):
        return obs['obs'].float(), obs['masks'].bool()


if __name__ == "__main__":
    register_policy('MicroRTSGridActorCritic', MicroRTSGridActorCritic)

    envs = CustomMicroRTSGridMode(
        num_selfplay_envs=0,
        max_steps=2000,
        render_theme=2,
        ai2s=[
            microrts_ai.randomBiasedAI,
            # microrts_ai.lightRushAI,
            # microrts_ai.workerRushAI,
            # microrts_ai.coacAI,
        ],
        map_paths=["maps/16x16/basesWorkers16x16.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    )
    envs = VecMonitor(envs)

    model = PPO(
        'MicroRTSGridActorCritic',
        envs,
        verbose=1,
        policy_kwargs=dict(ortho_init=False, features_extractor_class=NoopFeaturesExtractor)
    )
    model.learn(total_timesteps=10_000)