import argparse
from distutils.util import strtobool
import numpy as np
import os
from pathlib import Path
import time
import torch
from torch import nn
from tqdm import trange

from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper

from gym_microrts import microrts_ai

from ppo_gridnet_diverse_encode_decode_sb3 import (
    CustomMicroRTSGridMode,
    HierachicalMultiCategoricalDistribution,
    layer_init,
    ParseBotEnvs,
    _parse_bot_envs
)
from rendering import Window, GameStatePanel, GameStatePanelConfig, Tilemap

class OfflineDatasetRecorder(VecEnvWrapper):

    def __init__(self, venv, folder:str, verbose:int=1):
        now = int(time.time())
        self.folder = Path(folder, str(now))
        self.folder.mkdir(parents=True, exist_ok=True)
        self.iteration = 0
        self.obs_buffer = []
        self.mask_buffer = []
        self.action_buffer = []

        if verbose > 0:
            print(f"Saving offline dataset to {self.folder}")

        super(OfflineDatasetRecorder, self).__init__(venv)

    def reset(self):
        self.save_dataset()
        obs = self.venv.reset()
        self.obs_buffer.append(obs['obs'])
        self.mask_buffer.append(obs['masks'])
        return obs

    def step_async(self, actions):
        self.action_buffer.append(actions)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.obs_buffer.append(obs['obs'])
        self.mask_buffer.append(obs['masks'])
        return obs, rewards, dones, infos

    # xxx(okachaiev): move this to callbacks
    # or consider using "imitation" package
    def save_dataset(self):
        if self.obs_buffer:
            min_len = min(len(self.obs_buffer), len(self.action_buffer))
            with self.folder.joinpath(f"{self.iteration}.npz").open("wb") as fd:
                np.savez_compressed(
                    fd,
                    obs=np.concatenate(self.obs_buffer[:min_len], axis=0),
                    mask=np.concatenate(self.mask_buffer[:min_len], axis=0),
                    action=np.concatenate(self.action_buffer[:min_len], axis=0)
                )
        self.iteration += 1
        self.obs_buffer = []
        self.mask_buffer = []
        self.action_buffer = []

    def close(self):
        self.save_dataset()
        self.venv.close()


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-model-file", type=str, required=True)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument('--max-steps', type=int, default=2_000,
                        help='max number of steps per game environment')
    parser.add_argument('--bot-envs', nargs='*', action=ParseBotEnvs,
                        default=_parse_bot_envs('randomBiasedAI=2 lightRushAI=2 workerRushAI=2 coacAI=18'),
                        help='bot envs to setup following "bot_name=<num envs>" format')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--capture-offline-dataset', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture trajectories into offline dataset (check out `offline_rl` folder)')
    return parser


def create_env(args):
    env = CustomMicroRTSGridMode(
        num_selfplay_envs=0,
        max_steps=args.max_steps,
        render_theme=2,
        ai2s=args.bot_envs,
        map_paths=["maps/16x16/basesWorkers16x16.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        microrts_jar_path="/Users/okachaiev/Workspace/gym-microrts-fork/gym-microrts/gym_microrts/microrts/microrts.jar",
    )
    env = VecMonitor(env)

    if args.capture_video:
        env = VecVideoRecorder(env, "videos/", lambda _: True, video_length=args.max_steps)

    if args.capture_offline_dataset:
        env = OfflineDatasetRecorder(env, "offline_rl")

    return env


def create_agent(envs, input_channels=27, output_channels=78):
    class Transpose(nn.Module):
        def __init__(self, permutation):
            super().__init__()
            self.permutation = permutation
    
        def forward(self, x):
            return x.permute(self.permutation)

    class Encoder(nn.Module):
        def __init__(self, input_channels):
            super().__init__()
            self._encoder = nn.Sequential(
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
                nn.MaxPool2d(3, stride=2, padding=1),
            )
    
        def forward(self, x):
            return self._encoder(x)
    
    
    class Decoder(nn.Module):
        def __init__(self, output_channels):
            super().__init__()
            self.deconv = nn.Sequential(
                layer_init(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(32, output_channels, 3, stride=2, padding=1, output_padding=1)),
                Transpose((0, 2, 3, 1)),
            )
    
        def forward(self, x):
            batch_size = x.size(0)
            return self.deconv(x).reshape((batch_size, -1))
    
    class Agent(nn.Module):
        def __init__(self, device="auto"):
            super(Agent, self).__init__()

            self.device = get_device(device)
            self._mask_value = torch.tensor(-1e+8)

            self.action_space_size = env.action_space.nvec.sum()
            self.height, self.width, self.input_channels = env.observation_space['obs'].shape
            self.num_cells = self.height * self.width
            self.action_plane = env.action_space.nvec[:env.action_space.nvec.size // self.num_cells]
            self.action_dist = HierachicalMultiCategoricalDistribution(self.num_cells, self.action_plane)

            self.encoder = Encoder(self.input_channels)
            self.actor = Decoder(self.action_plane.sum())
            self.critic = nn.Sequential(
                nn.Flatten(),
                layer_init(nn.Linear(256, 128), std=1),
                nn.ReLU(),
                layer_init(nn.Linear(128, 1), std=1),
            )
    
        def forward(self, x):
            return self.encoder(x)  # "bhwc" -> "bchw"

        def _mask_action_logits(self, latent_pi, masks, mask_value=None):
            mask_value = mask_value or self._mask_value
            return torch.where(masks, latent_pi, mask_value)

        def predict(self, inputs, deterministic=False):
            obs, masks = inputs['obs'], inputs['masks']
            obs = torch.from_numpy(obs).float().to(self.device)
            masks = torch.from_numpy(masks).bool().to(self.device)
            latent_pi = self.encoder(obs)
            value = self.critic(latent_pi)
            logits = self._mask_action_logits(self.actor(latent_pi), masks)
            actions = self.action_dist.actions_from_params(action_logits=logits, deterministic=deterministic)
            return actions.cpu().numpy(), value.cpu().numpy()

    return Agent()


if __name__ == "__main__":
    args = make_parser().parse_args()

    env = create_env(args)

    print("Env is succesfully loaded")

    ext = Path(args.agent_model_file).suffix
    if ext == ".zip": # SB3 archieve
        model = PPO.load(args.agent_model_file)
    elif ext == ".pt": # PyTorch module
        model = create_agent(env)
        model.load_state_dict(torch.load(args.agent_model_file, map_location="cpu"))

    print(f"Model is succesfully loaded, device={model.device}")

    window = Window(1280, 980, "MicroRTS")

    # for a single game:
    # this API should be a part of "env.render()" and not visible for a user
    # game_panel = GameStatePanel(
    #     env.vec_client.clients[0],
    #     # xxx(okachaiev): I should be able to get mapsize from the client
    #     config=GameStatePanelConfig(mapsize=(16,16), players=[dict(name="ppo_gridnet"), dict(name="coacAI")])
    # )
    # window.add_panel(game_panel)

    # for multiple games:
    # this API should be a part of "env.render()" and not visible for a user
    unique_bots = {k.__name__:ind for ind, k in enumerate(args.bot_envs)}
    game_tiles = [
        GameStatePanel(
            env.vec_client.clients[game_client_ind],
            # xxx(okachaiev): I should be able to get mapsize from the client
            config=GameStatePanelConfig(mapsize=(16,16), players=[dict(name="ppo_gridnet"), dict(name=ai_name)])
        )
        for (ai_name, game_client_ind)
        in unique_bots.items()
    ]
    window.add_panel(Tilemap(game_tiles))

    print(f"Env rendering engine is loaded")

    obs = env.reset()

    # this API should be a part of "env.render()" and not visible for a user
    window.render()

    progress = trange(args.total_timesteps, desc="R=? V=? I=?")
    with torch.no_grad():
        for i in progress:
            action, value = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)

            # this API should be a part of "env.render()" and not visible for a user
            window.render()

            raw_reward = np.array([e['raw_rewards'] for e in info]).sum(axis=0)
            # xxx(okachaiev): this description definitely need some work
            progress.set_description(f"R={reward.mean():0.4f} V={value.mean():0.4f} I={raw_reward}")

    print("Finishing up...")

    # xxx(okachaiev): hack
    # technically, we should call `env.close` here but there's
    # something not exactly right about shutting down JVM on Mac
    if args.capture_video:
        env.close_video_recorder()

    if args.capture_offline_dataset:
        env.save_dataset()