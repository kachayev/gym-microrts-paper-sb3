import argparse
from distutils.util import strtobool
import numpy as np
import os
import time
import torch
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder

from gym_microrts import microrts_ai

from ppo_gridnet_diverse_encode_decode_sb3 import CustomMicroRTSGridMode, ParseBotEnvs, _parse_bot_envs


parser = argparse.ArgumentParser()
parser.add_argument("--agent-file", type=str, required=True)
parser.add_argument("--num-episodes", type=int, default=1)
parser.add_argument('--max-steps', type=int, default=2_000,
                    help='max number of steps per game environment')
parser.add_argument('--bot-envs', nargs='*', action=ParseBotEnvs,
                    default=_parse_bot_envs('randomBiasedAI=2 lightRushAI=2 workerRushAI=2 coacAI=18'),
                    help='bot envs to setup following "bot_name=<num envs>" format')
parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                    help='weather to capture videos of the agent performances (check out `videos` folder)')

args = parser.parse_args()

model = PPO.load(args.agent_file)

print(f"Model is succesfully loaded, device={model.device}")

# xxx(okachaiev): not exactly the hack but quite
# annoying detail. as I trained the policy using
# nummber of vec envs as a parameter (for re-shaping),
# I now cannot use it with different configuration
# I need to find a way to avoid doint it in the policy
env = CustomMicroRTSGridMode(
    num_selfplay_envs=0,
    max_steps=args.max_steps,
    render_theme=2,
    ai2s=args.bot_envs,
    map_paths=["maps/16x16/basesWorkers16x16.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)

if args.capture_video:
    env = VecVideoRecorder(env, "videos/", lambda _: True, video_length=args.max_steps)

print("Env is succesfully loaded")

obs = env.reset()
for i in range(1,args.num_episodes+1):
    done = np.zeros(len(args.bot_envs))
    total_reward = np.zeros(len(args.bot_envs))
    total_raw_reward = np.zeros(6)
    progress = tqdm(range(args.max_steps+1), desc=f"Episode #{i} R={0.0:0.3f} V={0.0:0.4f}")
    for _ in progress:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        total_raw_reward += np.array([e['raw_rewards'] for e in info]).sum(axis=0)
        with torch.no_grad():
            critic = model.policy.mlp_extractor.forward_critic(
                (torch.tensor(obs['obs']).float().to(model.device), None)
            ).mean()
        progress.set_description(
            f"Episode #{i} R={total_reward.mean():0.3f} V={critic:0.4f} I={total_raw_reward}")
        if done.all(): break
    print("Reward by envs:", total_reward)
    obs = env.reset()

print("Finishing up...")

# xxx(okachaiev): hack
# technically, we should call `env.close` here
# but there's something not exactly right about
# shutting down JVM on Mac
if args.capture_video:
    env.close_video_recorder()