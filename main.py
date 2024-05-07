import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch

# Parallel environments
train=True
vec_env = make_vec_env("CartPole-v1", n_envs=2)
if train:

    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda:2")
    model.learn(total_timesteps=1)
    model.ep_info_buffer.extend([torch.ones(10000,device="cuda:2")])
    model.save("ppo_cartpole")

    del model

model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda:3")
import torch
model.set_parameters("ppo_cartpole.zip", device="cuda:3")
torch.cuda.empty_cache()
import time
print("watch gpustat now")
time.sleep(10)
print("memory on ep info device:", torch.cuda.memory_allocated("cuda:2"))
