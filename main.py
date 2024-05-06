import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=100)

model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
model.learn(total_timesteps=1)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
import resource
for i in range(10):
    model.set_parameters("ppo_cartpole.zip")
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
