import gym
import math
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize


# multiprocess environment
# env = make_vec_env('CartPole-v1', n_envs=4)
env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=False,clip_obs=10.)

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000)
model.save("ppo2_cartpole")

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo2_cartpole")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()


