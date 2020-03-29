import gym
from stable_baselines.common.policies import CnnPolicy, FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env

# Params
env_name = 'CarRacing-v0'
total_timesteps = 100_000
model_name = f"ppo-{env_name}-{total_timesteps}"
tensorboard_log = f"./ppo2_{env_name}_tensorboard/"

# Custom policy 
class CustomPolicy(CnnPolicy):
    net_arch = [
        256,
        256,
        256,
        dict(
            pi=[256, 256, 256],
            vf=[256, 256, 256]
        )
    ]

    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=self.net_arch)

# Create and wrap the environment
env = gym.make(env_name)
# env = FrameStack(env, 16)

# print(check_env(env)) # dont work with FrameStack
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)
env = VecFrameStack(env, 24)

# RL algorithm
model = PPO2(
    CustomPolicy, env, 
    verbose=1, 
    tensorboard_log=tensorboard_log
)
model.learn(total_timesteps=total_timesteps)
model.save(model_name)
del model

# When loading a model with a custom policy
# you MUST pass explicitly the policy when loading the saved model
model = PPO2.load(model_name, policy=CustomPolicy)

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
