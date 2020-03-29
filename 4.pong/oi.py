import gym
from stable_baselines.common.policies import CnnPolicy, FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env

# Params
env_name = 'PongNoFrameskip-v4'
total_timesteps = 1_000_000
path = './4.pong'
model_name = f"{path}/ppo-{env_name}-{total_timesteps}"
tensorboard_log = f"{path}/ppo2_{env_name}_tensorboard/"
train = False
test = True

# Custom policy 
class CustomPolicy(CnnPolicy):
    net_arch = [
        512,
        512,
        512,
        dict(
            pi=[512, 256],
            vf=[512, 256]
        )
    ]

    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=self.net_arch)


if train:
    # Create and wrap the environment
    env = make_atari_env(env_name, num_env=4, seed=0)
    env = VecFrameStack(env, n_stack=4)
    # RL algorithm
    model = PPO2(
        CustomPolicy, env, 
        verbose=1, 
        tensorboard_log=tensorboard_log
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(model_name)
    del model

if test:
    # When loading a model with a custom policy
    # you MUST pass explicitly the policy when loading the saved model
    env = make_atari_env(env_name, num_env=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    model = PPO2.load(model_name, policy=CustomPolicy)
    # Enjoy trained agent
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
