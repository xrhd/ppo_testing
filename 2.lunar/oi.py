import gym

from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import A2C, PPO2

# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    net_arch = [
        128, 128, 128
        ,dict(
            pi=[128, 128, 128],
            vf=[128, 128, 128]
        )
    ]

    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=self.net_arch,
                                           feature_extraction="mlp")

# Create and wrap the environment
env = gym.make('LunarLander-v2')
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=False,clip_obs=10.)

model = PPO2(CustomPolicy, env, verbose=1, tensorboard_log="./ppo2_lunar_tensorboard/")

# Train the agent
model.learn(total_timesteps=100000)
# Save the agent
model.save("a2c-lunar")

del model
# When loading a model with a custom policy
# you MUST pass explicitly the policy when loading the saved model
model = A2C.load("a2c-lunar", policy=CustomPolicy)

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()


