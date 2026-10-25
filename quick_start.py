import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5")

obs, info = env.reset(seed=42)
print("观测空间：", env.observation_space)
print("动作空间：", env.action_space)
print("观测形状：", obs.shape)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("运行成功")