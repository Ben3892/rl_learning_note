# rl_learning_note

网球训练示例位于 [ppo_tennis](ppo_tennis/README.md)：

- [PPO 训练与辅助奖励](ppo_tennis/README.md)
- [带经验回放的 Q-learning / DQN](ppo_tennis/DQN.md)，支持 CUDA、评估和视频导出。

install package:
```bash
python -m pip install --upgrade pip
python -m pip install --upgrade "gymnasium[atari]"
```
quick start
```python
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
```
