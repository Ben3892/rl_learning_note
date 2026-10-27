# 带经验回放的 Q-learning：DQN Tennis

入口为 [dqn_tennis.py](dqn_tennis.py)，与 PPO 共用依赖、环境和辅助奖励。
图像状态无法方便地存成 Q 表，因此用 CNN 近似 `Q(s,a)`；这就是 Deep Q-learning。
使用 SB3 的 vanilla DQN 实现，包含均匀经验回放、目标网络和梯度裁剪，
不包含 Double DQN、Dueling 网络或优先级回放。

## 最小设计

1. 输入仍为 4×84×84 灰度帧，CNN 输出 Tennis 18 个离散动作的 Q 值。
2. 预热阶段直接均匀随机动作；之后使用 epsilon-greedy 交互：概率 epsilon 随机选择动作，
   否则选择最大 Q 值动作。
3. 把 `(s, action, training_reward, next_s, done, timeout)` 存入 CPU 环形回放池，
   满后覆盖旧数据。training_reward 默认包含辅助奖励。
4. 预热后每交互 4 步，从池中均匀随机抽取一个 batch，执行一次梯度更新。
5. 使用目标网络构造单步 TD 目标：

   `y = reward + 0.99 × (1 - terminal) × max_a Q_target(next_s, a)`

   最小化当前 `Q_online(s, action)` 与 y 的 Huber loss，目标值不参与反向传播。
   真正 game over 的 terminal 为 1；时间上限截断仍保留 bootstrap，
   SB3 使用终止前观测而非自动 reset 后的观测计算 next_s。
6. 默认每 1 万个交互步把在线网络复制给目标网络。

固定单环境，明确每一步及回放容量的含义。复用原有 `--device cuda / cuda:N / cpu / auto`。
CUDA 处理网络计算和采样 batch，环境、辅助奖励及完整回放池留在 CPU 内存。
默认参数是可运行起点，未针对 Tennis 调优，不保证优于 PPO。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| total-timesteps | 1000000 | 总交互步数，不是梯度更新次数 |
| buffer-size | 20000 | 最多保留 2 万条 transition |
| learning-starts | 10000 | 预热收集数据，之后才开始更新 |
| batch-size | 32 | 每次从回放池抽取的条数 |
| learning-rate | 0.0001 | Q 网络学习率 |
| train-freq / gradient-steps | 4 / 1 | 每 4 步训练一次、每次 1 次更新 |
| target-update-interval | 10000 | 硬同步目标网络的交互步数间隔 |
| exploration-fraction | 0.2 | 前 20% 预算内 epsilon 从 1 线性下降 |
| exploration-final-eps | 0.05 | 下降后的训练探索概率 |
| eval-freq / eval-episodes | 100000 / 1 | 每 10 万步评估 1 局 |
| eval-max-steps | 5000 | 单局评估上限，单位为 agent steps |
| log-freq | 1000 | 不依赖回合结束，每 1000 步输出日志 |

普通回放存储当前和下一观测两份 uint8 数组，2 万条约占 **1.05 GiB 主机内存**，
还有其他数组、模型和中间计算开销。10 万条约 5.26 GiB，100 万条约 52.57 GiB。
因此没有使用库中默认的百万条回放容量。buffer-size 不代表显存大小。
若内存较小可先用 `--buffer-size 5000 --learning-starts 1000`，但会改变训练行为。

## 运行

工作目录为 `ppo_tennis/`，依赖安装方式与 [README](README.md) 相同。
CUDA 机器先安装合适的 CUDA 版 PyTorch。

先运行 CPU 小测试，确认实际发生 Q 网络更新，而非只有预热采样：

```bash
python dqn_tennis.py train --run-dir runs/dqn_smoke_local \
  --total-timesteps 256 --buffer-size 512 --learning-starts 32 \
  --batch-size 16 --target-update-interval 64 --log-freq 64 \
  --eval-freq 256 --eval-max-steps 100 --save-replay-buffer
```

正式训练（需要一个新的实验目录）：

```bash
python dqn_tennis.py train --device cuda --run-dir runs/dqn_tennis_cuda \
  --total-timesteps 1000000
tensorboard --logdir runs/dqn_tennis_cuda
```

继续支持所有辅助奖励参数，包括 `--no-reward-shaping`、`--return-bonus`、
`--position-weight`、`--position-cap`、`--idle-penalty`、`--idle-grace`。
辅助奖励改变后应新建实验，不混用旧奖励下的回放数据。

关注 `train/loss`、`train/n_updates`、`train/exploration_rate`、
`replay/size`、`rewards/*` 和原生 `eval/mean_reward`。
预热期不应有 Q 网络梯度更新；回放池大小应增长，达到容量后保持固定。
回调在当前 transition 入池前运行，日志 replay/size 会比累计交互步数少 1；
最终准确统计见 `replay_stats.json`。epsilon 取决于预算与进度，短测试会很快衰减。

## 评估与视频

```bash
python dqn_tennis.py evaluate --device cuda \
  --model runs/dqn_tennis_cuda/best/best_model.zip \
  --episodes 5 --output runs/dqn_tennis_cuda/evaluation.json

python dqn_tennis.py play --device cuda \
  --model runs/dqn_tennis_cuda/best/best_model.zip \
  --video runs/dqn_tennis_cuda/demo.mp4
```

评估同时运行相同 seeds 的随机策略，只使用原生奖励。每局最多 5000 个决策步，
到上限会截断，记录 `truncated=true`；JSON 还报告等待发球步数和有效回击次数。
可调 `--eval-max-steps`。相同评估上限和 seeds 才适合比较不同模型。
定期评估会打印开始、每 1000 步进度、单局结果与结束消息；仍是同步评估，期间训练暂停。
诊断记录保存为 `eval/diagnostics_<训练步数>.json`。
超时回报可能偏好不发球的策略，best_model 仍按原生回报选取，
必须结合超时、等待与回击统计判断效果，不能把截断后少失分当成学会打球。

play 默认贪心选动作；用 `--stochastic --epsilon 0.1` 可观察 10% 随机探索的行为。
DQN 的 stochastic 是 epsilon-greedy，不是 PPO 那样按策略概率分布采样。
视频同时生成奖励分项 CSV 和配置 JSON；也支持 `--human`。
CPU/CUDA 间可以加载同一个 DQN 模型。PPO 和 DQN 的 zip 不可互换，使用对应的入口文件。

## 模型与回放的保存

- `final_model.zip`、`best/best_model.zip`、定期 checkpoints 保存网络及算法状态。
- `replay_stats.json` 记录池容量、实际条数、更新次数、最终 epsilon 和实际交互步数。
- **训练始终使用经验回放**；`--save-replay-buffer` 只控制结束时是否另外把池保存为
  `replay_buffer.pkl`，不传也有回放。它不是默认开启，避免输出大文件。
- checkpoint 和 best 模型不另存回放；模型 zip 不包含回放池。
- 当前 CLI 不提供断点续训。若用 Python API 继续训练，可以按下面方式加载自己保存的池：

```python
from stable_baselines3 import DQN
from tennis import make_env, model_reward_config

model_path = "runs/dqn_tennis_cuda/final_model.zip"
from pathlib import Path
env = make_env(shaping=True, reward_config=model_reward_config(Path(model_path)))
model = DQN.load(model_path, env=env, device="cuda")
model.load_replay_buffer("runs/dqn_tennis_cuda/replay_buffer.pkl")
# 根据原实验设置 shaping；原生奖励实验应使用 shaping=False。
# 这是继续学习，不是逐位复现原运行，探索计划也会按新的预算重新计算。
try:
    model.learn(total_timesteps=100000, reset_num_timesteps=False)
    model.save("runs/dqn_tennis_cuda/continued_model")
finally:
    env.close()
```

## 验证范围

`python -m unittest -v test_dqn` 使用真实 Tennis 环境验证 Q 网络权重变化、
目标网络同步、回放环形覆盖、随机 batch 采样、时间截断 bootstrap，
以及模型/回放保存重载后继续更新。
本地 CPU 短训练已通过；没有 NVIDIA 实卡测试，也没有长训练成绩结论。
更换算法仍可能出现停滞行为，经验回放本身不会消除奖励设计的漏洞。

参考：[SB3 DQN 官方文档](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)。
