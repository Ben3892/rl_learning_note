# PPO 网球游戏：最小 MVP 设计

## 1. 目标与范围

使用 Gymnasium 的 `ALE/Tennis-v5`（Atari 网球，对手由游戏控制），通过
Stable-Baselines3 的 PPO 学习从图像选择动作，最终提供模型、评估结果和游戏视频。
这里的网球是 Tennis；已有 quick_start.py 中的 Breakout 是打砖块。

第一阶段验收的是可运行的完整学习流程；第二阶段才检查策略是否优于随机策略。
短训练不承诺能接球或获胜。加入可关闭的辅助奖励；暂不实现自博弈或手写 PPO。

## 2. 数据与算法

`游戏 RGB → 4 帧动作重复/末两帧最大池化 → 84×84 灰度 → 4 帧堆叠 → CNN → PPO`

- 原始环境设 `frameskip=1`，仅在 AtariWrapper 中跳 4 帧，避免重复跳帧。
- 保留 v5 的 0.25 sticky-action 概率；动作空间包含移动、击球及其组合。
- 训练与评估共用同一图像预处理，不按失去生命划分回合，不裁剪原始奖励。
  训练默认使用原生奖励加辅助奖励，定期评估和最终评估始终使用原生奖励。
- AtariWrapper 在 reset 时执行 FIRE；后续每个得分后的发球仍由策略选择 FIRE。
  若画面长期静止且回报为零，应检查发球行为，不能直接当成学会防守。
- 使用 DummyVecEnv（单进程、多环境）保证初学者和 macOS 使用简单。
- 起始参数：4 个环境，rollout 128 步/环境，batch 256，4 epochs，
  learning rate 2.5e-4，gamma 0.99，GAE lambda 0.95，clip 0.1，entropy 0.01。
- 默认 CPU；有 NVIDIA GPU 时可选 `--device cuda`。参数是起点，未经过 Tennis 调优。

## 3. 最小交付

| 文件/产物 | 用途 |
| --- | --- |
| tennis.py | train / evaluate / play 三个命令 |
| reward_shaping.py | RAM 状态读取、事件检测、奖励塑形 |
| test_reward_shaping.py | 回击、等待、站位、换边等事件回归测试 |
| requirements.txt | 运行依赖 |
| runs/<名称>/config.json、versions.json | 参数、依赖版本 |
| runs/<名称>/reward_config.json | 辅助奖励开关、权重与检测器版本 |
| runs/<名称>/final_model.zip | 最后一次训练模型 |
| runs/<名称>/best/best_model.zip | 定期评估选出的模型（触发评估后才生成） |
| runs/<名称>/progress.csv、events.out.tfevents.* | 训练日志与 TensorBoard 学习曲线 |
| runs/<名称>/evaluation.json | 随机策略和 PPO 的原始回报对比 |
| runs/<名称>/demo.mp4 | 定长策略演示，可包含多局或未结束的一局 |
| runs/<名称>/demo.rewards.csv | 每步动作、原生/辅助奖励及发球状态 |

辅助奖励的事件条件、默认系数与限制详见 [运行说明](README.md#辅助奖励)。
RAM 只供训练奖励和诊断使用，不提供给 CNN 策略。回击检测按原始帧执行，
不会因为 4 帧跳帧而漏掉方向反转；奖励在跳帧后聚合成一个 agent step。

## 4. 验证与预算

1. 冒烟：512 个 agent transitions，完成 PPO 更新、保存、重新加载和短视频导出。
2. 试训：100 万 transitions，检查日志、回报、画面和吞吐量。
3. 正式实验：可从 1000 万 transitions 起尝试，按实际曲线决定是否继续；
   这不是保证收敛的步数。约每个 transition 对应 4 个模拟器帧（reset 步另计）。
4. 固定同组评估 seeds，比较 5 个以上回合的随机策略和确定性 PPO；
   报告均值、标准差和每局分数。多次训练 seeds 才能进一步判断稳定性。
5. 回合上限约 108000 原始环境步（约 30 分钟游戏时间），超时是截断，
   JSON 单独记录。回报不是胜率；只看 MP4 也不能证明策略提升。

通过冒烟测试只代表流程正常。只有较长训练后，独立评估回报稳定优于随机策略，
才能声称策略有提升。训练用 seed 0；定期选模型用 10000；最终评估默认 20000。
MVP 不自动启动长训练。先根据控制台 fps 估算本机预算。

## 5. 参考

- [ALE Tennis：环境、动作与发球](https://ale.farama.org/environments/tennis/)
- [SB3 AtariWrapper：预处理与参数](https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html)
- [SB3 示例：训练、保存与评估](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)
