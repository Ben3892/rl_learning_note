# RAM PPO 实验结果（2026-11-01）

已完成代码修改、9 项回归测试及两轮对照实验。以下主要比较修复发球对齐后的 v2 两组。
这是 RAM 状态输入、辅助发球的任务；不能将分数直接当作旧 CNN 原版任务上的提升。
每组仅一个训练 seed=0，验证 seeds=10000–10004，独立测试 seeds=20000–20004，各 5 局。

| 组别 | beta | 实际训练步数 | 最佳验证检查点 | 验证原始均分 | 独立测试原始均分 | 测试完成 | 测试发球停滞 | 最大单一决策动作占比 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tennis_ram_v2_raw_250k | 0.0 | 251904 | 200000 | -21.60 | -21.60 | 5/5 | 0 | 68.3% |
| tennis_ram_v2_potential_250k | 0.05 | 251904 | 250000 | -22.80 | -23.00 | 5/5 | 0 | 47.5% |

## 独立测试基线

| 策略 | 原始均分 | 完成 | 检测到回球总数 |
| --- | --- | --- | --- |
| random | -23.80 | 5/5 | 16 |
| downfire | -23.80 | 5/5 | 45 |
| upfire | -24.00 | 5/5 | 25 |
| fire | -24.00 | 5/5 | 41 |

## 原始奖励组逐局结果

| Seed | 原始奖励 | 决策步数 | 检测回球次数 | 结束原因 |
| --- | --- | --- | --- | --- |
| 20000 | -21.0 | 2257 | 18 | game_over |
| 20001 | -22.0 | 2094 | 14 | game_over |
| 20002 | -21.0 | 2249 | 16 | game_over |
| 20003 | -23.0 | 2094 | 14 | game_over |
| 20004 | -21.0 | 2205 | 17 | game_over |

## 结论与限制

- 本轮原始奖励组的最佳模型优于测试集固定向下策略（-21.6 对 -23.8），势函数组为 -23.0；未证实势函数在本预算下有优势。
- 原始奖励组在 20 万步时最好，25 万步验证又退化到 -24；需要使用 best 模型，不能把 final 模型当成最佳模型。
- 原始奖励组测试动作不再是 100% 同一个，但仍有 68.3% 为 DOWN，且多数比赛仍在输；尚未收敛，也不能声称已解决所有坏模式。
- 回球次数只是 RAM 事件启发式，动作多样性也不是能力证明；选择模型依据完整比赛原始得分。
- 新版自动发球在已确认己方发球时横向对齐球，避免只按 FIRE 却发不出的已复现故障；它只辅助发球阶段。
- 同样设置下的 5 个验证检查点、两种推理模式均有记录；没有把截断分数当成完整比赛得分。
- 这些是单次探索性结果，没有多训练 seeds 或置信区间支持统计显著性结论。
- 训练阶段约 4 分钟/组（含中途评估），机器上约 1100 steps/s；旧 CNN 任务约 425 steps/s，任务不同，不能据此单独量化样本效率。

## 产物

- 推荐检查的模型：[原始奖励组最佳模型](runs/tennis_ram_v2_raw_250k/best/best_model.zip)。
- [原始奖励组完整独立测试](runs/tennis_ram_v2_raw_250k/heldout_evaluation.json)。
- [势函数组完整独立测试](runs/tennis_ram_v2_potential_250k/heldout_evaluation.json)。
- [原始奖励组演示视频](runs/tennis_ram_v2_raw_250k/demo_best.mp4)。
- [视频逐步动作和坐标](runs/tennis_ram_v2_raw_250k/demo_best.actions.csv)。
- 每组保存 source_snapshot 和 source_sha256.json，可检查实验实际使用的代码。
- 第一轮 fire-only 自动发球试验在 runs/tennis_ram_v1_250k、runs/tennis_ram_raw_250k；发现发球缺陷后，独立新建 v2 重新对照，没有覆盖旧结果。

## 复现

```bash
conda activate yja
cd /home/ben/rl_learning_note/ppo_tennis
RUN_DIR=runs/ram_raw_repeat TARGET_MODE=intercept BETA=0 bash train_ppo_ram.sh
RUN_DIR=runs/ram_potential_repeat TARGET_MODE=intercept BETA=0.05 bash train_ppo_ram.sh
```

## 合并等价移动后的诊断

DOWN 和 DOWNFIRE 在回合中的移动方向相同，不能把两者当成独立的方向变化。

| 组别 | 最大同方向占比 | 合并后的决策动作 |
| --- | --- | --- |
| tennis_ram_v2_raw_250k | 80.2% | {'DOWN': 8707, 'DOWNRIGHT': 1354, 'UPLEFT': 789} |
| tennis_ram_v2_potential_250k | 47.5% | {'DOWNRIGHT': 4262, 'DOWNLEFT': 5993, 'UPRIGHT': 216, 'LEFT': 481, 'RIGHT': 1542, 'UPLEFT': 128} |

原始奖励组同方向 DOWN 仍约占 80.2%；得分有改善，但尚未解决明显的方向偏好。
