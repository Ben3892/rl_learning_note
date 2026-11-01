# RAM + MLP PPO Tennis 实验

入口为 `ram_ppo.py`，环境在 `ram_env.py`。这是与原图像版分开的实验，
目标是先验证策略能否根据位置、球速和换边状态进行控制。

## 当前默认：回中 / 追球引导

按用户指定的规则，默认训练模式为 `recover_chase`，`beta=0.05`：

- 球飞向对方（outgoing）：目标为己方可活动区域中心。上半场为 (80, 72.5)，下半场为 (80, 153.5)。
- 球飞向己方（incoming）：目标跟随球的地面/影子位置；x 限制在 [20, 148]，
  y 限制在己方可活动区（上半场 [47, 98]、下半场 [120, 187]），避免目标跨到对方半场。
- 方向按己方所在半场计算：上半场 dy<0 为来球，下半场 dy>0 为来球。
- 发球、得分/换边造成速度失效、方向不明或没有有效回合时不设移动目标。
- 坐标范围来自当前 ROM 的玩家脚部坐标，是可调整的经验值。
- 继续使用距离势函数差值作为辅助奖励；PPO 自己选择移动动作，不强制执行追球/回中脚本。
  该改动不保证收敛；效果仍需用完整比赛原始得分与固定动作基线检验。

```bash
cd /home/ben/rl_learning_note/ppo_tennis
bash train_ppo_ram.sh
```

脚本自动激活 yja，保留当前 DEVICE 默认值。可显式设置设备与奖励强度：

```bash
DEVICE=cpu BETA=0.05 bash train_ppo_ram.sh
TARGET_MODE=recover_chase BETA=0 bash train_ppo_ram.sh  # 相同新目标观测，但仅训练原始得分
TARGET_MODE=intercept BETA=0 bash train_ppo_ram.sh      # 复现旧版目标与原始奖励配置
```

目标模式写入 config.json 和 observation_schema.json，评估与录像读取同一配置。
旧配置缺少 target_mode 时继续使用 intercept，避免改变旧模型输入的目标特征。
旧模型需要重新训练才能学习新规则。视频动作 CSV 新增 ball_direction、target_x、target_y。

## 环境与奖励


- 输入为 128 个归一化 RAM 字节、19 个几何/速度/阶段特征，以及上一次实际动作的 18 维 one-hot，共 165 维。
- 坐标和特征裁剪到 [-1, 1]；得分、换边、阶段变化和不合理瞬移会清除球速。
- 球速是每个 agent step 的位移；预测拦截点时转换为原始帧速度。
- 默认自动发球：仅在确认己方等待发球时，对齐球的横向位置并按 FIRE，记录请求动作和实际动作。
  用 `--no-auto-serve` 可关闭。训练、评估和录像从模型旁的配置读取相同设置。
- 原始 RAM 会提供图像版没有的状态信息；自动发球也改变了任务，
  因此本实验的得分不能直接作为旧 CNN 原版任务的胜负比较。
- 保留 ALE 的 0.25 sticky actions，每次决策跳过 4 个原始帧。
- 不使用原有的站位、距离、等待惩罚或回球附加奖励。回球事件仅用于诊断。
- 塑形公式：`raw + beta * (gamma * Phi(next) - Phi(previous))`。
  有效目标阶段 `Phi = -min(目标距离 / 160, 1)`，没有目标时为 0；目标定义由 target_mode 决定。
  每个 agent step 计算一次，与 PPO 使用相同 gamma；真终止的下一势值为 0，
  时间截断保留下一势值，由 SB3 用 terminal_observation bootstrap。
- Monitor 始终位于塑形之前，`rollout/ep_rew_mean` 和评估都保留原始得分。

## 运行

脚本先初始化 Conda 并执行 `conda activate yja`：

```bash
cd /home/ben/rl_learning_note/ppo_tennis
bash train_ppo_ram.sh
```

默认 25 万 transitions、8 环境 × 256 steps、batch 256、4 epochs、
MLP 的 actor/critic 各 2 层 128 单元，gamma 0.99、GAE lambda 0.95、
clip 0.1、entropy 0.01、target KL 0.02。
学习率从 1e-4 线性减至 1e-5。PyTorch 单线程；脚本当前默认 CUDA（保留用户设置）。
小型 MLP 也可用 `DEVICE=cpu`，请根据本机实测吞吐量选择。
这些是实验起点，不能保证收敛。当前新规则默认 beta=0.05；用 BETA=0 关闭辅助奖励。旧版目标的历史对照见实验结果报告。

比较有/无势函数奖励，分别使用新目录：

```bash
RUN_DIR=runs/tennis_ram_potential BETA=0.05 bash train_ppo_ram.sh
RUN_DIR=runs/tennis_ram_raw BETA=0 bash train_ppo_ram.sh
```

自定义预算及关闭自动发球：

```bash
TOTAL_TIMESTEPS=1000000 bash train_ppo_ram.sh
bash train_ppo_ram.sh --no-auto-serve
```

脚本拒绝覆盖已有目录。实际步数向上取整到完整 rollout 的倍数。
按 Ctrl+C 保存为 `interrupted_model.zip` 并记录 interrupted 状态。

## 评估与选模

每 5 万 transitions 在固定 seeds 10000–10004 上分别运行确定性和随机采样 PPO。
每个 episode 都显式 reset(seed)，评估前后的 PyTorch RNG 状态保持一致。
对照策略包括随机、固定 DOWNFIRE、固定 UPFIRE、固定 FIRE，
与策略使用相同 RAM 环境和自动发球设置。

- `reward_observation.md`：基线和各检查点评估表，每次评估后更新。
- `baseline_validation.json`：固定验证 seeds 上的基线。
- `eval/diagnostics_<steps>.json`：每局得分、结束原因、实际/请求动作、分半场动作和得分。
- `best/best_model.zip`：确定性评估中所有比赛完整结束，且原始平均得分最好的模型。
  这是当前实验的相对最佳，不代表已经学会打球。
- `qualified/best_model.zip`：只有完整比赛的平均原始得分严格超过全部完整固定动作基线时才保存。
  文件夹可能为空；超过基线也不代表已收敛或泛化。
- `heldout_evaluation.json`：训练结束后，对已选最佳模型在独立 seeds 20000–20004 上运行所有基线及两种 PPO 推理。
  这些 seeds 不用于选模。
- `training_status.json`：训练完成或中断状态；`<run-dir>.exitcode` 是整个脚本退出码。
- `progress.csv`、TensorBoard、原始 Monitor、checkpoint、final_model 和控制台日志照常保存。

动作占比高只作诊断，不作为奖励惩罚，也不会直接禁止合法的连续动作。
角色位于合适位置时会自动回球，回球次数不能代替完整比赛得分和基线比较。
每轮只用单个训练 seed，不能据此断言统计显著性；正式结论还需要多次训练 seeds。

独立重评（模型路径替换为实际实验目录）：

```bash
conda activate yja
python ram_ppo.py evaluate --model runs/tennis_ram_potential/best/best_model.zip \
  --output runs/tennis_ram_potential/recheck.json --episodes 5 --seed 30000
```

## 视频

RAM 模型必须用新入口，不能直接交给旧图像版的 `tennis.py play`：

```bash
conda activate yja
python ram_ppo.py play --model runs/tennis_ram_potential/best/best_model.zip \
  --video runs/tennis_ram_potential/demo.mp4 --steps 2000 --seed 20000
```

默认确定性推理、15 fps，同时生成 `.actions.csv` 与配置 JSON。
可用 `--stochastic` 做诊断，但随机移动不能视为学会控制的证据。

## 验证

```bash
conda activate yja
python -m unittest -v test_ram_ppo
```

覆盖球速在得分/换边时重置、特征范围、势函数折扣望远镜恒等式、
真终止与截断的差别、Monitor 原始得分、自动发球条件、
真实环境 API、PPO 小训练及模型保存加载、固定 seed 评估与 RNG 恢复。


发球逻辑的早期 fire-only 试验仍可能因横向偏离抛球位置而停滞，正式 v2 使用 align_x_fire。
配置记录 serve_mode，旧模型缺少此项时按 fire 模式加载，避免静默改变旧实验任务。

原游戏中 FIRE 只用于发球，回合中角色在合适位置时自动挥拍；参见 [Activision 原始说明书](https://www.atariage.com/manual_html_page.php?SoftwareLabelID=555)。
因此动作名 DOWNFIRE 不代表策略学会了挥拍时机。

实际实验结果见 [RAM_EXPERIMENT_RESULTS.md](RAM_EXPERIMENT_RESULTS.md)。
诊断还会合并带 FIRE 和不带 FIRE 的同方向动作，记录 dominant_movement_fraction，避免低估方向偏好。
