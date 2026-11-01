# RAM DQN：与回中／追球 PPO 对照

入口 `dqn_ram.py`，启动脚本 `train_dqn_ram.sh`。
原有图像版 `dqn_tennis.py` 保留，不能直接用于这个 RAM 对照。

## 公平比较的部分

- 复用 `ram_env.make_ram_env`：165 维归一化状态、18 个动作、4 帧跳帧、0.25 sticky actions。
- 自动发球为 `align_x_fire`，目标为 `recover_chase`，gamma=0.99、beta=0.05。
- 原始得分加同一势函数奖励；不修改中心坐标、奖励符号或移动控制方式。
- 8 个环境，训练 seed=0；默认总预算 25 万 transitions。
- 每 5 万步在 seeds=10000–10004 上评估 5 局。每局显式 reset(seed)。
- 选出的最佳模型在 seeds=20000–20004 上另做 5 局独立测试。
- 比较对象为 `runs/tennis_ram_20261101_192426` 中的同任务 PPO，而非旧 CNN 模型。

这只是各算法一组参数、单个训练 seed 的探索性比较，不能据此认定某算法普遍更快。
训练更新方式、探索方式和学习率调度是算法各自的设置。
PPO 那次使用 CUDA；此脚本默认 CPU，小型 MLP 的耗时比较需注明设备差异。
框架会收集完整 rollout，因此实际训练步数可能略高于请求预算。

## DQN 参数

使用 SB3 vanilla DQN，包括均匀经验回放、目标网络、Huber loss 与梯度裁剪；
不包含 Double DQN、Dueling 或优先级回放。

- Q 网络：[128, 128] MLP；另有同结构的目标网络。
- 回放容量 10 万 transitions；float32 观测数组约 126 MiB，加上动作等数组约 128 MiB。
- 预热 1 万 transitions 后更新；batch=128、learning rate=1e-4。
- 每 4 次向量环境 step（8 环境合计 32 transitions）执行 8 次梯度更新，
  也就是平均每 4 个 transitions 更新一次。
- 每 1 万 transitions 硬同步目标网络，gamma=0.99。
- 前 40% 训练预算内，epsilon 从 1 线性降到 0.05，之后保持 0.05。
- 评估时始终使用贪心动作，关闭 epsilon 探索；录像默认 epsilon=0。
- 观察回中距离、动作同方向占比、原始得分和比赛是否完整完成；loss 降低不等于学会打球。

## 运行

脚本会初始化 Conda 并执行 `conda activate yja`：

```bash
cd /home/ben/rl_learning_note/ppo_tennis
bash train_dqn_ram.sh
```

修改设备、预算或奖励时，使用新的实验目录：

```bash
RUN_DIR=runs/dqn_ram_trial DEVICE=cpu TOTAL_TIMESTEPS=250000 bash train_dqn_ram.sh
DEVICE=cuda:0 bash train_dqn_ram.sh
```

不覆盖已有目录。Ctrl+C 保存 interrupted_model，并记录 interrupted 状态。
默认不保存回放池；可传 `--save-replay-buffer`。模型 ZIP 本身不包含回放池。

## 产物与选模

- `reward_observation.md`、`eval/diagnostics_<steps>.json`：验证曲线、逐局分数、结束原因、动作分布。
- `best/best_model.zip`：在自然结束或达到环境原生时间上限的回合中，原始平均得分最好的模型；同分保留最早模型。`step_limit` 和 `serve_stall` 不参与选模。
- `qualified/best_model.zip`：合格环境回合的原始均分严格超过固定动作基线时才保存；这不代表赢得每局比赛。
- `heldout_evaluation.json`：所选 best 的独立测试，包含随机、固定 DOWNFIRE/UPFIRE/FIRE 基线。
- `replay_stats.json`：总 transition 数、每个环境的行数、实际容量、分配字节数及梯度更新次数。
  SB3 多环境回放池的 size() 返回每个环境的行数；报告中明确乘以 n_envs。
- `training_status.json`：训练及独立测试完成状态；`<run-dir>.exitcode` 是整个脚本的退出码。
- 配置、版本、原始 Monitor、TensorBoard、检查点和控制台日志照常保存。

移动由 Q 网络决定，不会自动执行回中策略。FIRE 只用于发球；不能把 FIRE 变体视为额外移动方向。
原始奖励为负或动作变化更多，都不能单独证明已经收敛。

## 独立评估与 MP4

将模型路径替换为实际目录：

```bash
conda activate yja
python dqn_ram.py evaluate --model runs/dqn_ram_trial/best/best_model.zip \
  --output runs/dqn_ram_trial/recheck.json --episodes 5 --seed 30000
python dqn_ram.py play --model runs/dqn_ram_trial/best/best_model.zip \
  --video runs/dqn_ram_trial/demo.mp4 --steps 2000 --seed 20000
```

录像为 15 fps，默认约 133 秒，保存逐步动作/坐标及配置。
可显式传 `--epsilon 0.05` 做带探索的诊断录像，这不能替代贪心评估。

## 验证

```bash
conda activate yja
python -m unittest -v test_dqn_ram
```

测试真实 RAM 环境上的回放学习、Q 网络及目标网络更新、超时 bootstrap、
多环境回放容量、模型/回放保存加载和继续更新。
另外在 epsilon=1 时验证评估仍调用贪心动作，不消耗探索 RNG。

参考：[SB3 DQN 官方文档](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)。


## 长对打的评估上限

初次试验使用与 PPO 相同的 5000 步上限，20 万步后出现对打仍在进行但被截断的情况。
已保存所有检查点，并在验证 seeds 上统一改用 27000 步重新评估后再选模，随后使用独立测试 seeds。
新启动脚本默认 EVAL_MAX_STEPS=27000；底层原始帧时间上限仍生效，触发时仍记为截断。
可用 EVAL_MAX_STEPS=5000 复现初轮上限。报告必须区分完整比赛和未完成的正分片段。
完整比赛重评的模型在 best_full_match/，初轮 best/ 保留不覆盖。

选模需区分自然环境时限与人工提前终止：自然时限属于标准有限时域回合，可用于环境回报比较，仍记录 truncated；不能把它当作已打完整场网球比赛。
本轮原始 best/ 是 5000 步旧协议的产物；best_full_match/ 保留所有比赛均结束的严格重评，best_natural_episode/ 是统一自然时限重评后的环境回报最佳模型。
完整对照结果见 [DQN_RAM_RESULTS.md](DQN_RAM_RESULTS.md)。
