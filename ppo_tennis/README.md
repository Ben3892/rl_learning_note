# PPO Tennis MVP

新增：[带经验回放的 Q-learning / DQN 版本](DQN.md)，入口为 `dqn_tennis.py`，
支持训练、原生奖励评估、视频导出和 CUDA，与 PPO 使用独立的模型与实验目录。

先读 [最小 MVP 设计](MVP.md)，入口为 [tennis.py](tennis.py)，辅助奖励在
[reward_shaping.py](reward_shaping.py)。
使用 SB3 的 PPO 实现，适合先理解训练流程，再学习算法内部。

## 安装

在终端切换到本文件所在文件夹；使用已安装 Gymnasium 的 Python 环境：

```bash
cd /Users/ben/git_projects/rl_learning_note/ppo_tennis
# 如果你使用的是 yja 环境：conda activate yja
python -c "import sys; print(sys.executable)"
python -m pip install -r requirements.txt
python tennis.py --help
```

推荐 Python 3.10+。只有 Gymnasium 还不够，PPO 需要 SB3/PyTorch，
Atari 图像预处理需要 OpenCV，导出视频需要 imageio/FFmpeg。
依赖文件限制主要版本范围，每次实验在 versions.json 记录实际核心依赖版本。
若遇到 `No module named gymnasium`，先检查终端和 IDE 是否使用同一解释器。

## 在 NVIDIA CUDA 机器运行

把整个 `ppo_tennis/` 文件夹复制到有 NVIDIA GPU 的机器，使用相同的 Python 入口。
网络的 CNN 前向计算和 PPO 反向传播在 GPU 上执行；Atari 模拟器、RAM 奖励检测、
图像预处理以及 rollout buffer 仍在 CPU。当前 DummyVecEnv 在单进程中依次执行各环境，
增加 n_envs 主要扩大推理批次，不会自动让模拟器多进程并行。

先运行 `nvidia-smi` 检查驱动，再按
[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)
选择目标系统、Pip 和适合驱动的 CUDA 构建，执行页面生成的安装命令。
随后在同一环境安装本项目依赖并验证：

```bash
python -m pip install -r requirements.txt
python -c "import torch; print('torch:', torch.__version__, 'CUDA build:', torch.version.cuda); assert torch.cuda.is_available(), 'PyTorch 无法使用 CUDA'; print(torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda'))"
```

先做一次 CUDA 冒烟，再启动正式训练（工作目录为 ppo_tennis）：

```bash
python tennis.py train --device cuda --run-dir runs/cuda_smoke --total-timesteps 1024
python tennis.py train --device cuda:0 --run-dir runs/tennis_cuda_1m --total-timesteps 1000000
python tennis.py evaluate --device cuda:0 --model runs/tennis_cuda_1m/best/best_model.zip --episodes 5 --output runs/tennis_cuda_1m/evaluation.json
python tennis.py play --device cuda:0 --model runs/tennis_cuda_1m/best/best_model.zip --video runs/tennis_cuda_1m/demo.mp4
```

`--device cuda` 或 `cuda:N` 不可用时会直接报错，不会静默回退 CPU；
`--device auto` 则在 CUDA 可用时选择 CUDA，否则选择 CPU。
默认仍为 `cpu`，便于 Mac 本地调试。N 是进程可见设备编号：
例如 `CUDA_VISIBLE_DEVICES=1` 后使用 `--device cuda:0`，实际对应物理 GPU 1。
本实现使用单 GPU，不包含多 GPU 分布式训练。

启动时打印实际设备与 GPU 名称，训练目录 `device.json` 记录请求设备、
实际模型设备、PyTorch/CUDA 构建版本和 GPU 名称。
CUDA 模型也可以复制回 Mac，用 `--device cpu` 进行评估和视频导出。

先保持默认的 4 环境、batch 256 作为对照。若 GPU 利用率低，可试
`--n-envs 8 --batch-size 512`（rollout=1024，可被 batch 整除），
但这也改变训练批次设置，需同时观察原生评估回报和吞吐量；GPU 不保证总耗时更短。
无图形桌面的服务器直接导出 MP4，不使用 `--human`。

本地验证覆盖设备路由测试和 CPU 训练。当前开发机器没有 NVIDIA GPU，
未实测 CUDA 内核执行、显存占用或加速比；上面的 CUDA 冒烟命令用于目标机器验收。

## 先验证流程

```bash
python tennis.py train --run-dir runs/smoke --total-timesteps 512 --n-envs 1 --n-steps 128 --batch-size 64
python tennis.py play --model runs/smoke/final_model.zip --video runs/smoke/demo.mp4 --steps 150
```

这会执行 PPO 更新并生成约 10 秒的 MP4。模型还没学会打球是正常现象。
PPO 按完整 rollout 收集样本，实际 transitions 可能向上取整到 `n_envs * n_steps` 的倍数。
`--run-dir` 必须不存在，防止覆盖实验；可换用 smoke2 等名称重新运行。

## 训练并评估

```bash
python tennis.py train --run-dir runs/tennis_1m --total-timesteps 1000000
tensorboard --logdir runs/tennis_1m
```

浏览器访问 TensorBoard 输出的网址，关注 `rollout/ep_rew_mean`、
`eval/mean_reward` 和 `time/fps`。前两项都保留原生回报口径，训练实际使用的
奖励见 `rewards/shaped`。开头没有完整回合时，可能还没有回报曲线。
控制台 fps 是 transitions/s，剩余时间可粗估为剩余 transitions / fps；
独立评估耗时另计。CPU 较慢可先减到 `--n-envs 1 --batch-size 128`；
有 NVIDIA GPU 可使用 `--device cuda`。训练时不开窗口。

每 10 万 transitions 保存检查点并评估 3 局。短训练未到评估频率时没有 best 模型。
按 Ctrl+C 会保存当前模型为 final_model.zip（不包含精确恢复训练所需的 rollout/RNG 状态）。
MVP 不提供断点续训。

```bash
python tennis.py evaluate --model runs/tennis_1m/best/best_model.zip --episodes 5 --output runs/tennis_1m/evaluation.json
```

JSON 包含随机策略、PPO 的每局原始累计回报、均值、标准差与超时截断标记。
两者使用同一组独立评估 seeds。完整评估可能比短训练耗时更长。
均值变大才是更好的回报，回报不等于获胜场数。不要从一次短演示判断已收敛。
若 100 万步没有提升，可另建 `runs/tennis_10m` 尝试 1000 万步并观察曲线；
没有保证获胜的固定训练步数。

## 辅助奖励

训练默认启用；旧模型不会自动获得新能力，需要新建实验重新训练。

| 项目 | 默认设置 | 触发条件 |
| --- | --- | --- |
| 原生奖励 | 原样保留 | ALE 返回的得失分反馈 |
| 成功回击 | 每次 +0.1 | 来球在己方球员附近改变飞行方向，越网后首次在对方场内落地；一次只计一次 |
| 接球站位 | 系数 0.1 | 来球接近己方时，横向移动缩小预计接球点误差；偏离则小额扣分 |
| 等待发球 | 超时后每个 agent step -0.01 | 确认由己方发球，等待超过 120 个 agent steps（约 8 秒） |
| 随意移动、按 FIRE | 直接奖励为 0 | 只有有效结果触发其他奖励；移动也不能重置等待计时 |

`训练奖励 = 原生奖励 + 回击奖励 + 站位奖励 + 等待惩罚`。
评估与默认 play 返回的奖励始终为原生奖励，best_model 按原生评估回报选择。

站位项使用当前球的地面速度线性预测它到达球员当前纵深时的横向坐标。
针对同一个目标比较移动前后的横向误差：
`0.1 × (移动前误差 - 移动后误差) / 160`。
只站着不动不会因为球的移动拿到分；每分的正站位奖励总额最多 0.05，
站位奖励的净累计值限制在 [-0.05, 0.05]。
它是启发式反馈，不是严格保证最优策略不变的势函数奖励，也没有完整的落点物理模型。

状态从模拟器 RAM 读取，CNN 仍只看图像。坐标参考
[OCAtari Tennis 状态提取](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/tennis.py)，
无需额外安装 OCAtari。每个原始游戏帧都检测事件，随后累加到 4 帧的 agent step。
用球的地面位置和高度区分飞行、落地，得失分、换边、传送、终止或超时会清理待确认回击。
场内判断排除边线附近的模糊情况；可能漏报边线球或凌空回击，不把按键当成成功回击。

发球阶段 RAM 字节通过 ALE 0.11.2 的默认 Tennis ROM 实验验证，
不是 ALE 官方提供的事件接口。检测是保守启发式：状态未知时不发放对应奖励。
更换 ROM、游戏 mode 或状态提取规则后，应重新验证检测器。
默认系数仅为实验起点，不保证提高成绩；等待惩罚长期累计可能大于原生得失分，
应结合等待时长、回击次数及原生评估回报调节。

新建辅助奖励实验，或训练原生奖励对照组：

```bash
python tennis.py train --run-dir runs/tennis_shaped_1m --total-timesteps 1000000
python tennis.py train --run-dir runs/tennis_raw_1m --total-timesteps 1000000 --no-reward-shaping
```

可调参数：`--return-bonus`、`--position-weight`、`--position-cap`、
`--idle-penalty`、`--idle-grace`；奖励项系数设为 0 即可单独关闭该项。
参数保存在 `config.json` 与 `reward_config.json`，建议同一 seed 下比较每次只改一个奖励项的实验。

TensorBoard / progress.csv 中：`rewards/raw`、`rewards/return_reward`、
`rewards/position_reward`、`rewards/idle_reward`、`rewards/auxiliary`、
`rewards/shaped` 是日志区间内每个 agent step 的平均值。
`rewards/return_count` 是每步平均有效回击次数，`rewards/own_serve_frames`
是每个 agent step 中己方等待发球的平均原始帧数。
`rewards/episode_shaped` 是已结束回合的候选塑形回报；Monitor CSV 和
`rollout/ep_rew_mean` 保留原生回报，即使设置 `--no-reward-shaping` 也能观察候选辅助奖励。

## 展示成果

保存视频（默认约 133 秒，可跨多个回合）：

```bash
python tennis.py play --model runs/tennis_1m/best/best_model.zip --video runs/tennis_1m/demo.mp4 --steps 2000
```

打开 MP4 即可查看或分享。视频为 15 fps：一个决策重复 4 个游戏帧。
只记录每次决策前的一帧，回合结束后 VecEnv 自动重置，因此不是逐帧完整录像。
同时默认生成 `demo.rewards.csv` 和 `demo.rewards.config.json`，记录每步动作、
原生奖励、各辅助项、候选 shaped 奖励以及发球状态。
play 自动读取模型同目录或上一级实验目录的 `reward_config.json`；旧模型使用默认系数做诊断。
这些辅助奖励仅用于展示诊断，不影响 play 的动作选择或原生奖励。
用 `--reward-log 路径.csv` 可指定日志位置，窗口模式也支持。

本机游戏窗口（需要图形桌面和支持显示的 ALE 安装）：

```bash
python tennis.py play --model runs/tennis_1m/best/best_model.zip --human --steps 2000
```

默认使用确定性动作，与评估一致。若模型一直不发球，可用 `--stochastic`
观察采样动作下的行为；这只是诊断演示，不能替换确定性评估结果。
Tennis 每一分开始都需要 FIRE，模型可能学到不发球来避免失分，应同时检查录像和回报。
窗口遇到显示问题时可先使用 MP4 导出。

所有命令都支持 `--seed`、`--device`（cpu / cuda / cuda:N / auto）和 `--threads`；
具体参数见各子命令 `--help`。

## 本次实际验证（2026-10-25）

在 macOS CPU、Python 3.10 环境测试通过；核心版本为 Gymnasium 1.4.0、
ALE 0.11.2、SB3 2.9.0、PyTorch 2.14.1。依赖装在临时虚拟环境
`/private/tmp/ppo-tennis-venv`，未修改已有 yja 环境；日常使用请按上面的安装步骤配置。

- 单环境 512 步训练完成，模型保存并重新加载成功。
- 四环境 512 步训练完成，生成 checkpoint、best_model、评估 NPZ、CSV 与 TensorBoard 日志。
- MP4 解码检查通过：H.264、160×210、15 fps、150 帧、10 秒，抽帧检查确为 Tennis 画面。
- 一个完整评估回合（seed 20000）：随机策略 -23、短训 PPO -24，均正常结束。
  该结果不支持“PPO 已有提升”的结论。

本地测试产物在 `runs/smoke_verified/` 和 `runs/callback_verified/`，已被 .gitignore 忽略。
没有运行百万步长训练；`--human` 桌面窗口路径尚未实测。

辅助奖励版本另行验证：

```bash
python -m unittest -v test_reward_shaping
```

- 11 个事件测试覆盖等待宽限期、对方发球不罚、无效移动不加分、站位预算、
  两侧回击落地、发球不算回击、场外落地不加分、得失分与终止时清理事件。
- 5000 个随机 agent steps 检测到 10 次有效回击事件，原始帧轨迹保存在
  `runs/shaping_smoke/return_event_traces.json`；这不是策略学习成果。
- 对照同 seed、同动作的原生/塑形环境，2000 步的画面、原生回报和回合边界完全相同。
- 从同一个模拟器状态分支验证：己方发球在 NOOP 下等待、FIRE 后开始，对方发球在 NOOP 下自行开始。
- 已完成 512 步塑形 PPO 训练、分项日志和模型保存；未进行长训练验证效果。
