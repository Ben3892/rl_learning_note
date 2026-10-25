# PPO Tennis MVP

先读 [最小 MVP 设计](MVP.md)，代码集中在 [tennis.py](tennis.py)。
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
`eval/mean_reward` 和 `time/fps`。开头没有完整回合时，可能还没有回报曲线。
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

## 展示成果

保存视频（默认约 133 秒，可跨多个回合）：

```bash
python tennis.py play --model runs/tennis_1m/best/best_model.zip --video runs/tennis_1m/demo.mp4 --steps 2000
```

打开 MP4 即可查看或分享。视频为 15 fps：一个决策重复 4 个游戏帧。
只记录每次决策前的一帧，回合结束后 VecEnv 自动重置，因此不是逐帧完整录像。

本机游戏窗口（需要图形桌面和支持显示的 ALE 安装）：

```bash
python tennis.py play --model runs/tennis_1m/best/best_model.zip --human --steps 2000
```

默认使用确定性动作，与评估一致。若模型一直不发球，可用 `--stochastic`
观察采样动作下的行为；这只是诊断演示，不能替换确定性评估结果。
Tennis 每一分开始都需要 FIRE，模型可能学到不发球来避免失分，应同时检查录像和回报。
窗口遇到显示问题时可先使用 MP4 导出。

所有命令都支持 `--seed`、`--device` 和 `--threads`；具体参数见各子命令 `--help`。

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
