"""PPO Atari Tennis MVP. Run `python tennis.py --help` for commands."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

gym.register_envs(ale_py)
ROOT = Path(__file__).resolve().parent
ENV_ID = "ALE/Tennis-v5"
MAX_RAW_STEPS = 108_000


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_env(n_envs=1, seed=0, render_mode=None, monitor_dir=None):
    """Exactly the same observation/reward pipeline for all three commands."""
    def factory(rank):
        def create():
            env = gym.make(
                ENV_ID, frameskip=1, repeat_action_probability=0.25,
                render_mode=render_mode, max_episode_steps=MAX_RAW_STEPS,
            )
            env = AtariWrapper(env, frame_skip=4, terminal_on_life_loss=False,
                               clip_reward=False)
            filename = None
            if monitor_dir is not None:
                Path(monitor_dir).mkdir(parents=True, exist_ok=True)
                filename = str(Path(monitor_dir) / str(rank))
            # Outside AtariWrapper: counts agent steps, keeps unmodified rewards.
            return Monitor(env, filename=filename)
        return create

    env = DummyVecEnv([factory(rank) for rank in range(n_envs)])
    env = VecTransposeImage(VecFrameStack(env, n_stack=4))
    env.seed(seed)  # VecEnv applies this seed at the next reset.
    return env


def train(args):
    run = args.run_dir.resolve()
    # A new directory prevents mixing checkpoints/logs from different experiments.
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", {k: str(v) if isinstance(v, Path) else v
                                    for k, v in vars(args).items()})
    write_json(run / "versions.json", {name: importlib.metadata.version(name) for name in
                                      ["gymnasium", "ale-py", "stable-baselines3", "torch"]})
    env = make_env(args.n_envs, args.seed, monitor_dir=run / "monitor")
    eval_env = make_env(seed=10_000 + args.seed)
    try:
        model = PPO(
            "CnnPolicy", env, learning_rate=2.5e-4, n_steps=args.n_steps,
            batch_size=args.batch_size, n_epochs=4, gamma=0.99, gae_lambda=0.95,
            clip_range=0.1, ent_coef=0.01, seed=args.seed, device=args.device,
            verbose=1,
        )
        model.set_logger(configure(str(run), ["stdout", "csv", "tensorboard"]))
        callbacks = [
            CheckpointCallback(save_freq=max(args.save_freq // args.n_envs, 1),
                               save_path=str(run / "checkpoints"), name_prefix="ppo_tennis"),
            EvalCallback(eval_env, best_model_save_path=str(run / "best"),
                         log_path=str(run / "eval"),
                         eval_freq=max(args.eval_freq // args.n_envs, 1),
                         n_eval_episodes=args.eval_episodes, deterministic=True),
        ]
        try:
            model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
        except KeyboardInterrupt:
            print("训练已中断，保存当前模型。")
        model.save(run / "final_model")
        print(f"模型已保存：{run / 'final_model.zip'}")
    finally:
        env.close()
        eval_env.close()


def evaluate(args):
    model = PPO.load(args.model, device=args.device)
    result = {"model": str(args.model.resolve()), "env_id": ENV_ID,
              "max_raw_steps": MAX_RAW_STEPS, "deterministic": True,
              "note": "原始累计奖励，不等于胜率；truncated 表示时间上限。"}
    for label in ("random", "ppo"):
        env = make_env()
        episodes = []
        try:
            for index in range(args.episodes):
                seed = args.seed + index
                rng = np.random.default_rng(seed)
                env.seed(seed)
                obs = env.reset()
                score, length = 0.0, 0
                while True:
                    if label == "random":
                        action = np.array([rng.integers(env.action_space.n)])
                    else:
                        action, _ = model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = env.step(action)
                    score += float(rewards[0])
                    length += 1
                    if dones[0]:
                        episodes.append({"seed": seed, "reward": score, "steps": length,
                                         "truncated": bool(infos[0].get("TimeLimit.truncated"))})
                        print(f"{label}: seed={seed}, reward={score}, steps={length}")
                        break
            scores = [episode["reward"] for episode in episodes]
            result[label] = {"mean_reward": float(np.mean(scores)),
                             "std_reward": float(np.std(scores)), "episodes": episodes}
        finally:
            env.close()
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def play(args):
    import imageio.v2 as imageio

    model = PPO.load(args.model, device=args.device)
    env = make_env(seed=args.seed, render_mode="human" if args.human else "rgb_array")
    writer = None
    try:
        if not args.human:
            args.video.parent.mkdir(parents=True, exist_ok=True)
            # 60 emulator frames/s, 4 emulator frames per agent action.
            writer = imageio.get_writer(str(args.video), fps=15, codec="libx264",
                                        macro_block_size=2)
        obs = env.reset()
        score = 0.0
        for step in range(args.steps):
            if writer is not None:
                writer.append_data(env.get_images()[0])
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, rewards, dones, infos = env.step(action)
            score += float(rewards[0])
            if dones[0]:
                print(f"回合结束：reward={score}, step={step + 1}, "
                      f"truncated={infos[0].get('TimeLimit.truncated', False)}")
                score = 0.0  # VecEnv auto-resets; frame history is reset by VecFrameStack.
        if writer is not None:
            print(f"视频已保存：{args.video.resolve()}（定长片段，不代表完整比赛）")
    finally:
        if writer is not None:
            writer.close()
        env.close()


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def main():
    parser = argparse.ArgumentParser(description="PPO 训练 Atari Tennis、评估并展示")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="新建实验并训练 PPO")
    p.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "tennis")
    p.add_argument("--total-timesteps", type=positive, default=1_000_000)
    p.add_argument("--n-envs", type=positive, default=4)
    p.add_argument("--n-steps", type=positive, default=128)
    p.add_argument("--batch-size", type=positive, default=256)
    p.add_argument("--save-freq", type=positive, default=100_000)
    p.add_argument("--eval-freq", type=positive, default=100_000)
    p.add_argument("--eval-episodes", type=positive, default=3)
    p = sub.add_parser("evaluate", help="在相同 seeds 上比较随机策略与 PPO")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--episodes", type=positive, default=5)
    p.add_argument("--output", type=Path, default=ROOT / "runs" / "evaluation.json")
    p = sub.add_parser("play", help="导出 MP4，或使用 --human 打开游戏窗口")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--video", type=Path, default=ROOT / "runs" / "demo.mp4")
    p.add_argument("--steps", type=positive, default=2000)
    p.add_argument("--human", action="store_true")
    p.add_argument("--stochastic", action="store_true", help="随机采样动作，默认确定性演示")
    for p in sub.choices.values():
        p.add_argument("--seed", type=int, default=20_000 if p.prog.endswith("evaluate") else 0)
        p.add_argument("--device", default="cpu", help="cpu / cuda / auto")
        p.add_argument("--threads", type=positive, default=4)
    args = parser.parse_args()
    if args.command == "train":
        rollout = args.n_envs * args.n_steps
        if rollout < 2 or args.batch_size < 2 or rollout % args.batch_size:
            parser.error("n_envs * n_steps 必须 >= 2，且能被 batch_size（>=2）整除")
        if args.run_dir.exists():
            parser.error("run-dir 已存在，请使用一个新的实验目录")
    torch.set_num_threads(args.threads)
    set_random_seed(args.seed)
    {"train": train, "evaluate": evaluate, "play": play}[args.command](args)


if __name__ == "__main__":
    main()
