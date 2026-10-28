"""Deep Q-learning with uniform replay and a target network for Atari Tennis.

Uses SB3 vanilla DQN (not Double/Dueling/Prioritized DQN). See DQN.md.
"""

import argparse
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed

from device_utils import device_info, resolve_device
from reward_shaping import RewardConfig
from tennis import (ROOT, RewardLoggingCallback, evaluate, make_env, nonnegative_float,
                    play, positive, reward_config_from_args, write_json)


class VisibleEvalCallback(EvalCallback):
    """Single-environment, bounded evaluation with heartbeat and diagnostics."""

    def _on_step(self):
        triggered = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if triggered:
            self.eval_steps = 0
            self.wait_steps = 0
            self.return_count = 0
            self.episodes = []
            start = time.monotonic()
            print(f'\n[eval] 开始：训练步数={self.num_timesteps}，'
                  f'评估 {self.n_eval_episodes} 局；训练暂时等待', flush=True)
        result = super()._on_step()
        if triggered:
            summary = dict(timesteps=self.num_timesteps, steps=self.eval_steps,
                           elapsed_seconds=time.monotonic() - start,
                           own_serve_steps=self.wait_steps, returns=self.return_count,
                           episodes=self.episodes)
            write_json(Path(self.best_model_save_path).parent / 'eval' /
                       f'diagnostics_{self.num_timesteps}.json', summary)
            print(f'[eval] 结束：耗时 {summary["elapsed_seconds"]:.1f}s，继续训练\n', flush=True)
        return result

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        info = locals_['info']
        self.eval_steps += 1
        self.wait_steps += int(info['tennis_state']['own_serve'])
        self.return_count += int(info['reward_components']['return_count'])
        if self.eval_steps % 1000 == 0:
            print(f'[eval] 已执行 {self.eval_steps} 步，已结束 {len(self.episodes)} 局', flush=True)
        if locals_['done'] and 'episode' in info:
            entry = dict(reward=info['episode']['r'], steps=info['episode']['l'],
                         truncated=bool(info.get('TimeLimit.truncated')))
            self.episodes.append(entry)
            print(f'[eval] 第 {len(self.episodes)} 局：{entry}', flush=True)


class ReplayLoggingCallback(BaseCallback):
    def __init__(self, log_freq):
        super().__init__()
        self.log_freq = log_freq

    def _on_training_start(self):
        self.started = time.monotonic()
        self.start_steps = self.num_timesteps

    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            # Callback executes before SB3 stores this step's transition.
            self.logger.record('replay/size', self.model.replay_buffer.size())
            self.logger.record('replay/capacity', self.model.replay_buffer.buffer_size)
            self.logger.record('train/exploration_rate', self.model.exploration_rate)
            self.logger.record('train/n_updates', self.model._n_updates)
            self.logger.record('time/total_timesteps', self.num_timesteps)
            elapsed = time.monotonic() - self.started
            self.logger.record('time/fps', (self.num_timesteps - self.start_steps) / max(elapsed, 1e-6))
            self.logger.record('time/time_elapsed', elapsed)
            self.logger.dump(self.num_timesteps)
        return True


def build_model(args, env):
    return DQN(
        'CnnPolicy', env, device=args.device, seed=args.seed, verbose=1,
        learning_rate=args.learning_rate, buffer_size=args.buffer_size,
        learning_starts=args.learning_starts, batch_size=args.batch_size,
        gamma=0.99, train_freq=(args.train_freq, 'step'), gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval, tau=1.0,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=1.0, exploration_final_eps=args.exploration_final_eps,
        # Store next observations separately to correctly mask time-limit truncations.
        optimize_memory_usage=False,
        replay_buffer_kwargs={'handle_timeout_termination': True},
    )


def train(args):
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / 'config.json', {k: str(v) if isinstance(v, Path) else v
                                    for k, v in vars(args).items()})
    write_json(run / 'versions.json', {k: importlib.metadata.version(k) for k in
               ['gymnasium', 'ale-py', 'stable-baselines3', 'torch']})
    config = reward_config_from_args(args)
    write_json(run / 'reward_config.json', dict(enabled=args.reward_shaping,
               detector='ram-heuristics-v2', **asdict(config)))
    env = make_env(seed=args.seed, shaping=args.reward_shaping,
                   reward_config=config, monitor_dir=run / 'monitor')
    eval_env = make_env(seed=10_000 + args.seed, max_agent_steps=args.eval_max_steps,
                        reward_config=config)
    try:
        replay_bytes = 2 * args.buffer_size * int(np.prod(env.observation_space.shape))
        print(f'回放池观测数组约 {replay_bytes / 1024**3:.2f} GiB 主机内存；'
              f'预热 {args.learning_starts} 步后开始更新。', flush=True)
        model = build_model(args, env)
        model.set_logger(configure(str(run), ['stdout', 'csv', 'tensorboard']))
        write_json(run / 'device.json', device_info(args.requested_device, model.device))
        callbacks = [
            RewardLoggingCallback(), ReplayLoggingCallback(args.log_freq),
            CheckpointCallback(save_freq=args.save_freq, save_path=str(run / 'checkpoints'),
                               name_prefix='dqn_tennis'),
            VisibleEvalCallback(eval_env, best_model_save_path=str(run / 'best'),
                                log_path=str(run / 'eval'), eval_freq=args.eval_freq,
                                n_eval_episodes=args.eval_episodes, deterministic=True),
        ]
        try:
            # Step-based logging above still works when a policy never finishes a game.
            model.learn(total_timesteps=args.total_timesteps, callback=callbacks, log_interval=1)
        except KeyboardInterrupt:
            print('训练中断，保存模型及回放统计。', flush=True)
        model.save(run / 'final_model')
        write_json(run / 'replay_stats.json', dict(size=model.replay_buffer.size(),
                   capacity=model.replay_buffer.buffer_size, n_updates=model._n_updates,
                   exploration_rate=model.exploration_rate,
                   saved_buffer=args.save_replay_buffer, timesteps=model.num_timesteps))
        if args.save_replay_buffer:
            print('正在保存回放池，文件可能较大……', flush=True)
            model.save_replay_buffer(run / 'replay_buffer.pkl')
        print(f'模型：{run / "final_model.zip"}', flush=True)
    finally:
        env.close()
        eval_env.close()


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('必须为非负整数')
    return result


def probability(value):
    result = float(value)
    if not 0 <= result <= 1:
        raise argparse.ArgumentTypeError('必须在 [0, 1] 内')
    return result


def main():
    parser = argparse.ArgumentParser(description='Q-learning / DQN 网球：经验回放 + 目标网络')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('--run-dir', type=Path, default=ROOT / 'runs/dqn_tennis')
    p.add_argument('--total-timesteps', type=positive, default=1_000_000)
    p.add_argument('--buffer-size', type=positive, default=20_000)
    p.add_argument('--learning-starts', type=nonnegative_int, default=10_000)
    p.add_argument('--batch-size', type=positive, default=32)
    p.add_argument('--learning-rate', type=nonnegative_float, default=1e-4)
    p.add_argument('--train-freq', type=positive, default=4)
    p.add_argument('--gradient-steps', type=positive, default=1)
    p.add_argument('--target-update-interval', type=positive, default=10_000)
    p.add_argument('--exploration-fraction', type=probability, default=0.2)
    p.add_argument('--exploration-final-eps', type=probability, default=0.05)
    p.add_argument('--save-freq', type=positive, default=100_000)
    p.add_argument('--eval-freq', type=positive, default=100_000)
    p.add_argument('--eval-episodes', type=positive, default=1)
    p.add_argument('--log-freq', type=positive, default=1000)
    p.add_argument('--save-replay-buffer', action='store_true', help='结束时另存回放池（较大）')
    p.add_argument('--reward-shaping', action=argparse.BooleanOptionalAction, default=True)
    c = RewardConfig()
    for flag, default in [('return-bonus', c.return_bonus), ('position-weight', c.position_weight),
                          ('position-cap', c.position_cap), ('idle-penalty', c.idle_penalty)]:
        p.add_argument('--' + flag, type=nonnegative_float, default=default)
    p.add_argument('--idle-grace', type=positive, default=c.idle_grace)
    p.add_argument('--position-mode', choices=['intercept_2d', 'legacy_x'], default=c.position_mode)
    p.add_argument('--tracking-penalty', type=nonnegative_float, default=c.tracking_penalty)
    p.add_argument('--tracking-radius', type=nonnegative_float, default=c.tracking_radius)
    p = sub.add_parser('evaluate')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--episodes', type=positive, default=5)
    p.add_argument('--output', type=Path, default=ROOT / 'runs/dqn_evaluation.json')
    p = sub.add_parser('play')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--video', type=Path, default=ROOT / 'runs/dqn_demo.mp4')
    p.add_argument('--steps', type=positive, default=2000)
    p.add_argument('--human', action='store_true')
    p.add_argument('--stochastic', action='store_true', help='启用 epsilon-greedy 探索')
    p.add_argument('--epsilon', type=probability, default=0.05)
    p.add_argument('--reward-log', type=Path)
    for command, p in sub.choices.items():
        if command != 'play':
            p.add_argument('--eval-max-steps', type=positive, default=5000,
                           help='每局评估 agent step 上限，超时单独标记')
        p.add_argument('--seed', type=int, default=20_000 if command == 'evaluate' else 0)
        p.add_argument('--device', default='cpu', help='cpu / cuda / cuda:N / auto')
        p.add_argument('--threads', type=positive, default=4)
    args = parser.parse_args()
    if args.command == 'train':
        if args.run_dir.exists():
            parser.error('run-dir 已存在，请使用新的实验目录')
        if args.batch_size > args.buffer_size:
            parser.error('batch-size 不能大于 buffer-size')
        if args.learning_starts >= args.total_timesteps:
            parser.error('learning-starts 必须小于 total-timesteps，否则不会执行 Q 网络更新')
        if args.learning_rate <= 0 or args.exploration_fraction == 0:
            parser.error('learning-rate 和 exploration-fraction 必须大于 0')
    args.requested_device = args.device
    try:
        args.device = resolve_device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    torch.set_num_threads(args.threads)
    set_random_seed(args.seed)
    print(f'计算设备：{args.device}（请求：{args.requested_device}）', flush=True)
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args, algorithm=DQN, label='dqn')
    else:
        play(args, algorithm=DQN)


if __name__ == '__main__':
    main()
