"""Vanilla DQN with replay for the same assisted RAM Tennis task as RAM PPO."""
import argparse
import csv
import importlib.metadata
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from device_utils import device_info, resolve_device
from ram_env import FEATURE_NAMES, OBS_SIZE, TARGET_MODES, make_ram_env
from ram_ppo import (BASELINES, ROOT, positive, run_suite,
                     settings_for, write_json, write_observation)


def build_model(args, env):
    return DQN(
        'MlpPolicy', env, learning_rate=args.learning_rate,
        buffer_size=args.buffer_size, learning_starts=args.learning_starts,
        batch_size=args.batch_size, gamma=args.gamma,
        train_freq=(args.train_freq, 'step'), gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval, tau=1.0,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=1.0, exploration_final_eps=args.exploration_final_eps,
        policy_kwargs=dict(net_arch=[128, 128]),
        optimize_memory_usage=False,
        replay_buffer_kwargs=dict(handle_timeout_termination=True),
        seed=args.seed, device=args.device, verbose=1,
    )


def selection_score(result):
    """Standard bounded ALE episodes count; artificial eval/serve cutoffs do not."""
    episodes = result.get('episodes', [])
    if not episodes or len(episodes) != result['total']:
        return None
    if any(e['stop_reason'] not in ('game_over', 'environment_time_limit') for e in episodes):
        return None
    return result['mean_raw_reward']


class DQNMetrics(BaseCallback):
    def __init__(self, frequency):
        super().__init__()
        self.frequency = frequency

    def _on_training_start(self):
        self.started = time.monotonic()
        self.start_steps = self.num_timesteps

    def _on_step(self):
        for info in self.locals['infos']:
            for name, value in info['ram_rewards'].items():
                self.logger.record_mean('rewards/' + name, value)
            self.logger.record_mean('behavior/auto_serve', int(info['auto_serve']))
            if info['ball_direction'] == 'outgoing' and info['target'] is not None:
                state = info['tennis_state']
                distance = float(np.hypot(state['player_x'] - info['target'][0],
                                          state['player_y'] - info['target'][1]))
                self.logger.record_mean('behavior/recovery_distance', distance)
                self.logger.record_mean('behavior/at_recovery_center', int(distance <= 5))
        if self.n_calls % self.frequency == 0:
            replay = self.model.replay_buffer
            self.logger.record('replay/transitions', replay.size() * replay.n_envs)
            self.logger.record('replay/capacity', replay.buffer_size * replay.n_envs)
            self.logger.record('train/exploration_rate', self.model.exploration_rate)
            self.logger.record('train/n_updates', self.model._n_updates)
            self.logger.record('time/total_timesteps', self.num_timesteps)
            elapsed = time.monotonic() - self.started
            self.logger.record('time/fps', (self.num_timesteps - self.start_steps) / max(elapsed, 1e-6))
            self.logger.record('time/time_elapsed', elapsed)
            self.logger.dump(self.num_timesteps)
        return True


class DQNEvaluation(BaseCallback):
    def __init__(self, run, settings, baselines, frequency):
        super().__init__()
        self.run, self.settings, self.baselines = run, settings, baselines
        self.frequency = frequency
        self.best_score = self.best_qualified = -np.inf
        scores = [selection_score(baselines[name]) for name in ('downfire', 'upfire', 'fire')]
        self.constant_score = max((score for score in scores if score is not None), default=None)

    def _on_step(self):
        if self.n_calls % self.frequency:
            return True
        print(f'\n[eval] DQN transitions={self.num_timesteps}; exploration disabled', flush=True)
        seeds = list(range(10000, 10000 + self.settings['eval_episodes']))
        policies = run_suite(self.model, self.settings, ('dqn',), seeds,
                             self.settings['eval_max_steps'], self.settings['eval_idle_limit'])
        item = policies['dqn']
        score = selection_score(item)
        qualified = score is not None and self.constant_score is not None and score > self.constant_score
        record = dict(timesteps=self.num_timesteps, seeds=seeds,
                      constant_baseline_score=self.constant_score,
                      beats_constant_baseline=qualified, policies=policies,
                      selection_protocol='game_over_or_natural_environment_time_limit')
        write_json(self.run / 'eval' / f'diagnostics_{self.num_timesteps}.json', record)
        if score is not None and score > self.best_score:
            self.best_score = score
            self.model.save(self.run / 'best' / 'best_model')
            write_json(self.run / 'best' / 'evaluation.json', record)
        if qualified and score > self.best_qualified:
            self.best_qualified = score
            self.model.save(self.run / 'qualified' / 'best_model')
            write_json(self.run / 'qualified' / 'evaluation.json', record)
        self.logger.record('eval/dqn_raw', item['mean_raw_reward'])
        self.logger.record('eval/dqn_completed', item['completed'] / item['total'])
        self.logger.record('eval/dqn_dominant_movement', item['dominant_movement_fraction'])
        self.logger.record('eval/beats_constant_baseline', int(qualified))
        write_observation(self.run, self.baselines)
        return True


def replay_statistics(model):
    replay = model.replay_buffer
    arrays = ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts')
    return dict(
        transitions=int(replay.size() * replay.n_envs),
        capacity=int(replay.buffer_size * replay.n_envs),
        rows_per_env=int(replay.size()), n_envs=replay.n_envs,
        allocated_bytes=sum(getattr(replay, name).nbytes for name in arrays),
        n_updates=model._n_updates, exploration_rate=model.exploration_rate,
        timesteps=model.num_timesteps,
    )


def train(args):
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    for name in ('eval', 'best', 'qualified', 'monitor', 'checkpoints'):
        (run / name).mkdir()
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    settings.update(algorithm='DQN', observation='ram',
                    selection_protocol='game_over_or_natural_environment_time_limit')
    write_json(run / 'config.json', settings)
    write_json(run / 'versions.json', {name: importlib.metadata.version(name) for name in
               ('gymnasium', 'ale-py', 'stable-baselines3', 'torch')})
    write_json(run / 'observation_schema.json', dict(
        size=OBS_SIZE, ram_bytes=128, features=FEATURE_NAMES, executed_action_one_hot=18,
        frame_skip=4, target_mode=args.target_mode))
    print('[eval] fixed-seed baselines; same RAM, rewards and auto-serve as PPO', flush=True)
    baselines = run_suite(None, settings, BASELINES, range(10000, 10000 + args.eval_episodes),
                          args.eval_max_steps, args.eval_idle_limit)
    write_json(run / 'baseline_validation.json', baselines)
    write_observation(run, baselines)
    env = DummyVecEnv([
        lambda rank=rank: make_ram_env(
            auto_serve=args.auto_serve, shaping=True, gamma=args.gamma, beta=args.beta,
            serve_mode=args.serve_mode, target_mode=args.target_mode,
            monitor_file=str(run / 'monitor' / str(rank)))
        for rank in range(args.n_envs)
    ])
    env.seed(args.seed)
    model = None
    started = time.monotonic()
    try:
        model = build_model(args, env)
        model.set_logger(configure(str(run), ['stdout', 'csv', 'tensorboard']))
        write_json(run / 'device.json', device_info(args.requested_device, model.device))
        print(f"Replay arrays: {replay_statistics(model)['allocated_bytes'] / 1024**2:.1f} MiB; "
              f"learning starts after {args.learning_starts} transitions", flush=True)
        callbacks = [
            DQNMetrics(max(args.log_freq // args.n_envs, 1)),
            DQNEvaluation(run, settings, baselines, max(args.eval_freq // args.n_envs, 1)),
            CheckpointCallback(save_freq=max(args.eval_freq // args.n_envs, 1),
                               save_path=str(run / 'checkpoints'), name_prefix='dqn_ram'),
        ]
        try:
            model.learn(total_timesteps=args.total_timesteps, callback=callbacks, log_interval=1)
        except KeyboardInterrupt:
            model.save(run / 'interrupted_model')
            write_json(run / 'replay_stats.json', replay_statistics(model))
            write_json(run / 'training_status.json',
                       dict(status='interrupted', timesteps=model.num_timesteps))
            raise
        model.save(run / 'final_model')
        write_json(run / 'replay_stats.json', replay_statistics(model))
        if args.save_replay_buffer:
            model.save_replay_buffer(run / 'replay_buffer.pkl')
        write_json(run / 'training_status.json', dict(
            status='training_complete', timesteps=model.num_timesteps,
            train_seconds=time.monotonic() - started))
    finally:
        env.close()
    selected = run / 'best' / 'best_model.zip'
    if selected.exists():
        print('[eval] held-out seeds for selected best DQN', flush=True)
        chosen = DQN.load(selected, device=args.device)
        seeds = list(range(20000, 20000 + args.eval_episodes))
        results = run_suite(chosen, settings, (*BASELINES, 'dqn'), seeds,
                            args.eval_max_steps, args.eval_idle_limit)
        write_json(run / 'heldout_evaluation.json',
                   dict(model=str(selected), seeds=seeds, policies=results))
    write_json(run / 'training_status.json', dict(
        status='complete', timesteps=model.num_timesteps,
        train_seconds=time.monotonic() - started,
        note='Includes checkpoint and final held-out evaluation; initial baselines excluded.'))
    print(f'Run complete: {run}', flush=True)


def load_dqn(args):
    settings = settings_for(args.model.resolve())
    if settings.get('algorithm') != 'DQN':
        raise ValueError('Expected a RAM DQN model/config, not an image DQN or PPO model.')
    return DQN.load(args.model, device=args.device), settings


def evaluate(args):
    model, settings = load_dqn(args)
    seeds = list(range(args.seed, args.seed + args.episodes))
    result = run_suite(model, settings, (*BASELINES, 'dqn'), seeds,
                       args.eval_max_steps, args.eval_idle_limit)
    write_json(args.output, dict(model=str(args.model.resolve()), seeds=seeds, policies=result,
                                max_agent_steps=args.eval_max_steps, idle_limit=args.eval_idle_limit))


def play(args):
    import imageio.v2 as imageio
    model, settings = load_dqn(args)
    model.exploration_rate = args.epsilon
    env = make_ram_env(auto_serve=settings['auto_serve'], shaping=False,
                       gamma=settings['gamma'], beta=settings['beta'], render_mode='rgb_array',
                       serve_mode=settings['serve_mode'], target_mode=settings['target_mode'])
    args.video.parent.mkdir(parents=True, exist_ok=True)
    try:
        obs, _ = env.reset(seed=args.seed)
        names = env.unwrapped.get_action_meanings()
        fields = ['step', 'requested', 'executed', 'auto_serve', 'side', 'raw', 'returns',
                  'player_x', 'player_y', 'ball_x', 'ball_y', 'done',
                  'ball_direction', 'target_x', 'target_y']
        with imageio.get_writer(args.video, fps=15, codec='libx264', macro_block_size=2) as video, \
                args.video.with_suffix('.actions.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for step in range(args.steps):
                video.append_data(env.render())
                action, _ = model.predict(obs, deterministic=args.epsilon == 0)
                obs, reward, terminated, truncated, info = env.step(int(action))
                state, target = info['tennis_state'], info['target']
                writer.writerow(dict(
                    step=step + 1, requested=names[info['requested_action']],
                    executed=names[info['executed_action']], auto_serve=info['auto_serve'],
                    side=info['side_before'], raw=reward, returns=info['return_count'],
                    player_x=state['player_x'], player_y=state['player_y'],
                    ball_x=state['ball_x'], ball_y=state['ball_y'], done=terminated or truncated,
                    ball_direction=info['ball_direction'],
                    target_x=target[0] if target is not None else None,
                    target_y=target[1] if target is not None else None))
                if terminated or truncated:
                    obs, _ = env.reset()
    finally:
        env.close()
    write_json(args.video.with_suffix('.config.json'), dict(
        algorithm='DQN', observation='ram', model=str(args.model.resolve()), seed=args.seed,
        epsilon=args.epsilon, auto_serve=settings['auto_serve'], target_mode=settings['target_mode'],
        serve_mode=settings['serve_mode'], fps=15, steps=args.steps))
    print(f'Video: {args.video.resolve()}', flush=True)


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return result


def probability(value):
    result = float(value)
    if not 0 <= result <= 1:
        raise argparse.ArgumentTypeError('must be in [0, 1]')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--total-timesteps', type=positive, default=250000)
    p.add_argument('--n-envs', type=positive, default=8)
    p.add_argument('--buffer-size', type=positive, default=100000)
    p.add_argument('--learning-starts', type=nonnegative, default=10000)
    p.add_argument('--batch-size', type=positive, default=128)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--train-freq', type=positive, default=4)
    p.add_argument('--gradient-steps', type=positive)
    p.add_argument('--target-update-interval', type=positive, default=10000)
    p.add_argument('--exploration-fraction', type=probability, default=.4)
    p.add_argument('--exploration-final-eps', type=probability, default=.05)
    p.add_argument('--auto-serve', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--serve-mode', choices=['fire', 'align_x_fire'], default='align_x_fire')
    p.add_argument('--target-mode', choices=TARGET_MODES, default='recover_chase')
    p.add_argument('--gamma', type=float, default=.99)
    p.add_argument('--beta', type=float, default=.05)
    p.add_argument('--eval-freq', type=positive, default=50000)
    p.add_argument('--eval-episodes', type=positive, default=5)
    p.add_argument('--log-freq', type=positive, default=1000)
    p.add_argument('--save-replay-buffer', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p = sub.add_parser('evaluate')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=positive, default=5)
    p.add_argument('--seed', type=int, default=20000)
    p = sub.add_parser('play')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--steps', type=positive, default=2000)
    p.add_argument('--seed', type=int, default=20000)
    p.add_argument('--epsilon', type=probability, default=0.0)
    for child in sub.choices.values():
        child.add_argument('--device', default='cpu')
        child.add_argument('--threads', type=positive, default=1)
        if not child.prog.endswith('play'):
            child.add_argument('--eval-max-steps', type=positive, default=27000)
            child.add_argument('--eval-idle-limit', type=positive, default=150)
    args = parser.parse_args()
    if args.command == 'train':
        if args.run_dir.exists():
            parser.error('run-dir already exists')
        if args.buffer_size < max(args.batch_size, args.n_envs):
            parser.error('buffer-size must be >= batch-size and n-envs')
        if args.learning_starts >= args.total_timesteps:
            parser.error('learning-starts must be below total-timesteps')
        if not (np.isfinite(args.learning_rate) and args.learning_rate > 0
                and 0 < args.gamma < 1 and np.isfinite(args.beta) and args.beta >= 0
                and args.exploration_fraction > 0):
            parser.error('Invalid learning rate, gamma, beta or exploration fraction')
        if args.eval_freq % args.n_envs:
            parser.error('eval-freq must be divisible by n-envs')
        args.gradient_steps = args.gradient_steps or args.n_envs
    args.requested_device = args.device
    args.device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    set_random_seed(args.seed)
    {'train': train, 'evaluate': evaluate, 'play': play}[args.command](args)


if __name__ == '__main__':
    main()

