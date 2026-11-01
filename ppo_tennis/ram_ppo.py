"""Train/evaluate/render a separate RAM + MLP PPO Tennis baseline."""
import argparse
from collections import Counter, defaultdict
import csv
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from device_utils import device_info, resolve_device
from ram_env import FEATURE_NAMES, OBS_SIZE, TARGET_MODES, make_ram_env

ROOT = Path(__file__).resolve().parent
BASELINES = ('random', 'downfire', 'upfire', 'fire')


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def settings_for(model_path):
    for parent in (model_path.parent, model_path.parent.parent):
        file = parent / 'config.json'
        if file.exists():
            settings = json.loads(file.read_text())
            if settings.get('observation') != 'ram':
                raise ValueError('Expected a RAM-model config.json, not an image-model config.')
            return settings
    raise FileNotFoundError('RAM-model config.json must accompany the model.')


def summary(episodes):
    complete = [e for e in episodes if not e['truncated']]
    counts, decisions = Counter(), Counter()
    side_counts = defaultdict(Counter)
    for episode in episodes:
        counts.update(episode['executed_actions'])
        decisions.update(episode['decision_actions'])
        for side, actions in episode['actions_by_side'].items():
            side_counts[str(side)].update(actions)
    movement = Counter()
    for name, count in decisions.items():
        movement[name.replace('FIRE', '') or 'NOOP'] += count
    return dict(
        mean_raw_reward=float(np.mean([e['raw_reward'] for e in episodes])),
        mean_completed_reward=float(np.mean([e['raw_reward'] for e in complete])) if complete else None,
        completed=len(complete), total=len(episodes),
        stalled=sum(e['stop_reason'] == 'serve_stall' for e in episodes),
        truncated=sum(e['truncated'] for e in episodes),
        returns=sum(e['returns'] for e in episodes),
        executed_actions=dict(counts), decision_actions=dict(decisions),
        movement_actions=dict(movement),
        dominant_movement_fraction=max(movement.values(), default=0) / max(sum(movement.values()), 1),
        dominant_decision_fraction=max(decisions.values(), default=0) / max(sum(decisions.values()), 1),
        actions_by_side={key: dict(value) for key, value in side_counts.items()},
        episodes=episodes,
    )


def run_suite(model, settings, policies, seeds, max_steps=5000, idle_limit=150):
    """Each episode has an explicit seed; preserve the training policy's RNG state."""
    results = {}
    devices = [model.device.index] if model is not None and model.device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        for policy in policies:
            env = make_ram_env(auto_serve=settings['auto_serve'], shaping=False,
                               gamma=settings['gamma'], beta=settings['beta'],
                               serve_mode=settings.get('serve_mode', 'fire'),
                               target_mode=settings.get('target_mode', 'intercept'),
                               max_steps=max_steps, idle_limit=idle_limit)
            names = env.unwrapped.get_action_meanings()
            episodes = []
            try:
                for seed in seeds:
                    obs, _ = env.reset(seed=seed)
                    rng = np.random.default_rng(seed)
                    torch.manual_seed(seed)
                    raw, returns, overrides, waits = 0.0, 0, 0, 0
                    executed, decisions = Counter(), Counter()
                    actions_by_side, raw_by_side = defaultdict(Counter), defaultdict(float)
                    for length in range(1, max_steps + 1):
                        if policy == 'random':
                            action = int(rng.integers(env.action_space.n))
                        elif policy in ('downfire', 'upfire', 'fire'):
                            action = names.index(policy.upper())
                        else:
                            action, _ = model.predict(obs, deterministic=policy in ('ppo', 'dqn'))
                            action = int(action)
                        obs, reward, terminated, truncated, info = env.step(action)
                        raw += float(reward)
                        returns += info['return_count']
                        overrides += int(info['auto_serve'])
                        waits += int(info['tennis_state']['own_serve'])
                        actual = names[info['executed_action']]
                        executed[actual] += 1
                        if not info['auto_serve']:
                            decisions[names[action]] += 1
                            actions_by_side[str(info['side_before'])][names[action]] += 1
                        raw_by_side[str(info['side_before'])] += float(reward)
                        if terminated or truncated:
                            episodes.append(dict(
                                seed=seed, raw_reward=raw, steps=length, returns=returns,
                                auto_serve_steps=overrides, own_serve_steps=waits,
                                truncated=bool(truncated),
                                stop_reason=info.get('evaluation_stop_reason', 'game_over'),
                                executed_actions=dict(executed), decision_actions=dict(decisions),
                                actions_by_side={k: dict(v) for k, v in actions_by_side.items()},
                                raw_by_side=dict(raw_by_side),
                            ))
                            break
                    else:
                        raise RuntimeError('Evaluation limit failed to end an episode.')
            finally:
                env.close()
            results[policy] = summary(episodes)
            item = results[policy]
            print(f"[eval] {policy}: raw={item['mean_raw_reward']:.2f}, "
                  f"complete={item['completed']}/{item['total']}, "
                  f"returns={item['returns']}, dominant={item['dominant_decision_fraction']:.1%}",
                  flush=True)
    return results


def completed_score(result):
    return result['mean_raw_reward'] if result['completed'] == result['total'] else None


def write_observation(run, baselines, latest_step=None):
    files = sorted((run / 'eval').glob('diagnostics_*.json'),
                   key=lambda file: int(file.stem.split('_')[-1]))
    settings = json.loads((run / 'config.json').read_text())
    lines = [f"# RAM + MLP {settings.get('algorithm', 'PPO')} experiment", '',
             f"Target mode: {settings.get('target_mode', 'intercept')}; beta: {settings['beta']}.",
             'Assisted task: auto-serve setting is shared by training, evaluation and video.',
             'Inputs: normalized RAM, geometry/velocity/phase features and last executed action.',
             'Training reward: raw + beta * (gamma * Phi(next) - Phi(previous)).',
             'Evaluation: raw rewards; every episode explicitly resets a fixed seed.', '',
             '| Baseline | Mean raw | Completed | Detected returns |',
             '| --- | --- | --- | --- |']
    for name, item in baselines.items():
        lines.append(f"| {name} | {item['mean_raw_reward']:.2f} | "
                     f"{item['completed']}/{item['total']} | {item['returns']} |")
    lines += ['', '| Steps | Policy | Mean raw | Completed | Stalls | Returns | Dominant action |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for file in files:
        record = json.loads(file.read_text())
        for name, item in record['policies'].items():
            lines.append(f"| {record['timesteps']} | {name} | {item['mean_raw_reward']:.2f} | "
                         f"{item['completed']}/{item['total']} | {item['stalled']} | "
                         f"{item['returns']} | {item['dominant_decision_fraction']:.1%} |")
    lines += ['', 'Dominant action is a diagnostic, not a reward penalty or automatic rejection rule.',
              'Detected return counts are RAM heuristics, not proof of skilled play.',
              'Assisted RAM-task results are not directly comparable to the old image-only task.', '']
    target = run / 'reward_observation.md'
    temporary = target.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines))
    temporary.replace(target)


class RAMEvaluationCallback(BaseCallback):
    def __init__(self, run, settings, baselines, frequency, episodes):
        super().__init__()
        self.run, self.settings, self.baselines = run, settings, baselines
        self.frequency, self.episodes = frequency, episodes
        self.best_score = -np.inf
        self.best_qualified = -np.inf
        constant_scores = [completed_score(baselines[name]) for name in ('downfire', 'upfire', 'fire')]
        constant_scores = [score for score in constant_scores if score is not None]
        self.constant_score = max(constant_scores) if constant_scores else None

    def _on_step(self):
        for info in self.locals['infos']:
            for name, value in info['ram_rewards'].items():
                self.logger.record_mean('rewards/' + name, value)
            self.logger.record_mean('behavior/auto_serve', int(info['auto_serve']))
        if self.n_calls % self.frequency:
            return True
        print(f'\n[eval] transitions={self.num_timesteps}', flush=True)
        policies = run_suite(self.model, self.settings, ('ppo', 'ppo_sampled'),
                             range(10000, 10000 + self.episodes),
                             self.settings['eval_max_steps'], self.settings['eval_idle_limit'])
        score = completed_score(policies['ppo'])
        qualified = score is not None and self.constant_score is not None and score > self.constant_score
        record = dict(timesteps=self.num_timesteps, seeds=list(range(10000, 10000 + self.episodes)),
                      constant_baseline_score=self.constant_score,
                      beats_constant_baseline=qualified, policies=policies)
        write_json(self.run / 'eval' / f'diagnostics_{self.num_timesteps}.json', record)
        if score is not None and score > self.best_score:
            self.best_score = score
            self.model.save(self.run / 'best' / 'best_model')
            write_json(self.run / 'best' / 'evaluation.json', record)
        if qualified and score > self.best_qualified:
            self.best_qualified = score
            self.model.save(self.run / 'qualified' / 'best_model')
            write_json(self.run / 'qualified' / 'evaluation.json', record)
        for name, result in policies.items():
            self.logger.record(f'eval/{name}_raw', result['mean_raw_reward'])
            self.logger.record(f'eval/{name}_completed', result['completed'] / result['total'])
            self.logger.record(f'eval/{name}_dominant', result['dominant_decision_fraction'])
            self.logger.record(f'eval/{name}_dominant_movement', result['dominant_movement_fraction'])
        self.logger.record('eval/beats_constant_baseline', int(qualified))
        write_observation(self.run, self.baselines)
        return True


def train(args):
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    for name in ('eval', 'best', 'qualified', 'monitor', 'checkpoints'):
        (run / name).mkdir()
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings['observation'] = 'ram'
    write_json(run / 'config.json', settings)
    write_json(run / 'versions.json', {name: importlib.metadata.version(name) for name in
               ('gymnasium', 'ale-py', 'stable-baselines3', 'torch')})
    write_json(run / 'observation_schema.json',
               dict(size=OBS_SIZE, ram_bytes=128, features=FEATURE_NAMES,
                    executed_action_one_hot=18, frame_skip=4, target_mode=args.target_mode))
    print('[eval] fixed-seed baselines, same assisted environment', flush=True)
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
        model = PPO('MlpPolicy', env, n_steps=args.n_steps, batch_size=args.batch_size,
                    learning_rate=lambda remaining: 1e-5 + (args.learning_rate - 1e-5) * remaining,
                    n_epochs=4, gamma=args.gamma, gae_lambda=0.95, clip_range=0.1,
                    ent_coef=0.01, target_kl=0.02,
                    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
                    seed=args.seed, device=args.device, verbose=1)
        model.set_logger(configure(str(run), ['stdout', 'csv', 'tensorboard']))
        write_json(run / 'device.json', device_info(args.requested_device, model.device))
        evaluation = RAMEvaluationCallback(
            run, settings, baselines, max(args.eval_freq // args.n_envs, 1), args.eval_episodes)
        checkpoints = CheckpointCallback(
            save_freq=max(args.eval_freq // args.n_envs, 1),
            save_path=str(run / 'checkpoints'), name_prefix='ppo_ram')
        try:
            model.learn(total_timesteps=args.total_timesteps, callback=[evaluation, checkpoints])
        except KeyboardInterrupt:
            model.save(run / 'interrupted_model')
            write_json(run / 'training_status.json',
                       dict(status='interrupted', timesteps=model.num_timesteps))
            raise
        model.save(run / 'final_model')
        write_json(run / 'training_status.json', dict(
            status='complete', timesteps=model.num_timesteps,
            requested_timesteps=args.total_timesteps, train_seconds=time.monotonic() - started))
    finally:
        env.close()
    selected = run / 'best' / 'best_model.zip'
    if selected.exists():
        print('[eval] held-out seeds for selected best checkpoint', flush=True)
        best_model = PPO.load(selected, device=args.device)
        test = run_suite(best_model, settings, (*BASELINES, 'ppo', 'ppo_sampled'),
                         range(20000, 20000 + args.eval_episodes),
                         args.eval_max_steps, args.eval_idle_limit)
        write_json(run / 'heldout_evaluation.json',
                   dict(model=str(selected), seeds=list(range(20000, 20000 + args.eval_episodes)),
                        policies=test))
    print(f'Run complete: {run}', flush=True)


def evaluate(args):
    settings = settings_for(args.model.resolve())
    model = PPO.load(args.model, device=args.device)
    result = run_suite(model, settings, (*BASELINES, 'ppo', 'ppo_sampled'),
                       range(args.seed, args.seed + args.episodes),
                       args.eval_max_steps, args.eval_idle_limit)
    write_json(args.output, dict(model=str(args.model.resolve()), policies=result))


def play(args):
    import imageio.v2 as imageio
    settings = settings_for(args.model.resolve())
    model = PPO.load(args.model, device=args.device)
    env = make_ram_env(auto_serve=settings['auto_serve'], shaping=False,
                       gamma=settings['gamma'], beta=settings['beta'], render_mode='rgb_array',
                       serve_mode=settings.get('serve_mode', 'fire'),
                       target_mode=settings.get('target_mode', 'intercept'))
    args.video.parent.mkdir(parents=True, exist_ok=True)
    log = args.video.with_suffix('.actions.csv')
    try:
        obs, _ = env.reset(seed=args.seed)
        names = env.unwrapped.get_action_meanings()
        with imageio.get_writer(args.video, fps=15, codec='libx264', macro_block_size=2) as video, log.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                'step', 'requested', 'executed', 'auto_serve', 'side', 'raw', 'returns',
                'player_x', 'player_y', 'ball_x', 'ball_y', 'done',
                'ball_direction', 'target_x', 'target_y'])
            writer.writeheader()
            for step in range(args.steps):
                video.append_data(env.render())
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(int(action))
                state = info['tennis_state']
                target = info['target']
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
        model=str(args.model.resolve()), auto_serve=settings['auto_serve'],
        observation='ram', serve_mode=settings.get('serve_mode', 'fire'),
        target_mode=settings.get('target_mode', 'intercept'),
        seed=args.seed, stochastic=args.stochastic, fps=15, steps=args.steps))
    print(f'Video: {args.video.resolve()}\nActions: {log.resolve()}', flush=True)


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--total-timesteps', type=positive, default=250000)
    p.add_argument('--n-envs', type=positive, default=8)
    p.add_argument('--n-steps', type=positive, default=256)
    p.add_argument('--batch-size', type=positive, default=256)
    p.add_argument('--auto-serve', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--serve-mode', choices=['fire', 'align_x_fire'], default='align_x_fire')
    p.add_argument('--target-mode', choices=TARGET_MODES, default='recover_chase')
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--beta', type=float, default=0.05)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--eval-freq', type=positive, default=50000)
    p.add_argument('--eval-episodes', type=positive, default=5)
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
    p.add_argument('--stochastic', action='store_true')
    for child in sub.choices.values():
        child.add_argument('--device', default='cpu')
        child.add_argument('--threads', type=positive, default=1)
        if not child.prog.endswith('play'):
            child.add_argument('--eval-max-steps', type=positive, default=5000)
            child.add_argument('--eval-idle-limit', type=positive, default=150)
    args = parser.parse_args()
    if args.command == 'train':
        rollout = args.n_envs * args.n_steps
        if args.batch_size < 2 or rollout % args.batch_size:
            parser.error('n_envs * n_steps must be divisible by batch_size >= 2')
        if not (0 < args.gamma < 1 and np.isfinite(args.beta) and args.beta >= 0
                and np.isfinite(args.learning_rate) and args.learning_rate >= 1e-5):
            parser.error('Require 0 < gamma < 1, finite beta >= 0, learning_rate >= 1e-5')
        if args.eval_freq % args.n_envs:
            parser.error('eval-freq must be divisible by n-envs')
        if args.run_dir.exists():
            parser.error('run-dir already exists')
    args.requested_device = args.device
    args.device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    set_random_seed(args.seed)
    {'train': train, 'evaluate': evaluate, 'play': play}[args.command](args)


if __name__ == '__main__':
    main()

