"""Bounded, visible evaluation; stalled episodes cannot select a best model."""

import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class EvaluationLimits(gym.Wrapper):
    """Evaluation only. Stop confirmed serve stalls or an agent-step timeout."""

    def __init__(self, env, tracker, max_steps=None, idle_limit=None):
        super().__init__(env)
        self.tracker = tracker
        self.max_steps = max_steps
        self.idle_limit = idle_limit
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        reason = None
        if terminated:
            reason = 'game_over'
        elif truncated:
            reason = 'environment_time_limit'
        elif (self.idle_limit is not None and self.tracker.state.own_serve
              and self.tracker.signals.wait_frames >= self.idle_limit *
              self.tracker.signals.config.frame_skip):
            truncated, reason = True, 'serve_stall'
        elif self.max_steps is not None and self.steps >= self.max_steps:
            truncated, reason = True, 'step_limit'
        if reason is not None:
            info['evaluation_stop_reason'] = reason
        return obs, reward, terminated, truncated, info


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class VisibleEvalCallback(BaseCallback):
    """Single eval env; rank only batches whose games all ended naturally.

    All raw scores are logged, including cutoffs, so rejecting a checkpoint
    does not hide poor behavior. No auxiliary penalty is added to eval reward.
    """

    def __init__(self, eval_env, best_model_save_path, log_path, eval_freq,
                 n_eval_episodes=1, deterministic=True):
        super().__init__()
        self.eval_env = eval_env
        self.best_path = Path(best_model_save_path)
        self.log_path = Path(log_path)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf
        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.evaluations_length = []
        self.evaluations_truncated = []

    def _init_callback(self):
        if self.eval_env.num_envs != 1:
            raise ValueError('VisibleEvalCallback 使用单个评估环境')
        self.best_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)

    def _record_step(self, locals_, globals_):
        info = locals_['info']
        self.eval_steps += 1
        self.wait_steps += int(info['tennis_state']['own_serve'])
        self.return_count += int(info['reward_components']['return_count'])
        now = time.monotonic()
        if self.eval_steps % 1000 == 0 or now - self.last_print >= 5:
            print(f'[eval] 已执行 {self.eval_steps} 步，已结束 {len(self.episodes)} 局', flush=True)
            self.last_print = now
        if locals_['done'] and 'episode' in info:
            entry = dict(reward=info['episode']['r'], steps=info['episode']['l'],
                         truncated=bool(info.get('TimeLimit.truncated')),
                         stop_reason=info.get('evaluation_stop_reason', 'game_over'))
            self.episodes.append(entry)
            print(f'[eval] 第 {len(self.episodes)} 局：{entry}', flush=True)

    def _on_step(self):
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq:
            return True
        self.eval_steps = self.wait_steps = self.return_count = 0
        self.episodes = []
        start = self.last_print = time.monotonic()
        print(f'\n[eval] 开始：训练步数={self.num_timesteps}，'
              f'评估 {self.n_eval_episodes} 局；训练暂时等待', flush=True)
        rewards, lengths = evaluate_policy(self.model, self.eval_env,
            n_eval_episodes=self.n_eval_episodes, deterministic=self.deterministic,
            return_episode_rewards=True, callback=self._record_step)
        eligible = (len(self.episodes) == self.n_eval_episodes and all(
            not episode['truncated'] and episode['stop_reason'] == 'game_over'
            for episode in self.episodes))
        mean_reward = float(np.mean(rewards))
        summary = dict(timesteps=self.num_timesteps, steps=self.eval_steps,
                       elapsed_seconds=time.monotonic() - start,
                       own_serve_steps=self.wait_steps, returns=self.return_count,
                       mean_raw_reward=mean_reward, eligible_for_best=eligible,
                       episodes=self.episodes)
        self.evaluations_timesteps.append(self.num_timesteps)
        self.evaluations_results.append(rewards)
        self.evaluations_length.append(lengths)
        self.evaluations_truncated.append([e['truncated'] for e in self.episodes])
        np.savez(self.log_path / 'evaluations.npz', timesteps=self.evaluations_timesteps,
                 results=self.evaluations_results, ep_lengths=self.evaluations_length,
                 truncated=self.evaluations_truncated)
        write_json(self.log_path / f'diagnostics_{self.num_timesteps}.json', summary)
        self.logger.record('eval/mean_reward', mean_reward)
        self.logger.record('eval/mean_ep_length', float(np.mean(lengths)))
        self.logger.record('eval/eligible_for_best', int(eligible))
        self.logger.record('eval/truncated_rate', float(np.mean([e['truncated'] for e in self.episodes])))
        self.logger.record('eval/stall_rate', float(np.mean([
            e['stop_reason'] == 'serve_stall' for e in self.episodes])))
        self.logger.record('time/total_timesteps', self.num_timesteps)
        self.logger.dump(self.num_timesteps)
        if eligible and mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(self.best_path / 'best_model')
            write_json(self.best_path / 'evaluation.json', summary)
            print(f'[eval] 保存最佳完整比赛模型：raw_reward={mean_reward:.2f}', flush=True)
        elif not eligible:
            print('[eval] 包含未完成比赛：只记录诊断，不覆盖最佳模型。', flush=True)
        print(f'[eval] 结束：耗时 {summary["elapsed_seconds"]:.1f}s，继续训练\n', flush=True)
        return True
