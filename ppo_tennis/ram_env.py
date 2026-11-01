"""RAM observations and potential-based shaping for an independent Tennis task."""
from dataclasses import asdict
import math

import ale_py
import gymnasium as gym
import numpy as np
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, FireResetEnv
from stable_baselines3.common.monitor import Monitor

from evaluation import EvaluationLimits
from reward_shaping import RewardConfig, State, TennisRAMTracker, receiving_target

gym.register_envs(ale_py)
FRAME_SKIP = 4
FEATURE_NAMES = (
    'player_x', 'player_y', 'ball_x', 'ball_y', 'ball_height',
    'ball_dx', 'ball_dy', 'side', 'valid', 'waiting', 'own_serve', 'rally',
    'ball_relative_x', 'ball_relative_y', 'target_valid', 'target_x', 'target_y',
    'target_relative_x', 'target_relative_y',
)
OBS_SIZE = 128 + len(FEATURE_NAMES) + 18


def ball_velocity(previous, current, raw_reward=0.0, terminated=False):
    """Agent-step displacement; invalidate point/side/phase changes and teleports."""
    if (previous is None or raw_reward != 0 or terminated
            or current.side != previous.side
            or not (current.valid and previous.valid and current.rally and previous.rally)):
        return 0.0, 0.0
    dx, dy = current.ball_x - previous.ball_x, current.ball_y - previous.ball_y
    return (float(dx), float(dy)) if abs(dx) <= 32 and abs(dy) <= 32 else (0.0, 0.0)


# Empirical player-foot movement ranges for this Tennis ROM.
OWN_Y_BOUNDS = ((47.0, 98.0), (120.0, 187.0))
TARGET_MODES = ('intercept', 'recover_chase')


def flight_phase(state, dy):
    """Screen y increases downwards; the incoming sign reverses on a side switch."""
    if (state.side not in (0, 1) or not state.valid or not state.rally
            or state.waiting or not math.isfinite(dy) or dy == 0):
        return 'unknown'
    toward_player = -1 if state.side == 0 else 1
    return 'incoming' if dy * toward_player > 0 else 'outgoing'


def movement_target(state, dx, dy, target_mode='intercept'):
    if target_mode not in TARGET_MODES:
        raise ValueError(f'Unknown target_mode: {target_mode}')
    phase = flight_phase(state, dy)
    if phase == 'unknown':
        return None
    if target_mode == 'intercept':
        return receiving_target(state, dx / FRAME_SKIP, dy / FRAME_SKIP)
    low_y, high_y = OWN_Y_BOUNDS[state.side]
    if phase == 'outgoing':
        # Return to the center of our own half, never the center/net of the full court.
        return 80.0, (low_y + high_y) / 2
    # Follow the ball's ground/shadow position, not its airborne sprite.
    # While the ball is still on the opponent's side, do not target that half.
    return (float(np.clip(state.ball_x, 20.0, 148.0)),
            float(np.clip(state.ball_y, low_y, high_y)))


def target_and_potential(state, dx, dy, target_mode='intercept'):
    target = movement_target(state, dx, dy, target_mode)
    if target is None:
        return None, 0.0
    distance = math.hypot(state.player_x - target[0], state.player_y - target[1])
    return target, -min(distance / 160.0, 1.0)


def serve_action(state, action_names, mode):
    """A misaligned server must reach the ball horizontally while pressing FIRE."""
    if mode == 'fire':
        return action_names.index('FIRE')
    delta = state.ball_x - state.player_x
    direction = 'LEFT' if delta < -3 else 'RIGHT' if delta > 3 else ''
    return action_names.index(direction + 'FIRE')


def encode_observation(ram, state, dx, dy, target, last_action):
    target_x, target_y = target if target is not None else (state.player_x, state.player_y)
    features = [
        state.player_x / 80 - 1, state.player_y / 105 - 1,
        state.ball_x / 80 - 1, state.ball_y / 105 - 1, state.height / 127.5 - 1,
        dx / 32, dy / 32, 2 * state.side - 1,
        float(state.valid), float(state.waiting), float(state.own_serve), float(state.rally),
        (state.ball_x - state.player_x) / 160, (state.ball_y - state.player_y) / 210,
        float(target is not None), target_x / 80 - 1, target_y / 105 - 1,
        (target_x - state.player_x) / 160, (target_y - state.player_y) / 210,
    ]
    action = np.zeros(18, dtype=np.float32)
    action[last_action] = 1
    return np.clip(np.concatenate([
        np.asarray(ram, dtype=np.float32) / 127.5 - 1,
        np.asarray(features, dtype=np.float32), action,
    ]), -1, 1).astype(np.float32)


class RAMControlEnv(gym.Wrapper):
    """Raw rewards and engineered RAM state; auto-serve changes the task explicitly."""

    def __init__(self, env, tracker, auto_serve=True, serve_mode='align_x_fire',
                 target_mode='intercept'):
        super().__init__(env)
        self.tracker = tracker
        self.auto_serve = auto_serve
        self.serve_mode = serve_mode
        if target_mode not in TARGET_MODES:
            raise ValueError(f'Unknown target_mode: {target_mode}')
        self.target_mode = target_mode
        self.action_names = self.unwrapped.get_action_meanings()
        if len(self.action_names) != 18:
            raise ValueError('This RAM experiment expects the 18-action Tennis ROM.')
        self.fire_action = self.action_names.index('FIRE')
        self.observation_space = gym.spaces.Box(-1, 1, shape=(OBS_SIZE,), dtype=np.float32)

    def observation(self, state, dx=0.0, dy=0.0):
        target, potential = target_and_potential(state, dx, dy, self.target_mode)
        observation = encode_observation(
            self.unwrapped.ale.getRAM(), state, dx, dy, target, self.last_action)
        return observation, target, potential

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self.tracker.signals.reset(self.tracker.state)
        self.last_action = 0
        obs, target, potential = self.observation(self.tracker.state)
        info.update(tennis_state=asdict(self.tracker.state), potential=potential, target=target, ball_direction='unknown')
        return obs, info

    def step(self, action):
        previous = self.tracker.state
        requested = int(action)
        forced = bool(self.auto_serve and previous.valid and previous.own_serve)
        executed = serve_action(previous, self.action_names, self.serve_mode) if forced else requested
        before_returns = self.tracker.signals.totals['return_count']
        _, raw, terminated, truncated, info = self.env.step(executed)
        current = self.tracker.state
        dx, dy = ball_velocity(previous, current, raw, terminated)
        self.last_action = executed
        obs, target, potential = self.observation(current, dx, dy)
        info.update(
            tennis_state=asdict(current), side_before=previous.side,
            potential=potential, target=target, velocity=(dx, dy),
            ball_direction=flight_phase(current, dy),
            requested_action=requested, executed_action=executed, auto_serve=forced,
            return_count=int(self.tracker.signals.totals['return_count'] - before_returns),
            player_moved=bool(previous.side == current.side and (
                previous.player_x != current.player_x or previous.player_y != current.player_y)),
        )
        return obs, raw, terminated, truncated, info


def potential_transition(previous, current, gamma, beta, terminated):
    # A time-limit truncation keeps its potential; SB3 bootstraps terminal_observation.
    return beta * (gamma * (0.0 if terminated else current) - previous)


class PotentialReward(gym.Wrapper):
    """Outside Monitor, so episode.r always reports the unmodified game score."""

    def __init__(self, env, gamma=0.99, beta=0.05, enabled=False):
        super().__init__(env)
        self.gamma, self.beta, self.enabled = gamma, beta, enabled
        self.previous_potential = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_potential = float(info['potential'])
        return obs, info

    def step(self, action):
        obs, raw, terminated, truncated, info = self.env.step(action)
        current = float(info['potential'])
        shaping = potential_transition(
            self.previous_potential, current, self.gamma, self.beta, terminated)
        self.previous_potential = current
        info['ram_rewards'] = dict(raw=float(raw), potential=shaping, shaped=float(raw + shaping))
        return obs, raw + shaping if self.enabled else raw, terminated, truncated, info


def make_ram_env(seed=None, auto_serve=True, shaping=False, gamma=0.99, beta=0.05,
                 max_steps=None, idle_limit=None, monitor_file=None, render_mode=None,
                 serve_mode='align_x_fire', target_mode='intercept'):
    base = gym.make('ALE/Tennis-v5', obs_type='ram', frameskip=1,
                    repeat_action_probability=0.25, max_episode_steps=108000,
                    render_mode=render_mode)
    # Reuse only event diagnostics, never the previous auxiliary rewards.
    tracker = TennisRAMTracker(base, RewardConfig(
        return_bonus=0.0, position_weight=0.0, tracking_penalty=0.0, idle_penalty=0.0))
    env = FireResetEnv(MaxAndSkipEnv(NoopResetEnv(tracker, noop_max=30), skip=FRAME_SKIP))
    env = RAMControlEnv(env, tracker, auto_serve=auto_serve, serve_mode=serve_mode,
                        target_mode=target_mode)
    if max_steps is not None or idle_limit is not None:
        env = EvaluationLimits(env, tracker, max_steps=max_steps, idle_limit=idle_limit)
    env = Monitor(env, filename=monitor_file)
    env = PotentialReward(env, gamma=gamma, beta=beta, enabled=shaping)
    if seed is not None:
        env.action_space.seed(seed)
    return env

