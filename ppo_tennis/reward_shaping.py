"""Training-only Tennis shaping, with per-emulator-frame RAM event detection.

Coordinates: https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/tennis.py
No OCAtari dependency or source is copied. Phase bytes 32/52/75 were checked
experimentally on the default ALE Tennis ROM (ale-py 0.11.2); unknown states
fail closed. Events are conservative heuristics, not official ALE hit labels.
"""

from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    return_bonus: float = 0.1
    position_weight: float = 0.1
    position_cap: float = 0.05  # Positive position rewards per point, not per step.
    idle_penalty: float = 0.01  # Per agent step (4 emulator frames), only own serve.
    idle_grace: int = 120  # Agent steps; 8 seconds at 15 decisions/s.
    frame_skip: int = 4
    position_mode: str = 'intercept_2d'
    tracking_penalty: float = 0.01  # Per agent step, scaled by normalized error.
    tracking_radius: float = 12.0  # Pixel tolerance around the receiving position.

    @classmethod
    def from_saved(cls, data):
        # Old experiments used horizontal progress only. Preserve that reward
        # definition when replaying old models instead of silently changing it.
        values = asdict(cls(position_mode='legacy_x', tracking_penalty=0.0))
        values.update({key: data[key] for key in values if key in data})
        config = cls(**values)
        if config.position_mode not in ('legacy_x', 'intercept_2d'):
            raise ValueError(f'未知站位奖励模式：{config.position_mode}')
        return config


@dataclass(frozen=True)
class State:
    player_x: float
    player_y: float  # Feet on the ground plane, not sprite top.
    ball_x: float
    ball_y: float  # Shadow/ground y; avoids mistaking a bounce for a return.
    height: int
    side: int  # 0: orange player at top; 1: at bottom.
    valid: bool
    waiting: bool
    own_serve: bool
    rally: bool

    @classmethod
    def from_ram(cls, ram):
        r = [int(v) for v in ram]
        side = r[80]
        valid_side = side in (0, 1)
        yi, xi = (24, 26) if side == 0 else (25, 27)
        # 255 marks pre-serve; 52 selects the serving half; 75 is a
        # point-transition timer (255 when inactive). Toss height is not zero.
        waiting = valid_side and r[32] == 255 and r[75] == 255
        valid = valid_side and 2 < r[16] < 160 and 0 < r[55] < 189
        return cls(r[xi] + 5, 189 - r[yi], r[16] - 1, 189 - r[55],
                   r[17], side, valid, waiting,
                   waiting and r[52] == 1 - side,
                   valid and r[32] == 0 and r[75] == 255)


def in_court(x, y):
    """Conservative interior of the perspective court (exclude line ambiguity)."""
    left = 40 - (y - 49) * 0.12
    return 50 <= y <= 180 and left + 2 <= x <= 160 - left - 2


def receiving_target(state, dx, dy):
    """Predict an intercept at a fixed receiving depth, not the ball's current xy.

    Conservative linear ground-plane extrapolation; no feedback after the ball
    has passed this line or when the intercept is implausibly far in the future.
    A fixed target depth prevents moving the player from moving the target.
    """
    toward = -1 if state.side == 0 else 1
    if dy * toward <= 0:
        return None
    target_y = 55.0 if state.side == 0 else 165.0
    eta = (target_y - state.ball_y) / dy
    if not 0 <= eta <= 180:
        return None
    left = 40 - (target_y - 49) * 0.12
    target_x = float(np.clip(state.ball_x + dx * eta, left + 5, 155 - left))
    return target_x, target_y


class RewardSignals:
    """Pure state machine, separated from Gym for replayable event tests."""

    def __init__(self, config=None):
        self.config = config or RewardConfig()
        self.reset()

    def reset(self, state=None):
        self.previous = state
        self.last_dy = 0.0
        self.pending_return = None
        self.wait_frames = 0
        self.position_paid = 0.0
        self.position_total = 0.0
        self.target = None
        self.totals = dict(return_reward=0.0, position_reward=0.0, idle_reward=0.0,
                           tracking_reward=0.0, receiving_frames=0.0,
                           return_count=0.0, own_serve_frames=0.0)

    def update(self, state, raw_reward=0.0, terminated=False, truncated=False):
        c, prev = self.config, self.previous
        parts = dict.fromkeys(self.totals, 0.0)
        self.target = None
        # A score, side switch or terminal step must never look like a hit.
        boundary = (raw_reward != 0 or terminated or truncated or prev is None
                    or state.side != prev.side)
        if boundary:
            self.pending_return = None
            self.last_dy = 0.0
            self.wait_frames = 0
            self.position_paid = 0.0
            self.position_total = 0.0
        else:
            if state.own_serve and state.valid:
                self.wait_frames += 1
                parts['own_serve_frames'] = 1.0
                if self.wait_frames > c.idle_grace * c.frame_skip:
                    parts['idle_reward'] = -c.idle_penalty / c.frame_skip
            else:
                self.wait_frames = 0

            if not (state.rally and prev.rally and state.valid and prev.valid):
                self.pending_return = None
                self.last_dy = 0.0
            else:
                dx = state.ball_x - prev.ball_x
                dy = state.ball_y - prev.ball_y
                # Reset/teleport, rather than plausible one-frame flight.
                if abs(dx) > 8 or abs(dy) > 8:
                    self.pending_return = None
                    self.last_dy = 0.0
                else:
                    toward = -1 if state.side == 0 else 1
                    incoming = dy * toward > 0
                    outgoing = dy * toward < 0
                    near_player = any(abs(s.ball_x - s.player_x) <= 20
                                      and abs(s.ball_y - s.player_y) <= 24
                                      for s in (prev, state))
                    if outgoing and self.last_dy * toward > 0 and near_player:
                        self.pending_return = {'age': 0, 'crossed_net': False}

                    if self.pending_return is not None:
                        candidate = self.pending_return
                        candidate['age'] += 1
                        opponent_half = (state.ball_y > 109 if state.side == 0
                                         else state.ball_y < 101)
                        if opponent_half:
                            candidate['crossed_net'] = True
                        # First ground contact in the opposite court. Merely
                        # pressing FIRE, serving or crossing the net is insufficient.
                        landed = (prev.height > state.height and state.height <= 1
                                  and opponent_half and candidate['crossed_net']
                                  and in_court(state.ball_x, state.ball_y))
                        if landed:
                            parts['return_reward'] = c.return_bonus
                            parts['return_count'] = 1.0
                            self.pending_return = None
                        elif candidate['age'] > 180 or incoming:
                            self.pending_return = None

                    # The SAME target is used for old/new positions: moving ball
                    # alone cannot generate positive progress for a stationary player.
                    if incoming and abs(dy) > 0:
                        target = receiving_target(state, dx, dy)
                        gain = None
                        if c.position_mode == 'legacy_x':
                            eta = (state.player_y - state.ball_y) / dy
                            if 0 <= eta <= 180:
                                target_x = float(np.clip(state.ball_x + dx * eta, 25, 135))
                                gain = (abs(prev.player_x - target_x)
                                        - abs(state.player_x - target_x)) / 160
                        elif target is not None:
                            target_x, target_y = target
                            old_error = float(np.hypot(prev.player_x - target_x,
                                                       prev.player_y - target_y))
                            error = float(np.hypot(state.player_x - target_x,
                                                   state.player_y - target_y))
                            old_excess = max(old_error - c.tracking_radius, 0.0)
                            excess = max(error - c.tracking_radius, 0.0)
                            gain = (old_excess - excess) / 160
                            parts['receiving_frames'] = 1.0
                            # Unlike progress, distance feedback persists at a wall
                            # and after the position-reward budget is exhausted.
                            parts['tracking_reward'] = (-c.tracking_penalty / c.frame_skip
                                                        * min(excess / 160, 1.0))
                            self.target = dict(x=target_x, y=target_y, error_pixels=error)
                        if gain is not None:
                            position = c.position_weight * gain
                            if position > 0:
                                position = min(position, max(c.position_cap - self.position_paid, 0))
                                self.position_paid += position
                            else:
                                # Bound negative feedback too; don't let cumulative
                                # positioning feedback outweigh the original score.
                                position = max(position, -c.position_cap - self.position_total)
                            self.position_total += position
                            parts['position_reward'] = position
                    if dy:
                        self.last_dy = dy
        self.previous = state
        for name, value in parts.items():
            self.totals[name] += value
        return parts


class TennisRAMTracker(gym.Wrapper):
    """Observe every raw frame, inside AtariWrapper's frame-skip loop."""

    def __init__(self, env, config):
        super().__init__(env)
        self.signals = RewardSignals(config)
        self.state = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.state = State.from_ram(self.unwrapped.ale.getRAM())
        self.signals.reset(self.state)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.state = State.from_ram(self.unwrapped.ale.getRAM())
        self.signals.update(self.state, reward, terminated, truncated)
        return obs, reward, terminated, truncated, info


class TennisRewardWrapper(gym.Wrapper):
    """Outside Monitor: PPO may receive shaped reward, episode.r stays raw."""

    def __init__(self, env, tracker, enabled=False):
        super().__init__(env)
        self.tracker = tracker
        self.enabled = enabled
        self.episode_shaped = 0.0

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # Atari reset performs no-op/FIRE actions. Clear those events so they
        # cannot leak into the first agent action or receive a serve bonus.
        self.tracker.signals.reset(self.tracker.state)
        self.episode_shaped = 0.0
        return result

    def step(self, action):
        before = self.tracker.signals.totals.copy()
        obs, raw, terminated, truncated, info = self.env.step(action)
        totals = self.tracker.signals.totals
        components = {k: totals[k] - before[k] for k in totals}
        aux = sum(components[k] for k in
                  ('return_reward', 'position_reward', 'idle_reward', 'tracking_reward'))
        shaped = float(raw + aux)
        info['reward_components'] = dict(raw=float(raw), shaped=shaped, auxiliary=aux,
                                         **components)
        info['tennis_state'] = asdict(self.tracker.state)
        info['tennis_target'] = self.tracker.signals.target
        self.episode_shaped += shaped
        if terminated or truncated:
            info['episode_shaped_reward'] = self.episode_shaped
        return obs, shaped if self.enabled else raw, terminated, truncated, info
