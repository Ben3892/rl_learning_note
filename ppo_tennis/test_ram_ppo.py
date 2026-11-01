"""Regression tests for the RAM task, shaping boundaries and evaluation reproducibility."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ram_env import (OBS_SIZE, RAMControlEnv, PotentialReward, ball_velocity, encode_observation,
                     make_ram_env, potential_transition, target_and_potential)
from ram_ppo import completed_score, run_suite, summary
from reward_shaping import State


def state(**changes):
    return replace(State(70, 55, 90, 90, 3, 0, True, False, False, True), **changes)


class OneTransition(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (1,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, terminated=False, truncated=False):
        self.terminated, self.truncated = terminated, truncated

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, np.float32), {'potential': -0.2}

    def step(self, action):
        return (np.zeros(1, np.float32), 2.0, self.terminated, self.truncated,
                {'potential': -0.8})


class RAMTests(unittest.TestCase):
    def test_velocity_cleared_on_point_side_phase_and_teleport(self):
        first = state()
        self.assertEqual(ball_velocity(first, replace(first, ball_y=88)), (0, -2))
        for second, reward in [
            (replace(first, side=1), 0), (replace(first, rally=False), 0),
            (replace(first, ball_x=150), 0), (replace(first, ball_y=88), -1),
        ]:
            self.assertEqual(ball_velocity(first, second, reward), (0, 0))

    def test_observation_contains_side_velocity_and_valid_range(self):
        first = state()
        target, phi = target_and_potential(first, 0, -4)
        self.assertIsNotNone(target)
        self.assertTrue(-1 <= phi <= 0)
        obs = encode_observation(np.arange(128, dtype=np.uint8), first, 0, -4, target, 13)
        self.assertEqual(obs.shape, (OBS_SIZE,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all(np.abs(obs) <= 1))
        opposite = encode_observation(np.arange(128, dtype=np.uint8),
                                      replace(first, side=1), 0, 4, None, 13)
        self.assertNotEqual(obs[128 + 7], opposite[128 + 7])
        self.assertEqual(obs[-18 + 13], 1)

    def test_shaping_telescopes_and_terminal_potential_is_zero(self):
        gamma, beta = .99, .05
        for path in [[-.2, -.8, -.1], [-.2, -.1, -.9]]:
            rewards = [potential_transition(path[i], path[i + 1], gamma, beta, i == 1)
                       for i in range(2)]
            self.assertAlmostEqual(sum(gamma ** i * r for i, r in enumerate(rewards)), .01)

    def test_time_limit_keeps_potential_and_monitor_keeps_raw_reward(self):
        for terminated, truncated in [(True, False), (False, True)]:
            env = PotentialReward(Monitor(OneTransition(terminated, truncated)), enabled=True)
            env.reset()
            _, reward, _, _, info = env.step(0)
            expected = 2 + .05 * (.99 * (0 if terminated else -.8) + .2)
            self.assertAlmostEqual(reward, expected)
            self.assertEqual(info['episode']['r'], 2)
            env.close()

    def test_auto_serve_only_overrides_confirmed_own_serve(self):
        env = make_ram_env(auto_serve=True)
        try:
            env.reset(seed=0)
            tracker = env.get_wrapper_attr('tracker')
            forced = 0
            for _ in range(600):
                before = tracker.state
                _, _, terminated, truncated, info = env.step(0)
                if info['auto_serve']:
                    forced += 1
                    self.assertTrue(before.valid and before.own_serve)
                    self.assertIn(info['executed_action'], (1, 11, 12))
                else:
                    self.assertEqual(info['executed_action'], 0)
                if terminated or truncated:
                    env.reset()
            self.assertGreater(forced, 0)
        finally:
            env.close()

    def test_misaligned_server_needs_horizontal_fire(self):
        # Replay a real failure without depending on trained model weights.
        fixture = json.loads((Path(__file__).parent / 'testdata/ram_misaligned_serve.json').read_text())
        env = make_ram_env(auto_serve=True, serve_mode='fire')
        try:
            env.reset(seed=fixture['seed'])
            control = env
            while not isinstance(control, RAMControlEnv):
                control = control.env
            for action, count in fixture['actions_rle']:
                for _ in range(count):
                    env.step(action)
            self.assertTrue(control.tracker.state.own_serve)
            snapshot = env.unwrapped.ale.cloneSystemState()
            for _ in range(30):
                env.step(0)
            self.assertTrue(control.tracker.state.own_serve)
            env.unwrapped.ale.restoreSystemState(snapshot)
            control.tracker.state = State.from_ram(env.unwrapped.ale.getRAM())
            control.tracker.signals.reset(control.tracker.state)
            control.serve_mode = 'align_x_fire'
            for _ in range(30):
                env.step(0)
                if not control.tracker.state.own_serve:
                    break
            self.assertFalse(control.tracker.state.own_serve)
            self.assertTrue(control.tracker.state.rally)
        finally:
            env.close()

    def test_real_environment_api_and_model_roundtrip(self):
        torch.set_num_threads(1)
        raw = make_ram_env(auto_serve=True, shaping=True)
        try:
            check_env(raw, skip_render_check=True)
        finally:
            raw.close()
        env = DummyVecEnv([lambda: make_ram_env(auto_serve=True, shaping=True)])
        try:
            model = PPO('MlpPolicy', env, n_steps=16, batch_size=16, n_epochs=1, device='cpu')
            model.learn(32)
            with tempfile.TemporaryDirectory() as temp:
                file = Path(temp) / 'model.zip'
                model.save(file)
                restored = PPO.load(file, device='cpu')
                obs = env.reset()
                np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                              restored.predict(obs, deterministic=True)[0])
        finally:
            env.close()

    def test_fire_variants_do_not_hide_constant_movement(self):
        episode = dict(truncated=False, raw_reward=-24, stop_reason='game_over',
                       returns=0, executed_actions={'DOWN': 6, 'DOWNFIRE': 4},
                       decision_actions={'DOWN': 6, 'DOWNFIRE': 4},
                       actions_by_side={'0': {'DOWN': 6}, '1': {'DOWNFIRE': 4}})
        result = summary([episode])
        self.assertEqual(result['dominant_decision_fraction'], .6)
        self.assertEqual(result['dominant_movement_fraction'], 1)

    def test_fixed_seed_evaluation_and_rng_restoration(self):
        settings = dict(auto_serve=True, gamma=.99, beta=.05)
        torch.manual_seed(123)
        before = torch.get_rng_state().clone()
        first = run_suite(None, settings, ['random'], [20000], max_steps=40)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        second = run_suite(None, settings, ['random'], [20000], max_steps=40)
        self.assertEqual(first, second)
        self.assertIsNone(completed_score(first['random']))


if __name__ == '__main__':
    unittest.main()

