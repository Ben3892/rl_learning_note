"""RAM DQN integration: actual replay learning, target updates, timeouts and greedy evaluation."""
from argparse import Namespace
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from dqn_ram import build_model, replay_statistics, selection_score, DQNEvaluation
from ram_env import OBS_SIZE, make_ram_env
from ram_ppo import run_suite


def args(n_envs=2):
    return Namespace(
        device='cpu', seed=1, learning_rate=1e-4, buffer_size=64, learning_starts=8,
        batch_size=8, gamma=.99, train_freq=2, gradient_steps=n_envs,
        target_update_interval=16, exploration_fraction=.5, exploration_final_eps=.05)


class RAMDQNTests(unittest.TestCase):
    def test_natural_horizon_is_not_an_artificial_cutoff(self):
        result = dict(total=2, mean_raw_reward=6.0, episodes=[
            dict(stop_reason='game_over'), dict(stop_reason='environment_time_limit')])
        self.assertEqual(selection_score(result), 6.0)
        callback = DQNEvaluation(Path('/tmp/unused-dqn-test'), {},
                                 {name: result for name in ('downfire', 'upfire', 'fire')}, 1)
        self.assertEqual(callback.constant_score, 6.0)
        for reason in ('step_limit', 'serve_stall'):
            result['episodes'][1]['stop_reason'] = reason
            self.assertIsNone(selection_score(result))

    def test_replay_target_timeout_and_save_load(self):
        torch.set_num_threads(1)
        env = DummyVecEnv([lambda: make_ram_env(
            auto_serve=True, shaping=True, target_mode='recover_chase', max_steps=4)
            for _ in range(2)])
        env.seed(1)
        try:
            model = build_model(args(), env)
            initial_q = [p.detach().clone() for p in model.q_net.parameters()]
            initial_target = [p.detach().clone() for p in model.q_net_target.parameters()]
            model.learn(128)
            self.assertGreater(model._n_updates, 0)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(initial_q, model.q_net.parameters())))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(initial_target, model.q_net_target.parameters())))
            replay = model.replay_buffer
            self.assertTrue(replay.full)
            self.assertEqual(replay_statistics(model)['transitions'], 64)
            self.assertEqual(replay.observations.dtype, np.float32)
            self.assertGreater(float(replay.timeouts.sum()), 0)
            sample = replay.sample(8)
            self.assertEqual(tuple(sample.observations.shape), (8, OBS_SIZE))
            self.assertEqual(float(sample.dones.sum()), 0.0)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp)
                model.save(path / 'model')
                model.save_replay_buffer(path / 'replay.pkl')
                restored = DQN.load(path / 'model.zip', env=env, device='cpu')
                restored.load_replay_buffer(path / 'replay.pkl')
                observation = env.reset()
                np.testing.assert_array_equal(
                    model.predict(observation, deterministic=True)[0],
                    restored.predict(observation, deterministic=True)[0])
                self.assertEqual(replay_statistics(restored)['transitions'], 64)
                updates = restored._n_updates
                restored.learn(8, reset_num_timesteps=False)
                self.assertGreater(restored._n_updates, updates)
        finally:
            env.close()

    def test_greedy_evaluation_never_uses_epsilon_or_changes_rng(self):
        torch.set_num_threads(1)
        env = DummyVecEnv([lambda: make_ram_env(target_mode='recover_chase')])
        try:
            model = build_model(args(1), env)
            model.exploration_rate = 1.0
            settings = dict(auto_serve=True, serve_mode='align_x_fire',
                            target_mode='recover_chase', gamma=.99, beta=.05)
            numpy_before = pickle.dumps(np.random.get_state())
            torch_before = torch.get_rng_state().clone()
            action_rng = pickle.dumps(model.action_space.np_random.bit_generator.state)
            with patch.object(model.action_space, 'sample', side_effect=AssertionError('Unexpected exploration')):
                first = run_suite(model, settings, ['dqn'], [20000], max_steps=40)
                second = run_suite(model, settings, ['dqn'], [20000], max_steps=40)
            self.assertEqual(first, second)
            self.assertEqual(numpy_before, pickle.dumps(np.random.get_state()))
            self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
            self.assertEqual(action_rng, pickle.dumps(model.action_space.np_random.bit_generator.state))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()

