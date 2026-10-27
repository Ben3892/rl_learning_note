"""Integration regression: replay, target updates, truncations and persistence."""

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from stable_baselines3 import DQN

from dqn_tennis import build_model
from tennis import make_env


class DQNTests(unittest.TestCase):
    def test_replay_training_target_updates_and_reload(self):
        torch.set_num_threads(2)
        env = make_env(seed=1, max_agent_steps=5)
        args = Namespace(device='cpu', seed=1, learning_rate=1e-4, buffer_size=32,
                         learning_starts=8, batch_size=4, train_freq=4, gradient_steps=1,
                         target_update_interval=8, exploration_fraction=0.5,
                         exploration_final_eps=0.05)
        try:
            model = build_model(args, env)
            initial_q = [p.detach().clone() for p in model.q_net.parameters()]
            initial_target = [p.detach().clone() for p in model.q_net_target.parameters()]
            model.learn(total_timesteps=48)
            self.assertGreater(model._n_updates, 0)
            self.assertTrue(any(not torch.equal(a, b) for a, b in
                                zip(initial_q, model.q_net.parameters())))
            self.assertTrue(any(not torch.equal(a, b) for a, b in
                                zip(initial_target, model.q_net_target.parameters())))
            replay = model.replay_buffer
            self.assertEqual(replay.size(), 32)
            self.assertTrue(replay.full)  # FIFO overwrite after exceeding capacity.
            self.assertEqual(replay.observations.dtype, np.uint8)
            self.assertGreater(replay.timeouts.sum(), 0)
            samples = replay.sample(8)
            self.assertEqual(tuple(samples.observations.shape), (8, 4, 84, 84))
            # Time-limit cutoffs are bootstrapped, not treated as real game overs.
            self.assertEqual(float(samples.dones.sum()), 0)
            with tempfile.TemporaryDirectory(prefix='tennis-dqn-test-') as temp:
                path = Path(temp)
                model.save(path / 'model')
                model.save_replay_buffer(path / 'buffer.pkl')
                loaded = DQN.load(path / 'model.zip', env=env, device='cpu')
                loaded.load_replay_buffer(path / 'buffer.pkl')
                self.assertEqual(loaded.replay_buffer.size(), 32)
                obs = env.reset()
                np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                              loaded.predict(obs, deterministic=True)[0])
                old_updates = loaded._n_updates
                loaded.learn(total_timesteps=8, reset_num_timesteps=False)
                self.assertGreater(loaded._n_updates, old_updates)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
