"""Evaluation regressions: don't select a stalled policy as the best model."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from stable_baselines3.common.logger import configure

from evaluation import VisibleEvalCallback
from tennis import make_env


class EvaluationTests(unittest.TestCase):
    def test_noop_serve_stall_truncates_with_raw_score(self):
        env = make_env(seed=0, max_agent_steps=300, eval_idle_limit=8)
        try:
            env.reset()
            raw = 0.0
            for step in range(300):
                _, rewards, dones, infos = env.step(np.array([0]))
                raw += float(rewards[0])
                if dones[0]:
                    self.assertEqual(infos[0]['evaluation_stop_reason'], 'serve_stall')
                    self.assertTrue(infos[0]['TimeLimit.truncated'])
                    self.assertEqual(infos[0]['episode']['r'], raw)
                    self.assertLess(step + 1, 300)
                    break
            else:
                self.fail('Confirmed NOOP serve stall did not stop evaluation')
            # VecEnv reset must also clear the per-episode stall timer.
            self.assertFalse(env.step(np.array([0]))[2][0])
        finally:
            env.close()

    def test_step_limit_has_distinct_reason(self):
        env = make_env(seed=0, max_agent_steps=4, eval_idle_limit=150)
        try:
            env.reset()
            for _ in range(4):
                _, _, done, info = env.step(np.array([1]))
            self.assertTrue(done[0])
            self.assertEqual(info[0]['evaluation_stop_reason'], 'step_limit')
        finally:
            env.close()

    def test_stall_cannot_replace_completed_model_even_with_higher_score(self):
        with tempfile.TemporaryDirectory(prefix='tennis-eval-') as temp:
            root = Path(temp)
            model = SimpleNamespace(num_timesteps=1, logger=configure(temp, []), save=Mock())
            callback = VisibleEvalCallback(SimpleNamespace(num_envs=1), root/'best',
                                           root/'eval', eval_freq=1)
            callback.init_callback(model)

            def run(score, reason):
                truncated = reason != 'game_over'

                def evaluate(*args, **kwargs):
                    info = dict(episode=dict(r=score, l=10),
                                tennis_state=dict(own_serve=reason == 'serve_stall'),
                                reward_components=dict(return_count=0),
                                evaluation_stop_reason=reason)
                    info['TimeLimit.truncated'] = truncated
                    kwargs['callback'](dict(info=info, done=True), {})
                    return [score], [10]

                with patch('evaluation.evaluate_policy', side_effect=evaluate):
                    callback.on_step()
                model.num_timesteps += 1

            run(-24, 'game_over')
            self.assertEqual(callback.best_mean_reward, -24)
            self.assertEqual(model.save.call_count, 1)
            run(0, 'serve_stall')
            self.assertEqual(callback.best_mean_reward, -24)
            self.assertEqual(model.save.call_count, 1)
            run(0, 'step_limit')
            self.assertEqual(model.save.call_count, 1)
            run(-20, 'game_over')
            self.assertEqual(callback.best_mean_reward, -20)
            self.assertEqual(model.save.call_count, 2)
            rejected = json.loads((root/'eval/diagnostics_2.json').read_text())
            self.assertFalse(rejected['eligible_for_best'])
            self.assertEqual(rejected['mean_raw_reward'], 0)


if __name__ == '__main__':
    unittest.main()
