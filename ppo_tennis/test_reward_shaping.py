"""Reward-event regression tests: python -m unittest -v test_reward_shaping."""

from dataclasses import replace
import unittest

from reward_shaping import RewardConfig, RewardSignals, State


def state(**kwargs):
    return replace(State(77, 47, 77, 65, 8, 0, True, False, False, True), **kwargs)


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.config = RewardConfig(idle_grace=2)
        self.signals = RewardSignals(self.config)

    def test_only_confirmed_own_serve_is_penalized_after_grace(self):
        serve = state(waiting=True, own_serve=True, rally=False)
        self.signals.reset(serve)
        rewards = [self.signals.update(serve)['idle_reward'] for _ in range(12)]
        self.assertEqual(sum(rewards[:8]), 0)
        self.assertAlmostEqual(sum(rewards[8:]), -self.config.idle_penalty)
        for other in [replace(serve, own_serve=False), state(),
                      replace(serve, valid=False)]:
            self.signals.reset(other)
            self.assertEqual(sum(self.signals.update(other)['idle_reward']
                                 for _ in range(20)), 0)

    def test_movement_during_wait_does_not_cancel_penalty(self):
        serve = state(waiting=True, own_serve=True, rally=False)
        self.signals.reset(serve)
        for i in range(12):
            last = self.signals.update(replace(serve, player_x=50 + i))
        self.assertLess(last['idle_reward'], 0)

    def test_stationary_player_not_paid_for_ball_movement(self):
        self.signals.reset(state(ball_x=90, ball_y=80))
        r = self.signals.update(state(ball_x=89, ball_y=78))
        self.assertEqual(r['position_reward'], 0)

    def test_better_intercept_positive_worse_negative(self):
        previous = state(player_x=60, ball_x=100, ball_y=80)
        self.signals.reset(previous)
        self.assertGreater(self.signals.update(replace(previous, player_x=61,
                                                       ball_y=78))['position_reward'], 0)
        self.signals.reset(previous)
        self.assertLess(self.signals.update(replace(previous, player_x=59,
                                                    ball_y=78))['position_reward'], 0)

    def test_outgoing_ball_and_arbitrary_movement_not_paid(self):
        self.signals.reset(state(ball_y=80))
        r = self.signals.update(state(player_x=78, ball_y=82))
        self.assertEqual(r['position_reward'], 0)
        self.assertEqual(r['return_reward'], 0)

    def test_positive_position_budget_cannot_be_farmed_by_oscillation(self):
        self.signals.reset(state(player_x=20, ball_x=120, ball_y=100))
        for i in range(300):
            self.signals.update(state(player_x=20 + i % 2 * 10,
                                      ball_x=120, ball_y=100 - i % 2))
        self.assertLessEqual(self.signals.position_paid, self.config.position_cap)
        self.assertGreaterEqual(self.signals.position_total, -self.config.position_cap)

    def feed_return(self, bottom=False, out=False):
        # Approach the player's racket, reverse flight, cross net, then land.
        ys = [65, 63, 61, 63] + list(range(65, 138, 2))
        states = [state(ball_y=y) for y in ys]
        states[-1] = replace(states[-1], height=0)
        if out:
            states = [replace(s, ball_x=20, player_x=20) for s in states]
        if bottom:
            states = [replace(s, side=1, player_y=163, ball_y=210-s.ball_y)
                      for s in states]
        self.signals.reset(states[0])
        return [self.signals.update(s) for s in states[1:]], states[-1]

    def test_return_paid_after_landing_once_for_both_halves(self):
        for bottom in [False, True]:
            self.signals.reset()
            parts, last = self.feed_return(bottom)
            self.assertTrue(all(p['return_reward'] == 0 for p in parts[:-1]))
            self.assertEqual(parts[-1]['return_reward'], self.config.return_bonus)
            self.assertEqual(self.signals.update(last)['return_reward'], 0)

    def test_out_of_court_not_paid(self):
        parts, _ = self.feed_return(out=True)
        self.assertEqual(sum(p['return_reward'] for p in parts), 0)

    def test_serve_is_not_return(self):
        self.signals.reset(state(waiting=True, own_serve=True, rally=False))
        rewards = [self.signals.update(state(ball_y=y,
                    height=0 if y == 137 else 8)) for y in range(55, 138, 2)]
        self.assertEqual(sum(p['return_reward'] for p in rewards), 0)

    def test_score_teleport_and_side_switch_cancel_candidates(self):
        for transition in ['score', 'teleport', 'side', 'termination', 'truncation']:
            self.signals.reset(state())
            self.signals.pending_return = {'age': 1, 'crossed_net': True}
            s = state(ball_y=137, height=0)
            kwargs = {}
            if transition == 'score':
                kwargs['raw_reward'] = -1
            elif transition == 'side':
                s = replace(s, side=1)
            elif transition == 'termination':
                kwargs['terminated'] = True
            elif transition == 'truncation':
                kwargs['truncated'] = True
            # The remaining case is a physically impossible teleport.
            self.assertEqual(self.signals.update(s, **kwargs)['return_reward'], 0)
            self.assertIsNone(self.signals.pending_return)

    def test_ram_side_and_opponent_serve(self):
        ram = [0] * 128
        ram[16], ram[24], ram[25], ram[26], ram[27], ram[55] = 77, 142, 7, 72, 80, 143
        ram[32], ram[52], ram[75] = 255, 1, 255
        self.assertTrue(State.from_ram(ram).own_serve)
        ram[80] = 1
        s = State.from_ram(ram)
        self.assertFalse(s.own_serve)
        self.assertEqual(s.player_x, 85)
        ram[52] = 0
        self.assertTrue(State.from_ram(ram).own_serve)
        ram[75] = 30
        self.assertFalse(State.from_ram(ram).own_serve)
        ram[80] = 3
        self.assertFalse(State.from_ram(ram).valid)


if __name__ == '__main__':
    unittest.main()
