"""Reward-event regression tests: python -m unittest -v test_reward_shaping."""

from dataclasses import replace
import unittest

from reward_shaping import RewardConfig, RewardSignals, State, receiving_target


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

    def test_vertical_movement_rewarded_in_both_halves(self):
        for side, player_y, next_y, ball_y, next_ball_y in [
                (0, 30, 31, 80, 78), (1, 190, 189, 130, 132)]:
            previous = state(side=side, player_y=player_y, ball_y=ball_y)
            self.signals.reset(previous)
            toward = self.signals.update(replace(previous, player_y=next_y,
                                                 ball_y=next_ball_y))
            self.assertGreater(toward['position_reward'], 0)
            self.signals.reset(previous)
            away = self.signals.update(replace(previous, player_y=2*player_y-next_y,
                                               ball_y=next_ball_y))
            self.assertLess(away['position_reward'], 0)

    def test_wall_still_penalized_after_position_budget_exhausted(self):
        previous = state(player_x=20, ball_x=110, ball_y=100)
        self.signals.reset(previous)
        self.signals.position_paid = self.config.position_cap
        self.signals.position_total = -self.config.position_cap
        penalties = []
        for y in range(99, 79, -1):
            parts = self.signals.update(replace(previous, ball_y=y))
            self.assertEqual(parts['position_reward'], 0)
            penalties.append(parts['tracking_reward'])
        self.assertTrue(all(p < 0 for p in penalties))

    def test_stationary_player_in_receiving_zone_not_penalized(self):
        previous = state(player_y=55, ball_y=100)
        self.signals.reset(previous)
        for y in range(99, 79, -1):
            parts = self.signals.update(replace(previous, ball_y=y))
            self.assertEqual(parts['position_reward'], 0)
            self.assertEqual(parts['tracking_reward'], 0)
            self.assertEqual(parts['receiving_frames'], 1)

    def test_small_movement_inside_tolerance_cannot_farm_reward(self):
        previous = state(player_y=55, player_x=80, ball_y=100)
        self.signals.reset(previous)
        parts = self.signals.update(replace(previous, player_x=79, ball_y=99))
        self.assertEqual(parts['position_reward'], 0)
        self.assertEqual(parts['tracking_reward'], 0)

    def test_tracking_only_during_reachable_incoming_rally(self):
        cases = [
            (state(ball_y=80), state(ball_y=82)),  # Outgoing.
            (state(ball_y=50), state(ball_y=48)),  # Already past receiving line.
            (state(rally=False), state(rally=False, ball_y=63)),
            (state(waiting=True, own_serve=False, rally=False),
             state(waiting=True, own_serve=False, rally=False, ball_y=63)),
            (state(valid=False), state(valid=False, ball_y=63)),
        ]
        for previous, current in cases:
            self.signals.reset(previous)
            parts = self.signals.update(current)
            self.assertEqual(parts['tracking_reward'], 0)
            self.assertEqual(parts['receiving_frames'], 0)
            self.assertIsNone(self.signals.target)

    def test_approaching_target_reduces_distance_penalty(self):
        previous = state(player_x=20, ball_x=100, ball_y=80)
        self.signals.reset(previous)
        stationary = self.signals.update(replace(previous, ball_y=78))
        self.signals.reset(previous)
        closer = self.signals.update(replace(previous, player_x=21, ball_y=78))
        self.assertGreater(closer['tracking_reward'], stationary['tracking_reward'])

    def test_tracking_can_be_disabled_without_disabling_progress(self):
        signals = RewardSignals(replace(self.config, tracking_penalty=0))
        previous = state(player_x=20, ball_x=100, ball_y=80)
        signals.reset(previous)
        parts = signals.update(replace(previous, player_x=21, ball_y=78))
        self.assertEqual(parts['tracking_reward'], 0)
        self.assertGreater(parts['position_reward'], 0)

    def test_legacy_config_preserves_horizontal_rewards(self):
        config = RewardConfig.from_saved(dict(position_weight=0.2, position_cap=0.05))
        self.assertEqual(config.position_mode, 'legacy_x')
        signals = RewardSignals(config)
        previous = state(player_x=60, ball_x=100, ball_y=80)
        signals.reset(previous)
        parts = signals.update(replace(previous, player_x=61, ball_y=78))
        self.assertAlmostEqual(parts['position_reward'], 0.2/160)
        self.assertEqual(parts['tracking_reward'], 0)
        saved = dict(position_mode='intercept_2d', tracking_penalty=0.02, tracking_radius=8)
        self.assertEqual(RewardConfig.from_saved(saved).tracking_penalty, 0.02)

    def test_target_is_fixed_depth_and_within_reachable_court(self):
        s = state(player_y=25, ball_x=150, ball_y=80)
        target = receiving_target(s, 5, -2)
        self.assertEqual(target, receiving_target(replace(s, player_y=40), 5, -2))
        self.assertLess(target[0], 120)
        self.assertEqual(target[1], 55)

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
