
import os
import unittest

import numpy as np

import settings as s

from . import ladder, metrics, oracle, parity, stats, suites
from .runner import (
    TTA8_ENV,
    WEIGHTS_ENV,
    Board,
    RoundRow,
    arena_digest,
    assert_paired,
    run_match,
    tta8_override,
    weights_override,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))


def row(**kw) -> RoundRow:
    base = dict(round=1, agent='a', code_name='a', seat=0, score=0, coins=0,
                kills=0, suicides=0, crates=0, bombs=0, moves=0, invalid=0,
                steps=0, deaths_by_other=0, survived=1, round_steps=0, think=[])
    base.update(kw)
    return RoundRow(**base)


class StructureTests(unittest.TestCase):

    def test_eval_is_not_an_agent(self):
        self.assertFalse(os.path.exists(os.path.join(EVAL_DIR, 'callbacks.py')),
                         'a callbacks.py here would make the framework treat '
                         '_eval as an agent')

    def test_no_agent_imports_the_harness(self):
        import re
        pattern = re.compile(r'^\s*(?:from|import)\s+[\w.]*_eval\b', re.MULTILINE)
        offenders = []
        agent_root = os.path.dirname(EVAL_DIR)
        for name in sorted(os.listdir(agent_root)):
            path = os.path.join(agent_root, name)
            if not os.path.isdir(path) or name == '_eval':
                continue
            for fn in os.listdir(path):
                if not fn.endswith('.py'):
                    continue
                with open(os.path.join(path, fn), encoding='utf-8', errors='ignore') as fh:
                    if pattern.search(fh.read()):
                        offenders.append(f'{name}/{fn}')
        self.assertEqual(offenders, [], 'agent code must never import the harness')


class OracleTests(unittest.TestCase):

    def test_walks_straight_to_a_single_coin(self):
        field = np.zeros((7, 7), dtype=int)
        field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
        ref = oracle.gbr_play(field, [(1, 4)], (1, 1))
        self.assertEqual(ref['coins'], 1)
        self.assertEqual(ref['steps'], 3)
        self.assertEqual(ref['coin_steps'], [3])

    def test_collects_a_coin_underfoot(self):
        field = np.zeros((7, 7), dtype=int)
        field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
        ref = oracle.gbr_play(field, [(1, 1)], (1, 1))
        self.assertEqual((ref['steps'], ref['coins']), (1, 1))

    def test_stops_when_nothing_is_reachable(self):
        field = np.zeros((7, 7), dtype=int)
        field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
        field[3, :] = -1                       # wall the board in two
        ref = oracle.gbr_play(field, [(5, 5)], (1, 1))
        self.assertEqual(ref['coins'], 0)
        self.assertLess(ref['steps'], 400)

    def test_agrees_with_the_simulator_side_implementation(self):
        from agent_code.lin_agent.simenv import COIN_FREE, SoloWorld, build_arena
        for seed in (0, 1, 2, 3, 4):
            with self.subTest(seed=seed):
                field, coins, starts = build_arena(seed, 0.0, 50, 1)
                world = SoloWorld(crate_density=0.0, coin_count=50)
                world.load_round(field, coins, starts[0])
                sim_coin_steps = []
                while world.running:
                    _phi, info = world.step_action(world.greedy_step())
                    for _ in range(info['coins']):
                        sim_coin_steps.append(world.step)
                ref = oracle.gbr_play(
                    field, [tuple(p) for p in np.argwhere(coins == COIN_FREE)],
                    starts[0])
                self.assertEqual(ref['steps'], world.step)
                self.assertEqual(ref['coins'], world.stats['coins'])
                self.assertEqual(ref['coin_steps'], sim_coin_steps)

    def test_prefix_steps_matches_the_agent_prefix(self):
        ref = {'steps': 10, 'coins': 3, 'coin_steps': [2, 5, 10]}
        self.assertEqual(oracle.prefix_steps(ref, 1), 2.0)
        self.assertEqual(oracle.prefix_steps(ref, 3), 10.0)
        self.assertIsNone(oracle.prefix_steps(ref, 0))
        self.assertIsNone(oracle.prefix_steps(ref, 4))

    def test_crate_boards_have_no_static_reference(self):
        field = np.ones((5, 5), dtype=int)
        board = Board(round=1, field=field, coins=[], starts={'a': (1, 1)})
        self.assertIsNone(oracle.gbr_for_board(board, 'a'))


class StatsTests(unittest.TestCase):

    def test_summary_of_a_known_sample(self):
        out = stats.summary([1, 2, 3, 4])
        self.assertAlmostEqual(out['mu'], 2.5)
        self.assertAlmostEqual(out['sigma'], 1.2909944, places=6)
        self.assertAlmostEqual(out['sem'], 0.6454972, places=6)

    def test_single_sample_claims_no_spread(self):
        self.assertEqual(stats.summary([5.0])['sigma'], 0.0)

    def test_identical_samples_are_not_separated(self):
        d = stats.delta([1, 2, 3], [1, 2, 3])
        self.assertEqual(d['delta'], 0.0)
        self.assertFalse(d['separated'])

    def test_a_large_shift_is_separated(self):
        d = stats.delta([10] * 20, [0] * 20)
        self.assertTrue(d['separated'])

    def test_pairing_finds_what_the_unpaired_test_misses(self):
        base = [float(i) for i in range(20)]
        shifted = [v + 0.5 for v in base]
        self.assertFalse(stats.delta(shifted, base)['separated'])
        self.assertTrue(stats.paired_delta(shifted, base)['separated'])

    def test_paired_requires_equal_lengths(self):
        with self.assertRaises(ValueError):
            stats.paired_delta([1, 2], [1])

    def test_format_says_not_separated_in_words(self):
        text = stats.format_delta('a', 'b', stats.delta([1, 2, 3], [1, 2, 3]))
        self.assertIn('not separated by this many rounds', text)


class MetricTests(unittest.TestCase):

    def test_aggregate_computes_the_documented_columns(self):
        rows = [row(score=3, coins=3, steps=30, round_steps=30, invalid=1),
                row(round=2, score=5, coins=5, steps=50, round_steps=50, invalid=0)]
        out = metrics.aggregate(rows, boards=None)
        self.assertEqual(out['rounds'], 2)
        self.assertAlmostEqual(out['score_mu'], 4.0)
        self.assertAlmostEqual(out['coins'], 4.0)
        self.assertAlmostEqual(out['steps_per_coin'], 10.0)
        self.assertAlmostEqual(out['invalid_per_step'], 1 / 80)
        self.assertIsNone(out['ratio'], 'no boards means no honest denominator')

    def test_zero_coin_rounds_are_censored_not_averaged(self):
        field = np.zeros((7, 7), dtype=int)
        field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
        board = Board(round=1, field=field, coins=[(1, 4, True)], starts={'a': (1, 1)})
        rows = [row(coins=0, steps=40, round_steps=40)]
        out = metrics.aggregate(rows, [board])
        self.assertEqual(out['spc_censored'], 1)
        self.assertIsNone(out['ratio'])

    def test_ratio_is_one_for_reference_play(self):
        field = np.zeros((7, 7), dtype=int)
        field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
        board = Board(round=1, field=field, coins=[(1, 4, True)], starts={'a': (1, 1)})
        out = metrics.aggregate([row(coins=1, steps=3, round_steps=3)], [board])
        self.assertAlmostEqual(out['ratio'], 1.0)

    def test_bombing_columns_are_first_class(self):
        rows = [row(bombs=3, suicides=1, crates=5, steps=40, moves=30),
                row(round=2, bombs=1, crates=2, steps=20, moves=15)]
        out = metrics.aggregate(rows, boards=None)
        self.assertAlmostEqual(out['bombs'], 2.0)
        self.assertAlmostEqual(out['crates'], 3.5)
        self.assertAlmostEqual(out['suicides_per_bomb'], 0.25,
                               msg='the scale-free version of the S1 gate')
        self.assertAlmostEqual(out['idle_steps'], (7 + 4) / 2)

    def test_suicides_per_bomb_is_undefined_without_a_bomb(self):
        out = metrics.aggregate([row(suicides=0, bombs=0, steps=400, moves=0)], None)
        self.assertIsNone(out['suicides_per_bomb'],
                          'an agent that never bombs has no suicide *rate per '
                          'bomb*; reporting 0 there is how the old S1 gate was '
                          'passed by doing nothing')
        self.assertAlmostEqual(out['idle_steps'], 400.0,
                               '')

    def test_new_columns_are_in_the_frozen_order(self):
        for column in ('bombs', 'crates', 'suicides_per_bomb', 'idle_steps'):
            self.assertIn(column, metrics.METRIC_ORDER)

    def test_deaths_by_other_is_separate_from_suicide(self):
        out = metrics.aggregate([row(suicides=1, survived=0),
                                 row(round=2, deaths_by_other=1, survived=0)], None)
        self.assertAlmostEqual(out['suicides'], 0.5)
        self.assertAlmostEqual(out['deaths_by_other'], 0.5)
        self.assertAlmostEqual(out['survival_rate'], 0.0)

    def test_table_renders_missing_values_as_dashes(self):
        table = metrics.format_table({'a': metrics.aggregate([row()], None)})
        self.assertIn('| a |', table)
        self.assertIn('-', table)


class LadderTests(unittest.TestCase):

    def test_stage_names_cover_the_plan(self):
        self.assertEqual(sorted(ladder.STAGES), ['S0', 'S1', 'S2', 'S3', 'S4'])

    def test_s0_is_the_stock_solo_coin_heaven(self):
        st = ladder.STAGES['S0']
        self.assertEqual(st.scenario, 'coin-heaven')
        self.assertEqual(st.opponents, [])
        self.assertEqual(ladder.seats(st, 'lin_agent'), ['lin_agent'])

    def test_injection_is_temporary(self):
        st = ladder.STAGES['S1']
        self.assertNotIn(st.scenario, s.SCENARIOS)
        with ladder.stage_scenarios(st) as scenario:
            self.assertEqual(s.SCENARIOS[scenario]['CRATE_DENSITY'], 0.35)
        self.assertNotIn(st.scenario, s.SCENARIOS)

    def test_injection_is_removed_even_after_an_error(self):
        st = ladder.STAGES['S1']
        with self.assertRaises(RuntimeError):
            with ladder.stage_scenarios(st):
                raise RuntimeError('boom')
        self.assertNotIn(st.scenario, s.SCENARIOS)

    def test_stock_scenarios_are_never_overwritten(self):
        before = dict(s.SCENARIOS['classic'])
        with ladder.stage_scenarios(ladder.STAGES['S3']):
            pass
        self.assertEqual(s.SCENARIOS['classic'], before)


class RunnerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.makedirs('logs', exist_ok=True)
        cls.result = run_match(['coin_collector_agent'], n_rounds=2,
                               scenario='coin-heaven', seed=42)

    def test_one_row_per_agent_per_round(self):
        self.assertEqual(len(self.result.rows), 2)
        self.assertEqual(len(self.result.boards), 2)
        self.assertEqual(self.result.agent_names, ['coin_collector_agent'])

    def test_boards_are_captured_before_play(self):
        board = self.result.boards[0]
        self.assertEqual(len(board.coins), 50)
        self.assertTrue(all(c[2] for c in board.coins), 'S0 coins are all visible')
        self.assertIn('coin_collector_agent', board.starts)

    def test_think_times_are_recorded_per_call(self):
        rowobj = self.result.rows[0]
        self.assertEqual(len(rowobj.think), rowobj.steps)
        self.assertTrue(all(t >= 0 for t in rowobj.think))

    def test_the_reference_agent_lands_near_the_oracle(self):
        out = metrics.aggregate(self.result.rows, self.result.boards)
        self.assertIsNotNone(out['ratio'])
        self.assertLess(out['ratio'], 1.3,
                        'coin_collector_agent is greedy-BFS at S0; if this fails '
                        'the oracle is wrong, not the agent')

    def test_rejects_an_impossible_seat_count(self):
        with self.assertRaises(ValueError):
            run_match([], 1)
        with self.assertRaises(ValueError):
            run_match(['random_agent'] * 5, 1)


class WeightsOverrideTests(unittest.TestCase):

    def test_the_variable_is_set_and_then_restored(self):
        here = os.path.abspath(__file__)
        self.assertNotIn(WEIGHTS_ENV, os.environ)
        with weights_override(here) as path:
            self.assertEqual(os.environ[WEIGHTS_ENV], here)
            self.assertEqual(path, here)
        self.assertNotIn(WEIGHTS_ENV, os.environ)

    def test_no_path_changes_nothing(self):
        with weights_override(None):
            self.assertNotIn(WEIGHTS_ENV, os.environ)
        with weights_override(''):
            self.assertNotIn(WEIGHTS_ENV, os.environ)

    def test_a_missing_checkpoint_fails_loudly(self):
        with self.assertRaises(FileNotFoundError), weights_override('does/not/exist.pt'):
            pass

    def test_it_is_restored_even_after_an_error(self):
        with self.assertRaises(RuntimeError), weights_override(os.path.abspath(__file__)):
            raise RuntimeError('boom')
        self.assertNotIn(WEIGHTS_ENV, os.environ)

    def test_checkpoints_are_listed_in_env_step_order(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('ckpt_2000000.pt', 'ckpt_10000000.pt', 'ckpt_500000.pt',
                         'resume.pt', 'model.pt'):
                open(os.path.join(tmp, name), 'w').close()
            found = suites.checkpoints(tmp)
            self.assertEqual([steps for steps, _ in found],
                             [500_000, 2_000_000, 10_000_000])


class PairedTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.makedirs('logs', exist_ok=True)
        cls.rounds = 4
        cls.seats_a = ['coin_collector_agent', 'peaceful_agent']
        cls.seats_b = ['random_agent', 'peaceful_agent']
        cls.a = run_match(cls.seats_a, cls.rounds, scenario='classic', seed=7,
                          paired=True)
        cls.b = run_match(cls.seats_b, cls.rounds, scenario='classic', seed=7,
                          paired=True)
        cls.unpaired_a = run_match(cls.seats_a, cls.rounds, scenario='classic',
                                   seed=7)
        cls.unpaired_b = run_match(cls.seats_b, cls.rounds, scenario='classic',
                                   seed=7)

    def test_paired_rounds_are_played_on_identical_arenas(self):
        self.assertEqual(len(self.a.arena_hashes), self.rounds)
        self.assertEqual(self.a.arena_hashes, self.b.arena_hashes)
        assert_paired(self.a, self.b)

    def test_unpaired_rounds_share_only_the_first_board(self):
        self.assertEqual(self.unpaired_a.arena_hashes[0],
                         self.unpaired_b.arena_hashes[0])
        self.assertNotEqual(self.unpaired_a.arena_hashes,
                            self.unpaired_b.arena_hashes,
                            'if this ever passes, the engine stopped drawing a '
                            'per-step permutation and --paired can be revisited')
        with self.assertRaises(ValueError):
            assert_paired(self.unpaired_a, self.unpaired_b)

    def test_rows_are_numbered_and_ordered_by_round(self):
        rows = self.a.for_agent('coin_collector_agent')
        self.assertEqual([r.round for r in rows], list(range(1, self.rounds + 1)))
        self.assertEqual(len(self.a.scores('coin_collector_agent')), self.rounds)

    def test_a_length_mismatch_is_refused(self):
        short = run_match(self.seats_a, 2, scenario='classic', seed=7,
                          paired=True)
        with self.assertRaises(ValueError):
            assert_paired(self.a, short)

    def test_the_digest_ignores_who_played(self):
        board = Board(round=1, field=np.zeros((3, 3), dtype=int),
                      coins=[(1, 1, True)], starts={'a': (1, 1)})
        other = Board(round=9, field=np.zeros((3, 3), dtype=int),
                      coins=[(1, 1, True)], starts={'b': (2, 2)})
        self.assertEqual(arena_digest(board), arena_digest(other))


class Tta8ModeTests(unittest.TestCase):

    def test_the_variable_is_set_and_then_restored(self):
        self.assertNotIn(TTA8_ENV, os.environ)
        with tta8_override(True) as on:
            self.assertTrue(on)
            self.assertEqual(os.environ[TTA8_ENV], '1')
        self.assertNotIn(TTA8_ENV, os.environ)

    def test_off_changes_nothing(self):
        with tta8_override(False) as on:
            self.assertFalse(on)
            self.assertNotIn(TTA8_ENV, os.environ)

    def test_it_is_restored_even_after_an_error(self):
        with self.assertRaises(RuntimeError), tta8_override(True):
            raise RuntimeError('boom')
        self.assertNotIn(TTA8_ENV, os.environ)

    def test_a_match_records_the_mode_it_ran_in(self):
        result = run_match(['peaceful_agent'], 1, scenario='classic', seed=3,
                           tta8=True, paired=True)
        self.assertTrue(result.config['tta8'])
        self.assertTrue(result.config['paired'])
        self.assertNotIn(TTA8_ENV, os.environ, 'the mode leaked out of the match')


class RotatedArenaTests(unittest.TestCase):

    def test_every_agent_plays_every_seat_equally_often(self):
        agents = ['coin_collector_agent', 'peaceful_agent', 'random_agent',
                  'rule_based_agent']
        result = suites.arena(agents, n_rounds=4, seed=11, rotate=True)
        self.assertEqual(result.config['rotate'], True)
        self.assertEqual(len(result.boards), 4)
        by_agent = {}
        for row in result.rows:
            by_agent.setdefault(row.agent, []).append(row.seat)
        self.assertEqual(sorted(by_agent), sorted(agents))
        for name, seatlist in by_agent.items():
            self.assertEqual(sorted(seatlist), [0, 1, 2, 3],
                             f'{name} did not play every seat exactly once')

    def test_rounds_are_renumbered_across_the_blocks(self):
        agents = ['peaceful_agent', 'random_agent', 'coin_collector_agent',
                  'rule_based_agent']
        result = suites.arena(agents, n_rounds=4, seed=12, rotate=True)
        self.assertEqual(sorted(b.round for b in result.boards), [1, 2, 3, 4])

    def test_an_unbalanced_rotation_is_refused(self):
        with self.assertRaises(ValueError):
            suites.arena(['peaceful_agent', 'random_agent'], n_rounds=6,
                         rotate=True)


class ParityTests(unittest.TestCase):

    def test_derived_rows_reproduce_the_engine_totals(self):
        result = run_match(['coin_collector_agent', 'peaceful_agent'], n_rounds=2,
                           scenario='coin-heaven', seed=1)
        derived = parity.derive_from_rows(result)
        self.assertEqual(len(derived['by_round']), 2)
        for name, stats_dict in derived['by_agent'].items():
            rows = result.for_agent(name)
            self.assertEqual(stats_dict['score'], sum(r.score for r in rows))
            self.assertEqual(stats_dict['rounds'], len(rows))

    def test_full_parity_against_main_py(self):
        weights = os.path.join(REPO_ROOT, 'agent_code', 'lin_agent', 'weights.npy')
        if not os.path.isfile(weights):
            self.skipTest('needs a deterministic agent: run the training bundle '
                          'so lin_agent has weights.npy')
        verdict = parity.run_parity(['lin_agent'] * 2, n_rounds=2,
                                    scenario='coin-heaven', seed=42)
        self.assertTrue(verdict['match'], verdict['differences'])


if __name__ == '__main__':
    unittest.main()
