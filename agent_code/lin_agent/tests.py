
import logging
import os
import unittest
from types import SimpleNamespace

import numpy as np

import settings as s

from . import (
    callbacks,
    escape_oracle,
    features,
    handcheck,
    selfcheck,
    simenv,
    train_lin,
)
from .config import (
    ACTIONS,
    COLS,
    CONJ_PLACE_SLOT,
    CONJ_SLOT,
    CRATE_BLAST_NORM,
    CRATE_CLAWBACK,
    CRATE_CLAWBACK_MODES,
    CRATE_KAPPA,
    CRATE_KAPPA_LADDER,
    CRATE_REWARD_MODES,
    CRATE_SLOT,
    CURRICULA,
    DANGER_SLOT,
    D_NAV,
    ESCAPE_DIST_DEPTH,
    ESCAPE_POTENTIAL_CAP,
    ESCAPE_POTENTIAL_RHO,
    ESCAPE_REWARD_MODES,
    ESCAPE_SLOT,
    E_GATE_ESCAPE_NORM_FLOOR,
    E_GATE_SUICIDES_PER_BOMB_FLOOR,
    FEATURE_MODES,
    FEATURE_NAMES_FULL,
    FORCED_S2_BLOCK_N,
    FORCED_CEILING_L2,
    GAMMA,
    GATE_CRATES_MIN,
    INJECT_BOMB_TIMER,
    INJECT_MODES,
    MIX_TARGET,
    MIX_WAYPOINTS,
    MODE_DIM,
    MODE_SLOTS,
    N_ACTIONS,
    PLACE_SLOT,
    POST_BOMB_ROOM_DEPTH,
    R_CRATE,
    R_INVALID,
    ROOM_SLOT,
    ROWS,
    S2_CRATE_DENSITY,
    SB_ORACLE_CAP_FINAL,
    agent_path,
    dim_for,
    feature_names,
    load_variant,
)
from .model import LinearQ

KAPPA_MID, KAPPA_TOP = CRATE_KAPPA_LADDER[1], CRATE_KAPPA_LADDER[2]


def empty_board():
    field = np.zeros((COLS, ROWS), dtype=np.int8)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    for x in range(COLS):
        for y in range(ROWS):
            if (x + 1) * (y + 1) % 2 == 1:
                field[x, y] = -1
    return field


def phi_on(field, coin_positions, pos, mode='graded', bombs=(), explosions=(),
           bombs_left=True):
    coins = np.zeros((COLS, ROWS), dtype=np.uint8)
    for (x, y) in coin_positions:
        coins[x, y] = 1
    occupied = np.zeros((COLS, ROWS), dtype=np.uint8)
    for (x, y, _t) in bombs:
        occupied[x, y] = 1
    expl = np.zeros((COLS, ROWS))
    for (x, y, v) in explosions:
        expl[x, y] = v
    bx, by, bt = features.bomb_arrays([(x, y, t) for (x, y, t) in bombs])
    out = np.zeros(dim_for(mode))
    features.write_phi(out, field, coins, occupied, expl, bx, by, bt, len(bt),
                       pos[0], pos[1], bombs_left, features.MODE_ID[mode])
    return out


class FeatureTests(unittest.TestCase):

    def test_layout_is_the_documented_one(self):
        self.assertEqual(MODE_DIM, {'nav': 11, 'inblast1': 12, 'binary': 16,
                                    'graded': 16, 'room': 20, 'crate': 24,
                                    'place': 26, 'conj': 27, 'esc': 27,
                                    'crates_only': 25, 'conjesc': 27})
        self.assertEqual(FEATURE_NAMES_FULL[:11], [
            'blocked_up', 'blocked_right', 'blocked_down', 'blocked_left',
            'coin_up', 'coin_right', 'coin_down', 'coin_left',
            'coin_closeness', 'bomb_available', 'bias'])
        self.assertEqual(FEATURE_NAMES_FULL[DANGER_SLOT], 'danger_here')
        self.assertEqual(FEATURE_NAMES_FULL[ROOM_SLOT], 'room_up')
        self.assertEqual(FEATURE_NAMES_FULL[CRATE_SLOT], 'crate_up')
        self.assertEqual(FEATURE_NAMES_FULL[PLACE_SLOT], 'crates_in_blast')
        self.assertEqual(FEATURE_NAMES_FULL[PLACE_SLOT + 1], 'post_bomb_room')
        self.assertEqual(FEATURE_NAMES_FULL[CONJ_SLOT], 'conj_bomb_here')
        self.assertEqual(FEATURE_NAMES_FULL[ESCAPE_SLOT], 'escape_dist')
        self.assertEqual(len(FEATURE_NAMES_FULL), CONJ_PLACE_SLOT + 1)

    def test_the_feature_table_is_append_only(self):
        self.assertEqual(FEATURE_NAMES_FULL[:20],
                         feature_names('room'), 'twenty slots, unmoved')
        self.assertEqual(feature_names('crate')[:20], feature_names('room'))
        self.assertEqual(FEATURE_MODES[:5],
                         ('nav', 'inblast1', 'binary', 'graded', 'room'),
                         'the mode id is positional and consumed by numba: '
                         'a new encoding is appended, an existing id never moves')
        self.assertEqual(FEATURE_MODES[5], 'crate')
        self.assertEqual(features.MODE_ID['crate'], 5)
        self.assertEqual(feature_names('place')[:24], feature_names('crate'),
                         '')
        self.assertEqual(feature_names('conj')[:26], feature_names('place'))
        self.assertEqual(FEATURE_MODES[6:],
                         ('place', 'conj', 'esc', 'crates_only', 'conjesc'))
        self.assertEqual((features.MODE_ID['place'], features.MODE_ID['conj']),
                         (6, 7), 'the mode id is an ABI and is appended to')
        self.assertEqual((features.MODE_ID['esc'],
                          features.MODE_ID['crates_only']), (8, 9),
                         '')

    def test_l1_slots_kept_their_index_and_meaning(self):
        phi = phi_on(empty_board(), [(1, 4)], (1, 1))
        self.assertEqual(list(phi[:4]), [1.0, 0.0, 0.0, 1.0], 'blocked bits')
        self.assertEqual(list(phi[4:8]), [0.0, 0.0, 1.0, 0.0], 'coin is DOWN')
        self.assertAlmostEqual(phi[8], 1.0 - 3 / 10, msg='coin closeness')

    def test_every_slot_is_in_the_unit_interval(self):
        phi = phi_on(empty_board(), [(5, 5)], (1, 1), mode='room',
                     bombs=[(1, 3, 2)], explosions=[(5, 1, 1.0)])
        self.assertTrue(np.all(phi >= 0.0) and np.all(phi <= 1.0))
        self.assertEqual(phi[10], 1.0, 'the bias moved from slot 9 to slot 10')

    def test_bomb_available_is_the_self_tuple_flag(self):
        self.assertEqual(phi_on(empty_board(), [], (1, 1), bombs_left=True)[9], 1.0)
        self.assertEqual(phi_on(empty_board(), [], (1, 1), bombs_left=False)[9], 0.0)

    def test_unreachable_coin_is_representable(self):
        phi = phi_on(empty_board(), [], (1, 1))
        self.assertEqual(list(phi[4:8]), [0.0] * 4)
        self.assertEqual(phi[8], 0.0)

    def test_bfs_walks_around_the_pillar_lattice(self):
        field = empty_board()
        phi = phi_on(field, [(3, 4)], (2, 3))
        self.assertEqual(sum(phi[4:8]), 1.0)
        direction = int(np.argmax(phi[4:8]))
        self.assertEqual(field[2 + features.DX[direction], 3 + features.DY[direction]], 0)


    def test_danger_is_graded_by_time_to_detonation(self):
        field = empty_board()
        for timer, expected in ((3, 0.4), (2, 0.6), (1, 0.8), (0, 1.0)):
            with self.subTest(timer=timer):
                phi = phi_on(field, [], (1, 1), bombs=[(1, 1, timer)])
                self.assertAlmostEqual(phi[DANGER_SLOT], expected,
                                       msg='(5 - tau)/5, monotone in urgency')

    def test_danger_covers_own_tile_and_four_neighbours(self):
        phi = phi_on(empty_board(), [], (1, 1), bombs=[(1, 3, 3)])
        self.assertAlmostEqual(phi[DANGER_SLOT], 0.4, msg='own tile')
        self.assertAlmostEqual(phi[DANGER_SLOT + 3], 0.4, msg='DOWN, towards the bomb')
        self.assertEqual(phi[DANGER_SLOT + 1], 0.0, 'UP is wall, never in a blast')

    def test_blast_stops_at_walls_but_not_at_crates(self):
        field = empty_board()
        field[1, 3] = 1                       # a crate between bomb and agent
        phi = phi_on(field, [], (1, 4), bombs=[(1, 1, 3)])
        self.assertAlmostEqual(phi[DANGER_SLOT], 0.4,
                               msg='items.Bomb.get_blast_coords passes through crates')
        self.assertEqual(field[2, 2], -1)
        phi = phi_on(field, [], (2, 3), bombs=[(2, 1, 3)])
        self.assertEqual(phi[DANGER_SLOT], 0.0, 'a wall stops the ray')

    def test_active_explosion_is_lethal_now(self):
        phi = phi_on(empty_board(), [], (1, 1), explosions=[(1, 1, 1.0)])
        self.assertAlmostEqual(phi[DANGER_SLOT], 1.0, msg='tau = 0')

    def test_overlapping_blasts_take_the_most_urgent(self):
        phi = phi_on(empty_board(), [], (1, 1), bombs=[(1, 1, 3), (1, 2, 1)])
        self.assertAlmostEqual(phi[DANGER_SLOT], 0.8, msg='the deadline to beat')

    def test_binary_mode_is_the_same_tiles_without_the_timing(self):
        field = empty_board()
        graded = phi_on(field, [], (1, 1), mode='graded', bombs=[(1, 3, 2)])
        binary = phi_on(field, [], (1, 1), mode='binary', bombs=[(1, 3, 2)])
        self.assertEqual(len(graded), len(binary), 'same d, same slots')
        np.testing.assert_allclose(graded[:DANGER_SLOT], binary[:DANGER_SLOT],
                                   err_msg='only slots 11-15 may differ')
        self.assertAlmostEqual(graded[DANGER_SLOT], 0.6)
        self.assertEqual(binary[DANGER_SLOT], 1.0)
        np.testing.assert_allclose((graded[DANGER_SLOT:] > 0).astype(float),
                                   binary[DANGER_SLOT:])

    def test_inblast1_is_one_binary_flag_on_the_current_tile(self):
        phi = phi_on(empty_board(), [], (1, 1), mode='inblast1', bombs=[(1, 3, 2)])
        self.assertEqual(len(phi), 12)
        self.assertEqual(phi[DANGER_SLOT], 1.0)
        safe = phi_on(empty_board(), [], (5, 5), mode='inblast1', bombs=[(1, 3, 2)])
        self.assertEqual(safe[DANGER_SLOT], 0.0)

    def test_nav_mode_has_no_danger_group_at_all(self):
        phi = phi_on(empty_board(), [], (1, 1), mode='nav', bombs=[(1, 1, 0)])
        self.assertEqual(len(phi), D_NAV)


    def test_room_counts_reachable_tiles_and_saturates(self):
        field = empty_board()
        phi = phi_on(field, [], (1, 1), mode='room')
        self.assertEqual(phi[ROOM_SLOT], 0.0, 'UP is the wall ring')
        self.assertGreater(phi[ROOM_SLOT + 1], 0.0, 'RIGHT is open')
        self.assertLessEqual(max(phi[ROOM_SLOT:]), 1.0, 'clipped at 1')

    def test_room_is_zero_for_a_blocked_direction(self):
        field = empty_board()
        field[2, 1] = 1                       # a crate to the RIGHT
        phi = phi_on(field, [], (1, 1), mode='room')
        self.assertEqual(phi[ROOM_SLOT + 1], 0.0)

    def test_room_never_walks_back_through_the_agent(self):
        field = empty_board()
        field[1, 4] = 1
        field[2, 3] = 1
        field[2, 1] = 1
        phi = phi_on(field, [], (1, 1), mode='room')
        self.assertAlmostEqual(phi[ROOM_SLOT + 2], 2 / 9,
                               msg='only (1,2) and (1,3) are reachable that way; '
                                   '(2,2) is a pillar, (1,4) and (2,3) are crates, '
                                   'and the route back through (1,1) is closed')


    def test_crate_slots_say_which_neighbour_is_bombable(self):
        field = empty_board()
        field[2, 1] = 1                       # a crate to the RIGHT
        phi = phi_on(field, [], (1, 1), mode='crate')
        self.assertEqual(list(phi[CRATE_SLOT:CRATE_SLOT + 4]), [0, 1, 0, 0])
        self.assertEqual(phi[1], 1.0, 'and it is blocked, as it always was')
        self.assertEqual(phi[0], 1.0, 'UP is the wall ring: blocked')
        self.assertEqual(phi[CRATE_SLOT], 0.0, 'but not a crate -- that is the split')

    def test_crate_and_blocked_jointly_encode_wall_crate_free(self):
        field = empty_board()
        field[1, 2] = 1                       # crate DOWN; RIGHT is free
        phi = phi_on(field, [], (1, 1), mode='crate')
        wall = (phi[0], phi[CRATE_SLOT])            # UP
        crate = (phi[2], phi[CRATE_SLOT + 2])       # DOWN
        free = (phi[1], phi[CRATE_SLOT + 1])        # RIGHT
        self.assertEqual((wall, crate, free), ((1.0, 0.0), (1.0, 1.0), (0.0, 0.0)),
                         'the three cases are now distinguishable, which is what '
                         'makes "bomb here" expressible as a choice')

    def test_crate_mode_is_room_mode_plus_four_slots(self):
        field = empty_board()
        field[2, 1] = 1
        kw = dict(bombs=[(1, 3, 2)], explosions=[(5, 1, 1.0)])
        room = phi_on(field, [(5, 5)], (1, 1), mode='room', **kw)
        crate = phi_on(field, [(5, 5)], (1, 1), mode='crate', **kw)
        self.assertEqual(len(crate), 24)
        np.testing.assert_allclose(crate[:20], room,
                                   err_msg='appending must not move a slot')

    def test_a_bomb_on_a_neighbour_is_not_a_crate(self):
        phi = phi_on(empty_board(), [], (1, 1), mode='crate', bombs=[(1, 2, 3)])
        self.assertEqual(phi[2], 1.0, 'DOWN is blocked by the bomb')
        self.assertEqual(phi[CRATE_SLOT + 2], 0.0, 'and is not bombable')

    def test_a_dead_end_reads_smaller_than_open_space(self):
        field = empty_board()
        field[1, 2] = 1                       # wall the DOWN corridor off early
        dead_end = phi_on(field, [], (2, 1), mode='room')[ROOM_SLOT + 3]  # LEFT
        open_space = phi_on(empty_board(), [], (5, 5), mode='room')[ROOM_SLOT]
        self.assertLess(dead_end, open_space)


    def test_crates_in_blast_counts_the_blast_and_normalises(self):
        field = empty_board()
        field[2, 1] = field[3, 1] = field[1, 2] = 1
        self.assertEqual(features.crates_in_blast(field, 1, 1), 3)
        phi = phi_on(field, [], (1, 1), mode='place')
        self.assertAlmostEqual(phi[PLACE_SLOT], 3 / CRATE_BLAST_NORM)

    def test_crates_in_blast_saturates_at_the_divisor(self):
        field = empty_board()
        for x in (2, 3, 4):
            field[x, 1] = 1
        for y in (2, 3, 4):
            field[1, y] = 1
        self.assertEqual(features.crates_in_blast(field, 1, 1), 6)
        self.assertEqual(phi_on(field, [], (1, 1), mode='place')[PLACE_SLOT], 1.0,
                         'clipped at 1, like every other slot')

    def test_the_hypothetical_blast_stops_at_walls_and_passes_crates(self):
        field = empty_board()
        field[4, 3] = 1                     # behind the wall at (4,4)
        field[3, 5] = field[2, 5] = 1       # two crates in a row to the LEFT
        self.assertEqual(features.crates_in_blast(field, 4, 5), 2)

    def test_post_bomb_room_counts_only_tiles_outside_the_blast(self):
        field = empty_board()
        occupied = np.zeros((COLS, ROWS), dtype=np.uint8)
        value = features.post_bomb_room(field, occupied, 1, 1)
        self.assertGreater(value, 0.0)
        boxed = empty_board()
        boxed[2, 1] = boxed[1, 2] = 1
        self.assertEqual(features.post_bomb_room(boxed, occupied, 1, 1), 0.0,
                         'no reachable tile outside the blast at all')

    def test_post_bomb_room_is_bounded_by_the_fuse(self):
        field = empty_board()
        occupied = np.zeros((COLS, ROWS), dtype=np.uint8)
        far = empty_board()
        for y in (2, 3, 4, 5):
            far[1, y] = 1                    # block the DOWN ray with crates
        for x in (2, 3, 4):
            far[x, 1] = 0                    # keep the RIGHT corridor open
        reachable = features.post_bomb_room(far, occupied, 1, 1)
        self.assertLessEqual(reachable * 9.0, 4 * 4,
                             f'at most {POST_BOMB_ROOM_DEPTH} steps of walking')
        self.assertGreaterEqual(features.post_bomb_room(field, occupied, 5, 5),
                                reachable,
                                'open space has at least as much room as a corridor')

    def test_place_mode_is_crate_mode_plus_two_slots(self):
        field = empty_board()
        field[2, 1] = 1
        kw = dict(bombs=[(1, 3, 2)], explosions=[(5, 1, 1.0)])
        crate = phi_on(field, [(5, 5)], (1, 1), mode='crate', **kw)
        place = phi_on(field, [(5, 5)], (1, 1), mode='place', **kw)
        self.assertEqual(len(place), 26)
        np.testing.assert_allclose(place[:24], crate,
                                   err_msg='appending must not move a slot')

    def test_the_conjunction_is_the_product_of_its_three_factors(self):
        field = empty_board()
        field[2, 1] = field[3, 1] = 1
        phi = phi_on(field, [], (1, 1), mode='conj')
        self.assertEqual(len(phi), 27)
        expected = phi[9] * phi[PLACE_SLOT] * float(phi[PLACE_SLOT + 1] > 0)
        self.assertAlmostEqual(phi[CONJ_SLOT], expected)
        self.assertGreater(phi[CONJ_SLOT], 0.0)

    def test_the_conjunction_is_zero_when_any_factor_is(self):
        field = empty_board()
        field[2, 1] = field[3, 1] = 1
        no_bomb = phi_on(field, [], (1, 1), mode='conj', bombs_left=False)
        self.assertEqual(no_bomb[CONJ_SLOT], 0.0, 'no bomb available')
        empty = phi_on(empty_board(), [], (1, 1), mode='conj')
        self.assertEqual(empty[CONJ_SLOT], 0.0, 'no crate in range')
        boxed = empty_board()
        boxed[2, 1] = boxed[1, 2] = 1        # crates adjacent, and no way out
        walled = phi_on(boxed, [], (1, 1), mode='conj')
        self.assertGreater(walled[PLACE_SLOT], 0.0)
        self.assertEqual(walled[PLACE_SLOT + 1], 0.0)
        self.assertEqual(walled[CONJ_SLOT], 0.0, 'no exit: the AND is false')


class ComplianceTests(unittest.TestCase):

    def test_room_ignores_bomb_timers(self):
        field = empty_board()
        early = phi_on(field, [], (1, 1), mode='room', bombs=[(1, 3, 3)])
        late = phi_on(field, [], (1, 1), mode='room', bombs=[(1, 3, 0)])
        np.testing.assert_allclose(early[ROOM_SLOT:], late[ROOM_SLOT:],
                                   err_msg='a timer must not move a room slot')
        self.assertNotEqual(early[DANGER_SLOT], late[DANGER_SLOT],
                            'while the danger slots, which may, do move')

    def test_room_ignores_the_explosion_map(self):
        field = empty_board()
        calm = phi_on(field, [], (1, 1), mode='room')
        burning = phi_on(field, [], (1, 1), mode='room',
                         explosions=[(1, 2, 1.0), (1, 3, 1.0), (2, 1, 1.0)])
        np.testing.assert_allclose(calm[ROOM_SLOT:], burning[ROOM_SLOT:],
                                   err_msg='fire must not move a room slot')

    def test_room_treats_a_bomb_as_an_obstacle_only(self):
        field = empty_board()
        free = phi_on(field, [], (1, 1), mode='room')
        blocked = phi_on(field, [], (1, 1), mode='room', bombs=[(1, 2, 3)])
        self.assertEqual(blocked[ROOM_SLOT + 2], 0.0,
                         'DOWN is occupied, so there is no room that way')
        self.assertEqual(free[ROOM_SLOT + 1], blocked[ROOM_SLOT + 1],
                         'and no other direction changed')

    def test_no_feature_names_a_safe_action(self):
        for mode in FEATURE_MODES:
            for name in feature_names(mode):
                self.assertFalse(name.startswith('safe_') or name.endswith('_safe'),
                                 f'{name} reads like a decision, not information')


    def test_crate_slots_are_a_board_fact_and_nothing_else(self):
        field = empty_board()
        field[2, 1] = field[1, 2] = 1
        base = phi_on(field, [(5, 5)], (1, 1), mode='crate')
        loud = phi_on(field, [(5, 5)], (1, 1), mode='crate',
                      bombs=[(1, 3, 0), (5, 1, 2)],
                      explosions=[(1, 1, 1.0), (2, 1, 1.0)], bombs_left=False)
        np.testing.assert_allclose(base[CRATE_SLOT:], loud[CRATE_SLOT:],
                                   err_msg='crate_* reports the arena, exactly as '
                                           'blocked_* does; danger belongs to the '
                                           'danger slots')
        for k, (dx, dy) in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0))):
            self.assertEqual(base[CRATE_SLOT + k],
                             float(field[1 + dx, 1 + dy] == 1))

    def test_the_l3_encodings_still_count_no_crates_in_a_blast(self):
        field = empty_board()
        one = field.copy()
        one[1, 3] = 1
        many = field.copy()
        many[1, 3] = many[1, 4] = many[2, 1] = many[3, 1] = 1
        a = phi_on(one, [], (1, 1), mode='crate')
        b = phi_on(many, [], (1, 1), mode='crate')
        self.assertEqual(list(a[CRATE_SLOT:]), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(list(b[CRATE_SLOT:]), [0.0, 1.0, 0.0, 0.0])


    def test_post_bomb_room_ignores_bomb_timers(self):
        field = empty_board()
        early = phi_on(field, [], (1, 1), mode='place', bombs=[(4, 1, 3)])
        late = phi_on(field, [], (1, 1), mode='place', bombs=[(4, 1, 0)])
        self.assertEqual(early[PLACE_SLOT + 1], late[PLACE_SLOT + 1],
                         'a timer must not move post_bomb_room')
        self.assertNotEqual(early[DANGER_SLOT], late[DANGER_SLOT],
                            'while the danger slots, which may, do move')

    def test_post_bomb_room_ignores_the_explosion_map(self):
        field = empty_board()
        calm = phi_on(field, [], (1, 1), mode='place')
        burning = phi_on(field, [], (1, 1), mode='place',
                         explosions=[(3, 2, 1.0), (2, 3, 1.0), (5, 1, 1.0)])
        self.assertEqual(calm[PLACE_SLOT + 1], burning[PLACE_SLOT + 1],
                         'fire must not move post_bomb_room -- that would make '
                         'it "is it safe to bomb here", i.e. the answer')
        self.assertEqual(calm[PLACE_SLOT], burning[PLACE_SLOT])

    def test_post_bomb_room_treats_a_bomb_as_an_obstacle_only(self):
        field = empty_board()
        free = phi_on(field, [], (1, 1), mode='place')
        blocked = phi_on(field, [], (1, 1), mode='place', bombs=[(2, 1, 3)])
        self.assertLess(blocked[PLACE_SLOT + 1], free[PLACE_SLOT + 1],
                        'the bomb blocks the corridor, exactly as a crate would')
        as_crate = field.copy()
        as_crate[2, 1] = 1
        occupied = np.zeros((COLS, ROWS), dtype=np.uint8)
        bomb_occ = occupied.copy()
        bomb_occ[2, 1] = 1
        self.assertEqual(features.post_bomb_room(field, bomb_occ, 1, 1),
                         features.post_bomb_room(as_crate, occupied, 1, 1),
                         'a bomb is nothing but a tile that cannot be entered; '
                         'its blast geometry is a forbidden input')

    def test_post_bomb_room_is_one_number_for_one_tile(self):
        self.assertEqual(sum(1 for n in feature_names('conj')
                             if n.startswith('post_bomb_room')), 1)
        for suffix in ('_up', '_right', '_down', '_left'):
            self.assertNotIn(f'post_bomb_room{suffix}', feature_names('conj'))

    def test_the_placement_slots_do_not_depend_on_existing_bombs_geometry(self):
        field = empty_board()
        field[2, 1] = 1
        base = phi_on(field, [], (5, 5), mode='place')
        far_bomb = phi_on(field, [], (5, 5), mode='place', bombs=[(9, 9, 1)])
        self.assertEqual(base[PLACE_SLOT], far_bomb[PLACE_SLOT])
        self.assertEqual(base[PLACE_SLOT + 1], far_bomb[PLACE_SLOT + 1],
                         'a bomb four tiles away changes danger, not geometry')


    ESC = features.ESCAPE_OUT_INDEX

    def test_escape_dist_ignores_bomb_timers(self):
        field = empty_board()
        early = phi_on(field, [], (1, 1), mode='esc', bombs=[(4, 1, 3)])
        late = phi_on(field, [], (1, 1), mode='esc', bombs=[(4, 1, 0)])
        self.assertEqual(early[self.ESC], late[self.ESC],
                         'a timer must not move escape_dist')
        self.assertNotEqual(early[DANGER_SLOT], late[DANGER_SLOT],
                            'while the danger slots, which may, do move')

    def test_escape_dist_ignores_the_explosion_map(self):
        field = empty_board()
        calm = phi_on(field, [], (1, 1), mode='esc')
        burning = phi_on(field, [], (1, 1), mode='esc',
                         explosions=[(3, 2, 1.0), (2, 3, 1.0), (5, 1, 1.0)])
        self.assertEqual(calm[self.ESC], burning[self.ESC],
                         'fire must not move escape_dist -- that would make it '
                         '"how long until I am safe", i.e. the answer')

    def test_escape_dist_treats_a_bomb_as_an_obstacle_only(self):
        field = empty_board()
        as_crate = field.copy()
        as_crate[2, 1] = 1
        occupied = np.zeros((COLS, ROWS), dtype=np.uint8)
        bomb_occ = occupied.copy()
        bomb_occ[2, 1] = 1
        self.assertEqual(features.escape_dist(field, bomb_occ, 1, 1),
                         features.escape_dist(as_crate, occupied, 1, 1),
                         'a bomb is nothing but a tile that cannot be entered; '
                         'its blast geometry is a forbidden input')

    def test_escape_dist_is_one_number_for_one_tile(self):
        self.assertEqual(sum(1 for n in feature_names('esc')
                             if n.startswith('escape_dist')), 1)
        for suffix in ('_up', '_right', '_down', '_left'):
            self.assertNotIn(f'escape_dist{suffix}', feature_names('esc'))

    def test_escape_dist_does_not_depend_on_an_existing_bombs_geometry(self):
        field = empty_board()
        field[2, 1] = 1
        base = phi_on(field, [], (5, 5), mode='esc')
        far_bomb = phi_on(field, [], (5, 5), mode='esc', bombs=[(9, 9, 1)])
        self.assertEqual(base[self.ESC], far_bomb[self.ESC])


class EscapeFeatureTests(unittest.TestCase):

    ESC = features.ESCAPE_OUT_INDEX

    def test_the_slot_id_and_the_vector_index_are_the_stated_ones(self):
        self.assertEqual(ESCAPE_SLOT, 27)
        self.assertEqual(MODE_SLOTS['esc'][-1], ESCAPE_SLOT)
        self.assertEqual(MODE_SLOTS['esc'].index(ESCAPE_SLOT), self.ESC)
        self.assertEqual(feature_names('esc')[self.ESC], 'escape_dist')
        self.assertNotIn('conj_bomb_here', feature_names('esc'),
                         'esc skips slot 26; the slot ids do not move for it')

    def test_the_value_is_one_minus_the_capped_step_count_over_the_fuse(self):
        field = empty_board()
        occupied = np.zeros((COLS, ROWS), dtype=np.uint8)
        self.assertAlmostEqual(features.escape_dist(field, occupied, 1, 1),
                              1.0 - 3 / ESCAPE_DIST_DEPTH)
        near = field.copy()
        self.assertAlmostEqual(features.escape_dist(near, occupied, 2, 1),
                               1.0 - 2 / ESCAPE_DIST_DEPTH)

    def test_a_sealed_pocket_reads_zero(self):
        field = empty_board()
        field[1, 2] = 1        # crate below
        field[2, 1] = 1        # crate to the right
        occupied = np.zeros((COLS, ROWS), dtype=np.uint8)
        self.assertEqual(features.escape_dist(field, occupied, 1, 1), 0.0,
                         'walled in: no tile outside the blast is reachable '
                         'inside the fuse, which is exactly 0')

    def test_it_says_something_post_bomb_room_does_not(self):
        field = empty_board()
        occupied = np.zeros((COLS, ROWS), dtype=np.uint8)
        room_far = features.post_bomb_room(field, occupied, 1, 1)
        esc_far = features.escape_dist(field, occupied, 1, 1)
        near = field.copy()
        near[1, 4] = 1          # cap the corridor so the count drops...
        room_near = features.post_bomb_room(near, occupied, 2, 1)
        esc_near = features.escape_dist(near, occupied, 2, 1)
        self.assertGreater(esc_near, esc_far,
                           'the nearer exit reads higher on escape_dist')
        self.assertGreaterEqual(room_near, room_far)
        self.assertNotAlmostEqual(esc_near, esc_far,
                                  msg='the two features are not collinear on '
                                      'these boards, which is the premise of '
                                      'appending one to the other')

    def test_it_is_in_the_unit_interval_on_real_s2_boards(self):
        world = simenv.SoloWorld(crate_density=S2_CRATE_DENSITY, coin_count=50,
                                 feature_mode='esc')
        for seed in range(5):
            phi = world.reset(seed)
            self.assertTrue(0.0 <= phi[self.ESC] <= 1.0)
            for _ in range(20):
                if not world.running:
                    break
                phi, _info = world.step_action(4)     # WAIT
                self.assertTrue(0.0 <= phi[self.ESC] <= 1.0)


class ConjPlaceComplianceTests(unittest.TestCase):

    CP = features.CONJ_PLACE_OUT_INDEX

    @staticmethod
    def _bombable():
        field = empty_board()
        field[3, 2] = 1
        field[1, 2] = 1
        return field

    def test_the_construction_is_the_product_of_its_three_factors(self):
        field = self._bombable()
        phi = phi_on(field, [], (2, 2), mode='conjesc')
        esc = phi_on(field, [], (2, 2), mode='esc')
        self.assertGreater(phi[self.CP], 0.0, 'the anchor board must fire')
        self.assertAlmostEqual(phi[self.CP],
                               phi[9] * phi[PLACE_SLOT] * esc[features.ESCAPE_OUT_INDEX],
                               msg='conj_place IS bomb_available x '
                                   'crates_in_blast x escape_dist, and nothing '
                                   'else')

    def test_conj_place_ignores_bomb_timers(self):
        field = self._bombable()
        early = phi_on(field, [], (2, 2), mode='conjesc', bombs=[(2, 2, 3)])
        late = phi_on(field, [], (2, 2), mode='conjesc', bombs=[(2, 2, 0)])
        self.assertEqual(early[self.CP], late[self.CP],
                         'a timer must not move conj_place')
        self.assertNotEqual(early[DANGER_SLOT], late[DANGER_SLOT],
                            'while the danger slots, which may, do move')

    def test_conj_place_ignores_the_explosion_map(self):
        field = self._bombable()
        calm = phi_on(field, [], (2, 2), mode='conjesc')
        burning = phi_on(field, [], (2, 2), mode='conjesc',
                         explosions=[(2, 3, 1.0), (3, 3, 1.0), (5, 2, 1.0)])
        self.assertEqual(calm[self.CP], burning[self.CP],
                         'fire must not move conj_place -- that would make it '
                         '"is it safe to bomb here", i.e. the answer')

    def test_conj_place_treats_a_bomb_as_an_obstacle_only(self):
        field = self._bombable()
        occupied = np.zeros((COLS, ROWS), dtype=np.uint8)
        bomb_occ = occupied.copy()
        bomb_occ[2, 3] = 1
        as_crate = field.copy()
        as_crate[2, 3] = 1
        self.assertEqual(features.escape_dist(field, bomb_occ, 2, 2),
                         features.escape_dist(as_crate, occupied, 2, 2),
                         'a bomb is nothing but a tile that cannot be entered')
        phi = phi_on(field, [], (2, 2), mode='conjesc', bombs=[(2, 3, 3)])
        crates = min(features.crates_in_blast(field, 2, 2) / CRATE_BLAST_NORM,
                     1.0)
        self.assertAlmostEqual(
            phi[self.CP],
            1.0 * crates * features.escape_dist(field, bomb_occ, 2, 2),
            msg='the bomb enters through the escape BFS\'s occupancy mask and '
                'through nothing else: no timer, no explosion map, no blast '
                'geometry of the existing bomb')

    def test_conj_place_is_one_number_for_one_tile(self):
        names = feature_names('conjesc')
        self.assertEqual(sum(1 for n in names if n.startswith('conj_place')), 1)
        for suffix in ('_up', '_right', '_down', '_left'):
            self.assertNotIn(f'conj_place{suffix}', names)
        self.assertEqual(names[self.CP], 'conj_place')

    def test_conj_place_does_not_depend_on_an_existing_bombs_geometry(self):
        field = self._bombable()
        base = phi_on(field, [], (2, 2), mode='conjesc')
        far_bomb = phi_on(field, [], (2, 2), mode='conjesc', bombs=[(9, 9, 1)])
        self.assertEqual(base[self.CP], far_bomb[self.CP],
                         'a bomb four tiles away changes danger, not geometry')

    def test_it_is_zero_whenever_a_bomb_is_not_available(self):
        field = self._bombable()
        self.assertEqual(phi_on(field, [], (2, 2), mode='conjesc',
                                bombs_left=False)[self.CP], 0.0)

    def test_it_is_in_the_unit_interval_on_real_s2_boards(self):
        world = simenv.SoloWorld(crate_density=S2_CRATE_DENSITY, coin_count=50,
                                 feature_mode='conjesc')
        for seed in range(5):
            phi = world.reset(seed)
            self.assertTrue(0.0 <= phi[self.CP] <= 1.0)
            for _ in range(20):
                if not world.running:
                    break
                phi, _info = world.step_action(4)     # WAIT
                self.assertTrue(0.0 <= phi[self.CP] <= 1.0)


class ConjPlaceMechanismTests(unittest.TestCase):

    CP = features.CONJ_PLACE_OUT_INDEX

    def test_escape_dist_never_returns_as_its_own_slot(self):
        self.assertNotIn(ESCAPE_SLOT, MODE_SLOTS['conjesc'])
        self.assertNotIn('escape_dist', feature_names('conjesc'))
        self.assertEqual(MODE_SLOTS['conjesc'],
                         tuple(range(26)) + (CONJ_PLACE_SLOT,))
        self.assertEqual(CONJ_PLACE_SLOT, 28)

    def test_the_slot_is_silent_wherever_there_is_nothing_to_bomb(self):
        field = empty_board()          # no crates anywhere
        world = simenv.SoloWorld(crate_density=0.0, coin_count=50,
                                 feature_mode='conjesc')
        phi = world.reset(3)
        self.assertEqual(phi[self.CP], 0.0,
                         '')
        self.assertEqual(phi_on(field, [(5, 5)], (1, 1), mode='conjesc')[self.CP],
                         0.0)

    def test_it_is_graded_and_not_binary_like_l4s_conjunction(self):
        near = empty_board()
        near[3, 2] = 1
        near[1, 2] = 1
        far = empty_board()
        far[1, 2] = 1
        far[3, 2] = 1
        far[2, 5] = 1
        values = {pos: phi_on(near, [], pos, mode='conjesc')[self.CP]
                  for pos in ((2, 2), (2, 4))}
        self.assertNotEqual(len(set(values.values())), 0)
        self.assertTrue(any(0.0 < v < 1.0 for v in values.values()),
                        f'the slot must take intermediate values: {values}')

    def test_the_three_way_d27_collision_is_only_resolved_by_variant_json(self):
        self.assertEqual({MODE_DIM['conj'], MODE_DIM['esc'],
                          MODE_DIM['conjesc']}, {27})
        self.assertNotEqual(MODE_SLOTS['conj'], MODE_SLOTS['esc'])
        self.assertNotEqual(MODE_SLOTS['esc'], MODE_SLOTS['conjesc'])
        self.assertNotEqual(MODE_SLOTS['conj'], MODE_SLOTS['conjesc'])
        self.assertEqual({feature_names('conj')[26],
                          feature_names('esc')[26],
                          feature_names('conjesc')[26]},
                         {'conj_bomb_here', 'escape_dist', 'conj_place'})

    def test_the_vector_index_is_the_one_the_kernel_writes(self):
        self.assertEqual(features.CONJ_PLACE_OUT_INDEX, PLACE_SLOT + 2)
        self.assertEqual(MODE_SLOTS['conjesc'].index(CONJ_PLACE_SLOT),
                         features.CONJ_PLACE_OUT_INDEX)
        self.assertEqual(features.MODE_ID['conjesc'], 10)


class ModeSlotsRegressionTests(unittest.TestCase):

    PRE_EXISTING = {
        'nav': tuple(range(11)),
        'inblast1': tuple(range(12)),
        'binary': tuple(range(16)),
        'graded': tuple(range(16)),
        'room': tuple(range(20)),
        'crate': tuple(range(24)),
        'place': tuple(range(26)),
        'conj': tuple(range(27)),
        'esc': tuple(range(26)) + (27,),
        'crates_only': tuple(range(25)),
    }

    def test_every_pre_existing_mode_kept_its_slot_list(self):
        for mode, slots in self.PRE_EXISTING.items():
            self.assertEqual(MODE_SLOTS[mode], slots, mode)
            self.assertEqual(MODE_DIM[mode], len(slots), mode)

    def test_the_new_mode_is_the_one_the_plan_states(self):
        self.assertEqual(MODE_SLOTS['conjesc'], tuple(range(26)) + (28,))
        self.assertEqual(MODE_DIM['conjesc'], 27)
        self.assertEqual(len(FEATURE_MODES), 11)

    def test_every_mode_id_kept_its_position(self):
        self.assertEqual(FEATURE_MODES[:10],
                         ('nav', 'inblast1', 'binary', 'graded', 'room',
                          'crate', 'place', 'conj', 'esc', 'crates_only'))

    def test_every_width_the_chain_has_ever_shipped_still_loads(self):
        for mode in FEATURE_MODES:
            d = dim_for(mode)
            self.assertTrue(11 <= d <= 27, (mode, d))
            w = LinearQ(np.zeros((N_ACTIONS, d)), dim=d)
            self.assertEqual(w.w.shape, (N_ACTIONS, d))
            self.assertEqual(len(feature_names(mode)), d)

    def test_the_installed_pairs_still_match_their_variant(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        import json
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name, 'weights.npy')
            cfg = os.path.join(root, name, 'variant.json')
            if not (os.path.isfile(path) and os.path.isfile(cfg)):
                continue
            with open(cfg) as fh:
                mode = json.load(fh).get('features')
            if mode not in FEATURE_MODES:
                continue
            self.assertEqual(np.load(path).shape,
                             (N_ACTIONS, dim_for(mode)),
                             f'{name}: weights.npy and variant.json are '
                             f'installed as a pair and d = 27 is ambiguous '
                             f'between conj and esc')


class ArenaParityTests(unittest.TestCase):

    def test_matches_the_framework_for_several_seeds(self):
        with selfcheck.injected_scenarios():
            for seed in (0, 3, 42):
                for scenario in ('coin-heaven', 'classic', 'ladder-s1'):
                    with self.subTest(seed=seed, scenario=scenario):
                        world = selfcheck.make_world(seed, scenario)
                        world.new_round()
                        cfg = s.SCENARIOS[scenario]
                        field, coins, starts = simenv.build_arena(
                            seed, cfg['CRATE_DENSITY'], cfg['COIN_COUNT'], 1)
                        np.testing.assert_array_equal(np.asarray(world.arena),
                                                      field.astype(int))
                        np.testing.assert_array_equal(selfcheck.coin_array(world),
                                                      coins)
                        agent = world.agents[0]
                        self.assertEqual((agent.x, agent.y), starts[0])

    def test_s1_starts_with_no_collectable_coin(self):
        for seed in (0, 1, 2, 3, 4):
            _field, coins, _starts = simenv.build_arena(seed, 0.35, 9, 1)
            self.assertEqual(int((coins == simenv.COIN_FREE).sum()), 0,
                             'at density 0.35 every coin starts inside a crate')


class SelfCheckTests(unittest.TestCase):

    def test_s1_rounds_with_a_bombing_agent_match(self):
        os.makedirs('logs', exist_ok=True)
        with selfcheck.injected_scenarios():
            step, problems, coverage = selfcheck.run_round(
                0, 'ladder-s1', 'rule_based_agent', verbose=False)
        self.assertEqual(problems, [], f'diverged at step {step}')
        self.assertGreater(coverage['bombs'], 0, 'the check must exercise bombs')
        self.assertGreater(coverage['crates'], 0, 'and crate destruction')
        self.assertGreater(coverage['reveals'], 0, 'and coin reveals')
        self.assertGreater(coverage['explosion_steps'], 0, 'and live fire')

    def test_random_play_on_s1_matches(self):
        os.makedirs('logs', exist_ok=True)
        with selfcheck.injected_scenarios():
            for seed in (0, 1):
                step, problems, _cov = selfcheck.run_round(
                    seed, 'ladder-s1', 'random_agent', verbose=False)
                self.assertEqual(problems, [], f'diverged at step {step}')

    def test_coin_heaven_rounds_still_match(self):
        os.makedirs('logs', exist_ok=True)
        step, problems, _cov = selfcheck.run_round(0, 'coin-heaven', 'random_agent',
                                                   verbose=False)
        self.assertEqual(problems, [], f'diverged at step {step}')

    def test_the_injection_is_temporary(self):
        self.assertNotIn('ladder-s1', s.SCENARIOS)
        with selfcheck.injected_scenarios():
            self.assertEqual(s.SCENARIOS['ladder-s1']['CRATE_DENSITY'], 0.35)
        self.assertNotIn('ladder-s1', s.SCENARIOS, 'settings.py is never mutated '
                                                   'beyond the block')


class SimEnvTests(unittest.TestCase):

    def _world(self, **kw):
        world = simenv.SoloWorld(crate_density=0.0, coin_count=0, **kw)
        world.load_round(empty_board(), np.zeros((COLS, ROWS), dtype=np.uint8), (1, 1))
        return world

    def test_bomb_kills_its_owner_after_the_right_delay(self):
        world = self._world()
        world.step_action(ACTIONS.index('BOMB'))
        died_at = None
        for _ in range(10):
            if not world.running:
                break
            _phi, info = world.step_action(ACTIONS.index('WAIT'))
            if info['suicide']:
                died_at = world.step
        self.assertEqual(died_at, 5, 'dropped at step 1, detonates at step 5')
        self.assertTrue(world.dead)

    def test_four_steps_are_enough_to_walk_out_of_a_blast(self):
        world = self._world()
        world.step_action(ACTIONS.index('BOMB'))
        for _ in range(4):
            world.step_action(ACTIONS.index('RIGHT'))
        self.assertFalse(world.dead, 'the corner escape is four tiles right')
        self.assertTrue(world.safe_here())

    def test_safe_here_tracks_the_danger_map(self):
        world = self._world()
        self.assertTrue(world.safe_here())
        world.step_action(ACTIONS.index('BOMB'))
        self.assertFalse(world.safe_here(), 'standing on ones own bomb is not safe')

    def test_invalid_action_is_counted_not_moved(self):
        world = self._world()
        _phi, info = world.step_action(ACTIONS.index('UP'))  # into the wall ring
        self.assertTrue(info['invalid'])
        self.assertEqual((world.x, world.y), (1, 1))

    def test_round_ends_when_the_last_coin_is_taken(self):
        coins = np.zeros((COLS, ROWS), dtype=np.uint8)
        coins[1, 2] = simenv.COIN_FREE
        world = simenv.SoloWorld(crate_density=0.0, coin_count=1)
        world.load_round(empty_board(), coins, (1, 1))
        _phi, info = world.step_action(ACTIONS.index('DOWN'))
        self.assertEqual(info['coins'], 1)
        self.assertTrue(info['done'])
        self.assertFalse(world.running)

    def test_greedy_reference_collects_everything_at_s0(self):
        world = simenv.SoloWorld(crate_density=0.0, coin_count=50)
        world.reset(12345)
        while world.running:
            world.step_action(world.greedy_step())
        self.assertEqual(world.stats['coins'], 50)
        self.assertLess(world.step, s.MAX_STEPS)

    def test_idle_steps_are_counted(self):
        coins = np.zeros((COLS, ROWS), dtype=np.uint8)
        coins[9, 9] = simenv.COIN_FREE        # keeps the round running
        world = simenv.SoloWorld(crate_density=0.0, coin_count=1)
        world.load_round(empty_board(), coins, (1, 1))
        for _ in range(3):
            world.step_action(ACTIONS.index('WAIT'))
        world.step_action(ACTIONS.index('UP'))       # invalid
        world.step_action(ACTIONS.index('RIGHT'))    # a move
        stats = train_lin.episode_stats(world)
        self.assertEqual(stats['idle'], 4, 'waits plus rejected actions')


class ModelTests(unittest.TestCase):

    def test_dimension_is_enforced_when_it_is_stated(self):
        with self.assertRaises(ValueError):
            LinearQ(np.zeros((N_ACTIONS, 12)), dim=16)
        self.assertEqual(LinearQ(np.zeros((N_ACTIONS, 12))).dim, 12)

    def test_greedy_breaks_ties_deterministically(self):
        model = LinearQ(dim=16)
        self.assertEqual(model.greedy(np.ones(16)), 0)

    def test_save_load_roundtrip_keeps_the_width(self):
        model = LinearQ(np.arange(N_ACTIONS * 20, dtype=float).reshape(N_ACTIONS, 20))
        path = os.path.join(os.path.dirname(__file__), '_test_weights.npy')
        try:
            model.save(path)
            np.testing.assert_array_equal(LinearQ.load(path, dim=20).w, model.w)
            with self.assertRaises(ValueError):
                LinearQ.load(path, dim=16)
        finally:
            os.remove(path)

    def test_weight_table_renders_every_row(self):
        table = LinearQ(dim=20).table(feature_names('room'), ACTIONS)
        self.assertEqual(len(table.splitlines()), N_ACTIONS + 2)


class LearnerTests(unittest.TestCase):

    def _args(self, **kw):
        args = train_lin.build_parser().parse_args([])
        for k, v in kw.items():
            setattr(args, k, v)
        return args

    def test_one_step_update_moves_towards_the_target(self):
        model = LinearQ(dim=16)
        learner = train_lin.SemiGradientQ(model, alpha=1.0, gamma=0.9, n=1)
        phi = np.zeros(16)
        phi[10] = 1.0
        learner.begin_episode()
        learner.observe(phi, 0, 1.0, phi, True, False)
        learner.end_episode()
        self.assertAlmostEqual(model.w[0, 10], 0.5)

    def test_lstd_solves_a_known_two_state_chain(self):
        model = LinearQ(dim=16)
        learner = train_lin.LSTDQ(model, gamma=0.5, capacity=100, every=1, ridge=1e-9)
        s0 = np.zeros(16); s0[0] = 1.0
        s1 = np.zeros(16); s1[1] = 1.0
        for _ in range(20):
            learner.observe(s0, 0, 0.0, s1, False, False)
            learner.observe(s1, 0, 1.0, s1 * 0, True, False)
        learner.solve()
        self.assertAlmostEqual(float(model.w[0] @ s1), 1.0, places=4)
        self.assertAlmostEqual(float(model.w[0] @ s0), 0.5, places=4)

    def test_lstd_block_dimension_follows_the_feature_dimension(self):
        for dim in (11, 16, 20):
            learner = train_lin.LSTDQ(LinearQ(dim=dim), gamma=0.9)
            self.assertEqual(learner.dim, dim)

    def test_shaping_is_potential_based(self):
        info = {'coins': 0, 'crates': 0, 'invalid': False, 'suicide': False}
        r = train_lin.step_reward(info, prev_d=4, new_d=3, gamma=0.95,
                                  r_invalid=0.0, done=False)
        self.assertAlmostEqual(r, 0.95 * (-0.3) - (-0.4))
        self.assertGreater(r, 0.0, 'getting closer to a coin is rewarded')

    def test_coin_reward_is_the_game_score(self):
        info = {'coins': 1, 'crates': 0, 'invalid': False, 'suicide': False}
        r = train_lin.step_reward(info, prev_d=1, new_d=0, gamma=0.95,
                                  r_invalid=0.0, done=True)
        self.assertAlmostEqual(r, 1.0 + 0.0 - (-0.1))

    def test_death_is_a_terminal_event_reward_not_shaping(self):
        info = {'coins': 0, 'crates': 0, 'invalid': False, 'suicide': True}
        alive = {'coins': 0, 'crates': 0, 'invalid': False, 'suicide': False}
        died = train_lin.step_reward(info, prev_d=5, new_d=10, gamma=0.95,
                                     r_invalid=0.0, done=True, r_death=-1.0)
        survived = train_lin.step_reward(alive, prev_d=5, new_d=10, gamma=0.95,
                                         r_invalid=0.0, done=True, r_death=-1.0)
        self.assertAlmostEqual(died - survived, -1.0)
        harsher = train_lin.step_reward(info, prev_d=5, new_d=10, gamma=0.95,
                                        r_invalid=0.0, done=True, r_death=-5.0)
        self.assertAlmostEqual(harsher - died, -4.0, msg='the lin_d axis')

    def test_crates_are_worth_exactly_nothing_in_the_control_arm(self):
        with_crates = {'coins': 0, 'crates': 3, 'invalid': False, 'suicide': False}
        without = {'coins': 0, 'crates': 0, 'invalid': False, 'suicide': False}
        self.assertEqual(R_CRATE, 0.0)
        self.assertEqual(
            train_lin.step_reward(with_crates, 5, 5, 0.95, 0.0, False),
            train_lin.step_reward(without, 5, 5, 0.95, 0.0, False),
            'lin_a is the control arm and the baseline of every primary claim: '
            'under its configuration a destroyed crate is worth exactly nothing')
        self.assertEqual(train_lin.build_parser().parse_args([]).crate_reward,
                         'none')
        self.assertEqual(train_lin.build_parser().parse_args([]).crate_kappa, 0.0)

    def test_the_event_bonus_pays_kappa_per_crate_and_keeps_it(self):
        info = {'coins': 0, 'crates': 3, 'invalid': False, 'suicide': False}
        none = train_lin.step_reward(info, 5, 5, 0.95, 0.0, False)
        event = train_lin.step_reward(info, 5, 5, 0.95, 0.0, False,
                                      crate_mode='event', kappa=KAPPA_TOP,
                                      destroyed_before=0, destroyed_after=3)
        self.assertAlmostEqual(event - none, 3 * KAPPA_TOP)
        quiet = {'coins': 0, 'crates': 0, 'invalid': False, 'suicide': False}
        after = train_lin.step_reward(quiet, 5, 5, 0.95, 0.0, False,
                                      crate_mode='event', kappa=KAPPA_TOP,
                                      destroyed_before=3, destroyed_after=3)
        self.assertAlmostEqual(after, train_lin.step_reward(quiet, 5, 5, 0.95,
                                                            0.0, False))

    def test_the_potential_form_pays_gamma_kappa_and_claws_it_back(self):
        info = {'coins': 0, 'crates': 3, 'invalid': False, 'suicide': False}
        kw = dict(crate_mode='potential', kappa=KAPPA_TOP)
        at_detonation = train_lin.step_reward(info, 5, 5, 0.95, 0.0, False,
                                              destroyed_before=0,
                                              destroyed_after=3, **kw)
        none = train_lin.step_reward(info, 5, 5, 0.95, 0.0, False)
        self.assertAlmostEqual(at_detonation - none, 0.95 * 3 * KAPPA_TOP,
                               msg='gamma * kappa * Delta at the blast')
        quiet = {'coins': 0, 'crates': 0, 'invalid': False, 'suicide': False}
        later = train_lin.step_reward(quiet, 5, 5, 0.95, 0.0, False,
                                      destroyed_before=3, destroyed_after=3, **kw)
        self.assertAlmostEqual(later - train_lin.step_reward(quiet, 5, 5, 0.95,
                                                             0.0, False),
                               -(1 - 0.95) * 3 * KAPPA_TOP,
                               msg='the living clawback, -(1-gamma) kappa n')
        ending = train_lin.step_reward(quiet, 5, 5, 0.95, 0.0, True,
                                       destroyed_before=3, destroyed_after=3, **kw)
        self.assertAlmostEqual(ending - train_lin.step_reward(quiet, 5, 5, 0.95,
                                                              0.0, True),
                               -3 * KAPPA_TOP,
                               msg='Phi(terminal) = 0: the whole bonus is repaid')

    def test_the_two_forms_agree_at_the_moment_of_the_decision(self):
        info = {'coins': 0, 'crates': 4, 'invalid': False, 'suicide': False}
        event = train_lin.step_reward(info, 5, 5, 0.95, 0.0, False,
                                      crate_mode='event', kappa=0.1,
                                      destroyed_before=0, destroyed_after=4)
        pot = train_lin.step_reward(info, 5, 5, 0.95, 0.0, False,
                                    crate_mode='potential', kappa=0.1,
                                    destroyed_before=0, destroyed_after=4)
        self.assertAlmostEqual(pot - train_lin.step_reward(info, 5, 5, 0.95, 0.0,
                                                           False),
                               0.95 * (event - train_lin.step_reward(
                                   info, 5, 5, 0.95, 0.0, False)))
        fatal = {'coins': 0, 'crates': 4, 'invalid': False, 'suicide': True}
        e = train_lin.step_reward(fatal, 5, 10, 0.95, 0.0, True, r_death=-1.0,
                                  crate_mode='event', kappa=0.1,
                                  destroyed_before=0, destroyed_after=4)
        p = train_lin.step_reward(fatal, 5, 10, 0.95, 0.0, True, r_death=-1.0,
                                  crate_mode='potential', kappa=0.1,
                                  destroyed_before=0, destroyed_after=4)
        self.assertAlmostEqual(e - p, 4 * 0.1)

    def test_an_invalid_action_costs_one_tile_of_coin_distance(self):
        self.assertAlmostEqual(R_INVALID, -0.1, msg='base design')
        wait = {'coins': 0, 'crates': 0, 'invalid': False, 'suicide': False}
        bad = {'coins': 0, 'crates': 0, 'invalid': True, 'suicide': False}
        args = dict(prev_d=5, new_d=5, gamma=0.95, r_invalid=R_INVALID, done=False)
        r_wait = train_lin.step_reward(wait, **args)
        r_bad = train_lin.step_reward(bad, **args)
        self.assertAlmostEqual(r_bad - r_wait, -0.1)
        self.assertLess(r_bad, r_wait, 'strictly worse than waiting')
        coin = {'coins': 1, 'crates': 0, 'invalid': False, 'suicide': False}
        died = {'coins': 0, 'crates': 0, 'invalid': False, 'suicide': True}
        self.assertGreater(r_bad, train_lin.step_reward(died, r_death=-1.0, **args),
                           'and strictly better than dying')
        self.assertLess(r_bad, train_lin.step_reward(coin, **args),
                        'and strictly worse than a coin')

    def test_the_default_invalid_penalty_is_negative_in_the_trainer(self):
        args = train_lin.build_parser().parse_args([])
        self.assertAlmostEqual(args.r_invalid, -0.1)
        self.assertAlmostEqual(
            train_lin.build_parser().parse_args(['--r-invalid', '0']).r_invalid, 0.0,
            'ctrl_rinv0 prices the change from outside, without an edit')

    def test_board_seeds_are_paired_across_variants(self):
        self.assertEqual(train_lin.board_seed(0, 7), train_lin.board_seed(0, 7))
        self.assertNotEqual(train_lin.board_seed(0, 7), train_lin.board_seed(1, 7))

    def test_epsilon_anneals_then_flattens(self):
        self.assertAlmostEqual(train_lin.epsilon(0, 1.0, 0.05, 500), 1.0)
        self.assertAlmostEqual(train_lin.epsilon(250, 1.0, 0.05, 500), 0.525)
        self.assertAlmostEqual(train_lin.epsilon(5000, 1.0, 0.05, 500), 0.05)

    def test_the_three_stage_mixture_interpolates_and_then_holds(self):
        self.assertEqual(train_lin.mixture3(0, MIX_WAYPOINTS), (1.0, 0.0, 0.0))
        p150 = train_lin.mixture3(150, MIX_WAYPOINTS)
        self.assertAlmostEqual(p150[0], 0.575)
        self.assertAlmostEqual(p150[1], 0.425)
        self.assertAlmostEqual(p150[2], 0.0)
        p300 = train_lin.mixture3(300, MIX_WAYPOINTS)
        self.assertAlmostEqual(p300[0], 0.15)
        self.assertAlmostEqual(p300[1], 0.85)
        self.assertAlmostEqual(p300[2], 0.0)
        p550 = train_lin.mixture3(550, MIX_WAYPOINTS)
        self.assertAlmostEqual(p550[0], 0.15)
        self.assertAlmostEqual(p550[1], 0.5)
        self.assertAlmostEqual(p550[2], 0.35)
        for episode in (800, 4000):
            p = train_lin.mixture3(episode, MIX_WAYPOINTS)
            self.assertAlmostEqual(p[0], 0.15)
            self.assertAlmostEqual(p[1], 0.15)
            self.assertAlmostEqual(p[2], 0.70)

    def test_the_settled_lower_stage_share_did_not_move(self):
        last = MIX_WAYPOINTS[-1]
        self.assertAlmostEqual(last[1] + last[2], MIX_TARGET)
        self.assertAlmostEqual(last[1], last[2],
                               'an even split, because nothing says otherwise')

    def test_the_no_curriculum_control_is_pure_s2_from_episode_zero(self):
        for episode in (0, 300, 4000):
            self.assertEqual(train_lin.mixture3(episode, CURRICULA['s2only']),
                             (0.0, 0.0, 1.0))

    def test_one_draw_picks_the_stage(self):
        probs = (0.15, 0.15, 0.70)
        self.assertEqual(train_lin.draw_stage(0.0, probs), 'S0')
        self.assertEqual(train_lin.draw_stage(0.149, probs), 'S0')
        self.assertEqual(train_lin.draw_stage(0.2, probs), 'S1')
        self.assertEqual(train_lin.draw_stage(0.31, probs), 'S2')
        self.assertEqual(train_lin.draw_stage(0.999, probs), 'S2')

    def test_the_stage_stream_is_shared_by_every_variant(self):
        a = np.random.default_rng(77_000 + 1).random(50)
        b = np.random.default_rng(77_000 + 1).random(50)
        np.testing.assert_array_equal(a, b)
        c = np.random.default_rng(77_000 + 2).random(50)
        self.assertFalse(np.array_equal(a, c))


class ShapingInvarianceTests(unittest.TestCase):

    def _episode(self, crate_mode, kappa, seed=7, steps=60,
                 escape_mode='none', rho=0.0):
        world = simenv.SoloWorld(crate_density=S2_CRATE_DENSITY, coin_count=50,
                                 feature_mode='place')
        world.reset(seed)
        rng = np.random.default_rng(seed)
        k = kappa if crate_mode == 'potential' else 0.0
        e = rho if escape_mode == 'potential' else 0.0
        cap = ESCAPE_POTENTIAL_CAP
        prev_safe = world.safe_distance(cap)
        phi0 = train_lin.potential(world.coin_distance(), 0, k, prev_safe, e, cap)
        prev_d = world.coin_distance()
        destroyed = 0
        total_f = 0.0
        discount = 1.0
        for _ in range(steps):
            if not world.running:
                break
            action = int(rng.integers(N_ACTIONS))
            _phi, info = world.step_action(action)
            new_d = 10 if info['done'] else world.coin_distance()
            new_safe = 0 if info['done'] else world.safe_distance(cap)
            after = destroyed + info['crates']
            r = train_lin.step_reward(info, prev_d, new_d, GAMMA, R_INVALID,
                                      info['done'], -1.0, crate_mode=crate_mode,
                                      kappa=kappa, destroyed_before=destroyed,
                                      destroyed_after=after,
                                      escape_mode=escape_mode, rho=rho,
                                      safe_before=prev_safe, safe_after=new_safe,
                                      escape_cap=cap)
            unshaped = (1.0 * info['coins']
                        + (R_INVALID if info['invalid'] else 0.0)
                        + (-1.0 if info['suicide'] else 0.0)
                        + (kappa * info['crates'] if crate_mode == 'event' else 0.0))
            total_f += discount * (r - unshaped)
            discount *= GAMMA
            prev_d, destroyed, prev_safe = new_d, after, new_safe
        return total_f, phi0

    def test_the_potential_form_telescopes_to_minus_phi_of_the_start_state(self):
        for seed in (7, 11, 23, 41):
            with self.subTest(seed=seed):
                total_f, phi0 = self._episode('potential', KAPPA_TOP, seed)
                self.assertAlmostEqual(total_f, -phi0, places=9,
                                       msg='Ng, Harada & Russell 1999, episodic '
                                           'form: the shaping sum depends on '
                                           'nothing but the start state')

    def test_the_control_arm_telescopes_too(self):
        total_f, phi0 = self._episode('none', 0.0, 13)
        self.assertAlmostEqual(total_f, -phi0, places=9)

    def test_the_escape_potential_telescopes_too(self):
        for seed in (7, 11, 23, 41):
            with self.subTest(seed=seed):
                total_f, phi0 = self._episode('potential', KAPPA_TOP, seed,
                                              escape_mode='potential',
                                              rho=ESCAPE_POTENTIAL_RHO)
                self.assertAlmostEqual(total_f, -phi0, places=9)

    def test_the_escape_potential_alone_telescopes(self):
        total_f, phi0 = self._episode('none', 0.0, 19, escape_mode='potential',
                                      rho=ESCAPE_POTENTIAL_RHO)
        self.assertAlmostEqual(total_f, -phi0, places=9)

    def test_phi_e_is_non_negative_so_death_cannot_pay_out(self):
        for d_safe in range(0, ESCAPE_POTENTIAL_CAP + 2):
            phi_e = (train_lin.potential(0, 0, 0.0, d_safe,
                                         ESCAPE_POTENTIAL_RHO)
                     - train_lin.potential(0, 0, 0.0, d_safe, 0.0))
            self.assertGreaterEqual(phi_e, 0.0, d_safe)
            self.assertLessEqual(phi_e, ESCAPE_POTENTIAL_RHO, d_safe)
        death = {'coins': 0, 'crates': 0, 'invalid': 0, 'suicide': 1,
                 'done': True}
        r = train_lin.step_reward(death, 0, 0, GAMMA, R_INVALID, True, -1.0,
                                  escape_mode='potential',
                                  rho=ESCAPE_POTENTIAL_RHO,
                                  safe_before=0, safe_after=0)
        self.assertLess(r, -1.0,
                        'dying while safe forfeits Phi_e on top of R_DEATH; it '
                        'must never be cheaper than dying without the term')

    def test_the_escape_term_pays_for_progress_and_taxes_a_bad_bomb(self):
        step = {'coins': 0, 'crates': 0, 'invalid': 0, 'suicide': 0,
                'done': False}
        kw = dict(escape_mode='potential', rho=ESCAPE_POTENTIAL_RHO)
        progress = train_lin.step_reward(step, 0, 0, GAMMA, R_INVALID, False,
                                         safe_before=2, safe_after=1, **kw)
        self.assertAlmostEqual(progress,
                               GAMMA * ESCAPE_POTENTIAL_RHO * (1 - 1 / 4)
                               - ESCAPE_POTENTIAL_RHO * (1 - 2 / 4), places=9)
        self.assertGreater(progress, 0.0)
        tax = train_lin.step_reward(step, 0, 0, GAMMA, R_INVALID, False,
                                    safe_before=0, safe_after=3, **kw)
        self.assertLess(tax, 0.0,
                        'the escape potential is a tax on bombing where the '
                        'exit is far, made checkable')
        full = sum(
            GAMMA ** i * train_lin.step_reward(
                step, 0, 0, GAMMA, R_INVALID, False,
                safe_before=ESCAPE_POTENTIAL_CAP - i,
                safe_after=ESCAPE_POTENTIAL_CAP - i - 1, **kw)
            for i in range(ESCAPE_POTENTIAL_CAP))
        self.assertGreater(full, 0.3, 'about +0.4 telescoped over a full escape')
        self.assertLess(full, ESCAPE_POTENTIAL_RHO + 1e-9)

    def test_the_event_bonus_does_not_telescope(self):
        deltas = []
        for seed in (7, 11, 23, 41):
            world = simenv.SoloWorld(crate_density=S2_CRATE_DENSITY,
                                     coin_count=50, feature_mode='place')
            world.reset(seed)
            rng = np.random.default_rng(seed)
            phi0 = train_lin.potential(world.coin_distance(), 0, 0.0)
            prev_d, destroyed, total, discount = world.coin_distance(), 0, 0.0, 1.0
            unshaped_total = 0.0
            for _ in range(60):
                if not world.running:
                    break
                action = int(rng.integers(N_ACTIONS))
                _phi, info = world.step_action(action)
                new_d = 10 if info['done'] else world.coin_distance()
                after = destroyed + info['crates']
                total += discount * train_lin.step_reward(
                    info, prev_d, new_d, GAMMA, R_INVALID, info['done'], -1.0,
                    crate_mode='event', kappa=KAPPA_TOP,
                    destroyed_before=destroyed, destroyed_after=after)
                unshaped_total += discount * (
                    1.0 * info['coins']
                    + (R_INVALID if info['invalid'] else 0.0)
                    + (-1.0 if info['suicide'] else 0.0))
                discount *= GAMMA
                prev_d, destroyed = new_d, after
            deltas.append(abs((total - unshaped_total) - (-phi0)))
        self.assertGreater(max(deltas), 1e-6,
                           'the event bonus is path-dependent by construction; '
                           'if this ever telescoped, lin_b and lin_c would be '
                           'the same experiment')


class ClawbackTests(unittest.TestCase):

    DEATH = {'coins': 0, 'crates': 0, 'invalid': False, 'suicide': True,
             'done': True}

    def test_the_modes_are_the_two_the_plan_names(self):
        self.assertEqual(CRATE_CLAWBACK_MODES, ('terminal', 'suppressed'))
        self.assertEqual(CRATE_CLAWBACK, 'terminal',
                         'every VARIANT trains under the invariant form; only '
                         'the bundle-only control ctrl_noclaw does not')
        self.assertEqual(train_lin.build_parser().parse_args([]).crate_clawback,
                         'terminal')

    def test_the_effective_death_reward_scales_with_the_crates_opened(self):
        for destroyed in (0, 3, 10):
            r = train_lin.step_reward(self.DEATH, 0, 0, GAMMA, R_INVALID, True,
                                      -1.0, crate_mode='potential',
                                      kappa=KAPPA_TOP,
                                      destroyed_before=destroyed,
                                      destroyed_after=destroyed)
            self.assertAlmostEqual(r, -1.0 - KAPPA_TOP * destroyed, places=9,
                                   msg=f'{destroyed} crates at kappa = '
                                       f'{KAPPA_TOP}')
        at_ten = train_lin.step_reward(self.DEATH, 0, 0, GAMMA, R_INVALID, True,
                                       -1.0, crate_mode='potential',
                                       kappa=0.4, destroyed_before=10,
                                       destroyed_after=10)
        self.assertAlmostEqual(at_ten, -5.0, places=9,
                               msg='')

    def test_suppressing_the_clawback_leaves_only_the_living_cost(self):
        suppressed = train_lin.step_reward(self.DEATH, 0, 0, GAMMA, R_INVALID,
                                           True, -1.0, crate_mode='potential',
                                           kappa=0.4, destroyed_before=10,
                                           destroyed_after=10,
                                           clawback='suppressed')
        self.assertAlmostEqual(suppressed, -1.0 - (1 - GAMMA) * 0.4 * 10,
                               places=9,
                               msg='Phi(terminal) := Phi(s_T) keeps the per-step '
                                   'clawback -(1-gamma) kappa n and removes the '
                                   'terminal forfeiture -- that is the whole '
                                   'control')
        self.assertGreater(suppressed, -5.0)

    def test_the_clawback_only_touches_terminal_transitions(self):
        step = {'coins': 0, 'crates': 2, 'invalid': False, 'suicide': False,
                'done': False}
        kw = dict(crate_mode='potential', kappa=KAPPA_TOP,
                  destroyed_before=0, destroyed_after=2)
        self.assertAlmostEqual(
            train_lin.step_reward(step, 5, 5, GAMMA, R_INVALID, False, -1.0,
                                  **kw),
            train_lin.step_reward(step, 5, 5, GAMMA, R_INVALID, False, -1.0,
                                  clawback='suppressed', **kw), places=12,
            msg='ctrl_noclaw differs from the base on TERMINAL rows only; if it '
                'differed anywhere else the control would not be attributable')

    def test_the_suppressed_form_does_not_telescope(self):
        world = simenv.SoloWorld(crate_density=S2_CRATE_DENSITY, coin_count=50,
                                 feature_mode='place')
        deltas = []
        for seed in (7, 11, 23, 41):
            world.reset(seed)
            rng = np.random.default_rng(seed)
            phi0 = train_lin.potential(world.coin_distance(), 0, KAPPA_TOP)
            prev_d, destroyed, total_f, discount = (world.coin_distance(), 0,
                                                    0.0, 1.0)
            for _ in range(60):
                if not world.running:
                    break
                _phi, info = world.step_action(int(rng.integers(N_ACTIONS)))
                new_d = 10 if info['done'] else world.coin_distance()
                after = destroyed + info['crates']
                r = train_lin.step_reward(info, prev_d, new_d, GAMMA, R_INVALID,
                                          info['done'], -1.0,
                                          crate_mode='potential',
                                          kappa=KAPPA_TOP,
                                          destroyed_before=destroyed,
                                          destroyed_after=after,
                                          clawback='suppressed')
                unshaped = (1.0 * info['coins']
                            + (R_INVALID if info['invalid'] else 0.0)
                            + (-1.0 if info['suicide'] else 0.0))
                total_f += discount * (r - unshaped)
                discount *= GAMMA
                prev_d, destroyed = new_d, after
            deltas.append(abs(total_f - (-phi0)))
        self.assertGreater(max(deltas), 1e-6,
                           'ctrl_noclaw breaks policy invariance ON PURPOSE, and '
                           'the break is verified rather than assumed. If this '
                           'ever telescoped the control would be measuring '
                           'nothing.')

    def test_the_buffer_reports_the_effective_death_reward(self):
        model = LinearQ(dim=dim_for('place'))
        learner = train_lin.LSTDQ(model, GAMMA, capacity=50, every=1000,
                                  kappa=0.4, r_death=-1.0)
        phi = np.zeros(dim_for('place'))
        for destroyed, done in ((3, False), (10, True), (4, True)):
            learner.observe(phi, 5, -1.0, phi, done, False, stage=2,
                            destroyed=destroyed)
        comp = learner.composition()
        self.assertEqual(comp['n_terminal'], 2)
        self.assertAlmostEqual(comp['mean_destroyed_at_terminal'], 7.0)
        self.assertAlmostEqual(comp['effective_death_reward'], -1.0 - 0.4 * 7.0,
                               msg='the clawback instrument: this is the number '
                                   'the LSTD fit actually sees on its death '
                                   'rows, and it is what the dose ladder may '
                                   'really be a ladder of')

    def test_the_buffer_reports_the_conjunctions_support(self):
        dim = dim_for('conjesc')
        model = LinearQ(dim=dim)
        learner = train_lin.LSTDQ(model, GAMMA, capacity=50, every=1000,
                                  conj_index=features.CONJ_PLACE_OUT_INDEX)
        for value in (0.0, 0.0, 0.0, 0.5):
            phi = np.zeros(dim)
            phi[features.CONJ_PLACE_OUT_INDEX] = value
            learner.observe(phi, 5, 0.0, phi, False, False, stage=2)
        comp = learner.composition()
        self.assertAlmostEqual(comp['conj_place_support'], 0.25)
        self.assertAlmostEqual(comp['conj_place_support_S2'], 0.25)
        self.assertAlmostEqual(comp['conj_place_mean_positive'], 0.5)
        plain = train_lin.LSTDQ(LinearQ(dim=dim_for('place')), GAMMA,
                               capacity=10, every=1000)
        plain.observe(np.zeros(dim_for('place')), 5, 0.0,
                      np.zeros(dim_for('place')), False, False)
        self.assertTrue(np.isnan(plain.composition()['conj_place_support']))


class RetiredInjectionTests(unittest.TestCase):

    def _args(self, **kw):
        args = train_lin.build_parser().parse_args([])
        args.features = 'place'
        args.quiet = True
        for k, v in kw.items():
            setattr(args, k, v)
        return args

    def test_the_trainer_defaults_to_no_injection(self):
        args = train_lin.build_parser().parse_args([])
        self.assertEqual(args.inject, 'none')

    def test_the_shipped_variant_json_asks_for_no_injection(self):
        self.assertEqual(load_variant().get('inject'), 'none')

    def test_the_four_mechanisms_are_the_pre_registered_ones(self):
        self.assertEqual(INJECT_MODES, ('none', 'random', 'safe', 'seed'))

    def test_s0_episodes_are_never_injected(self):
        rng = np.random.default_rng(0)
        for mode in INJECT_MODES:
            for _ in range(50):
                plan = train_lin.plan_injection(mode, 1.0, 'S0', rng, 400)
                self.assertEqual(plan['mode'], 'none',
                                 '')

    def test_the_sampler_draws_the_same_number_of_randoms_whatever_the_mode(self):
        states = []
        for mode in INJECT_MODES:
            rng = np.random.default_rng(3)
            for _ in range(20):
                train_lin.plan_injection(mode, 0.25, 'S1', rng, 200)
            states.append(rng.random())
        self.assertEqual(len(set(states)), 1, 'one shared stream position')

    def test_no_injection_means_no_injected_bombs(self):
        args = self._args(inject='none', episodes=20, probe_every=10 ** 9,
                          forced_every=10 ** 9)
        summary = train_lin.train(args)
        self.assertEqual(summary['bomb_events'], 0)
        self.assertEqual(summary['injected_episodes'], 0)
        self.assertEqual(summary['inject'], 'none')

    def test_a_retired_mechanism_would_still_work_if_it_were_asked_for(self):
        world = simenv.SoloWorld(crate_density=0.35, coin_count=9,
                                 feature_mode='place')
        model = LinearQ(dim=26)
        learner = train_lin.make_learner('lstd', model, self._args())
        st = train_lin.run_training_episode(
            world, model, learner, np.random.default_rng(0), 0.0, self._args(),
            train_lin.FORCED_SEED_BASE, inject={'mode': 'safe', 'step': None},
            stage='S1')
        self.assertEqual(st['bomb_events'], 1)
        self.assertGreaterEqual(st['bombs'], 1, 'the override fires at step 1, '
                                                'exactly as forced_bomb_trial does')

    def test_the_seeded_start_state_is_the_real_post_bomb_state(self):
        seeded = simenv.SoloWorld(crate_density=0.35, coin_count=9,
                                  feature_mode='crate')
        phi = seeded.reset(train_lin.FORCED_SEED_BASE, inject_bomb=True)
        self.assertEqual(len(seeded.bombs), 1)
        self.assertEqual(seeded.bomb_list()[0][2], INJECT_BOMB_TIMER)
        self.assertFalse(seeded.bombs_left)
        self.assertEqual(phi[9], 0.0, 'bomb_available is consumed')
        self.assertEqual(seeded.stats['bombs'], 0, 'no BOMB action was taken')

        dropped = simenv.SoloWorld(crate_density=0.35, coin_count=9,
                                   feature_mode='crate')
        dropped.reset(train_lin.FORCED_SEED_BASE)
        dropped.step_action(ACTIONS.index('BOMB'))
        self.assertEqual(seeded.bomb_list(), dropped.bomb_list(),
                         'same tile, same timer as an actually dropped bomb')
        for world in (seeded, dropped):
            while world.running:
                world.step_action(ACTIONS.index('WAIT'))
        self.assertEqual(seeded.step + 1, dropped.step,
                         'the seeded episode starts one step later in the fuse, '
                         'which is exactly what makes the escape sub-problem the '
                         'same one the forced-bomb probe scores')

    def test_a_seeded_episode_contributes_nothing_to_the_bomb_row(self):
        args = self._args(inject='seed')
        world = simenv.SoloWorld(crate_density=0.35, coin_count=9,
                                 feature_mode='place')
        model = LinearQ(dim=26)
        learner = train_lin.make_learner('lstd', model, args)
        train_lin.run_training_episode(
            world, model, learner, np.random.default_rng(0), 0.0, args,
            train_lin.FORCED_SEED_BASE, inject={'mode': 'seed', 'step': None},
            stage='S1')
        injected = learner.injected[:learner.size]
        self.assertEqual(int(injected.sum()), 0,
                         'lin_d changes the initial-state distribution, not the '
                         'behaviour policy: no BOMB transition is injected')

    def test_injected_transitions_are_tagged_and_stage_split(self):
        args = self._args(inject='safe')
        world = simenv.SoloWorld(crate_density=0.35, coin_count=9,
                                 feature_mode='place')
        model = LinearQ(dim=26)
        learner = train_lin.make_learner('lstd', model, args)
        train_lin.run_training_episode(
            world, model, learner, np.random.default_rng(0), 0.0, args,
            train_lin.FORCED_SEED_BASE, inject={'mode': 'safe', 'step': None},
            stage='S1')
        composition = learner.composition()
        self.assertEqual(composition['n_BOMB_injected'], 1)
        self.assertEqual(composition['n_S0'], 0)
        self.assertEqual(composition['n_BOMB_S1'], composition['n_BOMB'])
        self.assertAlmostEqual(composition['share_BOMB'],
                               composition['n_BOMB'] / composition['buffer'])


class BufferCompositionTests(unittest.TestCase):

    def test_composition_counts_actions_stages_and_targets(self):
        model = LinearQ(dim=16)
        learner = train_lin.LSTDQ(model, gamma=0.5, capacity=100, every=1)
        phi = np.zeros(16)
        phi[10] = 1.0
        for _ in range(4):
            learner.observe(phi, 5, -1.0, phi, False, False, stage=0)
        for _ in range(6):
            learner.observe(phi, 0, 1.0, phi, True, False, stage=1)
        for _ in range(5):
            learner.observe(phi, 5, 2.0, phi, False, False, stage=2)
        c = learner.composition()
        self.assertEqual(c['buffer'], 15)
        self.assertEqual(c['n_BOMB'], 9)
        self.assertEqual(c['n_BOMB_S0'], 4)
        self.assertEqual(c['n_BOMB_S1'], 0)
        self.assertEqual(c['n_BOMB_S2'], 5, 'three stages now, not two')
        self.assertEqual(c['n_S2'], 5)
        self.assertAlmostEqual(c['mean_r_BOMB'], (4 * -1.0 + 5 * 2.0) / 9)
        self.assertAlmostEqual(c['mean_target_UP'], 1.0, msg='terminal: r only')

    def test_the_condition_number_is_recorded_at_every_solve(self):
        model = LinearQ(dim=27)
        learner = train_lin.LSTDQ(model, gamma=0.9, capacity=100, every=1)
        rng = np.random.default_rng(0)
        for _ in range(40):
            phi = rng.random(27)
            learner.observe(phi, int(rng.integers(6)), float(rng.random()),
                            rng.random(27), False, False, stage=2)
        learner.solve()
        self.assertTrue(np.isfinite(learner.last_cond))
        self.assertGreater(learner.last_cond, 1.0)
        self.assertEqual(learner.last_composition['cond'], learner.last_cond,
                         'and it is in the row the analysis reads')

    def test_every_solve_records_one_composition_row(self):
        model = LinearQ(dim=16)
        learner = train_lin.LSTDQ(model, gamma=0.9, capacity=100, every=1)
        self.assertIsNone(learner.last_composition)
        learner.observe(np.ones(16), 0, 1.0, np.ones(16), True, False)
        learner.episode_end_hook(0)
        self.assertEqual(learner.solves, 1)
        self.assertEqual(learner.last_composition['buffer'], 1)


class BombMarginTests(unittest.TestCase):

    def _args(self):
        args = train_lin.build_parser().parse_args([])
        args.features = 'crate'
        return args

    def test_the_margin_is_measured_on_s2_start_states_too(self):
        args = self._args()
        out = train_lin.bomb_margin_probe(handcheck.hand_weights('crate'), args, 4)
        self.assertIn('s2_start_mean', out)
        self.assertTrue(np.isfinite(out['s2_start_mean']))
        self.assertEqual(out['s2_start_positive'], 0.0,
                         'the escape-only anchor never wants to bomb anywhere')

    def test_the_margin_is_bomb_minus_the_best_alternative(self):
        w = np.zeros((N_ACTIONS, 24))
        w[ACTIONS.index('BOMB'), 10] = 2.0
        w[ACTIONS.index('WAIT'), 10] = 0.5
        phi = np.zeros(24)
        phi[10] = 1.0
        self.assertAlmostEqual(train_lin.bomb_margin(LinearQ(w), phi), 1.5)

    def test_a_bombing_policy_reads_positive_and_a_shy_one_negative(self):
        args = self._args()
        shy = train_lin.bomb_margin_probe(handcheck.hand_weights('crate'), args, 5)
        keen = np.array(handcheck.hand_weights('crate').w, copy=True)
        keen[ACTIONS.index('BOMB'), 10] = +10.0
        keen = train_lin.bomb_margin_probe(LinearQ(keen), args, 5)
        self.assertLess(shy['start_mean'], 0.0)
        self.assertEqual(shy['start_positive'], 0.0)
        self.assertGreater(keen['start_mean'], 0.0)
        self.assertEqual(keen['start_positive'], 1.0)
        self.assertEqual(shy['post_states'], 5, 'post-bomb states are measured too')


class EscapeOracleTests(unittest.TestCase):

    def test_verdicts_match_an_independent_replay_on_a_small_block(self):
        cache = escape_oracle.FeasibilityCache()
        world = simenv.SoloWorld(crate_density=0.35, coin_count=9,
                                 feature_mode='graded')
        for i in range(25):
            seed = train_lin.FORCED_SEED_BASE + i
            self.assertEqual(cache.verdict(seed),
                             escape_oracle.board_verdict(world, seed))

    def test_the_ceiling_is_the_one_l2_measured(self):
        cache = escape_oracle.FeasibilityCache()
        seeds = [train_lin.FORCED_SEED_BASE + i for i in range(100)]
        summary = cache.warm(seeds)
        self.assertEqual(summary['boards'], 100)
        self.assertEqual(summary['skipped'], 0, 'every S1 board offers a trial')
        self.assertGreater(summary['ceiling'], 0.6)
        self.assertLess(summary['ceiling'], 0.9)
        self.assertAlmostEqual(FORCED_CEILING_L2, 0.736)

    def test_the_hand_written_policy_is_contained_by_the_oracle(self):
        args = train_lin.build_parser().parse_args([])
        args.features = 'crate'
        cache = escape_oracle.FeasibilityCache()
        fb = train_lin.forced_bomb_probe(handcheck.hand_weights('crate'), args,
                                         60, cache)
        self.assertEqual(fb['containment_violations'], 0,
                         'a success on a board the oracle called inescapable '
                         'would mean the two are not playing the same game')
        self.assertGreater(fb['unescapable'], 0, 'and the decline is not vacuous')
        self.assertAlmostEqual(fb['rate_norm'], 1.0,
                               msg='')
        self.assertLess(fb['rate_raw'], fb['rate_norm'],
                        'which is exactly why the raw rate is never quoted alone')

    def test_a_cache_from_another_stage_is_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'escape_feasible.json')
            cache = escape_oracle.FeasibilityCache(path, crate_density=0.35)
            cache.warm([train_lin.FORCED_SEED_BASE])
            cache.save()
            with self.assertRaises(ValueError):
                escape_oracle.FeasibilityCache(path, crate_density=0.75)
            reused = escape_oracle.FeasibilityCache(path, crate_density=0.35)
            self.assertEqual(reused.verdicts, cache.verdicts)
            self.assertEqual(reused.computed, 0, 'memoised, not recomputed')


def regime(crates=0.0, coins=0.0, bombs=0.0, spb=None, suicides=0.0,
           ratio=1.0, invalid=0.0, norm=0.0):
    return ({'crates': crates, 'coins': coins, 'bombs': bombs,
             'suicides': suicides},
            {'bombs': bombs, 'suicides': suicides, 'suicides_per_bomb': spb,
             'coins': coins},
            {'ratio': ratio, 'invalid_per_step': invalid},
            {'rate_norm': norm},
            {'sb_escape_norm': norm, 'sb_placement_ok': norm,
             'sb_feasible': 10, 'sb_bombs': 10})


class CheckpointRuleTests(unittest.TestCase):

    def _args(self, **kw):
        args = train_lin.build_parser().parse_args([])
        for k, v in kw.items():
            setattr(args, k, v)
        return args

    NOTHING = regime()
    DIES = regime(crates=3.0, bombs=2.0, suicides=1.0, spb=0.5, coins=1.0,
                  ratio=1.2, norm=0.1)
    DIES_ALWAYS = regime(crates=9.0, bombs=4.0, suicides=4.0, spb=1.0,
                         coins=0.0, ratio=1.2, norm=0.0)
    ESCAPES = regime(norm=0.95)
    FARMS_BREAKS_S0 = regime(crates=30.0, coins=9.0, bombs=6.0, spb=0.0,
                             ratio=40.0, invalid=0.2, norm=0.95)
    ALL_GATES = regime(crates=12.0, coins=6.0, bombs=1.5, spb=0.0,
                       ratio=1.05, norm=0.95)

    def _rank(self, r, **kw):
        return train_lin.probe_rank(r[0], r[1], r[2], self._args(**kw), r[3],
                                    r[4])

    def test_the_intended_total_order(self):
        order = [self.ALL_GATES, self.FARMS_BREAKS_S0, self.ESCAPES,
                 self.DIES, self.DIES_ALWAYS, self.NOTHING]
        ranks = [self._rank(r) for r in order]
        self.assertEqual(ranks, sorted(ranks),
                         'all three gates > crate farmer > '
                         'escapes-but-never-bombs > bombs-and-sometimes-dies > '
                         'bombs-and-always-dies > nothing works')

    def test_nothing_works_is_never_preferred(self):
        for r in (self.ALL_GATES, self.FARMS_BREAKS_S0, self.ESCAPES, self.DIES):
            self.assertLess(self._rank(r), self._rank(self.NOTHING))

    def test_a_non_bomber_no_longer_wins_on_a_perfect_suicide_count(self):
        self.assertLess(self._rank(self.DIES), self._rank(self.NOTHING),
                        'the repaired key prefers a bomber that dies over a '
                        'policy that does nothing at all')
        self.assertEqual(self._rank(self.ESCAPES)[5], float('inf'),
                         'suicides/bomb is +inf for a non-bomber')
        args = self._args()
        population = [
            {'episode': 100, 'p2': self.ESCAPES[0], 'p1': self.ESCAPES[1],
             'p0': self.ESCAPES[2], 'fb2': self.ESCAPES[3],
             'sb': self.ESCAPES[4], 'w': np.zeros((N_ACTIONS, 26))},
            {'episode': 200, 'p2': self.DIES[0], 'p1': self.DIES[1],
             'p0': self.DIES[2], 'fb2': self.DIES[3], 'sb': self.DIES[4],
             'w': np.zeros((N_ACTIONS, 26))},
        ]
        out = train_lin.select_checkpoint(population, args)
        self.assertEqual(out['best']['episode'], 200,
                         'the pre-filter keeps only checkpoints that reach e1 '
                         'bombs/ep, so the empty-denominator escape score '
                         'cannot win')
        self.assertFalse(out['bomb_filter_empty'])

    def test_the_gate_conjunction_comes_first(self):
        self.assertLess(self._rank(self.ALL_GATES),
                        self._rank(self.FARMS_BREAKS_S0),
                        'a checkpoint that breaks S0 cannot outrank one that '
                        'passes all three conjuncts, however many crates it farms')

    def test_the_crate_term_is_capped_and_no_longer_leads(self):
        at_bar = regime(crates=8.0, coins=1.0, norm=0.5)
        farmer = regime(crates=40.0, coins=0.0, norm=0.5)
        self.assertEqual(self._rank(at_bar)[3], self._rank(farmer)[3],
                         'past GATE_CRATES_MIN more crates buy nothing')
        self.assertLess(self._rank(at_bar), self._rank(farmer),
                        'so coins break the tie, not crate count')
        few_crates_good_escape = regime(crates=2.0, bombs=2.0, spb=0.1,
                                        norm=0.704)
        many_crates_no_escape = regime(crates=7.0, bombs=2.0, spb=0.1,
                                       norm=0.025)
        self.assertLess(self._rank(few_crates_good_escape),
                        self._rank(many_crates_no_escape),
                        'the gate\'s own conjunct decides '
                        'before the capped crate count')
        self.assertAlmostEqual(GATE_CRATES_MIN, 8.0)

    def test_placement_is_the_second_term(self):
        p2, p1, p0, fb2, _sb = regime(crates=4.0, bombs=2.0, spb=0.1)
        args = self._args()

        def key(norm, placement):
            sb = {'sb_escape_norm': norm, 'sb_placement_ok': placement,
                  'sb_feasible': 10, 'sb_bombs': 10}
            return train_lin.probe_rank(p2, p1, p0, args, fb2, sb)

        self.assertLess(key(0.9, 0.8), key(0.9, 0.3),
                        'at equal escape quality the better PLACEMENT wins -- '
                        'sb_placement_ok is the statistic rung 5 is about')
        self.assertLess(key(0.95, 0.3), key(0.9, 0.9),
                        'but escape quality still leads placement')

    def test_the_clawback_field_is_not_a_ranking_term(self):
        self.assertEqual(len(self._rank(self.ALL_GATES)), 8,
                         'eight terms: gate, escape, placement, crates, bombs, '
                         'suicides/bomb, S0 ratio, coins -- and nothing else')

    def test_capability_outranks_propensity_and_both_outrank_the_s0_terms(self):
        better_capability = regime(ratio=9.9, norm=0.9)
        better_s0 = regime(coins=99.0, ratio=1.0, norm=0.2)
        self.assertLess(self._rank(better_capability), self._rank(better_s0))

    def test_the_propensity_term_is_capped_so_it_cannot_be_gamed(self):
        at_gate = regime(crates=1.0, coins=5.0, bombs=1.0, spb=0.0, norm=0.5)
        spamming = regime(crates=1.0, coins=0.0, bombs=40.0, spb=0.0, norm=0.5)
        self.assertLess(self._rank(at_gate), self._rank(spamming),
                        'past B_min, more bombs buy nothing -- coins break the tie')

    def test_an_unmeasured_capability_is_never_credited(self):
        r = self.ALL_GATES
        key = train_lin.probe_rank(r[0], r[1], r[2], self._args(), None, None)
        self.assertEqual(key[1], -0.0, 'no self-bomb measurement, no credit')
        self.assertEqual(key[2], -0.0, 'and none for placement either')

    def test_the_capability_term_is_now_the_self_bomb_statistic(self):
        p2, p1, p0, fb2, sb = regime(crates=4.0, norm=0.9)
        args = self._args()
        with_fb_only = train_lin.probe_rank(p2, p1, p0, args, fb2, None)
        with_sb = train_lin.probe_rank(p2, p1, p0, args, fb2, sb)
        self.assertEqual(with_fb_only[1], -0.0,
                         'fb_rate_norm no longer enters the key at all')
        self.assertEqual(with_sb[1], -0.9)
        self.assertLess(with_sb, with_fb_only)


class CheckpointSelectionTests(unittest.TestCase):

    def _args(self, **kw):
        args = train_lin.build_parser().parse_args([])
        for k, v in kw.items():
            setattr(args, k, v)
        return args

    def _c(self, episode, r):
        p2, p1, p0, fb2, sb = r
        return {'episode': episode, 'p2': p2, 'p1': p1, 'p0': p0, 'fb2': fb2,
                'sb': sb, 'w': np.full((N_ACTIONS, 26), float(episode))}

    def test_a_broken_s0_candidate_never_wins_while_a_sound_one_exists(self):
        population = [
            self._c(1300, regime(crates=30.0, coins=9.0, bombs=6.0, spb=0.2,
                                 ratio=4.0625, invalid=0.6, norm=0.9)),
            self._c(3500, regime(crates=2.0, coins=1.0, bombs=1.0, spb=0.3,
                                 ratio=1.05, invalid=0.001, norm=0.2)),
        ]
        out = train_lin.select_checkpoint(population, self._args())
        self.assertEqual(out['best']['episode'], 3500,
                         'the S0 filter is HARD: a checkpoint that breaks the '
                         'navigation gate cannot be installed while one that '
                         'does not exists, whatever its crate count')
        self.assertFalse(out['s0_filter_empty'])
        self.assertEqual(out['eligible'], 1)
        args = self._args()

        def key(c):
            return train_lin.probe_rank(c['p2'], c['p1'], c['p0'], args,
                                        c['fb2'], c['sb'])

        self.assertLess(key(population[0]), key(population[1]),
                        'the ordering is unchanged and still prefers the crate '
                        'farmer; what changed is that it is no longer asked')

    def test_all_candidates_break_s0_flags_the_arm(self):
        population = [
            self._c(500, regime(crates=4.0, ratio=9.0, invalid=0.3, norm=0.5)),
            self._c(1000, regime(crates=9.0, ratio=7.0, invalid=0.4, norm=0.5)),
        ]
        out = train_lin.select_checkpoint(population, self._args())
        self.assertTrue(out['s0_filter_empty'],
                        '')
        self.assertEqual(out['eligible'], 0)
        self.assertIsNotNone(out['best'])

    def test_farms_crates_but_breaks_s0(self):
        population = [
            self._c(200, regime(crates=40.0, coins=9.0, ratio=40.0,
                                invalid=0.2, norm=0.95)),
            self._c(400, regime(crates=8.0, coins=6.0, bombs=1.5, spb=0.0,
                                ratio=1.05, norm=0.95)),
        ]
        self.assertEqual(
            train_lin.select_checkpoint(population, self._args())['best']['episode'],
            400)

    def test_bombs_and_dies_every_time_loses_to_a_survivor(self):
        population = [
            self._c(100, regime(crates=4.0, bombs=4.0, suicides=4.0, spb=1.0,
                                ratio=1.1, norm=0.0)),
            self._c(200, regime(crates=4.0, bombs=4.0, suicides=0.1, spb=0.02,
                                ratio=1.1, norm=0.9)),
        ]
        self.assertEqual(
            train_lin.select_checkpoint(population, self._args())['best']['episode'],
            200)

    def test_nothing_works_still_returns_something_and_says_so(self):
        population = [self._c(100, regime(ratio=float('inf'))),
                      self._c(200, regime(ratio=float('inf')))]
        out = train_lin.select_checkpoint(population, self._args())
        self.assertTrue(out['s0_filter_empty'])
        self.assertEqual(out['best']['episode'], 100, 'ties -> earliest episode')

    def test_an_empty_population_is_not_a_crash(self):
        out = train_lin.select_checkpoint([], self._args())
        self.assertIsNone(out['best'])
        self.assertEqual(out['considered'], 0)

    def test_ties_everywhere_are_deterministic(self):
        population = [self._c(ep, regime(crates=4.0, bombs=1.0, spb=0.1,
                                         ratio=1.1, norm=0.5))
                      for ep in (500, 1000, 1500)]
        picks = {train_lin.select_checkpoint(list(reversed(population)),
                                             self._args())['best']['episode']
                 for _ in range(5)}
        self.assertEqual(picks, {500},
                         'identical candidates resolve to the earliest episode, '
                         'every time and in any input order')

    def test_a_gated_candidate_beats_every_ungated_one(self):
        population = [
            self._c(100, regime(crates=40.0, coins=99.0, bombs=9.0, spb=0.06,
                                suicides=0.5, ratio=1.2, norm=1.0)),
            self._c(900, regime(crates=12.0, coins=6.0, bombs=1.5, spb=0.0,
                                suicides=0.0, ratio=1.05, norm=0.95)),
        ]
        args = self._args(gate_suicides=0.02)
        self.assertEqual(
            train_lin.select_checkpoint(population, args)['best']['episode'], 900)


    def test_the_top_crate_candidate_with_no_escape_never_wins(self):
        population = [
            self._c(4500, regime(crates=7.0, coins=4.0, bombs=2.0, spb=0.3,
                                 ratio=1.05, norm=0.025)),
            self._c(8000, regime(crates=2.0, coins=1.0, bombs=2.0, spb=0.2,
                                 ratio=1.05, norm=0.704)),
            self._c(11000, regime(crates=3.0, coins=1.0, bombs=1.5, spb=0.25,
                                  ratio=1.05, norm=0.500)),
        ]
        out = train_lin.select_checkpoint(population, self._args())
        self.assertEqual(out['best']['episode'], 8000,
                         'the repaired key reads the gate\'s own conjunct '
                         'first, so the 0.704 checkpoint is installed')
        self.assertEqual(out['probe_rank_regret'], 0.0,
                         'and the regret column records that nothing better was '
                         'left behind')

    def test_probe_rank_regret_is_recorded_when_the_key_leaves_something_behind(self):
        population = [
            self._c(1000, regime(crates=12.0, coins=6.0, bombs=1.5, spb=0.0,
                                 suicides=0.0, ratio=1.05, norm=0.60)),
            self._c(2000, regime(crates=2.0, coins=1.0, bombs=2.0, spb=0.4,
                                 ratio=1.05, norm=0.95)),
        ]
        out = train_lin.select_checkpoint(population, self._args())
        self.assertEqual(out['best']['episode'], 1000, 'the gate leads')
        self.assertAlmostEqual(out['probe_rank_regret'], 0.35,
                               msg='0.95 - 0.60: the selection cost, measured '
                                   'instead of assumed away')

    def test_a_population_where_nothing_bombs_is_flagged_not_silently_ranked(self):
        population = [
            self._c(500, regime(crates=0.0, coins=8.0, bombs=0.0, ratio=1.02,
                                norm=0.0)),
            self._c(1000, regime(crates=0.0, coins=9.0, bombs=0.0, ratio=1.01,
                                 norm=0.0)),
        ]
        out = train_lin.select_checkpoint(population, self._args())
        self.assertTrue(out['bomb_filter_empty'],
                        'no candidate reaches e1 bombs/ep, so the pre-filter is '
                        'relaxed AND FLAGGED -- an escape statistic over an '
                        'empty denominator must never be ranked silently')
        self.assertFalse(out['s0_filter_empty'])
        self.assertIsNotNone(out['best'])
        self.assertEqual(out['bombing_candidates'], 0)

    def test_the_pre_filter_ranks_on_propensity_and_never_on_the_gates_conjunct(self):
        c = self._c(100, regime(crates=1.0, bombs=3.0, spb=0.2, norm=0.9))
        self.assertAlmostEqual(train_lin.bomb_propensity(c), 3.0)
        self.assertAlmostEqual(train_lin.bomb_propensity({'p2': {}}), 0.0)


class BimodalityTests(unittest.TestCase):

    def _args(self, **kw):
        args = train_lin.build_parser().parse_args([])
        for k, v in kw.items():
            setattr(args, k, v)
        return args

    def test_a_non_bombing_seed_is_recorded_as_one(self):
        out = train_lin.seed_bimodality([], {'bombs': 0.0}, None, [],
                                        self._args())
        self.assertEqual(out['final_bombs_s2'], 0.0)
        self.assertFalse(out['bombs_e1'])
        self.assertFalse(out['bombs_any'])
        self.assertIsNone(out['first_bomb_episode'])

    def test_the_two_proportions_have_different_thresholds(self):
        rare = train_lin.seed_bimodality([], {'bombs': 0.2}, None, [],
                                         self._args())
        self.assertFalse(rare['bombs_e1'], 'e1 is bombs/ep >= 1.0')
        self.assertTrue(rare['bombs_any'],
                        '')

    def test_the_first_bomb_episode_is_read_off_the_probe_curve(self):
        rows = [{'episode': 100, 'bombs': 0.0}, {'episode': 200, 'bombs': 0.4},
                {'episode': 300, 'bombs': 1.4}, {'episode': 400, 'bombs': 2.0}]
        out = train_lin.seed_bimodality(rows, {'bombs': 2.0}, None, [],
                                        self._args())
        self.assertEqual(out['first_bomb_episode'], 300)

    def test_the_diagnostics_come_from_the_last_buffer_solve(self):
        buffer_rows = [{'share_BOMB': 0.1, 'mean_r_BOMB': -0.5,
                        'effective_death_reward': -3.0,
                        'conj_place_support': 0.02, 'cond': 1e6},
                       {'share_BOMB': 0.2, 'mean_r_BOMB': -0.2,
                        'effective_death_reward': -1.4,
                        'conj_place_support': 0.04, 'cond': 2e6}]
        out = train_lin.seed_bimodality([], {'bombs': 3.0}, {'sb_bombs': 12,
                                                            'sb_feasible': 5},
                                       buffer_rows, self._args())
        self.assertEqual(out['buffer_share_BOMB'], 0.2)
        self.assertEqual(out['buffer_effective_death_reward'], -1.4)
        self.assertEqual(out['conj_place_support'], 0.04)
        self.assertEqual(out['final_sb_feasible'], 5)
        self.assertTrue(out['bombs_e1'])


class OracleCapTests(unittest.TestCase):

    def _args(self, **kw):
        args = train_lin.build_parser().parse_args([])
        for k, v in kw.items():
            setattr(args, k, v)
        return args

    def test_the_final_cap_is_zero_and_zero_means_uncapped(self):
        self.assertEqual(SB_ORACLE_CAP_FINAL, 0)
        self.assertEqual(self._args().sb_oracle_cap_final, 0)
        self.assertEqual(self._args().sb_oracle_cap, 4,
                         'the PERIODIC probes keep the cheap cap so the curves '
                         'stay cheap')

    def test_every_row_says_which_convention_produced_it(self):
        args = self._args(features='place', max_steps=60)
        model = handcheck.hand_weights('place', 'place')
        capped = train_lin.self_bomb_stats(model, args, 3, stage='S2')
        uncapped = train_lin.self_bomb_stats(model, args, 3, stage='S2',
                                             oracle_cap=0)
        self.assertEqual(capped['oracle_cap'], 4)
        self.assertEqual(uncapped['oracle_cap'], 0)
        self.assertEqual(uncapped['sb_oracle_capped'], 0,
                         '')
        row = train_lin.self_bomb_row(uncapped, 12_000, 1.0)
        self.assertIn('oracle_cap', row)
        self.assertEqual(row['oracle_cap'], 0)
        self.assertGreaterEqual(capped['sb_bombs_evaluated'], 0)


class EGateTests(unittest.TestCase):

    def _args(self, **kw):
        args = train_lin.build_parser().parse_args([])
        for k, v in kw.items():
            setattr(args, k, v)
        return args

    def _p2(self, bombs=2.0, spb=0.05):
        return {'bombs': bombs, 'suicides': bombs * (spb or 0.0),
                'suicides_per_bomb': spb, 'crates': 3.0, 'coins': 0.0}

    def _sb(self, norm=0.9, feasible=40):
        return {'sb_escape_norm': norm, 'sb_feasible': feasible}

    def test_all_three_conjuncts_must_hold(self):
        args = self._args()
        self.assertTrue(e_gate_pass(train_lin.e_gate(self._p2(), self._sb(), args)))
        self.assertFalse(e_gate_pass(train_lin.e_gate(self._p2(bombs=0.5),
                                                      self._sb(), args)))
        self.assertFalse(e_gate_pass(train_lin.e_gate(self._p2(spb=0.9),
                                                      self._sb(), args)))
        self.assertFalse(e_gate_pass(train_lin.e_gate(self._p2(),
                                                      self._sb(norm=0.5), args)))

    def test_a_non_bomber_fails_e1_rather_than_scoring_on_an_empty_denominator(self):
        out = train_lin.e_gate(self._p2(bombs=0.0, spb=None), None, self._args())
        self.assertFalse(out['e1']['pass'])
        self.assertFalse(out['e2']['pass'])
        self.assertIsNone(out['e3']['value'])
        self.assertFalse(out['pass'])

    def test_the_denominator_travels_with_the_rate(self):
        out = train_lin.e_gate(self._p2(), self._sb(norm=1.0, feasible=3),
                               self._args())
        self.assertEqual(out['e3']['denominator'], 3,
                         'a pass on three boards must not be quotable as a pass '
                         'without its three')

    def test_the_floors_are_the_pre_registered_ones(self):
        args = self._args()
        self.assertAlmostEqual(args.e_gate_bombs, 1.0)
        self.assertAlmostEqual(args.e_gate_suicides_per_bomb,
                               E_GATE_SUICIDES_PER_BOMB_FLOOR)
        self.assertAlmostEqual(args.e_gate_escape_norm, E_GATE_ESCAPE_NORM_FLOOR)
        self.assertAlmostEqual(E_GATE_SUICIDES_PER_BOMB_FLOOR, 0.10)
        self.assertAlmostEqual(E_GATE_ESCAPE_NORM_FLOOR, 0.75)


class SelfBombInstrumentTests(unittest.TestCase):

    def _args(self, **kw):
        args = train_lin.build_parser().parse_args([])
        args.features = 'place'
        for k, v in kw.items():
            setattr(args, k, v)
        return args

    def test_the_decomposition_identity_holds_on_a_real_block(self):
        model = handcheck.hand_weights('place', 'place')   # a bombing policy
        out = train_lin.self_bomb_stats(model, self._args(sb_oracle_cap=0), 12)
        self.assertGreater(out['sb_bombs'], 0, 'the anchor bombs, by design')
        self.assertEqual(out['sb_oracle_capped'], 0)
        self.assertEqual(out['sb_bombs_evaluated'], out['sb_bombs'])
        if out['sb_placement_ok'] is not None and out['sb_escape_norm'] is not None:
            predicted = 1.0 - out['sb_placement_ok'] * out['sb_escape_norm']
            self.assertAlmostEqual(predicted, 1.0 - out['sb_escape_raw'],
                                   places=9,
                                   msg='(1 - placement) + placement * (1 - '
                                       'escape) is an identity on the evaluated '
                                       'bombs, not an approximation')

    def test_the_oracle_is_sound_no_survival_on_an_infeasible_board(self):
        model = handcheck.hand_weights('place', 'place')
        out = train_lin.self_bomb_stats(model, self._args(sb_oracle_cap=0), 12)
        self.assertEqual(out['sb_survived_infeasible'], 0,
                         '')

    def test_the_cap_only_moves_the_denominator(self):
        model = handcheck.hand_weights('place', 'place')
        uncapped = train_lin.self_bomb_stats(model, self._args(sb_oracle_cap=0), 8)
        capped = train_lin.self_bomb_stats(model, self._args(sb_oracle_cap=1), 8)
        self.assertEqual(uncapped['sb_bombs'], capped['sb_bombs'],
                         'a capped bomb still counts into sb_bombs and '
                         'sb_escape_raw -- only the oracle-evaluated subset '
                         'shrinks')
        self.assertLessEqual(capped['sb_bombs_evaluated'],
                             uncapped['sb_bombs_evaluated'])
        if capped['sb_bombs'] > capped['sb_bombs_evaluated']:
            self.assertGreater(capped['sb_oracle_capped'], 0)

    def test_a_non_bomber_reports_nan_with_its_count_and_never_zero(self):
        model = handcheck.hand_weights('place', 'none')    # never bombs
        out = train_lin.self_bomb_stats(model, self._args(), 5)
        self.assertEqual(out['sb_bombs'], 0)
        self.assertIsNone(out['sb_escape_norm'],
                          'NaN with its count, never an imputed 0 or 1')
        self.assertIsNone(out['sb_placement_ok'])

    def test_it_is_deterministic(self):
        model = handcheck.hand_weights('place', 'place')
        a = train_lin.self_bomb_stats(model, self._args(), 6)
        b = train_lin.self_bomb_stats(model, self._args(), 6)
        for key in ('sb_bombs', 'sb_feasible', 'sb_survived_total',
                    'sb_survived_feasible'):
            self.assertEqual(a[key], b[key], key)

    def test_the_block_is_the_probes_own_so_the_rows_line_up(self):
        model = handcheck.hand_weights('place', 'place')
        args = self._args()
        sb = train_lin.self_bomb_stats(model, args, 5, stage='S2')
        p2 = train_lin.probe_s2(model, args, 5)
        self.assertAlmostEqual(sb['crates_per_episode'], p2['crates'], places=9,
                               msg='same policy, same boards, same episodes -- '
                                   'the instrument must not be measured on a '
                                   'different block from the probe it explains')
        self.assertAlmostEqual(sb['suicides_per_episode'], p2['suicides'],
                               places=9)

    def test_s0_has_no_self_bomb_block(self):
        model = handcheck.hand_weights('place', 'place')
        with self.assertRaises(ValueError):
            train_lin.self_bomb_stats(model, self._args(), 2, stage='S0')


def e_gate_pass(out) -> bool:
    return bool(out['pass'])


class GateTests(unittest.TestCase):

    def _args(self, **kw):
        args = train_lin.build_parser().parse_args([])
        for k, v in kw.items():
            setattr(args, k, v)
        return args

    def test_the_s2_conjunct_is_the_crate_count(self):
        args = self._args()
        self.assertTrue(train_lin.gate_s2({'crates': 8.0}, args))
        self.assertFalse(train_lin.gate_s2({'crates': 7.99}, args))
        self.assertFalse(train_lin.gate_s2({'crates': None}, args))

    def test_the_promotion_gate_is_all_three_conjuncts(self):
        args = self._args()
        p2 = {'crates': 12.0}
        p1 = {'bombs': 1.2, 'suicides': 0.0}
        p0 = {'ratio': 1.1, 'invalid_per_step': 0.0}
        self.assertTrue(train_lin.gate_all(p2, p1, p0, args))
        self.assertFalse(train_lin.gate_all({'crates': 2.0}, p1, p0, args),
                         '(a) fails')
        self.assertFalse(train_lin.gate_all(p2, {'bombs': 0.0, 'suicides': 0.0},
                                            p0, args), '(b) fails')
        self.assertFalse(train_lin.gate_all(p2, p1,
                                            {'ratio': 9.0,
                                             'invalid_per_step': 0.0}, args),
                         '(c) fails -- a crate farmer that forgot how to walk '
                         'is not a promotion')

    def test_a_non_bombing_agent_fails_the_s1_gate(self):
        args = self._args()
        frozen = {'bombs': 0.0, 'suicides': 0.0}
        self.assertFalse(train_lin.gate_s1(frozen, args),
                         '')

    def test_bombing_without_dying_passes(self):
        args = self._args()
        self.assertTrue(train_lin.gate_s1({'bombs': 1.4, 'suicides': 0.01}, args))
        self.assertFalse(train_lin.gate_s1({'bombs': 1.4, 'suicides': 0.4}, args))

    def test_the_gate_can_only_be_softened_from_outside(self):
        soft = self._args(gate_suicides=0.56)
        self.assertTrue(train_lin.gate_s1({'bombs': 1.0, 'suicides': 0.5}, soft))
        self.assertFalse(train_lin.gate_s1({'bombs': 1.0, 'suicides': 0.5},
                                           self._args()))

    def test_probe_rank_puts_the_gate_first(self):
        args = self._args()
        gated = regime(crates=9.0, coins=1.0, bombs=2.0, spb=0.0, ratio=1.1,
                       norm=0.5)
        not_gated = regime(crates=0.0, coins=5.0, ratio=1.0, norm=1.0)
        self.assertLess(
            train_lin.probe_rank(gated[0], gated[1], gated[2], args, gated[3]),
            train_lin.probe_rank(not_gated[0], not_gated[1], not_gated[2], args,
                                 not_gated[3]),
            'all three gates first, whatever the capability term says')
        cleaner = regime(crates=9.0, coins=0.0, bombs=2.0, spb=0.0, ratio=1.2,
                         norm=0.9)
        deadlier = regime(crates=9.0, coins=9.0, bombs=2.0, spb=0.05,
                          suicides=0.1, ratio=1.0, norm=0.9)
        self.assertLess(
            train_lin.probe_rank(cleaner[0], cleaner[1], cleaner[2], args,
                                 cleaner[3]),
            train_lin.probe_rank(deadlier[0], deadlier[1], deadlier[2], args,
                                 deadlier[3]),
            'at equal crates, capability and propensity, suicides/bomb decides')

    def test_episodes_to_gate_needs_two_consecutive_probes(self):
        rows = [{'episode': 100, 'gate_all': 1}, {'episode': 200, 'gate_all': 0},
                {'episode': 300, 'gate_all': 1}, {'episode': 400, 'gate_all': 1}]
        self.assertEqual(train_lin.episodes_to_gate(rows, 'gate_all'), 300)
        self.assertIsNone(train_lin.episodes_to_gate(rows[:1], 'gate_all'))


class ProbeTests(unittest.TestCase):

    def _args(self, **kw):
        args = train_lin.build_parser().parse_args([])
        args.probe_episodes = 5
        args.probe_episodes_final = 5
        args.forced_trials_final = 25
        for k, v in kw.items():
            setattr(args, k, v)
        return args

    def test_hand_written_weights_still_pass_the_s0_gate(self):
        args = self._args()
        p = train_lin.probe_s0(handcheck.hand_weights('graded'), args,
                               args.probe_episodes)
        self.assertTrue(p['gate'], '')
        self.assertLess(p['ratio'], 1.1)

    def test_untrained_weights_fail_both_gates(self):
        args = self._args()
        p0 = train_lin.probe_s0(LinearQ(dim=16), args, args.probe_episodes)
        p1 = train_lin.probe_s1(LinearQ(dim=16), args, args.probe_episodes)
        self.assertFalse(p0['gate'])
        self.assertFalse(p1['gate'])

    def test_s1_probe_has_no_ratio_column(self):
        args = self._args()
        p = train_lin.probe_s1(LinearQ(dim=16), args, 2)
        self.assertNotIn('ratio', p, 'a crate board has no static GBR reference')
        self.assertIn('suicides_per_bomb', p)

    def test_the_s2_probe_reports_the_gate_and_the_distortion_columns(self):
        args = self._args(features='conj')
        bomber = handcheck.hand_weights('conj')
        p2 = train_lin.probe_s2(bomber, args, 5)
        for key in ('crates', 'coins', 'bombs', 'suicides', 'suicides_per_bomb',
                    'idle', 'invalid_per_step', 'survival', 'gate'):
            self.assertIn(key, p2)
        self.assertGreater(p2['crates'], 0.0,
                           'the hand-written bombing anchor destroys crates')
        self.assertEqual(p2['gate'], p2['crates'] >= args.gate_crates)
        idle = train_lin.probe_s2(handcheck.hand_weights('crate'),
                                  self._args(features='crate'), 3)
        self.assertEqual(idle['crates'], 0.0, 'and the escape-only anchor does not')

    def test_the_s2_probe_plays_density_075_boards(self):
        args = self._args()
        self.assertAlmostEqual(args.s2_crate_density, 0.75)
        self.assertEqual(args.s2_coin_count, 50)
        world = simenv.SoloWorld(crate_density=args.s2_crate_density,
                                 coin_count=args.s2_coin_count)
        world.reset(train_lin.S2_PROBE_SEED_BASE)
        crates = int((world.field == simenv.CRATE).sum())
        self.assertGreater(crates, 90, 'loot-crate is a dense board')

    def test_the_s2_block_can_be_given_more_trials_for_the_same_precision(self):
        args = self._args(forced_trials=20, forced_trials_final=500,
                          forced_trials_s2=150, forced_trials_s2_final=500)
        self.assertEqual(train_lin.forced_trials_for(args, 20, False), 150)
        self.assertEqual(train_lin.forced_trials_for(args, 500, True), 500)
        default = self._args(forced_trials_s2=0, forced_trials_s2_final=0)
        self.assertEqual(train_lin.forced_trials_for(default, 20, False), 20,
                         '0 means "same as S1", so nothing changes by accident')

    def test_the_second_forced_bomb_block_is_at_s2_density(self):
        args = self._args()
        s1 = train_lin.forced_stage_config(args, 'S1')
        s2 = train_lin.forced_stage_config(args, 'S2')
        self.assertEqual(s1[0], 0.35)
        self.assertEqual(s2[0], 0.75)
        self.assertNotEqual(s1[2], s2[2], 'disjoint seed blocks')
        with self.assertRaises(ValueError):
            train_lin.forced_stage_config(args, 'S0')
        fb = train_lin.forced_bomb_probe(handcheck.hand_weights('crate'),
                                         self._args(features='crate'), 20,
                                         stage='S2')
        self.assertEqual(fb['stage'], 'S2')
        self.assertEqual(fb['containment_violations'], 0)
        self.assertGreater(fb['unescapable'], 0,
                           'at density 0.75 most spawn-corner bombs are '
                           'inescapable for any policy, which is exactly why '
                           'the rate is normalised')

    def test_forced_bomb_probe_separates_danger_from_no_danger(self):
        blind = train_lin.forced_bomb_probe(handcheck.hand_weights('nav'),
                                            self._args(features='nav'), 25)
        seeing = train_lin.forced_bomb_probe(handcheck.hand_weights('crate'),
                                             self._args(features='crate'), 25)
        self.assertEqual(blind['trials'], 25, 'every board offers a trial at step 1')
        self.assertGreater(seeing['rate_norm'], blind['rate_norm'],
                           'the instrument must register the danger features')

    def test_the_probe_declines_inescapable_boards_instead_of_failing_them(self):
        args = self._args(features='crate')
        fb = train_lin.forced_bomb_probe(handcheck.hand_weights('crate'), args, 40)
        self.assertEqual(fb['feasible'], fb['trials'] - fb['unescapable'])
        self.assertAlmostEqual(fb['rate_raw'], fb['successes'] / fb['trials'])
        self.assertAlmostEqual(fb['rate_norm'],
                               fb['feasible_successes'] / fb['feasible'])
        self.assertGreater(fb['rate_norm'], fb['rate_raw'],
                           'normalisation is what makes the threshold reachable')

    def test_the_bomb_margin_probe_reports_both_distributions(self):
        margin = train_lin.bomb_margin_probe(handcheck.hand_weights('crate'),
                                             self._args(features='crate'), 4)
        for key in ('start_mean', 'start_positive', 'post_mean', 'post_positive'):
            self.assertIn(key, margin)
        self.assertEqual(margin['states'], 4)

    def test_a_forced_trial_is_eight_steps_long(self):
        world = simenv.SoloWorld(crate_density=0.35, coin_count=9,
                                 feature_mode='room')
        out = train_lin.forced_bomb_trial(world, handcheck.hand_weights('room'),
                                          train_lin.FORCED_SEED_BASE, 8)
        self.assertEqual(out['trial'], 1)
        self.assertLessEqual(out['steps'], 8)

    def test_prefix_denominator_defeats_an_early_quitter(self):
        args = self._args()
        ref = train_lin.gbr_episode(args, train_lin.PROBE_SEED_BASE)
        self.assertAlmostEqual(train_lin.gbr_prefix_steps(ref, 1), ref['coin_steps'][0])
        self.assertIsNone(train_lin.gbr_prefix_steps(ref, 0))

    def test_probe_boards_are_disjoint_from_each_other(self):
        bases = sorted({train_lin.PROBE_SEED_BASE, train_lin.S1_PROBE_SEED_BASE,
                        train_lin.FORCED_SEED_BASE, train_lin.S2_PROBE_SEED_BASE,
                        train_lin.FORCED_S2_SEED_BASE})
        self.assertEqual(len(bases), 5, 'five blocks')
        for base in bases:
            self.assertGreater(base, 2 * train_lin.TRAIN_SEED_STRIDE,
                               'probe boards must never be training boards')
        for lo, hi in zip(bases, bases[1:]):
            self.assertGreaterEqual(hi - lo, 10_000_000,
                                    'and the blocks must not overlap each other')


class TrainingLoopTests(unittest.TestCase):

    def test_a_short_run_produces_every_artifact(self):
        import tempfile
        args = train_lin.build_parser().parse_args([])
        args.features = 'place'
        args.crate_reward = 'potential'
        args.crate_kappa = KAPPA_TOP
        args.episodes = 20
        args.probe_every = 10
        args.probe_episodes = 2
        args.probe_episodes_final = 2
        args.forced_every = 10
        args.forced_trials = 3
        args.forced_trials_final = 3
        args.lstd_every = 10
        args.quiet = True
        with tempfile.TemporaryDirectory() as tmp:
            args.out = os.path.join(tmp, 'weights.npy')
            args.log_csv = os.path.join(tmp, 'train_log.csv')
            args.probe_s2_csv = os.path.join(tmp, 'probe_s2.csv')
            args.probe_s1_csv = os.path.join(tmp, 'probe_s1.csv')
            args.probe_s0_csv = os.path.join(tmp, 'probe_s0.csv')
            args.forced_csv = os.path.join(tmp, 'forced_bomb.csv')
            args.forced_s2_csv = os.path.join(tmp, 'forced_bomb_s2.csv')
            args.bomb_margin_csv = os.path.join(tmp, 'bomb_margin.csv')
            args.buffer_composition_csv = os.path.join(tmp,
                                                       'buffer_composition.csv')
            args.escape_feasible = os.path.join(tmp, 'escape_feasible.json')
            args.escape_feasible_s2 = os.path.join(tmp, 'escape_feasible_s2.json')
            args.summary_json = os.path.join(tmp, 'summary.json')
            args.weights_table = os.path.join(tmp, 'weights_table.md')
            summary = train_lin.train(args)
            for name in ('weights.npy', 'weights_best.npy', 'train_log.csv',
                         'probe_s2.csv', 'probe_s1.csv', 'probe_s0.csv',
                         'forced_bomb.csv', 'forced_bomb_s2.csv',
                         'bomb_margin.csv', 'buffer_composition.csv',
                         'escape_feasible.json', 'escape_feasible_s2.json',
                         'summary.json', 'weights_table.md'):
                self.assertTrue(os.path.isfile(os.path.join(tmp, name)), name)
            self.assertEqual(sum(summary['stage_counts'].values()), 20)
            self.assertEqual(LinearQ.load(args.out).dim, 26)
            with open(os.path.join(tmp, 'train_log.csv')) as fh:
                header = fh.readline()
            for column in ('stage', 'crates', 'crate_reward', 'idle', 'p_s2'):
                self.assertIn(column, header, 'the plan\'s train_log columns')
            with open(os.path.join(tmp, 'buffer_composition.csv')) as fh:
                buffer_header = fh.readline()
            self.assertIn('cond', buffer_header)
            self.assertIn('n_BOMB_S2', buffer_header)
        self.assertIn('lexicographic', summary['selection'])
        self.assertIn('sb_escape_norm', summary['selection'])
        self.assertIn('hard filter on the S0 gate', summary['selection'])
        self.assertIn('final_s2', summary)
        self.assertIn('escape_ceiling_s2', summary)

    def test_the_mixture_actually_draws_all_three_stages(self):
        args = train_lin.build_parser().parse_args([])
        args.features = 'place'
        args.episodes = 40
        args.probe_every = 10_000
        args.forced_every = 10_000
        args.quiet = True
        import unittest.mock as mock
        with mock.patch.object(train_lin, 'CURRICULA',
                               {'mix': ((0, 1.0, 0.0), (10, 0.15, 0.85),
                                        (20, 0.15, 0.15))}):
            summary = train_lin.train(args)
        self.assertGreater(summary['stage_counts']['S0'], 0)
        self.assertGreater(summary['stage_counts']['S1'], 0)
        self.assertGreater(summary['stage_counts']['S2'], 0)

    def test_the_no_curriculum_control_never_leaves_s2(self):
        args = train_lin.build_parser().parse_args([])
        args.features = 'place'
        args.curriculum = 's2only'
        args.episodes = 8
        args.probe_every = 10_000
        args.forced_every = 10_000
        args.quiet = True
        summary = train_lin.train(args)
        self.assertEqual(summary['stage_counts'], {'S0': 0, 'S1': 0, 'S2': 8})


class CallbackTests(unittest.TestCase):

    def _fake_self(self):
        return SimpleNamespace(train=False, logger=logging.getLogger('lin_agent_test'))

    def _state(self, pos=(1, 1), coins=((1, 4),), bombs=(), explosion=None):
        return {
            'round': 1, 'step': 1, 'field': empty_board().astype(int),
            'self': ('me', 0, True, pos), 'others': [], 'bombs': list(bombs),
            'coins': list(coins),
            'explosion_map': (np.zeros((COLS, ROWS)) if explosion is None
                              else explosion),
            'user_input': None,
        }

    def test_act_returns_a_legal_action_name(self):
        fake = self._fake_self()
        callbacks.setup(fake)
        self.assertIn(callbacks.act(fake, self._state()), ACTIONS)

    def test_act_handles_a_dead_agent(self):
        fake = self._fake_self()
        callbacks.setup(fake)
        self.assertEqual(callbacks.act(fake, None), 'WAIT')

    def test_act_follows_the_weights(self):
        fake = self._fake_self()
        callbacks.setup(fake)
        fake.model = handcheck.hand_weights(fake.mode)
        self.assertEqual(callbacks.act(fake, self._state()), 'DOWN')

    def test_act_leaves_a_blast_when_the_weights_say_so(self):
        fake = self._fake_self()
        callbacks.setup(fake)
        fake.model = handcheck.hand_weights(fake.mode)
        state = self._state(pos=(1, 1), coins=(), bombs=[((1, 1), 3)])
        self.assertIn(callbacks.act(fake, state), ('RIGHT', 'DOWN'),
                      'the only two unblocked directions out of the corner')

    def test_act_is_far_inside_the_time_budget(self):
        from time import perf_counter
        fake = self._fake_self()
        callbacks.setup(fake)          # pays the numba compilation here
        fake.model = handcheck.hand_weights(fake.mode)
        state = self._state(bombs=[((5, 1), 2), ((1, 5), 0)])
        times = []
        for _ in range(200):
            t0 = perf_counter()
            callbacks.act(fake, state)
            times.append(perf_counter() - t0)
        p95 = float(np.percentile(times, 95))
        self.assertLess(p95, 0.005, f'p95 {p95 * 1e3:.2f} ms exceeds the 5 ms budget')
        self.assertLess(max(times), 0.5, 'a single call may never reach the timeout')

    def test_every_encoding_is_also_inside_the_budget(self):
        from time import perf_counter
        fake = self._fake_self()
        callbacks.setup(fake)
        state = self._state(bombs=[((5, 1), 2)])
        for mode in ('room', 'crate', 'place', 'conj'):
            with self.subTest(mode=mode):
                fake.mode = mode
                fake.dim = dim_for(mode)
                fake.model = handcheck.hand_weights(mode)
                callbacks.act(fake, state)     # compile this mode's kernels
                times = []
                for _ in range(200):
                    t0 = perf_counter()
                    callbacks.act(fake, state)
                    times.append(perf_counter() - t0)
                p95 = float(np.percentile(times, 95))
                self.assertLess(p95, 0.005,
                                '')

    def test_a_weight_file_of_the_wrong_width_is_refused(self):
        fake = self._fake_self()
        callbacks.setup(fake)
        self.assertIsNotNone(fake.model, 'the canonical agent ships weights')
        self.assertEqual(fake.model.dim, dim_for(fake.mode))


class WeightsPairingTests(unittest.TestCase):

    def test_the_shipped_weights_match_the_shipped_variant(self):
        variant = load_variant()
        weights = np.load(agent_path('weights.npy'))
        self.assertEqual(weights.shape, (N_ACTIONS, dim_for(variant['features'])),
                         f'{variant["features"]} wants d = '
                         f'{dim_for(variant["features"])}, the file is '
                         f'{weights.shape}')

    def test_the_canonical_agent_loads_its_variant(self):
        variant = load_variant()
        self.assertEqual(variant['features'], 'graded')

    def test_a_mismatched_pair_cannot_load_silently(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'weights.npy')
            LinearQ(dim=16).save(path)
            for mode in ('crate', 'place', 'conj'):
                with self.assertRaises(ValueError):
                    LinearQ.load(path, dim=dim_for(mode))


class FrameworkIntegrationTests(unittest.TestCase):

    def test_one_headless_round_in_training_mode(self):
        from main import main as engine_main
        engine_main(['play', '--no-gui', '--n-rounds', '1', '--scenario', 'coin-heaven',
                     '--agents', 'lin_agent', '--train', '1', '--seed', '7'])

    def test_one_headless_round_on_a_crate_board(self):
        from main import main as engine_main
        engine_main(['play', '--no-gui', '--n-rounds', '1', '--scenario', 'classic',
                     '--agents', 'lin_agent', '--train', '1', '--seed', '7'])


if __name__ == '__main__':
    unittest.main()
