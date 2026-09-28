
import argparse
import os
import sys
from contextlib import contextmanager

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    __package__ = 'agent_code.lin_agent'

import settings as s  # noqa: E402
from environment import BombeRLeWorld, WorldArgs  # noqa: E402

from .config import (  # noqa: E402
    ACTIONS,
    FEATURE_MODES,
    MAX_STEPS,
    S1_COIN_COUNT,
    S1_CRATE_DENSITY,
    S1_SCENARIO,
)
from .features import state_to_features  # noqa: E402
from .simenv import COIN_FREE, COIN_HIDDEN, SoloWorld  # noqa: E402

ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}

EXTRA_SCENARIOS = {S1_SCENARIO: {'CRATE_DENSITY': S1_CRATE_DENSITY,
                                 'COIN_COUNT': S1_COIN_COUNT}}


@contextmanager
def injected_scenarios():
    added = [k for k in EXTRA_SCENARIOS if k not in s.SCENARIOS]
    for k in added:
        s.SCENARIOS[k] = dict(EXTRA_SCENARIOS[k])
    try:
        yield s.SCENARIOS
    finally:
        for k in added:
            s.SCENARIOS.pop(k, None)


def make_world(seed, scenario, agent='random_agent'):
    args = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False,
        continue_without_training=True, log_dir='logs', save_stats=False,
        match_name=None, seed=seed, silence_errors=False, scenario=scenario)
    return BombeRLeWorld(args, [(agent, False)])


def coin_array(world):
    arr = np.zeros((s.COLS, s.ROWS), dtype=np.uint8)
    for c in world.coins:
        if c.collectable:
            arr[c.x, c.y] = COIN_FREE
        elif world.arena[c.x, c.y] == 1:
            arr[c.x, c.y] = COIN_HIDDEN
    return arr


def explosion_map(world):
    m = np.zeros((s.COLS, s.ROWS))
    for exp in world.explosions:
        if exp.is_dangerous():
            for (x, y) in exp.blast_coords:
                m[x, y] = max(m[x, y], exp.timer - 1)
    return m


def compare(world, sim, agent):
    problems = []
    if not np.array_equal(np.asarray(world.arena), sim.field.astype(int)):
        problems.append('arena')
    if not np.array_equal(coin_array(world), sim.coins):
        problems.append('coins')
    if sorted((b.x, b.y, b.timer) for b in world.bombs) != sim.bomb_list():
        problems.append(f'bombs {[(b.x, b.y, b.timer) for b in world.bombs]} '
                        f'!= {sim.bomb_list()}')
    if not np.array_equal(explosion_map(world), sim.explosion_map()):
        problems.append('explosion_map')

    alive = not agent.dead
    if alive != (not sim.dead):
        problems.append(f'alive {alive} != {not sim.dead}')
    if alive:
        if (agent.x, agent.y) != (sim.x, sim.y):
            problems.append(f'pos {(agent.x, agent.y)} != {(sim.x, sim.y)}')
        if bool(agent.bombs_left) != bool(sim.bombs_left):
            problems.append(f'bombs_left {agent.bombs_left} != {sim.bombs_left}')
    if agent.score != sim.score:
        problems.append(f'score {agent.score} != {sim.score}')

    state = world.get_state_for_agent(agent)
    if state is not None and alive:
        for mode in FEATURE_MODES:
            fw = state_to_features(state, mode)
            sm = sim.features_for(mode)
            if not np.allclose(fw, sm):
                problems.append(f'phi[{mode}] {fw.tolist()} != {sm.tolist()}')
    return problems


def run_round(seed, scenario, agent='random_agent', verbose=True):
    coverage = {'rounds': 1, 'steps': 0, 'bombs': 0, 'explosion_steps': 0,
                'crates': 0, 'reveals': 0, 'deaths': 0}
    world = make_world(seed, scenario, agent)
    world.new_round()
    world.user_input = None
    the_agent = world.agents[0]

    cfg = s.SCENARIOS[scenario]
    sim = SoloWorld(crate_density=cfg['CRATE_DENSITY'], coin_count=cfg['COIN_COUNT'],
                    max_steps=s.MAX_STEPS)
    sim.load_round(np.asarray(world.arena), coin_array(world), (the_agent.x, the_agent.y))

    problems = compare(world, sim, the_agent)
    if problems:
        return 0, ['initial state: ' + p for p in problems], coverage

    applied = []
    original = world.perform_agent_action

    def recording(a, action):
        applied.append(action)
        original(a, action)

    world.perform_agent_action = recording

    for step in range(MAX_STEPS):
        if not world.running:
            break
        applied.clear()
        hidden_before = int((sim.coins == COIN_HIDDEN).sum())
        world.do_step()
        action = ACTION_INDEX.get(applied[0], ACTION_INDEX['WAIT']) if applied else None
        if action is None:
            return step, ['framework applied no action'], coverage
        sim.step_action(action)

        coverage['steps'] += 1
        coverage['explosion_steps'] += int(bool(sim.explosions))
        coverage['reveals'] += max(0, hidden_before -
                                   int((sim.coins == COIN_HIDDEN).sum()))

        problems = compare(world, sim, the_agent)
        if problems:
            return step + 1, problems, coverage
        if world.running != sim.running:
            return (step + 1,
                    [f'round end: framework={world.running} sim={sim.running}'],
                    coverage)

    coverage['bombs'] = sim.stats['bombs']
    coverage['crates'] = sim.stats['crates']
    coverage['deaths'] = int(sim.dead)
    if verbose:
        print(f'  seed {seed}: {world.step} steps, score {the_agent.score}, '
              f'coins {the_agent.statistics["coins"]}, '
              f'invalid {the_agent.statistics["invalid"]}, '
              f'bombs {coverage["bombs"]}, crates {coverage["crates"]}, '
              f'reveals {coverage["reveals"]}, died {coverage["deaths"]}')
    return None, [], coverage


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--scenario', default=S1_SCENARIO)
    p.add_argument('--agent', default='random_agent')
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--no-coverage-gate', action='store_true',
                   help='')
    p.add_argument('--require', default='bombs,explosion_steps,crates',
                   help='')
    args = p.parse_args(argv)

    os.makedirs('logs', exist_ok=True)
    failures = 0
    total = {'rounds': 0, 'steps': 0, 'bombs': 0, 'explosion_steps': 0,
             'crates': 0, 'reveals': 0, 'deaths': 0}
    with injected_scenarios() as scenarios:
        if args.scenario not in scenarios:
            print(f'unknown scenario {args.scenario!r}; have {sorted(scenarios)}')
            return 1
        for seed in range(args.rounds):
            step, problems, coverage = run_round(seed, args.scenario, args.agent,
                                                 not args.quiet)
            for k, v in coverage.items():
                total[k] += v
            if problems:
                failures += 1
                print(f'MISMATCH seed={seed} step={step}:')
                for pr in problems:
                    print(f'    {pr}')

    print(f'\ncoverage: {total["rounds"]} rounds, {total["steps"]} steps, '
          f'{total["bombs"]} bombs, {total["explosion_steps"]} steps with live fire, '
          f'{total["crates"]} crates destroyed, {total["reveals"]} coins revealed, '
          f'{total["deaths"]} deaths')
    if failures:
        print(f'{failures}/{args.rounds} rounds diverged')
        return 1

    if not args.no_coverage_gate:
        labels = {'bombs': 'bombs', 'explosion_steps': 'explosions',
                  'crates': 'crate destruction', 'reveals': 'coins revealed',
                  'deaths': 'deaths'}
        required = [k.strip() for k in args.require.split(',') if k.strip()]
        unknown = [k for k in required if k not in total]
        if unknown:
            print(f'unknown coverage counter(s) {unknown}; have {sorted(total)}')
            return 1
        missing = [labels.get(key, key) for key in required if total[key] == 0]
        if missing:
            print(f'FAIL: these rule paths were never exercised: {", ".join(missing)}. '
                  f'A check that verified nothing must not report success.')
            return 1

    print(f'OK: {args.rounds} rounds of "{args.scenario}" reproduced exactly '
          f'(state, score, features in {len(FEATURE_MODES)} encodings)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
