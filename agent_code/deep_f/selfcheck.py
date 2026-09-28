
import argparse
import os
import sys

import numpy as np

if __package__ in (None, ''):
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
    __package__ = 'agent_code.' + os.path.basename(_HERE)

import settings as s
from environment import BombeRLeWorld, WorldArgs

from .config import ACTIONS, MAX_STEPS, N_AGENTS
from .features import state_to_features
from .vecenv import ST_SCORE, VecBomberman

ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}


def make_world(seed, scenario, agent='random_agent'):
    args = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False,
        continue_without_training=True, log_dir='logs', save_stats=False,
        match_name=None, seed=seed, silence_errors=False, scenario=scenario)
    return BombeRLeWorld(args, [(agent, False)] * N_AGENTS)


def collectable_set(world):
    return sorted((c.x, c.y) for c in world.coins if c.collectable)


def bomb_list(world):
    return sorted((b.x, b.y, b.timer) for b in world.bombs)


def explosion_map(world):
    m = np.zeros((s.COLS, s.ROWS), dtype=np.int64)
    for exp in world.explosions:
        if exp.is_dangerous():
            for (x, y) in exp.blast_coords:
                m[x, y] = max(m[x, y], exp.timer - 1)
    return m


def compare(world, env, index_of, step, canonical=False):
    problems = []
    if not np.array_equal(np.asarray(world.arena), env.field[0].astype(int)):
        problems.append('arena')
    if collectable_set(world) != sorted(map(tuple, np.argwhere(env.coins[0] == 2))):
        problems.append('coins')
    if bomb_list(world) != env.bomb_list(0):
        problems.append(f'bombs {bomb_list(world)} != {env.bomb_list(0)}')
    if not np.array_equal(explosion_map(world), env.explosion_map(0)):
        problems.append('explosion_map')

    for a in world.agents:
        i = index_of[a.name]
        alive = not a.dead
        if alive != bool(env.alive[0, i]):
            problems.append(f'alive[{a.name}] {alive} != {bool(env.alive[0, i])}')
        elif alive and (a.x, a.y) != (env.agx[0, i], env.agy[0, i]):
            problems.append(f'pos[{a.name}] {(a.x, a.y)} != '
                            f'{(env.agx[0, i], env.agy[0, i])}')
        if alive and bool(a.bombs_left) != bool(env.bleft[0, i]):
            problems.append(f'bombs_left[{a.name}]')
        if a.score != env.stats[0, i, ST_SCORE]:
            problems.append(f'score[{a.name}] {a.score} != {env.stats[0, i, ST_SCORE]}')

    for a in world.active_agents:
        i = index_of[a.name]
        obs_fw, mask_fw, _ = state_to_features(world.get_state_for_agent(a), canonical)
        if not np.array_equal(obs_fw, env.obs[0, i]):
            bad = np.unique(np.argwhere(obs_fw != env.obs[0, i])[:, 0])
            problems.append(f'observation[{a.name}] channels {bad.tolist()}')
        if not np.array_equal(mask_fw, env.masks[0, i]):
            problems.append(f'action_mask[{a.name}] {mask_fw} != {env.masks[0, i]}')
    return problems


def run_round(seed, scenario, verbose, agent='random_agent', canonical=False):
    world = make_world(seed, scenario, agent)
    index_of = {a.name: i for i, a in enumerate(world.agents)}
    world.new_round()

    env = VecBomberman(1, seed=seed, rewards=np.zeros(9), canonical=canonical)
    env.reset()
    env.load_round(0, np.asarray(world.arena),
                   [((c.x, c.y), c.collectable) for c in world.coins],
                   [(a.x, a.y) for a in world.agents])
    env._observe()

    applied = []
    original = world.perform_agent_action

    def recording(agent, action):
        applied.append((agent.name, action))
        original(agent, action)

    world.perform_agent_action = recording

    for step in range(MAX_STEPS):
        if not world.running:
            break
        applied.clear()
        world.do_step()

        order = [index_of[name] for name, _ in applied]
        order += [i for i in range(N_AGENTS) if i not in order]
        actions = np.zeros((1, N_AGENTS), dtype=np.int64)
        for name, act in applied:
            actions[0, index_of[name]] = ACTION_INDEX.get(act, ACTION_INDEX['WAIT'])
        env.order_in[0] = order
        env.step(actions, autoreset=False)

        problems = compare(world, env, index_of, step, canonical)
        if problems:
            return step, problems
        if bool(env.env_done[0]) != (not world.running):
            return step, [f'round end: framework={not world.running} '
                          f'fast={bool(env.env_done[0])}']
    if verbose:
        print(f'  seed {seed}: {world.step} steps, '
              f'scores {[a.score for a in world.agents]}')
    return None, []


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--scenario', default='classic', choices=list(s.SCENARIOS))
    p.add_argument('--agent', default='random_agent',
                   help='')
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--canonical', action='store_true',
                   help='')
    args = p.parse_args(argv)

    os.makedirs('logs', exist_ok=True)
    failures = 0
    for seed in range(args.rounds):
        step, problems = run_round(seed, args.scenario, not args.quiet, args.agent,
                                   args.canonical)
        if problems:
            failures += 1
            print(f'MISMATCH seed={seed} step={step}:')
            for pr in problems:
                print(f'    {pr}')
    if failures:
        print(f'\n{failures}/{args.rounds} rounds diverged')
        return 1
    print(f'\nOK: {args.rounds} rounds of "{args.scenario}" reproduced '
          f'exactly (state, scores, observations)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
