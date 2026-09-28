
import argparse
import copy
import json
import os
import sys

if __package__ in (None, ''):  # allow `python escape_oracle.py` from inside the dir
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    __package__ = 'agent_code.lin_agent'

from .config import (  # noqa: E402
    A_BOMB,
    FORCED_BOMB_HORIZON,
    MAX_STEPS,
    N_ACTIONS,
    S1_COIN_COUNT,
    S1_CRATE_DENSITY,
    S2_COIN_COUNT,
    S2_CRATE_DENSITY,
)
from .simenv import SoloWorld  # noqa: E402

SKIPPED = -1


def snapshot(world) -> dict:
    return {
        'field': world.field.copy(),
        'coins': world.coins.copy(),
        'x': world.x, 'y': world.y,
        'bombs': copy.deepcopy(world.bombs),
        'explosions': copy.deepcopy(world.explosions),
        'bombs_left': world.bombs_left,
        'dead': world.dead,
        'score': world.score,
        'step': world.step,
        'running': world.running,
        'stats': dict(world.stats),
    }


def restore(world, state):
    world.field = state['field'].copy()
    world.coins = state['coins'].copy()
    world.x, world.y = state['x'], state['y']
    world.bombs = copy.deepcopy(state['bombs'])
    world.explosions = copy.deepcopy(state['explosions'])
    world.bombs_left = state['bombs_left']
    world.dead = state['dead']
    world.score = state['score']
    world.step = state['step']
    world.running = state['running']
    world.stats = dict(state['stats'])


def escapable(world, state, dropped_at: int, horizon: int) -> bool:
    seen = set()

    def walk(st) -> bool:
        if st['dead']:
            return False
        elapsed = st['step'] - dropped_at
        if elapsed >= horizon:
            return True
        if not st['running']:
            return True                       # alive and the round ended
        key = (st['x'], st['y'], st['step'])
        if key in seen:
            return False
        seen.add(key)
        for action in range(N_ACTIONS):
            restore(world, st)
            world.step_action(action)
            if walk(snapshot(world)):
                return True
        return False

    return walk(state)


def board_verdict(world, seed: int, horizon: int = FORCED_BOMB_HORIZON) -> int:
    world.reset(seed)
    if not (world.bombs_left and world.safe_here()):
        return SKIPPED
    dropped_at = world.step + 1
    world.step_action(A_BOMB)
    return int(escapable(world, snapshot(world), dropped_at, horizon))


class FeasibilityCache:

    def __init__(self, path='', horizon=FORCED_BOMB_HORIZON,
                 crate_density=S1_CRATE_DENSITY, coin_count=S1_COIN_COUNT,
                 max_steps=MAX_STEPS):
        self.path = path
        self.horizon = int(horizon)
        self.crate_density = float(crate_density)
        self.coin_count = int(coin_count)
        self.max_steps = int(max_steps)
        self.verdicts = {}
        self.computed = 0
        self._world = None
        if path and os.path.isfile(path):
            self.load(path)

    @property
    def meta(self) -> dict:
        return {'horizon': self.horizon, 'crate_density': self.crate_density,
                'coin_count': self.coin_count, 'max_steps': self.max_steps}

    def load(self, path):
        with open(path) as fh:
            blob = json.load(fh)
        meta = blob.get('meta', {})
        if meta and meta != self.meta:
            raise ValueError(f'{path} holds verdicts for {meta}, not for {self.meta}')
        self.verdicts = {int(k): int(v) for k, v in blob.get('verdicts', {}).items()}

    def save(self, path=''):
        path = path or self.path
        if not path:
            return {}
        blob = {'meta': self.meta, 'summary': self.summary(),
                'verdicts': {str(k): v for k, v in sorted(self.verdicts.items())},
                'method': 'exhaustive search over action sequences inside '
                          'lin_agent.simenv.SoloWorld, memoised on (x, y, step); '
                          'success = alive `horizon` steps after the forced bomb, '
                          'identical to train_lin.forced_bomb_trial'}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(blob, fh, indent=2)
        return blob

    def summary(self, seeds=None) -> dict:
        keys = list(self.verdicts) if seeds is None else list(seeds)
        vals = [self.verdicts[k] for k in keys if k in self.verdicts]
        skipped = sum(1 for v in vals if v == SKIPPED)
        trials = len(vals) - skipped
        feasible = sum(1 for v in vals if v == 1)
        return {'boards': len(vals), 'skipped': skipped, 'trials': trials,
                'escapable': feasible,
                'ceiling': (feasible / trials) if trials else 0.0}

    def verdict(self, seed: int) -> int:
        seed = int(seed)
        if seed not in self.verdicts:
            if self._world is None:
                self._world = SoloWorld(crate_density=self.crate_density,
                                        coin_count=self.coin_count,
                                        max_steps=self.max_steps,
                                        feature_mode='graded')
            self.verdicts[seed] = board_verdict(self._world, seed, self.horizon)
            self.computed += 1
        return self.verdicts[seed]

    def warm(self, seeds) -> dict:
        seeds = [int(s) for s in seeds]
        for seed in seeds:
            self.verdict(seed)
        return self.summary(seeds)


def main(argv=None):
    from .train_lin import FORCED_S2_SEED_BASE, FORCED_SEED_BASE

    stages = {
        's1': (FORCED_SEED_BASE, S1_CRATE_DENSITY, S1_COIN_COUNT),
        's2': (FORCED_S2_SEED_BASE, S2_CRATE_DENSITY, S2_COIN_COUNT),
    }
    p = argparse.ArgumentParser(description='')
    p.add_argument('--stage', default='s1', choices=sorted(stages),
                   help='')
    p.add_argument('--seed-base', type=int, default=None)
    p.add_argument('--n', type=int, default=500)
    p.add_argument('--horizon', type=int, default=FORCED_BOMB_HORIZON)
    p.add_argument('--crate-density', type=float, default=None)
    p.add_argument('--coin-count', type=int, default=None)
    p.add_argument('--out', default='')
    p.add_argument('--expect-ceiling', type=float, default=None,
                   help='')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args(argv)
    seed_base, density, coins = stages[args.stage]
    if args.seed_base is not None:
        seed_base = args.seed_base
    if args.crate_density is not None:
        density = args.crate_density
    if args.coin_count is not None:
        coins = args.coin_count

    cache = FeasibilityCache(args.out, horizon=args.horizon,
                             crate_density=density, coin_count=coins)
    seeds = [seed_base + i for i in range(args.n)]
    summary = cache.warm(seeds)
    cache.save()
    summary['stage'] = args.stage
    summary['crate_density'] = density
    summary['seed_base'] = seed_base
    summary['computed_now'] = cache.computed
    if not args.quiet:
        print(json.dumps(summary, indent=2))

    if args.expect_ceiling is not None:
        got = summary['escapable'] / summary['boards'] if summary['boards'] else 0.0
        if abs(got - args.expect_ceiling) > 1e-12:
            print(f'FAIL: ceiling {summary["escapable"]}/{summary["boards"]} = {got!r} '
                  f'!= expected {args.expect_ceiling!r}. The oracle drifted, and '
                  f'every fb_rate_norm in this bundle depends on it.')
            return 1
        print(f'OK: ceiling reproduced exactly ({summary["escapable"]}/'
              f'{summary["boards"]} = {got})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
