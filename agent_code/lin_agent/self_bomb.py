
import time

from .config import (
    A_BOMB,
    FORCED_BOMB_HORIZON,
    MAX_STEPS,
    S1_COIN_COUNT,
    S1_CRATE_DENSITY,
    S2_COIN_COUNT,
    S2_CRATE_DENSITY,
)
from .escape_oracle import escapable, snapshot
from .simenv import SoloWorld

ORACLE_CAP = 4


def self_bomb_episode(world, scratch, model, seed, horizon=FORCED_BOMB_HORIZON,
                      oracle_cap=ORACLE_CAP):
    phi = world.reset(seed)
    pending = []          # [dropped_at, feasible (bool | None)]
    out = {'bombs': 0, 'evaluated': 0, 'feasible': 0, 'survived_total': 0,
           'survived_feasible': 0, 'survived_infeasible': 0, 'capped': 0,
           'oracle_calls': 0, 'oracle_seconds': 0.0}
    calls = 0

    def settle(alive):
        keep = []
        for rec in pending:
            done = (not alive) or (world.step >= rec[0] + horizon) or (not world.running)
            if not done:
                keep.append(rec)
                continue
            survived = bool(alive)
            out['survived_total'] += int(survived)
            if rec[1] is True:
                out['survived_feasible'] += int(survived)
            elif rec[1] is False:
                out['survived_infeasible'] += int(survived)
        pending[:] = keep

    while world.running:
        action = model.greedy(phi)
        legal_bomb = (action == A_BOMB and world.bombs_left)
        dropped_at = world.step + 1
        phi, _info = world.step_action(action)
        if legal_bomb:
            out['bombs'] += 1
            feasible = None
            if oracle_cap and calls >= oracle_cap:
                out['capped'] += 1
            else:
                t0 = time.perf_counter()
                feasible = bool(escapable(scratch, snapshot(world), dropped_at,
                                          horizon))
                out['oracle_seconds'] += time.perf_counter() - t0
                calls += 1
                out['oracle_calls'] += 1
                out['evaluated'] += 1
                out['feasible'] += int(feasible)
            pending.append([dropped_at, feasible])
        settle(alive=not world.dead)

    settle(alive=not world.dead)        # the round ended: alive == survived
    st = world.stats
    out.update({'steps': world.step, 'crates': st['crates'], 'coins': st['coins'],
                'bombs_dropped': st['bombs'], 'suicide': st['suicide'],
                'invalid': st['invalid']})
    return out


def aggregate(rows, episodes) -> dict:
    tot = {k: sum(r[k] for r in rows) for k in
           ('bombs', 'evaluated', 'feasible', 'survived_total',
            'survived_feasible', 'survived_infeasible', 'capped',
            'oracle_calls', 'crates', 'coins', 'suicide', 'steps', 'invalid')}
    oracle_seconds = sum(r['oracle_seconds'] for r in rows)
    placement_ok = (tot['feasible'] / tot['evaluated']) if tot['evaluated'] else None
    escape_norm = (tot['survived_feasible'] / tot['feasible']) if tot['feasible'] else None
    escape_raw = (tot['survived_total'] / tot['bombs']) if tot['bombs'] else None
    suicides_per_bomb = (tot['suicide'] / tot['bombs']) if tot['bombs'] else None
    predicted = None
    if placement_ok is not None and escape_norm is not None:
        predicted = (1.0 - placement_ok) + placement_ok * (1.0 - escape_norm)
    return {
        'episodes': episodes,
        'sb_bombs': tot['bombs'],
        'sb_bombs_evaluated': tot['evaluated'],
        'sb_feasible': tot['feasible'],
        'sb_survived_total': tot['survived_total'],
        'sb_survived_feasible': tot['survived_feasible'],
        'sb_survived_infeasible': tot['survived_infeasible'],
        'sb_oracle_capped': tot['capped'],
        'sb_placement_ok': placement_ok,
        'sb_escape_norm': escape_norm,
        'sb_escape_raw': escape_raw,
        'sb_escape_norm_denominator': tot['feasible'],
        'crates_per_episode': tot['crates'] / episodes,
        'coins_per_episode': tot['coins'] / episodes,
        'bombs_per_episode': tot['bombs'] / episodes,
        'suicides_per_episode': tot['suicide'] / episodes,
        'suicides_per_bomb': suicides_per_bomb,
        'steps_per_episode': tot['steps'] / episodes,
        'invalid_per_step': (tot['invalid'] / tot['steps']) if tot['steps'] else 0.0,
        'decomposition_identity': {
            'suicides_per_bomb_measured': suicides_per_bomb,
            'one_minus_sb_escape_raw': (1.0 - escape_raw) if escape_raw is not None
                                       else None,
            'suicides_per_bomb_predicted': predicted,
            'note': 'predicted = (1 - sb_placement_ok) + sb_placement_ok * '
                    '(1 - sb_escape_norm) = 1 - placement_ok * escape_norm, '
                    'and it is a statement about the ORACLE-EVALUATED bombs '
                    'only. `one_minus_sb_escape_raw` is the same quantity '
                    'measured directly over all bombs and is what '
                    'suicides/bomb should track; the two differ when a bomb '
                    'window outlives the episode, when several bombs share one '
                    'death, or when the evaluated subset is not all of them '
                    '(sb_oracle_capped > 0). The gaps are reported, never '
                    'absorbed.',
        },
        'oracle_soundness': {
            'survived_after_an_infeasible_drop': tot['survived_infeasible'],
            'must_be': 0,
            'ok': tot['survived_infeasible'] == 0,
            'why': 'the oracle says NO policy survives the horizon from that '
                   'post-bomb board; if this policy did, the oracle and the '
                   'simulator disagree and every normalised number below is '
                   'void. It is a check, not a statistic.',
        },
        'oracle_calls': tot['oracle_calls'],
        'oracle_seconds': oracle_seconds,
        'oracle_ms_per_call': (1000.0 * oracle_seconds / tot['oracle_calls'])
                              if tot['oracle_calls'] else None,
    }


def self_bomb_probe(model, feature_mode, episodes, seed_base=None,
                    crate_density=S2_CRATE_DENSITY, coin_count=S2_COIN_COUNT,
                    max_steps=MAX_STEPS, horizon=FORCED_BOMB_HORIZON,
                    oracle_cap=ORACLE_CAP, progress=None) -> dict:
    if seed_base is None:
        from .train_lin import S2_PROBE_SEED_BASE
        seed_base = S2_PROBE_SEED_BASE
    world = SoloWorld(crate_density=crate_density, coin_count=coin_count,
                      max_steps=max_steps, feature_mode=feature_mode)
    scratch = SoloWorld(crate_density=crate_density, coin_count=coin_count,
                        max_steps=max_steps, feature_mode=feature_mode)
    rows = []
    for i in range(episodes):
        rows.append(self_bomb_episode(world, scratch, model, seed_base + i,
                                      horizon=horizon, oracle_cap=oracle_cap))
        if progress is not None:
            progress(i + 1, episodes)
    out = aggregate(rows, episodes)
    out['oracle_cap'] = int(oracle_cap)
    out['block'] = {'seed_base': seed_base, 'episodes': episodes,
                    'crate_density': crate_density, 'coin_count': coin_count,
                    'max_steps': max_steps, 'horizon': horizon,
                    'oracle_cap': oracle_cap, 'feature_mode': feature_mode}
    out['episodes_detail'] = rows
    return out
