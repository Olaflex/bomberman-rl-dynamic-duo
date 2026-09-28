
import argparse
import json
import os
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    __package__ = 'agent_code.lin_agent'

from .config import (  # noqa: E402
    A_BOMB,
    A_WAIT,
    CONJ_SLOT,
    DANGER_SLOT,
    D_NAV,
    PLACE_SLOT,
    ROOM_SLOT,
    N_ACTIONS,
    dim_for,
)
from .escape_oracle import FeasibilityCache  # noqa: E402
from .features import CONJ_PLACE_OUT_INDEX, ESCAPE_OUT_INDEX  # noqa: E402
from .model import LinearQ  # noqa: E402
from .train_lin import (  # noqa: E402
    bomb_margin_probe,
    build_parser,
    e_gate,
    forced_bomb_probe,
    forced_trials_for,
    gate_all,
    jsonable,
    probe_s0,
    probe_s1,
    probe_s2,
    self_bomb_stats,
)

BLOCKED_PENALTY = -5.0

COIN_BONUS = 1.0

IDLE_PENALTY = -0.5

BOMB_PENALTY = -1.0

DANGER_TARGET_PENALTY = -4.0

DANGER_HERE_URGENCY = 3.0

DANGER_HERE_STAY = -3.0

ROOM_BONUS = 0.5

BOMB_BIAS_ADDITIVE = -1.0

BOMB_AVAILABLE_BONUS = 0.4

CRATES_IN_BLAST_BONUS = 1.2

POST_BOMB_ROOM_BONUS = 0.4

BOMB_BIAS_ESC = -1.5

ESCAPE_DIST_BONUS = 0.8

BOMB_BIAS_CONJ = -0.6

CONJ_BONUS = 1.6

BOMB_BIAS_CONJ_PLACE = -0.6

CONJ_PLACE_BONUS = 1.6

BOMBING_MODES = ('auto', 'none', 'place', 'conj', 'esc', 'conjesc')


def bombing_for(mode: str, bombing: str = 'auto') -> str:
    if bombing == 'auto':
        return {'place': 'place', 'conj': 'conj', 'esc': 'esc',
                'conjesc': 'conjesc'}.get(mode, 'none')
    if bombing == 'place' and dim_for(mode) <= PLACE_SLOT + 1:
        raise ValueError(f'mode {mode!r} has no placement slots to bomb from')
    if bombing == 'conj' and mode != 'conj':
        raise ValueError(f'mode {mode!r} has no conjunction slot')
    if bombing == 'esc' and mode != 'esc':
        raise ValueError(f'mode {mode!r} has no escape_dist slot')
    if bombing == 'conjesc' and mode != 'conjesc':
        raise ValueError(f'mode {mode!r} has no conj_place slot')
    return bombing


def hand_weights(mode, bombing: str = 'auto') -> LinearQ:
    dim = dim_for(mode)
    bombing = bombing_for(mode, bombing)
    w = np.zeros((N_ACTIONS, dim))
    for a in range(4):
        w[a, a] = BLOCKED_PENALTY
        w[a, 4 + a] = COIN_BONUS
    w[A_WAIT, D_NAV - 1] = IDLE_PENALTY      # the bias slot
    w[A_BOMB, D_NAV - 1] = BOMB_PENALTY

    if dim <= D_NAV:
        return LinearQ(w, dim=dim)

    for a in range(4):
        w[a, DANGER_SLOT] = DANGER_HERE_URGENCY
    w[A_WAIT, DANGER_SLOT] = DANGER_HERE_STAY
    w[A_BOMB, DANGER_SLOT] = DANGER_HERE_STAY

    if dim >= DANGER_SLOT + 5:               # the four neighbour danger slots
        for a in range(4):
            w[a, DANGER_SLOT + 1 + a] = DANGER_TARGET_PENALTY
    if dim >= ROOM_SLOT + 4:
        for a in range(4):
            w[a, ROOM_SLOT + a] = ROOM_BONUS

    if bombing == 'place':
        w[A_BOMB, D_NAV - 1] = BOMB_BIAS_ADDITIVE
        w[A_BOMB, D_NAV - 2] = BOMB_AVAILABLE_BONUS      # slot 9
        w[A_BOMB, PLACE_SLOT] = CRATES_IN_BLAST_BONUS
        w[A_BOMB, PLACE_SLOT + 1] = POST_BOMB_ROOM_BONUS
    elif bombing == 'conj':
        w[A_BOMB, D_NAV - 1] = BOMB_BIAS_CONJ
        w[A_BOMB, CONJ_SLOT] = CONJ_BONUS
    elif bombing == 'esc':
        w[A_BOMB, D_NAV - 1] = BOMB_BIAS_ESC
        w[A_BOMB, D_NAV - 2] = BOMB_AVAILABLE_BONUS      # slot 9
        w[A_BOMB, PLACE_SLOT] = CRATES_IN_BLAST_BONUS
        w[A_BOMB, ESCAPE_OUT_INDEX] = ESCAPE_DIST_BONUS  # slot id 27, index 26
    elif bombing == 'conjesc':
        w[A_BOMB, D_NAV - 1] = BOMB_BIAS_CONJ_PLACE
        w[A_BOMB, CONJ_PLACE_OUT_INDEX] = CONJ_PLACE_BONUS  # slot id 28, index 26
    return LinearQ(w, dim=dim)


def main(argv=None):
    parser = build_parser()
    parser.add_argument('--json', default='', help='')
    parser.add_argument('--bombing', default='auto', choices=list(BOMBING_MODES),
                        help='')
    args = parser.parse_args(argv)   # None means sys.argv, not "no arguments"

    oracle = FeasibilityCache(args.escape_feasible, horizon=args.forced_horizon,
                              crate_density=args.crate_density,
                              coin_count=args.coin_count, max_steps=args.max_steps)
    oracle_s2 = FeasibilityCache(args.escape_feasible_s2,
                                 horizon=args.forced_horizon,
                                 crate_density=args.s2_crate_density,
                                 coin_count=args.s2_coin_count,
                                 max_steps=args.max_steps)
    bombing = bombing_for(args.features, args.bombing)
    model = hand_weights(args.features, bombing)
    s2 = probe_s2(model, args, args.probe_episodes_final)
    sb = (self_bomb_stats(model, args, args.sb_episodes_final, stage='S2')
          if args.sb_episodes_final else None)
    s1 = probe_s1(model, args, args.probe_episodes_final)
    s0 = probe_s0(model, args, args.probe_episodes_final)
    forced = forced_bomb_probe(model, args, args.forced_trials_final, oracle,
                               stage='S1')
    forced_s2 = forced_bomb_probe(model, args,
                                  forced_trials_for(args,
                                                    args.forced_trials_final,
                                                    True),
                                  oracle_s2, stage='S2')
    margin = bomb_margin_probe(model, args, args.probe_episodes_final)
    if args.escape_feasible:
        oracle.save()
    if args.escape_feasible_s2:
        oracle_s2.save()

    result = {
        'features': args.features,
        'bombing': bombing,
        'dim': model.dim,
        'self_bomb': jsonable(sb) if sb else None,
        'sb_escape_norm': (sb or {}).get('sb_escape_norm'),
        'sb_placement_ok': (sb or {}).get('sb_placement_ok'),
        'sb_feasible': (sb or {}).get('sb_feasible'),
        'sb_bombs': (sb or {}).get('sb_bombs'),
        'suicides_per_bomb_s2': s2.get('suicides_per_bomb'),
        'e_gate': e_gate(s2, sb, args),
        's2': jsonable(s2),
        's1': jsonable(s1),
        's0': jsonable(s0),
        'forced_bomb': jsonable(forced),
        'forced_bomb_s2': jsonable(forced_s2),
        'bomb_margin': jsonable(margin),
        'gate_s2': bool(s2['gate']),
        'gate_s1': bool(s1['gate']),
        'gate_s0': bool(s0['gate']),
        'gate_all': bool(gate_all(s2, s1, s0, args)),
        'crates_per_episode_s2': s2['crates'],
        'bombs_per_episode_s2': s2['bombs'],
        'suicides_per_episode_s2': s2['suicides'],
        'forced_escape_rate': forced['rate_raw'],
        'forced_escape_rate_norm': forced['rate_norm'],
        'forced_escape_rate_norm_s2': forced_s2['rate_norm'],
        'forced_unescapable': forced['unescapable'],
        'forced_unescapable_s2': forced_s2['unescapable'],
        'containment_violations': forced['containment_violations'],
        'containment_violations_s2': forced_s2['containment_violations'],
        'propensity_rule': '',
        'threshold_rule': '',
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w') as fh:
            fh.write(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
