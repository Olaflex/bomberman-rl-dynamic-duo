
import os
from typing import List, Optional, Tuple

from . import metrics as m
from .ladder import STAGES, Stage, seats, stage_scenarios
from .runner import Board, MatchResult, RoundRow, arena_digest, run_match

GAUNTLET_SCENARIO = 'classic'

GAUNTLET_OPPONENTS = 3

DEFAULT_SEED = 42


def gauntlet(agent: str, n_rounds: int = 50, seed: int = DEFAULT_SEED,
             log_dir: str = 'logs', weights: Optional[str] = None,
             paired: bool = False, tta8: bool = False) -> MatchResult:
    agents = [agent] + ['rule_based_agent'] * GAUNTLET_OPPONENTS
    return run_match(agents, n_rounds, GAUNTLET_SCENARIO, seed,
                     match_name=f'gauntlet-{agent}', log_dir=log_dir,
                     weights=weights, paired=paired, tta8=tta8)


def arena(variants: List[str], n_rounds: int = 50, seed: int = DEFAULT_SEED,
          scenario: str = GAUNTLET_SCENARIO, log_dir: str = 'logs',
          weights: Optional[str] = None, paired: bool = False,
          tta8: bool = False, rotate: bool = False) -> MatchResult:
    if not rotate:
        return run_match(list(variants), n_rounds, scenario, seed,
                         match_name='arena-' + '-'.join(variants), log_dir=log_dir,
                         weights=weights, paired=paired, tta8=tta8)

    if n_rounds < 4 or n_rounds % 4:
        raise ValueError(f'a seat-rotated arena needs a multiple of 4 rounds, '
                         f'got {n_rounds}')
    block = n_rounds // 4
    rows: List[RoundRow] = []
    boards: List[Board] = []
    order = list(variants)
    for i in range(4):
        rotated = order[i:] + order[:i]
        part = run_match(rotated, block, scenario, seed,
                         match_name='arena-rot%d-' % i + '-'.join(order),
                         log_dir=log_dir, weights=weights, paired=paired, tta8=tta8)
        offset = i * block
        for row in part.rows:
            row.round += offset
            rows.append(row)
        for board in part.boards:
            board.round += offset
            boards.append(board)
    return MatchResult(rows=rows, boards=boards, config={
        'agents': list(variants), 'n_rounds': n_rounds, 'scenario': scenario,
        'seed': seed, 'train': 0, 'match_name': 'arena-rotated',
        'weights': weights or None, 'paired': bool(paired), 'tta8': bool(tta8),
        'rotate': True, 'block': block,
    }, arena_hashes=[arena_digest(b) for b in boards])


def probe(agent: str, stage: str = 'S0', n_rounds: int = 200,
          seed: int = DEFAULT_SEED, log_dir: str = 'logs',
          weights: Optional[str] = None, paired: bool = False,
          tta8: bool = False) -> MatchResult:
    st: Stage = STAGES[stage]
    with stage_scenarios(st) as scenario:
        return run_match(seats(st, agent), n_rounds, scenario, seed,
                         match_name=f'probe-{stage}-{agent}', log_dir=log_dir,
                         weights=weights, paired=paired, tta8=tta8)


def checkpoints(ckpt_dir: str, pattern: str = 'ckpt_*.pt') -> List[Tuple[int, str]]:
    import glob
    import re
    out = []
    for path in glob.glob(os.path.join(ckpt_dir, pattern)):
        m = re.search(r'ckpt_(\d+)\.pt$', os.path.basename(path))
        out.append((int(m.group(1)) if m else -1, path))
    return sorted(out)


def summarise(result: MatchResult, agent: Optional[str] = None) -> dict:
    if agent is None:
        return m.aggregate_match(result)
    return m.aggregate(result.for_agent(agent), result.boards)
