
from typing import Dict, List, Optional

from . import oracle
from .runner import Board, MatchResult, RoundRow
from .stats import summary

METRIC_ORDER = [
    'score_mu', 'score_sigma', 'score_sem', 'coins', 'kills', 'suicides',
    'deaths_by_other', 'bombs', 'crates', 'suicides_per_bomb', 'idle_steps',
    'invalid', 'invalid_per_step', 'survival_rate',
    'steps_per_coin', 'gbr_steps_per_coin', 'ratio', 'ratio_sem',
    'spc_censored', 'episode_length', 'rounds',
]


def _boards_by_round(boards: List[Board]) -> Dict[int, Board]:
    return {b.round: b for b in boards}


def ratio_per_round(rows: List[RoundRow], boards: List[Board]) -> List[Optional[float]]:
    index = _boards_by_round(boards)
    out = []
    for row in rows:
        board = index.get(row.round)
        ref = oracle.gbr_for_board(board, row.agent) if board else None
        den = oracle.prefix_steps(ref, row.coins) if ref else None
        out.append(row.steps / den if den else None)
    return out


def aggregate(rows: List[RoundRow], boards: Optional[List[Board]] = None) -> dict:
    if not rows:
        return {}
    score = summary([r.score for r in rows])
    steps = sum(r.steps for r in rows)
    coins = sum(r.coins for r in rows)
    bombs = sum(r.bombs for r in rows)
    suicides = sum(r.suicides for r in rows)
    idle = sum(max(0, r.steps - r.moves - r.bombs) for r in rows)
    out = {
        'agent': rows[0].agent,
        'code_name': rows[0].code_name,
        'rounds': len(rows),
        'score_mu': score['mu'],
        'score_sigma': score['sigma'],
        'score_sem': score['sem'],
        'coins': coins / len(rows),
        'kills': sum(r.kills for r in rows) / len(rows),
        'suicides': suicides / len(rows),
        'deaths_by_other': sum(r.deaths_by_other for r in rows) / len(rows),
        'bombs': bombs / len(rows),
        'crates': sum(r.crates for r in rows) / len(rows),
        'suicides_per_bomb': (suicides / bombs) if bombs else None,
        'idle_steps': idle / len(rows),
        'invalid': sum(r.invalid for r in rows) / len(rows),
        'invalid_per_step': (sum(r.invalid for r in rows) / steps) if steps else 0.0,
        'survival_rate': sum(r.survived for r in rows) / len(rows),
        'episode_length': sum(r.round_steps for r in rows) / len(rows),
        'steps_per_coin': (steps / coins) if coins else None,
    }

    if boards is not None:
        ratios = [r for r in ratio_per_round(rows, boards) if r is not None]
        ref_steps = ref_coins = 0
        index = _boards_by_round(boards)
        for row in rows:
            ref = oracle.gbr_for_board(index.get(row.round), row.agent) \
                if row.round in index else None
            if ref:
                ref_steps += ref['steps']
                ref_coins += ref['coins']
        s = summary(ratios) if ratios else {'mu': None, 'sem': None}
        out['ratio'] = s['mu']
        out['ratio_sem'] = s['sem']
        out['spc_censored'] = len(rows) - len(ratios)
        out['gbr_steps_per_coin'] = (ref_steps / ref_coins) if ref_coins else None
    else:
        out['ratio'] = out['ratio_sem'] = out['gbr_steps_per_coin'] = None
        out['spc_censored'] = None
    return out


def aggregate_match(result: MatchResult) -> Dict[str, dict]:
    return {name: aggregate(result.for_agent(name), result.boards)
            for name in result.agent_names}


def format_table(metrics: Dict[str, dict], columns=None) -> str:
    columns = list(columns or METRIC_ORDER)
    head = '| agent | ' + ' | '.join(columns) + ' |'
    rule = '| --- | ' + ' | '.join(['---'] * len(columns)) + ' |'
    lines = [head, rule]
    for name, m in metrics.items():
        cells = []
        for c in columns:
            v = m.get(c)
            cells.append('-' if v is None else (f'{v:.4g}' if isinstance(v, float) else str(v)))
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)
