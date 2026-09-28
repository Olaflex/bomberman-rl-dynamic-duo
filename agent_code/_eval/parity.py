
import json
import os
import subprocess
import sys
import tempfile
from typing import List

from .runner import run_match

IGNORED_FIELDS = ('time',)


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in IGNORED_FIELDS}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _round_values(results):
    by_round = results.get('by_round', {})

    def number(key):
        head = key.split('(')[0].strip()
        return int(head.rsplit(' ', 1)[-1])

    return [by_round[k] for k in sorted(by_round, key=number)]


def derive_from_rows(result) -> dict:
    by_agent = {}
    for row in result.rows:
        a = by_agent.setdefault(row.agent, {
            'score': 0, 'rounds': 0, 'steps': 0, 'coins': 0, 'kills': 0,
            'suicides': 0, 'crates': 0, 'bombs': 0, 'moves': 0, 'invalid': 0,
        })
        a['score'] += row.score
        a['rounds'] += 1
        for key in ('steps', 'coins', 'kills', 'suicides', 'crates', 'bombs',
                    'moves', 'invalid'):
            a[key] += getattr(row, key)

    by_round = []
    for board in result.boards:
        rows = [r for r in result.rows if r.round == board.round]
        by_round.append({
            'steps': rows[0].round_steps if rows else 0,
            'coins': sum(r.coins for r in rows),
            'kills': sum(r.kills for r in rows),
            'suicides': sum(r.suicides for r in rows),
        })
    return {'by_agent': by_agent, 'by_round': by_round}


def _drop_zero_counters(by_agent):
    return {name: {k: v for k, v in stats.items() if v}
            for name, stats in by_agent.items()}


DEFAULT_AGENTS = ['lin_agent'] * 4


def run_parity(agents: List[str] = None, n_rounds: int = 10,
               scenario: str = 'classic', seed: int = 42,
               log_dir: str = 'logs') -> dict:
    agents = list(agents or DEFAULT_AGENTS)
    tmp = tempfile.mkdtemp(prefix='eval-parity-')
    in_proc_json = os.path.join(tmp, 'in_process.json')
    cli_json = os.path.join(tmp, 'cli.json')

    result = run_match(agents, n_rounds, scenario, seed, log_dir=log_dir,
                       save_stats=in_proc_json)

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cmd = [sys.executable, 'main.py', 'play', '--no-gui',
           '--n-rounds', str(n_rounds), '--scenario', scenario,
           '--seed', str(seed), '--agents', *agents,
           '--save-stats', cli_json, '--log-dir', os.path.abspath(log_dir)]
    subprocess.run(cmd, cwd=repo_root, check=True, capture_output=True)

    with open(in_proc_json) as fh:
        a = _strip(json.load(fh))
    with open(cli_json) as fh:
        b = _strip(json.load(fh))
    derived = derive_from_rows(result)

    diffs = []
    if _drop_zero_counters(a['by_agent']) != _drop_zero_counters(b['by_agent']):
        diffs.append(f'by_agent: in-process {a["by_agent"]} != cli {b["by_agent"]}')
    if _round_values(a) != _round_values(b):
        diffs.append('by_round differs between in-process and cli')
    if _drop_zero_counters(derived['by_agent']) != _drop_zero_counters(a['by_agent']):
        diffs.append(f'derived rows: {derived["by_agent"]} != engine {a["by_agent"]}')
    if derived['by_round'] != _round_values(a):
        diffs.append('derived per-round aggregates differ from the engine\'s')

    return {'match': not diffs, 'differences': diffs, 'in_process': a, 'cli': b,
            'derived': derived,
            'config': {'agents': agents, 'n_rounds': n_rounds,
                       'scenario': scenario, 'seed': seed}}
