
import math
from typing import Sequence


def summary(values: Sequence[float]) -> dict:
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {'n': 0, 'mu': float('nan'), 'sigma': float('nan'), 'sem': float('nan')}
    mu = sum(vals) / n
    if n == 1:
        return {'n': 1, 'mu': mu, 'sigma': 0.0, 'sem': 0.0}
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    sigma = math.sqrt(var)
    return {'n': n, 'mu': mu, 'sigma': sigma, 'sem': sigma / math.sqrt(n)}


def delta(a: Sequence[float], b: Sequence[float]) -> dict:
    sa, sb = summary(a), summary(b)
    sem = math.sqrt(sa['sem'] ** 2 + sb['sem'] ** 2)
    d = sa['mu'] - sb['mu']
    z = d / sem if sem > 0 else float('inf') if d else 0.0
    return {'delta': d, 'sem': sem, 'z': z, 'separated': abs(z) >= 2.0,
            'a': sa, 'b': sb}


def paired_delta(a: Sequence[float], b: Sequence[float]) -> dict:
    if len(a) != len(b):
        raise ValueError(f'paired samples must have equal length: {len(a)} vs {len(b)}')
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    s = summary(diffs)
    z = s['mu'] / s['sem'] if s['sem'] > 0 else float('inf') if s['mu'] else 0.0
    return {'delta': s['mu'], 'sem': s['sem'], 'z': z, 'separated': abs(z) >= 2.0,
            'a': summary(a), 'b': summary(b), 'n_pairs': len(diffs)}


def format_delta(name_a: str, name_b: str, d: dict, unit: str = '') -> str:
    verdict = 'separated' if d['separated'] else 'not separated by this many rounds'
    return (f'{name_a} - {name_b} = {d["delta"]:+.3f}{unit} +- {d["sem"]:.3f} '
            f'(z={d["z"]:+.2f}, {verdict})')
