
import os
from typing import Dict, Optional, Sequence


def _pyplot():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except Exception:  # noqa: BLE001 - a missing plot must never fail a run
        return None


def sample_efficiency(curves: Dict[str, dict], path: str, gate_ratio: float = 1.3,
                      gate_invalid: float = 0.005,
                      floor: Optional[Dict[str, Sequence[float]]] = None) -> bool:
    plt = _pyplot()
    if plt is None:
        return False
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)

    def draw(ax, name, c, key, lo, hi, dashed=False):
        ax.plot(c['episodes'], c[key], label=name, linestyle='--' if dashed else '-')
        if lo in c and hi in c:
            ax.fill_between(c['episodes'], c[lo], c[hi], alpha=0.15)

    for name, c in curves.items():
        draw(axes[0], name, c, 'median', 'lo', 'hi')
        draw(axes[1], name, c, 'invalid_median', 'invalid_lo', 'invalid_hi')
    if floor:
        draw(axes[0], floor.get('name', 'ctrl_blind'), floor, 'median', 'lo', 'hi', True)
        draw(axes[1], floor.get('name', 'ctrl_blind'), floor, 'invalid_median',
             'invalid_lo', 'invalid_hi', True)

    axes[0].axhline(gate_ratio, color='k', lw=0.8, ls=':')
    axes[0].set_ylabel('steps-per-coin / GBR')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=8)
    axes[0].set_title('S0 sample efficiency (median over seeds, IQR band)')

    axes[1].axhline(gate_invalid, color='k', lw=0.8, ls=':')
    axes[1].set_ylabel('invalid actions / step')
    axes[1].set_xlabel('training episodes')
    axes[1].set_xscale('log')

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def ladder_bars(values: Dict[str, float], path: str, ylabel: str = '',
                title: str = '', hline: Optional[float] = None) -> bool:
    plt = _pyplot()
    if plt is None or not values:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(list(values), [v if v is not None else 0.0 for v in values.values()])
    if hline is not None:
        ax.axhline(hline, color='k', lw=0.8, ls=':')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis='x', labelrotation=30)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True
