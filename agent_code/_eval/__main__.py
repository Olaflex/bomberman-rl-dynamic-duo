
import argparse
import csv
import json
import os
from typing import List

import numpy as np

from . import metrics as m
from . import plots
from .ladder import STAGES
from .parity import DEFAULT_AGENTS, run_parity
from . import stats as st
from .runner import MatchResult, assert_paired
from .suites import DEFAULT_SEED, arena, checkpoints, gauntlet, probe

REFERENCE_AGENTS = ['coin_collector_agent', 'rule_based_agent', 'random_agent']


def write_rows(result: MatchResult, path: str):
    rows = [r.as_dict(with_think=True) for r in result.rows]
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_arenas(result: MatchResult, path: str):
    if not result.arena_hashes:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['round', 'arena_sha256_16'])
        for board, digest in zip(result.boards, result.arena_hashes):
            writer.writerow([board.round, digest])


def write_latency(result: MatchResult, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['round', 'agent', 'call', 'seconds'])
        for row in result.rows:
            for i, t in enumerate(row.think):
                writer.writerow([row.round, row.agent, i, f'{t:.6f}'])


def latency_summary(result: MatchResult) -> dict:
    out = {}
    for name in result.agent_names:
        vals = np.array([t for r in result.rows if r.agent == name for t in r.think])
        if vals.size:
            out[name] = {'calls': int(vals.size), 'p50': float(np.percentile(vals, 50)),
                         'p95': float(np.percentile(vals, 95)), 'max': float(vals.max())}
    return out


def emit(result: MatchResult, out_dir: str, label: str) -> dict:
    table = m.aggregate_match(result)
    latency = latency_summary(result)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        write_rows(result, os.path.join(out_dir, 'rows.csv'))
        write_latency(result, os.path.join(out_dir, 'latency.csv'))
        write_arenas(result, os.path.join(out_dir, 'arenas.csv'))
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as fh:
            json.dump({'label': label, 'config': result.config, 'metrics': table,
                       'latency': latency}, fh, indent=2, default=float)
        with open(os.path.join(out_dir, 'metrics.md'), 'w') as fh:
            fh.write(f'### {label}\n\n`{result.config}`\n\n')
            fh.write(m.format_table(table))
            fh.write('\n\n#### act() latency (seconds)\n\n')
            fh.write('| agent | calls | p50 | p95 | max |\n| --- | --- | --- | --- | --- |\n')
            for name, s in latency.items():
                fh.write(f'| {name} | {s["calls"]} | {s["p50"]:.5f} | '
                         f'{s["p95"]:.5f} | {s["max"]:.5f} |\n')
    print(f'\n### {label}\n')
    print(m.format_table(table))
    for name, s in latency.items():
        print(f'  latency[{name}]: p50={s["p50"] * 1e3:.2f} ms  '
              f'p95={s["p95"] * 1e3:.2f} ms  max={s["max"] * 1e3:.2f} ms  '
              f'({s["calls"]} calls)')
    return table


def cmd_probe(args) -> int:
    result = probe(args.agent, args.stage, args.rounds, args.seed, args.log_dir,
                   weights=args.weights, paired=args.paired, tta8=args.tta8)
    emit(result, args.out_dir, f'probe {args.stage} — {args.agent} '
                               f'(N={args.rounds}, seed={args.seed})')
    return 0


def cmd_calibrate(args) -> int:
    table = {}
    for agent in REFERENCE_AGENTS:
        result = probe(agent, 'S0', args.rounds, args.seed, args.log_dir)
        out_dir = os.path.join(args.out_dir, agent) if args.out_dir else ''
        table[agent] = emit(result, out_dir, f'S0 reference — {agent}')[agent]

    cc = table.get('coin_collector_agent', {}).get('ratio')
    gate = 1.3 if (cc is not None and cc <= 1.3) else \
        (max(1.3, cc * 1.05) if cc is not None else None)
    verdict = {
        'references': table,
        'coin_collector_ratio': cc,
        'gate_ratio': gate,
        'gate_invalid': 0.005,
        'rule': 'keep 1.3 if coin_collector_agent lands inside it, '
                'else max(1.3, ratio(coin_collector) * 1.05)',
    }
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, 'gate_calibration.json'), 'w') as fh:
            json.dump(verdict, fh, indent=2, default=float)
    print(f'\ncalibrated gate: ratio <= {gate} (coin_collector_agent at {cc})')
    return 0


def cmd_gauntlet(args) -> int:
    result = gauntlet(args.agent, args.rounds, args.seed, args.log_dir,
                      weights=args.weights, paired=args.paired, tta8=args.tta8)
    emit(result, args.out_dir, f'gauntlet — {args.agent} vs 3x rule_based_agent '
                               f'(N={args.rounds}, seed={args.seed}'
                               f'{", paired" if args.paired else ""}'
                               f'{", tta8" if args.tta8 else ""})')
    return 0


def cmd_pair(args) -> int:
    runs = {}
    for label, agent, weights in (('a', args.agent, args.weights),
                                  ('b', args.baseline, args.baseline_weights)):
        runs[label] = gauntlet(agent, args.rounds, args.seed, args.log_dir,
                               weights=weights, paired=args.paired,
                               tta8=args.tta8 and label == 'a')
    try:
        assert_paired(runs['a'], runs['b'])
    except ValueError as exc:
        print(f'NOT PAIRED: {exc}')
        return 1

    name_a = runs['a'].agent_names[0]
    name_b = runs['b'].agent_names[0]
    scores_a = runs['a'].scores(name_a)
    scores_b = runs['b'].scores(name_b)
    verdict = (st.paired_delta(scores_a, scores_b) if args.paired
               else st.delta(scores_a, scores_b))
    verdict['paired'] = bool(args.paired)
    verdict['agent'] = args.agent
    verdict['baseline'] = args.baseline
    verdict['rounds'] = args.rounds
    verdict['seed'] = args.seed
    verdict['tta8'] = bool(args.tta8)
    verdict['caveat'] = ('pairing fixes the arena, not the opponents: '
                         'rule_based_agent reseeds from OS entropy, a residual '
                         'that is symmetric and unbiased between the two sides')

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        emit(runs['a'], os.path.join(args.out_dir, args.agent), f'gauntlet — {args.agent}')
        emit(runs['b'], os.path.join(args.out_dir, args.baseline),
             f'gauntlet — {args.baseline}')
        with open(os.path.join(args.out_dir, 'paired_delta.json'), 'w') as fh:
            json.dump(verdict, fh, indent=2, default=float)
    print('\n' + st.format_delta(args.agent, args.baseline, verdict))
    print(f"  {'paired' if args.paired else 'UNPAIRED (pooled-SEM z)'} "
          f"over N={args.rounds} rounds, seed {args.seed}")
    print(f"  {verdict['caveat']}")
    return 0


def cmd_sweep(args) -> int:
    ckpts = checkpoints(args.ckpt_dir)
    if args.include_final:
        final = os.path.join(args.ckpt_dir, 'model.pt')
        if os.path.isfile(final) and all(p != final for _, p in ckpts):
            ckpts.append((-1, final))
    if not ckpts:
        print(f'no checkpoints in {args.ckpt_dir}')
        return 1
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for env_steps, path in ckpts:
        result = gauntlet(args.agent, args.rounds, args.seed, args.log_dir,
                          weights=path)
        name = result.agent_names[0]
        me = m.aggregate(result.for_agent(name), result.boards)
        summary = st.summary([r.score for r in result.for_agent(name)])
        rows.append({
            'env_steps': env_steps, 'checkpoint': os.path.basename(path),
            'rounds': args.rounds, 'score_mu': summary['mu'],
            'score_sigma': summary['sigma'], 'score_sem': summary['sem'],
            'coins': me['coins'], 'kills': me['kills'], 'suicides': me['suicides'],
            'deaths_by_other': me['deaths_by_other'],
            'survival_rate': me['survival_rate'],
            'invalid_per_step': me['invalid_per_step'],
            'episode_length': me['episode_length'],
        })
        print(f'{os.path.basename(path):>22}  env_steps={env_steps:>10}  '
              f"score_mu={summary['mu']:.3f} +- {summary['sem']:.3f}  "
              f"(N={args.rounds})")

    if args.out_dir:
        path = os.path.join(args.out_dir, 'sweep.csv')
        with open(path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        curve = {r['checkpoint']: r['score_mu'] for r in rows}
        plots.ladder_bars(curve, os.path.join(args.out_dir, 'sweep.png'),
                          ylabel='gauntlet score mu',
                          title=f'matched-step curve — {args.agent}')
        print(f'\nwrote {path}')
    return 0


def cmd_arena(args) -> int:
    result = arena(args.agents, args.rounds, args.seed, args.scenario, args.log_dir,
                   weights=args.weights, paired=args.paired, tta8=args.tta8,
                   rotate=args.rotate)
    table = emit(result, args.out_dir,
                 f'arena ({args.scenario}) — {", ".join(args.agents)} '
                 f'(N={args.rounds}, seed={args.seed}'
                 f'{", seat-rotated 4x%d" % (args.rounds // 4) if args.rotate else ""}'
                 f'{", paired" if args.paired else ""})')
    if args.out_dir:
        plots.ladder_bars({k: v['score_mu'] for k, v in table.items()},
                          os.path.join(args.out_dir, 'arena_scores.png'),
                          ylabel='score mu', title=f'arena ({args.scenario})')
    return 0


def cmd_parity(args) -> int:
    verdict = run_parity(args.agents, args.rounds, args.scenario, args.seed,
                         args.log_dir)
    text = json.dumps(verdict, indent=2, default=float)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as fh:
            fh.write(text)
    print(text)
    return 0 if verdict['match'] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='python -m agent_code._eval',
                                description='')
    p.add_argument('--log-dir', default='logs')
    sub = p.add_subparsers(dest='command', required=True)

    def common(sp, rounds):
        sp.add_argument('--rounds', type=int, default=rounds)
        sp.add_argument('--seed', type=int, default=DEFAULT_SEED)
        sp.add_argument('--out-dir', default='')
        return sp

    def weighted(sp):
        sp.add_argument('--weights', default='',
                        help='')
        return sp

    def measured(sp):
        sp.add_argument('--paired', action='store_true',
                        help='')
        sp.add_argument('--tta8', action='store_true',
                        help='')
        return sp

    sp = measured(weighted(common(sub.add_parser('probe'), 200)))
    sp.add_argument('--agent', required=True)
    sp.add_argument('--stage', default='S0', choices=list(STAGES))
    sp.set_defaults(func=cmd_probe)

    sp = common(sub.add_parser('calibrate'), 200)
    sp.set_defaults(func=cmd_calibrate)

    sp = measured(weighted(common(sub.add_parser('gauntlet'), 50)))
    sp.add_argument('--agent', required=True)
    sp.set_defaults(func=cmd_gauntlet)

    sp = measured(weighted(common(sub.add_parser('pair'), 200)))
    sp.add_argument('--agent', required=True, help='')
    sp.add_argument('--baseline', default='deep_agent',
                    help='')
    sp.add_argument('--baseline-weights', default='',
                    help='')
    sp.set_defaults(func=cmd_pair)

    sp = common(sub.add_parser('sweep'), 50)
    sp.add_argument('--agent', required=True,
                    help='')
    sp.add_argument('--ckpt-dir', required=True,
                    help='')
    sp.add_argument('--include-final', action='store_true',
                    help='')
    sp.set_defaults(func=cmd_sweep)

    sp = measured(weighted(common(sub.add_parser('arena'), 50)))
    sp.add_argument('--agents', nargs='+', required=True)
    sp.add_argument('--scenario', default='classic')
    sp.add_argument('--rotate', action='store_true',
                    help='')
    sp.set_defaults(func=cmd_arena)

    sp = sub.add_parser('parity')
    sp.add_argument('--agents', nargs='+', default=DEFAULT_AGENTS,
                    help='')
    sp.add_argument('--rounds', type=int, default=10)
    sp.add_argument('--seed', type=int, default=DEFAULT_SEED)
    sp.add_argument('--scenario', default='classic')
    sp.add_argument('--out', default='')
    sp.set_defaults(func=cmd_parity)

    return p


def main(argv: List[str] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
