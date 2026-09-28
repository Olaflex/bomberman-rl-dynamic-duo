
import argparse
import contextlib
import copy
import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np
import torch

if __package__ in (None, ''):  # allow `python agent_code/<arm>/train_ppo.py`
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
    __package__ = 'agent_code.' + os.path.basename(_HERE)

from .config import (
    A_BOMB,
    BOMB_TIMER,
    CHANNELS,
    COLS,
    FORCED_BOMB_HORIZON,
    GATE_S1_BOMBS,
    GATE_S1_FORCED_SURVIVAL,
    GATE_S1_SUICIDES,
    INJECT_ANNEAL_FRAC,
    INJECT_ANNEAL_STEPS,
    LATEST_FILE,
    MAX_STEPS,
    MODEL_FILE,
    N_ACTIONS,
    N_AGENTS,
    REWARD_ANNEAL_FRAC,
    REWARD_ANNEAL_STEPS,
    ROWS,
    anneal_steps,
    get_device,
    get_num_threads,
    variant_settings,
)
from .features import D4_ACTION_INV
from .model import ActorCritic
from .vecenv import (
    DEFAULT_REWARDS,
    D1_REWARDS_FINAL,
    D1_REWARDS_START,
    N_REWARDS,
    N_STAGE_COLS,
    NOT_SAFE,
    OPP_MIXED,
    OPP_POLICY,
    OPP_RULE,
    SC_ACTIVE,
    SC_OPP,
    ST_BOMBOPP,
    ST_BOMBS,
    ST_COINS,
    ST_CRATES,
    ST_DEATH_OTHER,
    ST_INVALID,
    ST_KILLS,
    ST_SCORE,
    ST_STEPS,
    ST_SUICIDE,
    ST_SURVIVED,
    VecBomberman,
    d1_rewards_at,
)

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Stage:

    name: str
    density: float
    coins: int
    seats: int
    opponent: int
    cap: int
    gate: str = ''
    holds: tuple = dc_field(default_factory=tuple)


LADDER = (
    Stage('S0', 0.00, 50, 1, OPP_POLICY, 1_500_000,
          'steps-per-coin <= 1.3 x greedy-tour reference and invalid/step < 0.5%'),
    Stage('S1', 0.35, 9, 1, OPP_POLICY, 2_000_000,
          f'bombs/episode >= {GATE_S1_BOMBS} and suicides/episode < g and '
          f'forced-bomb survival >= {GATE_S1_FORCED_SURVIVAL} '
          f'and the S0 gate still holds', holds=('S0',)),
    Stage('S2', 0.75, 50, 1, OPP_POLICY, 2_500_000,
          'crates/episode >= 8 and the S1 and S0 gates still hold',
          holds=('S1', 'S0')),
    Stage('S3', 0.75, 9, 4, OPP_MIXED, 2_000_000,
          'kills/episode > 0.2 and deaths-by-other/episode < 0.5'),
    Stage('S4', 0.75, 9, 4, OPP_POLICY, 1 << 62,
          'terminal'),
)

NO_LADDER = (Stage('S4', 0.75, 9, 4, OPP_POLICY, 1 << 62, 'terminal'),)

SCRIPTED_ROW = Stage('S4R', 0.75, 9, 4, OPP_RULE, 1 << 62,
                     'terminal (scripted rule-based opposition)')


def d1_stages(opp_frac):
    return NO_LADDER if opp_frac <= 0 else (NO_LADDER[0], SCRIPTED_ROW)


def d1_stage_weights(n_stages, opp_frac):
    if n_stages == 1:
        return np.array([1.0])
    return np.array([1.0 - opp_frac, opp_frac], dtype=np.float64)

RB_ROW = Stage('P-RB', 0.75, 9, 4, OPP_RULE, 1 << 62,
               'probe only (classic, seats 1-3 = the in-kernel rule_based proxy)')

PROBES = (
    ('P-S0', LADDER[0]),
    ('P-S1', LADDER[1]),
    ('P-S2', LADDER[2]),
    ('P-S3', LADDER[3]),
    ('P-REF', LADDER[4]),
)

FAST_PROBES = (
    ('P-RB', RB_ROW),
    ('P-REF', LADDER[4]),
)

SLOW_PROBES = (
    ('P-S0', LADDER[0]),
    ('P-S1', LADDER[1]),
    ('P-S2', LADDER[2]),
    ('P-S3', LADDER[3]),
)


def probe_sets(name):
    if name == 'round':
        return FAST_PROBES, ()
    if name == 'long':
        return FAST_PROBES, SLOW_PROBES
    if name == 'd1':
        return PROBES, ()
    raise ValueError(f'unknown probe set {name!r}')

FORCED_BOMB_PROBE = ('P-FB', LADDER[1])


def stage_table_array(stages):
    table = np.zeros((len(stages), N_STAGE_COLS), dtype=np.float64)
    for i, st in enumerate(stages):
        table[i] = (st.density, st.coins, st.seats, st.opponent)
    return table


def mixture_weights(n_stages, stage, mix, ramp):
    w = np.zeros(n_stages, dtype=np.float64)
    if stage <= 0:
        w[0] = 1.0
        return w
    top = (1.0 - mix) * max(0.0, min(1.0, ramp))
    w[stage] = top
    w[:stage] = (1.0 - top) / stage
    return w


def _s0_ok(m):
    ratio = m.get('spc_ratio')
    inv = m.get('invalid_per_step')
    ok = ratio is not None and ratio <= 1.3 and inv is not None and inv < 0.005
    return bool(ok), {'spc_ratio': ratio, 'invalid_per_step': inv}


def _s1_ok(m, g=GATE_S1_SUICIDES):
    bombs = m.get('bombs')
    suicides = m.get('suicides')
    fb = m.get('fb_survival')
    detail = {'bombs': bombs, 'suicides': suicides, 'fb_survival': fb,
              'fb_trials': m.get('fb_trials')}
    ok = (bombs is not None and bombs >= GATE_S1_BOMBS
          and suicides is not None and suicides < g
          and fb is not None and fb >= GATE_S1_FORCED_SURVIVAL)
    return bool(ok), detail


def _s2_ok(m):
    crates = m.get('crates')
    return bool(crates is not None and crates >= 8.0), {'crates': crates}


def _s3_ok(m):
    kills = m.get('kills')
    deaths = m.get('deaths_by_other')
    ok = (kills is not None and kills > 0.2
          and deaths is not None and deaths < 0.5)
    return bool(ok), {'kills': kills, 'deaths_by_other': deaths}


STAGE_GATES = {'S0': _s0_ok, 'S1': _s1_ok, 'S2': _s2_ok, 'S3': _s3_ok}


def stage_gate_ok(name, m, g=GATE_S1_SUICIDES):
    fn = STAGE_GATES.get(name)
    if fn is None:
        return False, {}
    return fn(m, g) if name == 'S1' else fn(m)


def gate_verdict(stage, metrics, held, g=GATE_S1_SUICIDES):
    ok, detail = stage_gate_ok(stage.name, metrics, g)
    detail = dict(detail)
    for name in stage.holds:
        if name not in held:
            continue
        h_ok, h_detail = stage_gate_ok(name, held[name], g)
        ok = ok and h_ok
        for k, v in h_detail.items():
            detail[f'{name.lower()}_{k}'] = v
    return bool(ok), detail


def _env_float(name, default):
    try:
        return float(os.environ.get(name, '').strip() or default)
    except ValueError:
        return float(default)


def load_variant(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='',
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variant-json', default=os.path.join(HERE, 'variant.json'))
    p.add_argument('--out-dir', default=HERE,
                   help='')
    p.add_argument('--total-steps', type=int, default=19_500_000,
                   help='')
    p.add_argument('--hours', type=float, default=0.0,
                   help='')
    p.add_argument('--envs', type=int, default=512, help='')
    p.add_argument('--rollout', type=int, default=48, help='')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--minibatch', type=int, default=2048)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lr-final-frac', type=float, default=0.1)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lam', type=float, default=0.95)
    p.add_argument('--ent', type=float, default=0.02)
    p.add_argument('--ent-final', type=float, default=0.003)
    p.add_argument('--vf', type=float, default=0.5)
    p.add_argument('--max-grad-norm', type=float, default=0.5)
    p.add_argument('--target-kl', type=float, default=0.03,
                   help='')
    p.add_argument('--width', type=int, default=64, help='')
    p.add_argument('--blocks', type=int, default=6, help='')
    p.add_argument('--league-frac', type=float, default=0.3,
                   help='')
    p.add_argument('--snapshot-every', type=int, default=150, help='')
    p.add_argument('--snapshots', type=int, default=4, help='')
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--amp', default='bf16', choices=['off', 'bf16', 'fp16'])
    p.add_argument('--compile', action='store_true', help='')
    p.add_argument('--no-fused-adam', dest='fused_adam', action='store_false',
                   help='')
    p.set_defaults(fused_adam=True)

    p.add_argument('--rewards', default=None, choices=['d1', 'gen1'],
                   help='')
    p.add_argument('--reward-anneal-frac', type=float, default=None,
                   help='')
    p.add_argument('--reward-anneal-steps', type=int, default=None,
                   help='')
    p.add_argument('--bombopp', type=float, default=None,
                   help='')
    p.add_argument('--pot-opp', type=float, default=None,
                   help='')
    p.add_argument('--opp-frac', type=float, default=None,
                   help='')
    p.add_argument('--inject-dose', type=float, default=None,
                   help='')
    p.add_argument('--inject-anneal-frac', type=float, default=None,
                   help='')
    p.add_argument('--inject-anneal-steps', type=int, default=None,
                   help='')
    p.add_argument('--round-id', default='',
                   help='')
    p.add_argument('--ref-arm', default='',
                   help='')
    p.add_argument('--canonical', dest='canonical', action='store_true', default=None,
                   help='')
    p.add_argument('--no-canonical', dest='canonical', action='store_false',
                   help='')

    p.add_argument('--curriculum', default=None, choices=['gated', 'none'],
                   help='')
    p.add_argument('--mix', type=float, default=None,
                   help='')
    p.add_argument('--ramp-steps', type=int, default=250_000,
                   help='')
    p.add_argument('--gate-every', type=int, default=500_000, help='')
    p.add_argument('--gate-episodes', type=int, default=200)
    p.add_argument('--gate-s1-suicides', type=float, default=_env_float(
        'GATE_S1_SUICIDES', GATE_S1_SUICIDES),
        help='')
    p.add_argument('--fb-trials', type=int, default=128,
                   help='')
    p.add_argument('--probe-every', type=int, default=1_000_000, help='')
    p.add_argument('--probe-slow-every', type=int, default=5_000_000,
                   help='')
    p.add_argument('--probe-set', default='round', choices=['round', 'long', 'd1'],
                   help='')
    p.add_argument('--probe-weights', default='',
                   help='')
    p.add_argument('--probe-tag', default='',
                   help='')
    p.add_argument('--probe-episodes', type=int, default=256)
    p.add_argument('--eval-envs', type=int, default=64,
                   help='')
    p.add_argument('--eval-sample', action='store_true',
                   help='')
    p.add_argument('--ckpt-every', type=int, default=2_000_000, help='')
    p.add_argument('--resume-every', type=int, default=500_000, help='')
    p.add_argument('--ref-weights', default='',
                   help='')
    p.add_argument('--no-resume', action='store_true',
                   help='')
    p.add_argument('--bench', type=float, default=0.0,
                   help='')
    p.add_argument('--bench-json', default='')
    return p.parse_args(argv)


BASE_KNOBS = (
    'total_steps', 'envs', 'rollout', 'epochs', 'minibatch', 'lr',
    'lr_final_frac', 'clip', 'gamma', 'lam', 'ent', 'ent_final', 'vf',
    'max_grad_norm', 'target_kl', 'width', 'blocks', 'league_frac',
    'snapshot_every', 'snapshots', 'amp', 'rewards', 'reward_anneal_frac',
    'inject_anneal_frac', 'inject_dose', 'opp_frac', 'canonical', 'bombopp',
    'pot_opp', 'probe_set', 'probe_episodes', 'seed',
)


def base_knobs(args):
    return {k: getattr(args, k, None) for k in BASE_KNOBS}


class Rollout:

    def __init__(self, T, B, device):
        def z(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=device)
        self.obs = z(T, B, CHANNELS, COLS, ROWS, dtype=torch.uint8)
        self.mask = z(T, B, N_ACTIONS, dtype=torch.bool)
        self.act = z(T, B, dtype=torch.int64)
        self.logp = z(T, B)
        self.val = z(T, B)
        self.rew = z(T, B)
        self.done = z(T, B)
        self.live = z(T, B)
        self.adv = z(T, B)
        self.ret = z(T, B)


def compute_gae(buf, last_val, gamma, lam):
    T = buf.rew.shape[0]
    nextval = last_val
    lastgae = torch.zeros_like(last_val)
    for t in reversed(range(T)):
        nonterminal = 1.0 - buf.done[t]
        delta = buf.rew[t] + gamma * nextval * nonterminal - buf.val[t]
        lastgae = delta + gamma * lam * nonterminal * lastgae
        buf.adv[t] = lastgae
        nextval = buf.val[t]
    buf.ret.copy_(buf.adv + buf.val)


def report_gpu(device, args):
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    gib = 1024 ** 3
    print(f'gpu={props.name}  cc={props.major}.{props.minor}  '
          f'total={total / gib:.1f} GiB  free={free / gib:.1f} GiB  '
          f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")}')
    per_sample_mb = 2.7 * (args.width / 64) * ((args.blocks + 1) / 7)
    need = args.minibatch * per_sample_mb / 1024
    rollout_gib = (args.rollout * args.envs * N_AGENTS * CHANNELS * COLS * ROWS) / gib
    print(f'estimated update memory: {need:.1f} GiB for --minibatch {args.minibatch} '
          f'(~{per_sample_mb:.1f} MB/sample) + {rollout_gib:.2f} GiB rollout buffer')
    if need > free / gib * 0.9:
        print(f'  warning: that is close to or above the {free / gib:.1f} GiB free; '
              f'the trainer will halve the minibatch and retry if it OOMs')


def metrics_from_stats(rows, ref_spc):
    rows = np.asarray(rows, dtype=np.float64)
    if rows.size == 0:
        return {'episodes': 0}
    n = rows.shape[0]
    steps = rows[:, ST_STEPS]
    coins = rows[:, ST_COINS]
    out = {
        'episodes': int(n),
        'score_mu': float(rows[:, ST_SCORE].mean()),
        'score_sigma': float(rows[:, ST_SCORE].std(ddof=1)) if n > 1 else 0.0,
        'score_sem': float(rows[:, ST_SCORE].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        'coins': float(coins.mean()),
        'kills': float(rows[:, ST_KILLS].mean()),
        'suicides': float(rows[:, ST_SUICIDE].mean()),
        'deaths_by_other': float(rows[:, ST_DEATH_OTHER].mean()),
        'crates': float(rows[:, ST_CRATES].mean()),
        'bombs': float(rows[:, ST_BOMBS].mean()),
        'invalid': float(rows[:, ST_INVALID].mean()),
        'invalid_per_step': float(rows[:, ST_INVALID].sum() / max(steps.sum(), 1.0)),
        'survival_rate': float(rows[:, ST_SURVIVED].mean()),
        'episode_length': float(steps.mean()),
        'steps_per_coin': float(steps.sum() / coins.sum()) if coins.sum() > 0 else None,
    }
    if ref_spc is None:
        out['spc_ratio'] = None
        out['spc_ref'] = None
        out['spc_censored'] = None
        return out
    ref = np.asarray(ref_spc, dtype=np.float64)
    ok = (coins > 0) & (ref > 0)
    out['spc_censored'] = int(n - ok.sum())
    if ok.any():
        out['spc_ratio'] = float(np.mean((steps[ok] / coins[ok]) / ref[ok]))
        out['spc_ref'] = float(ref[ok].mean())
    else:
        out['spc_ratio'] = None
        out['spc_ref'] = None
    return out


@torch.no_grad()
def play_episodes(model, stage, n_episodes, device, args, seed, opponent=None,
                  autocast=contextlib.nullcontext):
    n_envs = args.eval_envs
    table = stage_table_array((stage,))
    env = VecBomberman(n_envs, stage_table=table, stage_weights=[1.0], seed=seed,
                       rewards=np.zeros(N_REWARDS), gamma=args.gamma,
                       canonical=bool(getattr(args, 'canonical', False)))
    obs_np, mask_np, _ = env.reset(stagger=False)
    B = n_envs * N_AGENTS

    want_ref = stage.density <= 0.0
    ref_now = np.zeros(n_envs, dtype=np.float64)

    def refresh_ref(idx=None):
        steps, got = env.reference_tour()
        sel = slice(None) if idx is None else idx
        ref_now[sel] = np.where(got[sel] > 0, steps[sel] / np.maximum(got[sel], 1), 0.0)

    if want_ref:
        refresh_ref()

    rows = []
    refs = []
    steps_used = 0
    guard = 0
    while len(rows) < n_episodes and guard < 20_000:
        guard += 1
        obs_t = torch.from_numpy(obs_np.reshape(B, CHANNELS, COLS, ROWS)).to(device)
        mask_t = torch.from_numpy(mask_np.reshape(B, N_ACTIONS)).to(device).bool()
        with autocast():
            logits, _ = model(obs_t, mask_t)
        logits = logits.float()
        if args.eval_sample:
            action = torch.distributions.Categorical(logits=logits).sample()
        else:
            action = logits.argmax(-1)
        act_np = action.to('cpu').numpy().reshape(n_envs, N_AGENTS)
        if env.canonical:
            act_np = D4_ACTION_INV[env.gsel, act_np]

        if opponent is not None:
            with autocast():
                olog, _ = opponent(obs_t, mask_t)
            olog = olog.float()
            if args.eval_sample:
                oact = torch.distributions.Categorical(logits=olog).sample()
            else:
                oact = olog.argmax(-1)
            oact_np = oact.to('cpu').numpy().reshape(n_envs, N_AGENTS)
            if env.canonical:
                oact_np = D4_ACTION_INV[env.gsel, oact_np]
            act_np[:, 1:] = oact_np[:, 1:]

        obs_np, mask_np, _, _, _, env_done = env.step(act_np)
        steps_used += n_envs
        done_idx = np.nonzero(env_done)[0]
        if len(done_idx):
            finished = env.last_stats[done_idx][:, 0, :]
            rows.extend(finished.tolist())
            refs.extend(ref_now[done_idx].tolist())
            if want_ref:
                refresh_ref(done_idx)
    rows = rows[:n_episodes]
    refs = refs[:n_episodes] if want_ref else None
    return metrics_from_stats(rows, refs), steps_used


def injection_fire(pending, may_bomb, escape):
    return pending & may_bomb & (escape != NOT_SAFE) & (escape <= BOMB_TIMER)


def inject_dose_at(env_steps, dose0, anneal_steps):
    if dose0 <= 0 or anneal_steps <= 0:
        return 0.0
    return dose0 * max(0.0, 1.0 - env_steps / anneal_steps)


def mask_injected(live_mask, fire):
    live_mask[fire, 0] = False
    return live_mask


@torch.no_grad()
def forced_bomb_trials(model, stage, n_trials, device, args, seed,
                       horizon=FORCED_BOMB_HORIZON,
                       autocast=contextlib.nullcontext):
    n_envs = args.eval_envs
    env = VecBomberman(n_envs, stage_table=stage_table_array((stage,)),
                       stage_weights=[1.0], seed=seed,
                       rewards=np.zeros(N_REWARDS), gamma=args.gamma,
                       canonical=bool(getattr(args, 'canonical', False)))
    obs_np, mask_np, _ = env.reset(stagger=False)
    B = n_envs * N_AGENTS

    waiting, counting, spent = 0, 1, 2          # per-game trial state
    state = np.zeros(n_envs, dtype=np.int64)
    left = np.zeros(n_envs, dtype=np.int64)
    declined = np.zeros(n_envs, dtype=bool)     # already counted as unescapable
    trials = survived = no_chance = unescapable = 0
    episodes = n_envs
    steps_used = 0
    guard = 0
    while trials < n_trials and guard < 8 * MAX_STEPS:
        guard += 1
        obs_t = torch.from_numpy(obs_np.reshape(B, CHANNELS, COLS, ROWS)).to(device)
        mask_t = torch.from_numpy(mask_np.reshape(B, N_ACTIONS)).to(device).bool()
        with autocast():
            logits, _ = model(obs_t, mask_t)
        logits = logits.float()
        if args.eval_sample:
            action = torch.distributions.Categorical(logits=logits).sample()
        else:
            action = logits.argmax(-1)
        act_np = action.to('cpu').numpy().reshape(n_envs, N_AGENTS)
        if env.canonical:
            act_np = D4_ACTION_INV[env.gsel, act_np]

        escape = env.bomb_escape_distance()[:, 0]
        may = mask_np[:, 0, A_BOMB] != 0
        offered = (state == waiting) & may & (escape != NOT_SAFE)
        arm = injection_fire(state == waiting, may, escape)
        skipped = offered & ~arm & ~declined    # count a game once, not per step
        unescapable += int(skipped.sum())
        declined |= skipped
        act_np[arm, 0] = A_BOMB
        state[arm] = counting
        left[arm] = horizon

        obs_np, mask_np, _, _, agent_done, env_done = env.step(act_np)
        steps_used += n_envs

        ended = env_done != 0
        died_now = np.where(ended, env.last_stats[:, 0, ST_SURVIVED] == 0,
                            agent_done[:, 0] != 0)
        live = state == counting
        died = live & died_now
        left[live & ~died] -= 1
        made_it = live & ~died & ((left <= 0) | ended)
        trials += int(died.sum()) + int(made_it.sum())
        survived += int(made_it.sum())
        state[died | made_it] = spent

        fin = np.nonzero(env_done)[0]
        if len(fin):
            episodes += len(fin)
            no_chance += int((state[fin] == waiting).sum())
            state[fin] = waiting
            left[fin] = 0
            declined[fin] = False
        if trials < n_trials and not np.any(state == counting) and np.any(state == spent):
            no_chance += int((state == waiting).sum())
            obs_np, mask_np, _ = env.reset(stagger=False)
            episodes += n_envs
            state[:] = waiting
            left[:] = 0
            declined[:] = False

    return {
        'fb_trials': trials,
        'fb_survived': survived,
        'fb_survival': (survived / trials) if trials else None,
        'fb_episodes': episodes,
        'fb_no_opportunity': no_chance,
        'fb_unescapable': unescapable,
        'fb_horizon': horizon,
    }, steps_used


def open_appending_csv(path, header):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    fh = open(path, 'a', newline='')  # noqa: SIM115 - open for the whole run
    writer = csv.writer(fh)
    if fresh:
        writer.writerow(header)
        fh.flush()
    return fh, writer


PROBE_COLUMNS = [
    'env_steps', 'update', 'probe', 'episodes', 'score_mu', 'score_sigma',
    'score_sem', 'coins', 'kills', 'suicides', 'deaths_by_other', 'crates',
    'bombs', 'invalid', 'invalid_per_step', 'survival_rate', 'episode_length',
    'steps_per_coin', 'spc_ref_greedy_tour_approx', 'spc_ratio_vs_greedy_tour',
    'spc_censored', 'fb_trials', 'fb_survived', 'fb_survival',
    'fb_no_opportunity', 'fb_unescapable', 'probe_env_steps', 'wall_s',
]

GATE_COLUMNS = [
    'env_steps', 'update', 'stage', 'stage_name', 'stage_steps', 'cap',
    'episodes', 'spc_ratio_vs_greedy_tour', 'invalid_per_step', 'bombs',
    'suicides', 'fb_trials', 'fb_survival', 'crates', 'kills',
    'deaths_by_other', 'held_stages', 'held_detail', 'gate_pass', 'promoted',
    'promotion_reason', 'gate', 'gate_s1_suicides',
]


def probe_row(env_steps, update, name, m, probe_steps, wall):
    def g(k):
        v = m.get(k)
        return '' if v is None else (round(v, 5) if isinstance(v, float) else v)
    return [env_steps, update, name, m.get('episodes', 0), g('score_mu'),
            g('score_sigma'), g('score_sem'), g('coins'), g('kills'), g('suicides'),
            g('deaths_by_other'), g('crates'), g('bombs'), g('invalid'),
            g('invalid_per_step'), g('survival_rate'), g('episode_length'),
            g('steps_per_coin'), g('spc_ref'), g('spc_ratio'), g('spc_censored'),
            g('fb_trials'), g('fb_survived'), g('fb_survival'),
            g('fb_no_opportunity'), g('fb_unescapable'), probe_steps,
            round(wall, 1)]


def git_sha(root):
    try:
        out = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def maybe_plot(out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'(no curves: {exc!r})')
        return
    try:
        train = os.path.join(out_dir, 'train_log.csv')
        probe = os.path.join(out_dir, 'eval_ladder.csv')
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        if os.path.isfile(train):
            with open(train) as fh:
                rows = list(csv.DictReader(fh))
            xs = [float(r['env_steps']) for r in rows]
            ax[0].plot(xs, [float(r['score']) for r in rows], lw=1)
            ax[0].set_title('training score (stage-relative)')
            ax[0].set_xlabel('env steps')
        if os.path.isfile(probe):
            with open(probe) as fh:
                rows = list(csv.DictReader(fh))
            for name in sorted({r['probe'] for r in rows}):
                sel = [r for r in rows if r['probe'] == name and r['score_mu'] != '']
                ax[1].plot([float(r['env_steps']) for r in sel],
                           [float(r['score_mu']) for r in sel], marker='o', ms=3,
                           label=name)
            ax[1].legend(fontsize=7)
            ax[1].set_title('frozen probe (stage-independent)')
            ax[1].set_xlabel('env steps')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'curves.png'), dpi=120)
        plt.close(fig)
    except Exception as exc:
        print(f'(no curves: {exc!r})')


def bench_env(args):
    env = VecBomberman(args.envs, stage_table=stage_table_array((LADDER[-1],)),
                       stage_weights=[1.0], seed=args.seed, gamma=args.gamma)
    env.reset()
    rng = np.random.default_rng(args.seed)
    acts = rng.integers(0, N_ACTIONS, size=(args.envs, N_AGENTS))
    env.step(acts)                       # pay the compilation once
    t0 = time.time()
    steps = 0
    while time.time() - t0 < args.bench:
        env.step(acts)
        steps += args.envs
    out = {'env_sps': steps / max(time.time() - t0, 1e-9), 'envs': args.envs,
           'seconds': args.bench}
    print(json.dumps(out, indent=2))
    if args.bench_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.bench_json)), exist_ok=True)
        with open(args.bench_json, 'w') as fh:
            json.dump(out, fh, indent=2)
    return out


def probe_weights(args, device, ref_model, autocast, fast_probes):
    path = args.probe_weights
    if not os.path.isfile(path):
        print(f'ERROR: --probe-weights {path} does not exist')
        return 2
    model, ckpt = ActorCritic.load(path, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tag = args.probe_tag or os.path.basename(os.path.dirname(os.path.abspath(path)))
    steps = int(ckpt.get('env_steps', 0) or 0)
    updates = int(ckpt.get('updates', 0) or 0)
    print(f'probe-only: {path}  tag={tag}  env_steps={steps}  updates={updates}')

    os.makedirs(args.out_dir, exist_ok=True)
    fh, writer = open_appending_csv(os.path.join(args.out_dir, 'eval_ladder.csv'),
                                    PROBE_COLUMNS)
    summary = {'weights': os.path.abspath(path), 'tag': tag, 'env_steps': steps,
               'updates': updates, 'episodes': args.probe_episodes, 'probes': {}}
    try:
        for probe_name, probe_stage in fast_probes:
            opponent = ref_model if probe_name == 'P-REF' else None
            if probe_name == 'P-REF' and opponent is None:
                print('  (P-REF skipped: no --ref-weights)')
                continue
            t0 = time.time()
            m, used = play_episodes(model, probe_stage, args.probe_episodes, device,
                                    args, seed=args.seed + 1000, opponent=opponent,
                                    autocast=autocast)
            writer.writerow(probe_row(steps, updates, f'{probe_name}@{tag}', m, used,
                                      time.time() - t0))
            summary['probes'][probe_name] = m
            print(f'  {probe_name:6s} score={m["score_mu"]:.3f} '
                  f'+-{m["score_sem"]:.3f}  coins={m["coins"]:.3f} '
                  f'kills={m["kills"]:.3f} bombs={m["bombs"]:.2f} '
                  f'survival={m["survival_rate"]:.3f}')
        fb_name, fb_stage = FORCED_BOMB_PROBE
        t0 = time.time()
        m, used = forced_bomb_trials(model, fb_stage, args.fb_trials, device, args,
                                     seed=args.seed + 4000, autocast=autocast)
        m['episodes'] = m['fb_episodes']
        writer.writerow(probe_row(steps, updates, f'{fb_name}@{tag}', m, used,
                                  time.time() - t0))
        summary['probes'][fb_name] = m
        print(f'  {fb_name:6s} survival={m["fb_survival"]} of {m["fb_trials"]} trials')
        fh.flush()
    finally:
        fh.close()
    with open(os.path.join(args.out_dir, f'probe_{tag}.json'), 'w') as out:
        json.dump(summary, out, indent=2, default=str)
    return 0


def main(argv=None):
    args = parse_args(argv)
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(line_buffering=True)

    variant = load_variant(args.variant_json)
    name = variant.get('name', os.path.basename(HERE))
    curriculum = args.curriculum or variant.get('curriculum', 'none')
    mix = args.mix if args.mix is not None else float(variant.get('mix', 0.3))

    vset = variant_settings(variant)
    if args.rewards is not None:
        vset['rewards'] = args.rewards
    if args.reward_anneal_frac is not None:
        vset['reward_anneal_frac'] = args.reward_anneal_frac
    if args.reward_anneal_steps is not None:
        vset['reward_anneal_steps'] = args.reward_anneal_steps
    if args.inject_anneal_frac is not None:
        vset['inject_anneal_frac'] = args.inject_anneal_frac
    if args.inject_anneal_steps is not None:
        vset['inject_anneal_steps'] = args.inject_anneal_steps
    if args.opp_frac is not None:
        vset['opp_frac'] = args.opp_frac
    if args.inject_dose is not None:
        vset['inject_dose'] = args.inject_dose
    if args.bombopp is not None:
        vset['bombopp'] = args.bombopp
    if args.pot_opp is not None:
        vset['pot_opp'] = args.pot_opp
    if args.canonical is not None:
        vset['canonical'] = bool(args.canonical)
    if vset['minibatch'] > 0 and '--minibatch' not in (argv if argv is not None
                                                       else sys.argv[1:]):
        args.minibatch = int(vset['minibatch'])
    else:
        vset['minibatch'] = int(args.minibatch)

    args.reward_anneal_frac = float(vset['reward_anneal_frac'])
    args.inject_anneal_frac = float(vset['inject_anneal_frac'])
    args.reward_anneal_steps = (
        int(vset['reward_anneal_steps']) if int(vset['reward_anneal_steps']) >= 0
        else anneal_steps(args.reward_anneal_frac, args.total_steps))
    args.inject_anneal_steps = (
        int(vset['inject_anneal_steps']) if int(vset['inject_anneal_steps']) >= 0
        else anneal_steps(args.inject_anneal_frac, args.total_steps))
    vset['reward_anneal_steps'] = args.reward_anneal_steps
    vset['inject_anneal_steps'] = args.inject_anneal_steps
    args.rewards = vset['rewards']
    args.bombopp = float(vset['bombopp'])
    args.pot_opp = float(vset['pot_opp'])
    args.opp_frac = float(vset['opp_frac'])
    args.inject_dose = float(vset['inject_dose'])
    args.canonical = bool(vset['canonical'])
    fast_probes, slow_probes = probe_sets(args.probe_set)

    if curriculum == 'gated':
        stages = LADDER
    else:
        stages = d1_stages(args.opp_frac)
    table = stage_table_array(stages)
    n_stages = len(stages)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(get_device('cpu'))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    nthreads = get_num_threads(0)
    if nthreads > 0:
        import numba
        numba.set_num_threads(nthreads)

    if args.bench > 0:
        bench_env(args)
        return 0

    amp_dtype = {'off': None, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.amp]
    if device.type != 'cuda':
        amp_dtype = None
    autocast = ((lambda: torch.autocast('cuda', dtype=amp_dtype)) if amp_dtype
                else contextlib.nullcontext)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp == 'fp16' and device.type == 'cuda'))

    print(f'arm={name}  round={args.round_id or "-"}  ref={args.ref_arm or "-"}  '
          f'curriculum={curriculum}  stages={[s.name for s in stages]}')
    print(f'rewards={args.rewards} (anneal {args.reward_anneal_frac:.6f} of budget '
          f'= {args.reward_anneal_steps} steps)  opp_frac={args.opp_frac}  '
          f'inject_dose={args.inject_dose} (anneal {args.inject_anneal_frac:.6f} '
          f'= {args.inject_anneal_steps} steps)  bombopp={args.bombopp}  '
          f'pot_opp={args.pot_opp}  minibatch={args.minibatch}  '
          f'canonical={args.canonical}')
    print(f'probe set={args.probe_set}: '
          f'{[p for p, _ in fast_probes]} + P-FB every {args.probe_every} steps'
          + (f', {[p for p, _ in slow_probes]} every {args.probe_slow_every}'
             if slow_probes else '')
          + f'  ({args.probe_episodes} episodes)')
    print(f'device={device}  envs={args.envs}  rollout={args.rollout}  '
          f'batch={args.envs * N_AGENTS * args.rollout}  amp={args.amp}  '
          f'budget={args.total_steps} env steps')
    if device.type == 'cuda':
        report_gpu(device, args)

    model = ActorCritic(width=args.width, blocks=args.blocks).to(device)
    opt_kwargs = {'lr': args.lr, 'eps': 1e-5}
    if args.fused_adam and device.type == 'cuda':
        opt_kwargs['fused'] = True
    try:
        opt = torch.optim.Adam(model.parameters(), **opt_kwargs)
    except (TypeError, RuntimeError) as exc:
        print(f'(fused Adam unavailable: {exc!r}); falling back to the default')
        opt_kwargs.pop('fused', None)
        args.fused_adam = False
        opt = torch.optim.Adam(model.parameters(), **opt_kwargs)

    resume_path = os.path.join(args.out_dir, 'resume.pt')
    state = None
    if os.path.isfile(resume_path) and not args.no_resume:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state['state_dict'])
        opt.load_state_dict(state['optimizer'])
        print(f"resumed from {resume_path} at update {state['update']}, "
              f"{state['env_steps']} env steps, stage {state['stage']}")
    model.train()
    print(f'parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    net = torch.compile(model) if args.compile else model

    ref_model = None
    if args.ref_weights and os.path.isfile(args.ref_weights):
        ref_model, _ = ActorCritic.load(args.ref_weights, device)
        for p in ref_model.parameters():
            p.requires_grad_(False)
        print(f'P-REF opponent: {args.ref_weights}')
    elif args.ref_weights:
        print(f'WARNING: --ref-weights {args.ref_weights} not found; P-REF is skipped')

    if args.probe_weights:
        return probe_weights(args, device, ref_model, autocast, fast_probes)

    stage = state['stage'] if state else 0
    stage_start = state['stage_start'] if state else 0
    promotion_reason = state['promotion_reason'] if state else 'start'
    update = state['update'] if state else 0
    env_steps = state['env_steps'] if state else 0
    probe_steps = state['probe_steps'] if state else 0
    episodes_total = state['episodes_total'] if state else 0
    next_gate = state['next_gate'] if state else args.gate_every
    next_probe = state['next_probe'] if state else 0
    next_slow_probe = (state.get('next_slow_probe', 0) if state else 0)
    next_ckpt = state['next_ckpt'] if state else args.ckpt_every
    next_resume = state['next_resume'] if state else args.resume_every
    mb_size = state['mb_size'] if state else args.minibatch
    train_samples = state.get('train_samples', 0) if state else 0
    inject_total = state.get('inject_total', 0) if state else 0
    if state and state.get('numpy_rng') is not None:
        np.random.set_state(state['numpy_rng'])
    if state and state.get('torch_rng') is not None:
        torch.set_rng_state(state['torch_rng'].cpu().to(torch.uint8))

    base_rewards = (d1_rewards_at(env_steps, args.reward_anneal_steps,
                                  args.bombopp, args.pot_opp)
                    if args.rewards == 'd1' else DEFAULT_REWARDS)
    rewards = np.tile(base_rewards, (n_stages, 1))
    init_weights = (mixture_weights(n_stages, stage, mix, 1.0) if curriculum == 'gated'
                    else d1_stage_weights(n_stages, args.opp_frac))
    env = VecBomberman(args.envs, stage_table=table, stage_weights=init_weights,
                       seed=args.seed, rewards=rewards, gamma=args.gamma,
                       canonical=args.canonical)
    obs_np, mask_np, _ = env.reset()

    inject_armed = np.zeros(args.envs, dtype=bool)
    inject_fired = np.zeros(args.envs, dtype=bool)

    def redraw_inject(idx, dose):
        idx = np.asarray(idx)
        if idx.size == 0:
            return
        inject_armed[idx] = np.random.random(idx.size) < dose if dose > 0 else False
        inject_fired[idx] = False

    redraw_inject(np.arange(args.envs),
                  inject_dose_at(env_steps, args.inject_dose, args.inject_anneal_steps))

    B = args.envs * N_AGENTS
    T = args.rollout
    buf = Rollout(T, B, device)

    ctrl = np.zeros((args.envs, N_AGENTS), dtype=np.int64)
    snapshots = []
    if state:
        for sd in state.get('snapshots', []):
            snap = ActorCritic(width=args.width, blocks=args.blocks).to(device)
            snap.load_state_dict(sd)
            snap.eval()
            for p in snap.parameters():
                p.requires_grad_(False)
            snapshots.append(snap)

    def selfplay_envs(idx):
        return (env.n_active[idx] == N_AGENTS) & (env.opp_kind[idx] == OPP_POLICY)

    def reassign(env_idx):
        ctrl[env_idx] = 0
        if not snapshots or args.league_frac <= 0:
            return
        eligible = np.asarray(env_idx)[selfplay_envs(np.asarray(env_idx))]
        for n in eligible:
            if np.random.random() < args.league_frac:
                k = np.random.randint(len(snapshots)) + 1
                ctrl[n, 0] = 0
                ctrl[n, 1:] = k

    obs_dev = torch.zeros(B, CHANNELS, COLS, ROWS, dtype=torch.uint8, device=device)
    mask_dev = torch.zeros(B, N_ACTIONS, dtype=torch.bool, device=device)
    pin = torch.zeros(B, CHANNELS, COLS, ROWS, dtype=torch.uint8,
                      pin_memory=(device.type == 'cuda'))
    pin_mask = torch.zeros(B, N_ACTIONS, dtype=torch.uint8,
                           pin_memory=(device.type == 'cuda'))

    def upload():
        pin.copy_(torch.from_numpy(obs_np.reshape(B, CHANNELS, COLS, ROWS)))
        pin_mask.copy_(torch.from_numpy(mask_np.reshape(B, N_ACTIONS)))
        obs_dev.copy_(pin, non_blocking=True)
        mask_dev.copy_(pin_mask, non_blocking=True)

    @torch.no_grad()
    def policy_step():
        upload()
        with autocast():
            logits, value = net(obs_dev, mask_dev)
        logits = logits.float()
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        if snapshots:
            ctrl_flat = torch.from_numpy(ctrl.reshape(-1)).to(device)
            for k, snap in enumerate(snapshots, start=1):
                sel = torch.nonzero(ctrl_flat == k, as_tuple=False).squeeze(-1)
                if sel.numel() == 0:
                    continue
                with autocast():
                    slogits, _ = snap(obs_dev[sel], mask_dev[sel])
                action[sel] = torch.distributions.Categorical(
                    logits=slogits.float()).sample()
        return action, logp, value.float()

    reward_cols = ['r_coin', 'r_kill', 'r_selfkill', 'r_killed', 'r_crate',
                   'r_invalid', 'r_step', 'r_survive', 'r_shape', 'r_bombopp',
                   'r_potopp']
    log_header = (['time_s', 'update', 'env_steps', 'probe_env_steps', 'sps',
                   'stage', 'stage_name', 'mix', 'promotion_reason']
                  + [f'w_{s.name}' for s in stages]
                  + [f'ep_{s.name}' for s in stages]
                  + ['score', 'coins', 'kills', 'suicides', 'crates', 'ep_len',
                     'reward', 'policy_loss', 'value_loss', 'entropy', 'kl', 'lr',
                     'ent_coef', 'episodes', 'episodes_total', 'bomb_rate',
                     'invalid_rate',
                     'train_samples', 'inject_events', 'inject_total',
                     'inject_dose', 'opp_frac', 'rollout_s', 'opt_s',
                     'bombopp_events', 'minibatch']
                  + reward_cols)
    log_fh, log_writer = open_appending_csv(os.path.join(args.out_dir, 'train_log.csv'),
                                            log_header)
    probe_fh, probe_writer = open_appending_csv(
        os.path.join(args.out_dir, 'eval_ladder.csv'), PROBE_COLUMNS)
    gate_fh, gate_writer = open_appending_csv(
        os.path.join(args.out_dir, 'gate_log.csv'), GATE_COLUMNS)

    ep_score = deque(maxlen=400)
    ep_coins = deque(maxlen=400)
    ep_kills = deque(maxlen=400)
    ep_suicide = deque(maxlen=400)
    ep_crates = deque(maxlen=400)
    ep_bombopp = deque(maxlen=400)
    ep_invalid = deque(maxlen=400)
    ep_steps = deque(maxlen=400)
    ep_len = deque(maxlen=100)
    ep_stage_counts = np.zeros(n_stages, dtype=np.int64)

    meta = {
        'variant': name, 'curriculum': curriculum, 'mix': mix,
        'stages': [s.name for s in stages], 'args': vars(args),
        'config': variant.get('name', ''),
        'round_id': args.round_id,
        'ref_arm': args.ref_arm,
        'variant_settings': vset,
        'base_knobs': base_knobs(args),
        'schedule_fractions': {
            'reward_anneal': args.reward_anneal_frac,
            'inject_anneal': args.inject_anneal_frac,
            'lr': [args.lr, args.lr * args.lr_final_frac],
            'entropy': [args.ent, args.ent_final],
            'total_steps': args.total_steps,
            'note': '',
        },
        'probe_set': {'name': args.probe_set,
                      'fast': [p for p, _ in fast_probes],
                      'slow': [p for p, _ in slow_probes],
                      'every': args.probe_every,
                      'slow_every': args.probe_slow_every,
                      'episodes': args.probe_episodes},
        'reward_schedule': {
            'mode': args.rewards,
            'start': D1_REWARDS_START.tolist(),
            'final': D1_REWARDS_FINAL.tolist(),
            'anneal_env_steps': args.reward_anneal_steps,
            'anneal_frac': args.reward_anneal_frac,
            'bombopp0': args.bombopp,
            'pot_opp': args.pot_opp,
            'columns': reward_cols,
            'note': 'linear from step 0 to anneal_env_steps, constant after; '
                    'survive is removed outright, not annealed; r_bombopp '
                    'anneals on the same ramp, r_potopp is a potential and '
                    'does not anneal',
        },
        'injection': {
            'dose0': args.inject_dose,
            'anneal_env_steps': args.inject_anneal_steps,
            'anneal_frac': args.inject_anneal_frac,
            'masked_from_policy_loss': True,
            'kept_in_value_and_gae': True,
        },
        'opposition': {'opp_frac': args.opp_frac,
                       'kind': 'OPP_RULE (deterministic, in-kernel)',
                       'rows': [s.name for s in stages]},
        'canonical': args.canonical,
        'task0': {'sync_free_update': True,
                  'fused_adam': bool(args.fused_adam and device.type == 'cuda'),
                  'compile': bool(args.compile)},
        'seed': args.seed, 'device': str(device),
        'numba_threads': nthreads, 'git_sha': git_sha(os.path.dirname(os.path.dirname(HERE))),
        'torch': torch.__version__, 'variant_json': variant,
        'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'reward_table': rewards.tolist(),
        'gate_s1': {'bombs': GATE_S1_BOMBS, 'suicides': args.gate_s1_suicides,
                    'forced_bomb_survival': GATE_S1_FORCED_SURVIVAL,
                    'forced_bomb_horizon': FORCED_BOMB_HORIZON,
                    'forced_bomb_trials': args.fb_trials,
                    'revised': ''},
        'note': 'determinism is statistical, not bitwise: numba.prange '
                'interleaves games by thread count',
    }

    def write_meta(**extra):
        meta.update(extra)
        with open(os.path.join(args.out_dir, 'meta.json'), 'w') as fh:
            json.dump(meta, fh, indent=2, default=str)

    write_meta()

    def save(path, **extra):
        model.save(path, updates=update, env_steps=env_steps,
                   probe_steps=probe_steps, variant=name, args=vars(args), **extra)

    def save_resume():
        torch.save({
            'config': model.config(), 'state_dict': model.state_dict(),
            'optimizer': opt.state_dict(), 'update': update, 'env_steps': env_steps,
            'probe_steps': probe_steps, 'stage': stage, 'stage_start': stage_start,
            'promotion_reason': promotion_reason, 'episodes_total': episodes_total,
            'train_samples': train_samples, 'inject_total': inject_total,
            'next_gate': next_gate, 'next_probe': next_probe,
            'next_slow_probe': next_slow_probe, 'next_ckpt': next_ckpt,
            'next_resume': next_resume, 'mb_size': mb_size,
            'snapshots': [s.state_dict() for s in snapshots],
            'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
            'args': vars(args),
        }, resume_path)

    def run_probe(slow=False):
        nonlocal probe_steps
        model.eval()
        todo = list(fast_probes) + (list(slow_probes) if slow else [])
        for probe_name, probe_stage in todo:
            opponent = ref_model if probe_name == 'P-REF' else None
            if probe_name == 'P-REF' and opponent is None:
                continue
            t0p = time.time()
            m, used = play_episodes(model, probe_stage, args.probe_episodes, device,
                                    args, seed=args.seed + 1000, opponent=opponent,
                                    autocast=autocast)
            probe_steps += used
            probe_writer.writerow(probe_row(env_steps, update, probe_name, m, used,
                                            time.time() - t0p))
        fb_name, fb_stage = FORCED_BOMB_PROBE
        t0p = time.time()
        m, used = forced_bomb_trials(model, fb_stage, args.fb_trials, device, args,
                                     seed=args.seed + 4000, autocast=autocast)
        probe_steps += used
        m['episodes'] = m['fb_episodes']
        probe_writer.writerow(probe_row(env_steps, update, fb_name, m, used,
                                        time.time() - t0p))
        probe_fh.flush()
        model.train()

    def run_gate():
        nonlocal stage, stage_start, promotion_reason, probe_steps
        st = stages[stage]
        terminal = stage >= n_stages - 1
        model.eval()
        used = 0

        def measure(target, seed):
            nonlocal used
            mm, u = play_episodes(model, target, args.gate_episodes, device, args,
                                  seed=seed, autocast=autocast)
            used += u
            if target.name == 'S1':
                fb, u = forced_bomb_trials(model, target, args.fb_trials, device,
                                           args, seed=seed + 500, autocast=autocast)
                used += u
                mm.update(fb)
            return mm

        m = measure(st, args.seed + 2000)
        held = {}
        for i, hold in enumerate(st.holds):
            hold_stage = next(s for s in stages if s.name == hold)
            held[hold] = measure(hold_stage, args.seed + 3000 + 100 * i)
        model.train()
        probe_steps += used

        passed, detail = gate_verdict(st, m, held, args.gate_s1_suicides)
        stage_steps = env_steps - stage_start
        capped = (not terminal) and stage_steps >= st.cap
        promoted = (not terminal) and (passed or capped)
        reason = 'gate' if (promoted and passed) else ('cap' if promoted else '')
        held_detail = {k: v for k, v in detail.items()
                       if any(k.startswith(f'{h.lower()}_') for h in st.holds)}
        gate_writer.writerow([
            env_steps, update, stage, st.name, stage_steps,
            (st.cap if not terminal else ''), m.get('episodes', 0),
            detail.get('spc_ratio', ''), detail.get('invalid_per_step', ''),
            detail.get('bombs', m.get('bombs', '')),
            detail.get('suicides', m.get('suicides', '')),
            m.get('fb_trials', ''), m.get('fb_survival', ''),
            detail.get('crates', m.get('crates', '')),
            detail.get('kills', m.get('kills', '')),
            detail.get('deaths_by_other', m.get('deaths_by_other', '')),
            ';'.join(st.holds), json.dumps(held_detail, default=str),
            int(passed), int(promoted), reason, st.gate, args.gate_s1_suicides,
        ])
        gate_fh.flush()
        if promoted:
            stage += 1
            stage_start = env_steps
            promotion_reason = reason
            print(f'--- promoted to {stages[stage].name} at {env_steps} env steps '
                  f'({reason}); gate detail {detail}')
            if reason == 'cap':
                print('    NOTE: this is a *failed* promotion -- the cap expired '
                      'before the gate was met (PLAN_COMMON.md §2 rule 1)')
        return promoted

    print(f'{"time":>8} {"upd":>6} {"steps":>11} {"sps":>7} {"stg":>4} '
          f'{"score":>6} {"coin":>5} {"kill":>5} {"suic":>5} {"crate":>6} '
          f'{"len":>5} {"rew":>7} {"ent":>5} {"kl":>6} {"eps":>5} {"bomb":>5}')

    t0 = time.time()
    guard_deadline = t0 + args.hours * 3600.0 if args.hours > 0 else None
    sps = 0.0
    try:
        while env_steps < args.total_steps:
            if guard_deadline is not None and time.time() > guard_deadline:
                print('--- wall-clock safety guard hit; stopping (this is a guard, '
                      'not a schedule)')
                break

            frac = min(1.0, env_steps / max(args.total_steps, 1))

            if (curriculum == 'gated' and stage < n_stages - 1
                    and env_steps >= next_gate):
                run_gate()
                next_gate = env_steps + args.gate_every
            if curriculum == 'gated':
                ramp = 1.0 if args.ramp_steps <= 0 else \
                    min(1.0, (env_steps - stage_start) / args.ramp_steps)
                weights = mixture_weights(n_stages, stage, mix, ramp)
            else:
                weights = d1_stage_weights(n_stages, args.opp_frac)
            env.set_stage_weights(weights)

            lr = args.lr * (1.0 - frac * (1.0 - args.lr_final_frac))
            ent_coef = args.ent + frac * (args.ent_final - args.ent)
            for g in opt.param_groups:
                g['lr'] = lr

            if args.rewards == 'd1':
                live_rewards = d1_rewards_at(env_steps, args.reward_anneal_steps,
                                             args.bombopp, args.pot_opp)
                env.set_rewards(live_rewards)
            else:
                live_rewards = DEFAULT_REWARDS
            dose = inject_dose_at(env_steps, args.inject_dose,
                                  args.inject_anneal_steps)

            model.eval()
            rollout_t0 = time.time()
            episodes = 0
            bomb_acts = live_acts = 0
            inject_events = 0
            for t in range(T):
                action, logp, value = policy_step()
                buf.obs[t].copy_(obs_dev)
                buf.mask[t].copy_(mask_dev)
                buf.act[t].copy_(action)
                buf.logp[t].copy_(logp)
                buf.val[t].copy_(value)

                act_np = action.to('cpu').numpy().reshape(args.envs, N_AGENTS)
                if env.canonical:
                    act_np = D4_ACTION_INV[env.gsel, act_np]

                seat_ok = np.ones((args.envs, N_AGENTS), dtype=bool)
                scripted = env.opp_kind != OPP_POLICY
                seat_ok[scripted, 1:] = False
                live_mask = (ctrl == 0) & (env.alive != 0) & seat_ok

                if dose > 0:
                    pending = inject_armed & ~inject_fired & (env.alive[:, 0] != 0)
                    if pending.any():
                        escape = env.bomb_escape_distance(seat=0)
                        fire = injection_fire(pending, mask_np[:, 0, A_BOMB] != 0,
                                              escape)
                        if fire.any():
                            act_np = act_np.copy()
                            act_np[fire, 0] = A_BOMB
                            inject_fired[fire] = True
                            mask_injected(live_mask, fire)
                            inject_events += int(fire.sum())

                live_np = live_mask.reshape(-1)
                if live_mask.any():
                    bomb_acts += int(((act_np == A_BOMB) & live_mask).sum())
                    live_acts += int(live_mask.sum())
                train_samples += int(live_mask.sum())

                obs_np, mask_np, _, rew, agent_done, env_done = env.step(act_np)

                buf.rew[t].copy_(torch.from_numpy(rew.reshape(-1).astype(np.float32)))
                buf.done[t].copy_(torch.from_numpy(agent_done.reshape(-1).astype(np.float32)))
                buf.live[t].copy_(torch.from_numpy(live_np.astype(np.float32)))
                env_steps += args.envs

                done_idx = np.nonzero(env_done)[0]
                if len(done_idx):
                    episodes += len(done_idx)
                    episodes_total += len(done_idx)
                    st_rows = env.last_stats[done_idx]
                    done_stages = env.last_stage[done_idx]
                    ep_stage_counts += np.bincount(done_stages, minlength=n_stages)
                    seats = table[done_stages, SC_ACTIVE].astype(int)
                    opps = table[done_stages, SC_OPP].astype(int)
                    idx = np.arange(N_AGENTS)[None, :]
                    learner = (ctrl[done_idx] == 0) & (idx < seats[:, None])
                    learner[opps > OPP_POLICY, 1:] = False
                    if learner.any():
                        sel = st_rows[learner]
                        ep_score.extend(sel[:, ST_SCORE].tolist())
                        ep_coins.extend(sel[:, ST_COINS].tolist())
                        ep_kills.extend(sel[:, ST_KILLS].tolist())
                        ep_suicide.extend(sel[:, ST_SUICIDE].tolist())
                        ep_crates.extend(sel[:, ST_CRATES].tolist())
                        ep_bombopp.extend(sel[:, ST_BOMBOPP].tolist())
                        ep_invalid.extend(sel[:, ST_INVALID].tolist())
                        ep_steps.extend(sel[:, ST_STEPS].tolist())
                        ep_len.extend(st_rows[:, 0, ST_STEPS].tolist())
                    reassign(done_idx)
                    redraw_inject(done_idx, dose)

            with torch.no_grad():
                upload()
                with autocast():
                    _, last_val = net(obs_dev, mask_dev)
            compute_gae(buf, last_val.float(), args.gamma, args.lam)
            rollout_time = time.time() - rollout_t0

            model.train()

            def flat(x):
                return x.reshape(-1, *x.shape[2:])

            obs_f, mask_f = flat(buf.obs), flat(buf.mask)
            act_f, logp_f = flat(buf.act), flat(buf.logp)
            adv_f, ret_f, val_f = flat(buf.adv), flat(buf.ret), flat(buf.val)
            live_f = flat(buf.live)

            index = torch.nonzero(live_f > 0, as_tuple=False).squeeze(-1)
            n = index.numel()
            opt_t0 = time.time()
            while True:
                kl_acc = torch.zeros((), device=device)
                pl_acc = torch.zeros((), device=device)
                vl_acc = torch.zeros((), device=device)
                ent_acc = torch.zeros((), device=device)
                n_batches = 0
                stop = False
                try:
                    for _ in range(args.epochs):
                        perm = index[torch.randperm(n, device=device)]
                        for s in range(0, n, mb_size):
                            mb = perm[s:s + mb_size]
                            if mb.numel() < 2:
                                continue
                            with autocast():
                                logits, value = net(obs_f[mb], mask_f[mb])
                            logits = logits.float()
                            value = value.float()
                            dist = torch.distributions.Categorical(logits=logits)
                            new_logp = dist.log_prob(act_f[mb])
                            entropy = dist.entropy().mean()

                            ratio = torch.exp(new_logp - logp_f[mb])
                            mb_adv = adv_f[mb]
                            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                            pg1 = -mb_adv * ratio
                            pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                            policy_loss = torch.max(pg1, pg2).mean()

                            v_clipped = val_f[mb] + torch.clamp(
                                value - val_f[mb], -args.clip, args.clip)
                            value_loss = 0.5 * torch.max((value - ret_f[mb]) ** 2,
                                                         (v_clipped - ret_f[mb]) ** 2).mean()
                            loss = policy_loss + args.vf * value_loss - ent_coef * entropy

                            opt.zero_grad(set_to_none=True)
                            scaler.scale(loss).backward()
                            scaler.unscale_(opt)
                            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           args.max_grad_norm)
                            scaler.step(opt)
                            scaler.update()

                            with torch.no_grad():
                                logratio = new_logp - logp_f[mb]
                                kl_acc += ((torch.exp(logratio) - 1) - logratio).mean()
                                pl_acc += policy_loss.detach()
                                vl_acc += value_loss.detach()
                                ent_acc += entropy.detach()
                            n_batches += 1
                        if (args.target_kl > 0 and n_batches
                                and kl_acc.item() / n_batches > args.target_kl):
                            stop = True
                        if stop:
                            break
                    break
                except torch.OutOfMemoryError:
                    if mb_size <= 256:
                        raise
                    opt.zero_grad(set_to_none=True)
                    logits = value = loss = None  # drop refs before freeing
                    torch.cuda.empty_cache()
                    mb_size //= 2
                    print(f'--- CUDA OOM during the update: retrying with '
                          f'--minibatch {mb_size}')

            opt_time = time.time() - opt_t0
            approx_kl = float(kl_acc.item())
            pl = float(pl_acc.item())
            vl = float(vl_acc.item())
            ent_val = float(ent_acc.item())

            update += 1
            inject_total += inject_events
            n_batches = max(n_batches, 1)
            elapsed = time.time() - t0
            sps = (T * args.envs) / max(rollout_time, 1e-6)
            mean_rew = (buf.rew * buf.live).sum().item() / max(buf.live.sum().item(), 1)

            def mean_of(dq, default=0.0):
                return float(np.mean(dq)) if dq else default

            m_score = mean_of(ep_score)
            m_coins = mean_of(ep_coins)
            m_kills = mean_of(ep_kills)
            m_suicide = mean_of(ep_suicide)
            m_crates = mean_of(ep_crates)
            m_bombopp = mean_of(ep_bombopp)
            m_len = mean_of(ep_len)
            invalid_rate = (sum(ep_invalid) / max(sum(ep_steps), 1)) if ep_steps else 0.0
            bomb_rate = bomb_acts / max(live_acts, 1)

            row = ([round(elapsed, 1), update, env_steps, probe_steps, int(sps),
                    stage, stages[stage].name, round(mix, 3), promotion_reason]
                   + [round(float(w), 4) for w in weights]
                   + [int(c) for c in ep_stage_counts]
                   + [round(m_score, 2), round(m_coins, 2), round(m_kills, 3),
                      round(m_suicide, 3), round(m_crates, 2), round(m_len, 1),
                      round(mean_rew, 4), round(pl / n_batches, 4),
                      round(vl / n_batches, 4), round(ent_val / n_batches, 4),
                      round(approx_kl / n_batches, 5), round(lr, 7), round(ent_coef, 5),
                      episodes, episodes_total, round(bomb_rate, 4),
                      round(invalid_rate, 5), train_samples, inject_events,
                      inject_total, round(dose, 4), round(args.opp_frac, 3),
                      round(rollout_time, 2), round(opt_time, 2),
                      round(m_bombopp, 3), mb_size]
                   + [round(float(w), 5) for w in live_rewards])
            log_writer.writerow(row)
            log_fh.flush()

            if update % 5 == 0 or update <= 5:
                print(f'{elapsed/60:7.1f}m {update:6d} {env_steps:11d} {int(sps):7d} '
                      f'{stages[stage].name:>4} {m_score:6.2f} {m_coins:5.2f} '
                      f'{m_kills:5.3f} {m_suicide:5.3f} {m_crates:6.2f} {m_len:5.1f} '
                      f'{mean_rew:7.3f} {ent_val/n_batches:5.3f} '
                      f'{approx_kl/n_batches:6.4f} {episodes:5d} '
                      f'{bomb_rate:5.3f} (env {rollout_time:.1f}s)')

            if update % args.snapshot_every == 0 and args.league_frac > 0:
                snap = copy.deepcopy(model).eval()
                for p in snap.parameters():
                    p.requires_grad_(False)
                snapshots.append(snap)
                if len(snapshots) > args.snapshots:
                    snapshots.pop(0)
                reassign(np.arange(args.envs))
                print(f'--- snapshot added (pool={len(snapshots)})')

            if env_steps >= next_probe:
                slow_due = bool(slow_probes) and env_steps >= next_slow_probe
                run_probe(slow=slow_due)
                if slow_due:
                    next_slow_probe = env_steps + args.probe_slow_every
                next_probe = env_steps + args.probe_every

            if env_steps >= next_ckpt:
                save(os.path.join(args.out_dir, f'ckpt_{env_steps}.pt'))
                next_ckpt = env_steps + args.ckpt_every
                save(os.path.join(args.out_dir, MODEL_FILE))
                save(os.path.join(args.out_dir, LATEST_FILE))

            if env_steps >= next_resume:
                save_resume()
                save(os.path.join(args.out_dir, MODEL_FILE))
                save(os.path.join(args.out_dir, LATEST_FILE))
                write_meta(env_steps=env_steps, updates=update,
                           measured_sps=round(sps, 1),
                           wall_clock_h=round((time.time() - t0) / 3600, 3))
                next_resume = env_steps + args.resume_every

    except KeyboardInterrupt:
        print('\ninterrupted -- saving')

    run_probe(slow=bool(slow_probes))
    save(os.path.join(args.out_dir, MODEL_FILE), final=True)
    save(os.path.join(args.out_dir, LATEST_FILE), final=True)
    save(os.path.join(args.out_dir, f'ckpt_{env_steps}.pt'), final=True)
    save_resume()
    write_meta(env_steps=env_steps, updates=update, probe_env_steps=probe_steps,
               measured_sps=round(sps, 1), finished=time.strftime('%Y-%m-%dT%H:%M:%S'),
               wall_clock_h=round((time.time() - t0) / 3600, 3),
               final_stage=stages[stage].name)
    log_fh.close()
    probe_fh.close()
    gate_fh.close()
    maybe_plot(args.out_dir)
    print(f'done: {update} updates, {env_steps} env steps (+{probe_steps} probe), '
          f'{(time.time()-t0)/3600:.2f} h -> {os.path.join(args.out_dir, MODEL_FILE)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
