
import argparse
import csv
import json
import os
import platform
import sys
import time

import numpy as np

if __package__ in (None, ''):  # allow `python train_lin.py` from inside the dir
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    __package__ = 'agent_code.lin_agent'

from .config import (  # noqa: E402
    ACTIONS,
    A_BOMB,
    CONJ_PLACE_SLOT,
    CRATE_CLAWBACK,
    CRATE_CLAWBACK_MODES,
    CRATE_KAPPA,
    CRATE_KAPPA_LADDER,
    CRATE_REWARD_MODE,
    CRATE_REWARD_MODES,
    CURRICULA,
    DEFAULT_MODE,
    DIST_CAP,
    EPS_DECAY_EPISODES,
    EPS_END,
    EPS_START,
    ESCAPE_POTENTIAL_CAP,
    ESCAPE_POTENTIAL_RHO,
    ESCAPE_REWARD_MODE,
    ESCAPE_REWARD_MODES,
    E_GATE_ANCHOR_FACTOR,
    E_GATE_BOMBS_MIN,
    E_GATE_ESCAPE_NORM_FLOOR,
    E_GATE_SUICIDES_PER_BOMB_FLOOR,
    FEATURE_MODES,
    FORCED_BOMB_HORIZON,
    FORCED_ESCAPE_TARGET,
    FORCED_ESCAPE_TARGET_S2,
    GAMMA,
    GATE_BOMBS_MIN,
    GATE_CRATES_MIN,
    GATE_INVALID,
    GATE_RATIO,
    GATE_SUICIDES,
    INJECT_MODES,
    INJECT_PROB,
    INJECT_SEED_BASE,
    INJECT_STAGES,
    MAX_STEPS,
    MODE_SLOTS,
    N_ACTIONS,
    POTENTIAL_SCALE,
    R_DEATH,
    R_INVALID,
    REWARD_COIN,
    S0_COIN_COUNT,
    S0_CRATE_DENSITY,
    S1_COIN_COUNT,
    S1_CRATE_DENSITY,
    S2_COIN_COUNT,
    S2_CRATE_DENSITY,
    SB_ORACLE_CAP_FINAL,
    STAGE_SEED_BASE,
    dim_for,
    feature_names,
)
from .escape_oracle import SKIPPED, FeasibilityCache  # noqa: E402
from .model import LinearQ  # noqa: E402
from .self_bomb import ORACLE_CAP, self_bomb_probe  # noqa: E402
from .simenv import SoloWorld  # noqa: E402

PROBE_SEED_BASE = 900_000_000

S1_PROBE_SEED_BASE = 910_000_000

FORCED_SEED_BASE = 920_000_000

S2_PROBE_SEED_BASE = 930_000_000

FORCED_S2_SEED_BASE = 940_000_000

TRAIN_SEED_STRIDE = 10_000_000


def board_seed(run_seed: int, episode: int) -> int:
    return run_seed * TRAIN_SEED_STRIDE + episode


def epsilon(episode: int, eps_start: float, eps_end: float, decay: int) -> float:
    if episode >= decay:
        return eps_end
    return eps_start + (eps_end - eps_start) * (episode / decay)


def mixture3(episode: int, waypoints) -> tuple:
    pts = [(int(e), float(p0), float(p1)) for (e, p0, p1) in waypoints]
    if episode <= pts[0][0]:
        p0, p1 = pts[0][1], pts[0][2]
    elif episode >= pts[-1][0]:
        p0, p1 = pts[-1][1], pts[-1][2]
    else:
        p0 = p1 = 0.0
        for (e_lo, a_lo, b_lo), (e_hi, a_hi, b_hi) in zip(pts, pts[1:]):
            if e_lo <= episode <= e_hi:
                t = (episode - e_lo) / (e_hi - e_lo) if e_hi > e_lo else 0.0
                p0 = a_lo + (a_hi - a_lo) * t
                p1 = b_lo + (b_hi - b_lo) * t
                break
    return p0, p1, max(0.0, 1.0 - p0 - p1)


def draw_stage(u: float, probs) -> str:
    if u < probs[0]:
        return 'S0'
    if u < probs[0] + probs[1]:
        return 'S1'
    return 'S2'


class SemiGradientQ:

    def __init__(self, model, alpha, gamma, n=1):
        self.model = model
        self.alpha = alpha
        self.gamma = gamma
        self.n = n
        self._ep = []

    def begin_episode(self):
        self._ep = []

    def observe(self, phi, action, reward, phi_next, done, exploratory_next,
                stage=1, injected=False, destroyed=0):
        self._ep.append([phi, action, reward, phi_next, done, exploratory_next])
        if len(self._ep) >= self.n:
            self._backup(len(self._ep) - self.n)

    def mark_exploratory(self, exploratory):
        if self._ep:
            self._ep[-1][5] = exploratory

    def end_episode(self):
        start = max(0, len(self._ep) - self.n + 1)
        for tau in range(start, len(self._ep)):
            self._backup(tau)
        self._ep = []

    def _backup(self, tau):
        g = 0.0
        discount = 1.0
        bootstrap_phi = None
        end = min(len(self._ep), tau + self.n)
        for k in range(tau, end):
            _phi, _a, r, phi_next, done, exploratory_next = self._ep[k]
            g += discount * r
            discount *= self.gamma
            if done:
                bootstrap_phi = None
                break
            if exploratory_next or k == end - 1:
                bootstrap_phi = phi_next
                break
        if bootstrap_phi is not None:
            g += discount * float(np.max(self.model.q(bootstrap_phi)))

        phi, action = self._ep[tau][0], self._ep[tau][1]
        delta = g - float(self.model.w[action] @ phi)
        alpha_eff = self.alpha / (float(phi @ phi) + 1.0)
        self.model.w[action] += alpha_eff * delta * phi

    def episode_end_hook(self, episode):
        pass


class LSTDQ:

    def __init__(self, model, gamma, capacity=100_000, every=25, ridge=1e-3,
                 conj_index=None, kappa=0.0, r_death=0.0):
        self.model = model
        self.dim = model.dim
        self.gamma = gamma
        self.capacity = capacity
        self.every = every
        self.ridge = ridge
        self.phi = np.zeros((capacity, self.dim))
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity)
        self.nphi = np.zeros((capacity, self.dim))
        self.done = np.zeros(capacity, dtype=bool)
        self.stage = np.zeros(capacity, dtype=np.int8)
        self.injected = np.zeros(capacity, dtype=bool)
        self.destroyed = np.zeros(capacity, dtype=np.int32)
        self.conj_index = conj_index
        self.kappa = float(kappa)
        self.r_death = float(r_death)
        self.size = 0
        self.head = 0
        self.solves = 0
        self.solve_seconds = 0.0
        self.last_composition = None
        self.last_cond = float('nan')

    def begin_episode(self):
        pass

    def mark_exploratory(self, exploratory):
        pass

    def observe(self, phi, action, reward, phi_next, done, exploratory_next,
                stage=1, injected=False, destroyed=0):
        i = self.head
        self.phi[i] = phi
        self.act[i] = action
        self.rew[i] = reward
        self.nphi[i] = phi_next
        self.done[i] = done
        self.stage[i] = stage
        self.injected[i] = injected
        self.destroyed[i] = destroyed
        self.head = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def composition(self) -> dict:
        n = self.size
        out = {'buffer': n}
        if n == 0:
            return out
        act = self.act[:n]
        rew = self.rew[:n]
        done = self.done[:n]
        stage = self.stage[:n]
        q_next = self.nphi[:n] @ self.model.w.T
        target = rew + self.gamma * np.where(done, 0.0, q_next.max(axis=1))
        for a, name in enumerate(ACTIONS):
            sel = act == a
            k = int(sel.sum())
            out[f'n_{name}'] = k
            out[f'share_{name}'] = k / n
            out[f'mean_r_{name}'] = float(rew[sel].mean()) if k else float('nan')
            out[f'mean_target_{name}'] = float(target[sel].mean()) if k else float('nan')
        bomb = act == A_BOMB
        for tag, value in (('S0', 0), ('S1', 1), ('S2', 2)):
            in_stage = stage == value
            k = int(in_stage.sum())
            out[f'n_{tag}'] = k
            out[f'n_BOMB_{tag}'] = int((bomb & in_stage).sum())
            out[f'share_BOMB_in_{tag}'] = (out[f'n_BOMB_{tag}'] / k) if k else 0.0
            out[f'mean_target_BOMB_{tag}'] = (
                float(target[bomb & in_stage].mean())
                if (bomb & in_stage).any() else float('nan'))
        out['n_BOMB_injected'] = int((bomb & self.injected[:n]).sum())

        destroyed = self.destroyed[:n]
        term = done
        k = int(term.sum())
        out['n_terminal'] = k
        out['mean_r_terminal'] = float(rew[term].mean()) if k else float('nan')
        out['mean_destroyed_at_terminal'] = (float(destroyed[term].mean())
                                             if k else float('nan'))
        out['crate_kappa'] = self.kappa
        out['effective_death_reward'] = (
            self.r_death - self.kappa * float(destroyed[term].mean())
            if k else float('nan'))

        if self.conj_index is None:
            out['conj_place_support'] = float('nan')
            for tag in ('S0', 'S1', 'S2'):
                out[f'conj_place_support_{tag}'] = float('nan')
            out['conj_place_support_BOMB'] = float('nan')
        else:
            col = self.phi[:n, self.conj_index]
            positive = col > 0.0
            out['conj_place_support'] = float(positive.mean())
            out['conj_place_mean_positive'] = (float(col[positive].mean())
                                               if positive.any() else float('nan'))
            for tag, value in (('S0', 0), ('S1', 1), ('S2', 2)):
                in_stage = stage == value
                out[f'conj_place_support_{tag}'] = (
                    float(positive[in_stage].mean()) if in_stage.any()
                    else float('nan'))
            out['conj_place_support_BOMB'] = (float(positive[bomb].mean())
                                              if bomb.any() else float('nan'))
        out['cond'] = self.last_cond
        return out

    def end_episode(self):
        pass

    def episode_end_hook(self, episode):
        if (episode + 1) % self.every == 0 and self.size > 0:
            self.solve()

    def solve(self):
        t0 = time.perf_counter()
        n = self.size
        d = self.dim
        phi = self.phi[:n]
        act = self.act[:n]
        rew = self.rew[:n]
        nphi = self.nphi[:n]
        done = self.done[:n]

        nxt = np.argmax(nphi @ self.model.w.T, axis=1)

        block = N_ACTIONS * d
        a_mat = np.zeros((block, block))
        b_vec = np.zeros(block)
        for a in range(N_ACTIONS):
            sel = act == a
            if not sel.any():
                continue
            p = phi[sel]
            sl = slice(a * d, (a + 1) * d)
            a_mat[sl, sl] += p.T @ p
            b_vec[sl] += p.T @ rew[sel]
            live = ~done[sel]
            if not live.any():
                continue
            p_live = p[live]
            n_live = nphi[sel][live]
            nxt_live = nxt[sel][live]
            for a2 in range(N_ACTIONS):
                m = nxt_live == a2
                if not m.any():
                    continue
                a_mat[sl, a2 * d:(a2 + 1) * d] -= self.gamma * (p_live[m].T @ n_live[m])

        a_mat[np.diag_indices(block)] += self.ridge
        try:
            self.last_cond = float(np.linalg.cond(a_mat))
        except np.linalg.LinAlgError:
            self.last_cond = float('inf')
        self.last_composition = self.composition()
        try:
            w = np.linalg.solve(a_mat, b_vec)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]
        self.model.w[:] = w.reshape(N_ACTIONS, d)
        self.solves += 1
        self.solve_seconds += time.perf_counter() - t0


def make_learner(rule, model, args):
    if rule == 'q1':
        return SemiGradientQ(model, args.alpha, args.gamma, n=1)
    if rule == 'nstep':
        return SemiGradientQ(model, args.alpha, args.gamma, n=args.n_step)
    if rule == 'lstd':
        return LSTDQ(model, args.gamma, capacity=args.lstd_buffer,
                     every=args.lstd_every, ridge=args.lstd_ridge,
                     conj_index=conj_place_index(args.features),
                     kappa=(args.crate_kappa
                            if args.crate_reward == 'potential' else 0.0),
                     r_death=args.r_death)
    raise ValueError(f'unknown rule {rule!r}')


def conj_place_index(features: str):
    slots = MODE_SLOTS[features]
    return slots.index(CONJ_PLACE_SLOT) if CONJ_PLACE_SLOT in slots else None


def make_worlds(args) -> dict:
    return {
        'S0': SoloWorld(crate_density=args.s0_crate_density,
                        coin_count=args.s0_coin_count,
                        max_steps=args.max_steps, feature_mode=args.features),
        'S1': SoloWorld(crate_density=args.crate_density,
                        coin_count=args.coin_count,
                        max_steps=args.max_steps, feature_mode=args.features),
        'S2': SoloWorld(crate_density=args.s2_crate_density,
                        coin_count=args.s2_coin_count,
                        max_steps=args.max_steps, feature_mode=args.features),
    }


STAGE_ID = {'S0': 0, 'S1': 1, 'S2': 2}


def potential(coin_distance, destroyed, kappa, safe_distance=0, rho=0.0,
              cap=ESCAPE_POTENTIAL_CAP) -> float:
    phi = -POTENTIAL_SCALE * coin_distance + kappa * destroyed
    if rho:
        d = safe_distance if safe_distance < cap else cap
        phi += rho * (1.0 - d / cap)
    return phi


def step_reward(info, prev_d, new_d, gamma, r_invalid, done, r_death=0.0,
                crate_mode='none', kappa=0.0,
                destroyed_before=0, destroyed_after=0,
                escape_mode='none', rho=0.0,
                safe_before=0, safe_after=0,
                escape_cap=ESCAPE_POTENTIAL_CAP,
                clawback=CRATE_CLAWBACK):
    r = REWARD_COIN * info['coins']
    if info['invalid']:
        r += r_invalid
    if info['suicide']:
        r += r_death
    if crate_mode == 'event':
        r += kappa * info['crates']
    k = kappa if crate_mode == 'potential' else 0.0
    e = rho if escape_mode == 'potential' else 0.0
    phi_old = potential(prev_d, destroyed_before, k, safe_before, e, escape_cap)
    if done and clawback == 'terminal':
        phi_new = 0.0
    else:
        phi_new = potential(new_d, destroyed_after, k, safe_after, e, escape_cap)
    return r + gamma * phi_new - phi_old


def plan_injection(mode, prob, stage, inject_rng, length_estimate,
                   pending=False, max_steps=MAX_STEPS, inject_stage='S1') -> dict:
    u = float(inject_rng.random())
    span = int(min(max(1, round(length_estimate)), max_steps))
    step = int(inject_rng.integers(1, span + 1))
    if mode == 'none' or stage == 'S0' or stage != inject_stage or prob <= 0.0:
        return {'mode': 'none', 'step': None}
    if not pending and u >= prob:
        return {'mode': 'none', 'step': None}
    return {'mode': mode, 'step': step if mode == 'random' else None}


def run_training_episode(world, model, learner, rng, eps, args, seed,
                         inject=None, stage='S1'):  # noqa: C901
    inject = inject or {'mode': 'none', 'step': None}
    mechanism = inject['mode']
    phi = world.reset(seed, inject_bomb=(mechanism == 'seed'))
    prev_d = world.coin_distance()
    escape_on = getattr(args, 'escape_reward', 'none') != 'none'
    escape_cap = getattr(args, 'escape_cap', ESCAPE_POTENTIAL_CAP)
    prev_safe = world.safe_distance(escape_cap) if escape_on else 0
    learner.begin_episode()
    total = 0.0
    stage_id = STAGE_ID[stage]
    destroyed = world.stats['crates']
    bomb_events = int(mechanism == 'seed' and world.stats['seeded_bombs'] > 0)
    fired = mechanism == 'seed'

    while world.running:
        override = False
        if not fired and world.bombs_left:
            if mechanism == 'random' and world.step + 1 == inject['step']:
                override = True
            elif mechanism == 'safe' and world.safe_here():
                override = True

        if override:
            action, exploratory = A_BOMB, False
            fired = True
            bomb_events += 1
        else:
            exploratory = rng.random() < eps
            action = int(rng.integers(N_ACTIONS)) if exploratory else model.greedy(phi)
        learner.mark_exploratory(exploratory)

        phi_next, info = world.step_action(action)
        new_d = DIST_CAP if info['done'] else world.coin_distance()
        destroyed_after = destroyed + info['crates']
        new_safe = 0
        if escape_on and not info['done']:
            new_safe = world.safe_distance(escape_cap)
        r = step_reward(info, prev_d, new_d, args.gamma, args.r_invalid,
                        info['done'], args.r_death,
                        crate_mode=args.crate_reward, kappa=args.crate_kappa,
                        destroyed_before=destroyed, destroyed_after=destroyed_after,
                        escape_mode=getattr(args, 'escape_reward', 'none'),
                        rho=getattr(args, 'escape_rho', 0.0),
                        safe_before=prev_safe, safe_after=new_safe,
                        escape_cap=escape_cap,
                        clawback=getattr(args, 'crate_clawback',
                                         CRATE_CLAWBACK))
        total += r
        learner.observe(phi, action, r, phi_next, info['done'], False,
                        stage=stage_id, injected=override,
                        destroyed=destroyed_after)
        phi, prev_d, destroyed, prev_safe = phi_next, new_d, destroyed_after, new_safe

    learner.end_episode()
    st = episode_stats(world, ret=total)
    st['inject'] = mechanism if bomb_events else 'none'
    st['planned_inject'] = mechanism
    st['bomb_events'] = bomb_events
    return st


def episode_stats(world, ret=None) -> dict:
    st = world.stats
    out = {
        'steps': world.step, 'coins': st['coins'], 'crates': st['crates'],
        'bombs': st['bombs'], 'invalid': st['invalid'],
        'idle': world.step - st['moves'] - st['bombs'],
        'suicide': st['suicide'],
    }
    if ret is not None:
        out['ret'] = ret
    return out


def run_greedy_episode(world, model, seed):
    phi = world.reset(seed)
    while world.running:
        phi, _info = world.step_action(model.greedy(phi))
    return episode_stats(world)


_GBR_CACHE = {}


def gbr_episode(args, seed):
    key = (seed, args.s0_crate_density, args.s0_coin_count, args.max_steps)
    if key in _GBR_CACHE:
        return _GBR_CACHE[key]
    world = SoloWorld(crate_density=args.s0_crate_density,
                      coin_count=args.s0_coin_count, max_steps=args.max_steps,
                      feature_mode=args.features)
    world.reset(seed)
    coin_steps = []
    while world.running:
        _phi, info = world.step_action(world.greedy_step())
        for _ in range(info['coins']):
            coin_steps.append(world.step)
    out = {'steps': world.step, 'coins': world.stats['coins'], 'coin_steps': coin_steps}
    _GBR_CACHE[key] = out
    return out


def gbr_prefix_steps(ref, k):
    if k <= 0 or k > len(ref['coin_steps']):
        return None
    return float(ref['coin_steps'][k - 1])


def probe_s0(model, args, n_episodes) -> dict:
    world = SoloWorld(crate_density=args.s0_crate_density,
                      coin_count=args.s0_coin_count, max_steps=args.max_steps,
                      feature_mode=args.features)
    steps = coins = invalid = suicides = 0
    matched_agent_steps = 0.0
    matched_gbr_steps = 0.0
    ratios = []
    censored = 0
    for i in range(n_episodes):
        seed = PROBE_SEED_BASE + i
        ep = run_greedy_episode(world, model, seed)
        ref = gbr_episode(args, seed)
        steps += ep['steps']
        coins += ep['coins']
        invalid += ep['invalid']
        suicides += ep['suicide']
        den = gbr_prefix_steps(ref, ep['coins'])
        if den:
            matched_agent_steps += ep['steps']
            matched_gbr_steps += den
            ratios.append(ep['steps'] / den)
        else:
            censored += 1

    out = {
        'ratio': (matched_agent_steps / matched_gbr_steps) if matched_gbr_steps
                 else float('inf'),
        'ratio_mean': float(np.mean(ratios)) if ratios else float('inf'),
        'censored': censored,
        'invalid_per_step': invalid / steps if steps else 0.0,
        'coins': coins / n_episodes,
        'steps': steps / n_episodes,
        'suicides': suicides / n_episodes,
        'episodes': n_episodes,
    }
    out['gate'] = bool(gate_s0(out, args))
    return out


def probe_s2(model, args, n_episodes) -> dict:
    world = SoloWorld(crate_density=args.s2_crate_density,
                      coin_count=args.s2_coin_count, max_steps=args.max_steps,
                      feature_mode=args.features)
    steps = coins = crates = bombs = invalid = idle = suicides = survived = 0
    for i in range(n_episodes):
        ep = run_greedy_episode(world, model, S2_PROBE_SEED_BASE + i)
        steps += ep['steps']
        coins += ep['coins']
        crates += ep['crates']
        bombs += ep['bombs']
        invalid += ep['invalid']
        idle += ep['idle']
        suicides += ep['suicide']
        survived += int(ep['suicide'] == 0)

    out = {
        'episodes': n_episodes,
        'crates': crates / n_episodes,
        'coins': coins / n_episodes,
        'bombs': bombs / n_episodes,
        'suicides': suicides / n_episodes,
        'suicides_per_bomb': (suicides / bombs) if bombs else None,
        'idle': idle / n_episodes,
        'steps': steps / n_episodes,
        'invalid_per_step': invalid / steps if steps else 0.0,
        'survival': survived / n_episodes,
    }
    out['gate'] = bool(gate_s2(out, args))
    return out


def probe_s1(model, args, n_episodes) -> dict:
    world = SoloWorld(crate_density=args.crate_density, coin_count=args.coin_count,
                      max_steps=args.max_steps, feature_mode=args.features)
    steps = coins = crates = bombs = invalid = idle = suicides = survived = 0
    for i in range(n_episodes):
        ep = run_greedy_episode(world, model, S1_PROBE_SEED_BASE + i)
        steps += ep['steps']
        coins += ep['coins']
        crates += ep['crates']
        bombs += ep['bombs']
        invalid += ep['invalid']
        idle += ep['idle']
        suicides += ep['suicide']
        survived += int(ep['suicide'] == 0)

    out = {
        'episodes': n_episodes,
        'suicides': suicides / n_episodes,
        'bombs': bombs / n_episodes,
        'suicides_per_bomb': (suicides / bombs) if bombs else None,
        'crates': crates / n_episodes,
        'coins': coins / n_episodes,
        'idle': idle / n_episodes,
        'steps': steps / n_episodes,
        'invalid_per_step': invalid / steps if steps else 0.0,
        'survival': survived / n_episodes,
    }
    out['gate'] = bool(gate_s1(out, args))
    return out


def forced_bomb_trial(world, model, seed, horizon):
    phi = world.reset(seed)
    dropped_at = None
    time_to_safety = None

    while world.running:
        if dropped_at is None and world.bombs_left and world.safe_here():
            action = A_BOMB
            dropped_at = world.step + 1
        else:
            action = model.greedy(phi)
        phi, _info = world.step_action(action)

        if dropped_at is None:
            continue
        elapsed = world.step - dropped_at
        if world.dead:
            return {'trial': 1, 'success': 0, 'time_to_safety': time_to_safety,
                    'steps': elapsed}
        if time_to_safety is None and elapsed >= 1 and world.safe_here():
            time_to_safety = elapsed
        if elapsed >= horizon:
            return {'trial': 1, 'success': 1, 'time_to_safety': time_to_safety,
                    'steps': elapsed}

    if dropped_at is None:
        return {'trial': 0, 'success': 0, 'time_to_safety': None, 'steps': world.step}
    return {'trial': 1, 'success': int(not world.dead),
            'time_to_safety': time_to_safety, 'steps': world.step - dropped_at}


def forced_stage_config(args, stage: str) -> tuple:
    if stage == 'S1':
        return (args.crate_density, args.coin_count, FORCED_SEED_BASE,
                args.forced_target)
    if stage == 'S2':
        return (args.s2_crate_density, args.s2_coin_count, FORCED_S2_SEED_BASE,
                args.forced_target_s2)
    raise ValueError(f'no forced-bomb block for stage {stage!r}')


def forced_bomb_probe(model, args, n_trials, oracle=None, stage='S1') -> dict:
    density, coins, seed_base, target = forced_stage_config(args, stage)
    world = SoloWorld(crate_density=density, coin_count=coins,
                      max_steps=args.max_steps, feature_mode=args.features)
    if oracle is None:
        oracle = FeasibilityCache(horizon=args.forced_horizon,
                                  crate_density=density,
                                  coin_count=coins,
                                  max_steps=args.max_steps)
    trials = successes = skipped = unescapable = 0
    feasible_successes = violations = 0
    histogram = {}
    for i in range(n_trials):
        seed = seed_base + i
        out = forced_bomb_trial(world, model, seed, args.forced_horizon)
        if not out['trial']:
            skipped += 1
            continue
        trials += 1
        successes += out['success']
        verdict = oracle.verdict(seed)
        if verdict == 0:
            unescapable += 1
            violations += int(out['success'])
        elif verdict == 1:
            feasible_successes += out['success']
        if out['success']:
            key = 'never' if out['time_to_safety'] is None else str(out['time_to_safety'])
            histogram[key] = histogram.get(key, 0) + 1
    feasible = trials - unescapable
    rate_raw = (successes / trials) if trials else 0.0
    rate_norm = (feasible_successes / feasible) if feasible else 0.0
    return {'stage': stage, 'trials': trials, 'successes': successes,
            'skipped': skipped,
            'unescapable': unescapable, 'feasible': feasible,
            'feasible_successes': feasible_successes,
            'rate_raw': rate_raw, 'rate_norm': rate_norm,
            'rate': rate_raw,
            'containment_violations': violations,
            'target': target,
            'pass': bool(feasible and rate_norm >= target),
            'time_to_safety': histogram}


def self_bomb_stats(model, args, n_episodes, stage='S2', oracle_cap=None) -> dict:
    if stage == 'S2':
        density, coins, base = (args.s2_crate_density, args.s2_coin_count,
                                S2_PROBE_SEED_BASE)
    elif stage == 'S1':
        density, coins, base = (args.crate_density, args.coin_count,
                                S1_PROBE_SEED_BASE)
    else:
        raise ValueError(f'no self-bomb block for stage {stage!r} -- S0 has no '
                         f'crates and a bomb there is pure downside')
    cap = args.sb_oracle_cap if oracle_cap is None else oracle_cap
    out = self_bomb_probe(model, args.features, n_episodes, seed_base=base,
                          crate_density=density, coin_count=coins,
                          max_steps=args.max_steps, horizon=args.forced_horizon,
                          oracle_cap=cap)
    out.pop('episodes_detail', None)
    out['stage'] = stage
    out['oracle_cap'] = int(cap)
    return out


def self_bomb_row(sb, episode, wall_s) -> dict:
    def r(value, nd=4):
        return '' if value is None else round(float(value), nd)

    return {
        'episode': episode, 'stage': sb['stage'], 'episodes_probed': sb['episodes'],
        'sb_bombs': sb['sb_bombs'],
        'sb_bombs_evaluated': sb['sb_bombs_evaluated'],
        'sb_feasible': sb['sb_feasible'],
        'sb_survived_total': sb['sb_survived_total'],
        'sb_survived_feasible': sb['sb_survived_feasible'],
        'sb_placement_ok': r(sb['sb_placement_ok']),
        'sb_escape_norm': r(sb['sb_escape_norm']),
        'sb_escape_raw': r(sb['sb_escape_raw']),
        'sb_oracle_capped': sb['sb_oracle_capped'],
        'oracle_cap': sb.get('oracle_cap', ''),
        'sb_survived_infeasible': sb['sb_survived_infeasible'],
        'bombs_per_episode': r(sb['bombs_per_episode']),
        'suicides_per_episode': r(sb['suicides_per_episode']),
        'suicides_per_bomb': r(sb['suicides_per_bomb']),
        'crates_per_episode': r(sb['crates_per_episode'], 3),
        'coins_per_episode': r(sb['coins_per_episode'], 3),
        'oracle_calls': sb['oracle_calls'],
        'oracle_ms_per_call': r(sb['oracle_ms_per_call'], 3),
        'wall_s': round(wall_s, 3),
    }


def bomb_margin(model, phi) -> float:
    q = model.q(phi)
    return float(q[A_BOMB] - np.max(np.delete(q, A_BOMB)))


def bomb_margin_probe(model, args, n_states) -> dict:
    world = SoloWorld(crate_density=args.crate_density, coin_count=args.coin_count,
                      max_steps=args.max_steps, feature_mode=args.features)
    starts, post = [], []
    for i in range(n_states):
        seed = S1_PROBE_SEED_BASE + i
        phi = world.reset(seed)
        starts.append(bomb_margin(model, phi))
        if world.bombs_left and world.safe_here():
            phi_after, info = world.step_action(A_BOMB)
            if not info['done']:
                post.append(bomb_margin(model, phi_after))

    s2_world = SoloWorld(crate_density=args.s2_crate_density,
                         coin_count=args.s2_coin_count, max_steps=args.max_steps,
                         feature_mode=args.features)
    s2_starts = [bomb_margin(model, s2_world.reset(S2_PROBE_SEED_BASE + i))
                 for i in range(n_states)]
    return {
        'states': n_states,
        'start_mean': float(np.mean(starts)) if starts else float('nan'),
        'start_positive': float(np.mean([m > 0 for m in starts])) if starts else 0.0,
        'post_states': len(post),
        'post_mean': float(np.mean(post)) if post else float('nan'),
        'post_positive': float(np.mean([m > 0 for m in post])) if post else 0.0,
        's2_start_mean': float(np.mean(s2_starts)) if s2_starts else float('nan'),
        's2_start_positive': (float(np.mean([m > 0 for m in s2_starts]))
                              if s2_starts else 0.0),
    }


def gate_s0(p, args) -> bool:
    return p['ratio'] <= args.gate_ratio and p['invalid_per_step'] < args.gate_invalid


def gate_s1(p, args) -> bool:
    return p['bombs'] >= args.gate_bombs and p['suicides'] < args.gate_suicides


def gate_s2(p, args) -> bool:
    return float(p.get('crates') or 0.0) >= args.gate_crates


def e_gate(p2, sb, args) -> dict:
    bombs = float(p2.get('bombs') or 0.0)
    spb = p2.get('suicides_per_bomb')
    spb = None if (not bombs or spb is None) else float(spb)
    norm = (sb or {}).get('sb_escape_norm')
    feasible = int((sb or {}).get('sb_feasible') or 0)
    e1 = bombs >= args.e_gate_bombs
    e2 = spb is not None and spb <= args.e_gate_suicides_per_bomb
    e3 = norm is not None and float(norm) >= args.e_gate_escape_norm
    return {
        'e1': {'statistic': 'bombs/episode (S2 probe)', 'value': bombs,
               'threshold': args.e_gate_bombs, 'pass': bool(e1)},
        'e2': {'statistic': 'suicides/bomb (S2 probe)', 'value': spb,
               'threshold': args.e_gate_suicides_per_bomb, 'pass': bool(e2)},
        'e3': {'statistic': 'sb_escape_norm (self-chosen bombs, S2)',
               'value': None if norm is None else float(norm),
               'denominator': feasible,
               'threshold': args.e_gate_escape_norm, 'pass': bool(e3)},
        'pass': bool(e1 and e2 and e3),
        'rule': 'e1 and e2 and e3. b = max(0.10, measured suicides/bomb of '
                'rule_based_agent on solo loot-crate at N = 200) and the e3 '
                'threshold = max(0.75, 0.90 x the Step-0 anchor); both were '
                'fixed before any arm number existed and are never tightened.',
    }


def gate_all(p2, p1, p0, args) -> bool:
    return bool(gate_s2(p2, args) and gate_s1(p1, args) and gate_s0(p0, args))


def s0_eligible(p0, args) -> bool:
    return bool(gate_s0(p0, args))


def probe_rank(p2, p1, p0, args, fb2=None, sb=None) -> tuple:
    ratio = p0['ratio'] if np.isfinite(p0['ratio']) else float('inf')
    gated = gate_all(p2, p1, p0, args)
    crates = float(p2.get('crates') or 0.0)
    norm = (sb or {}).get('sb_escape_norm')
    norm = 0.0 if norm is None else float(norm)
    placement = (sb or {}).get('sb_placement_ok')
    placement = 0.0 if placement is None else float(placement)
    bombs = float(p1.get('bombs') or 0.0)
    spb = p1.get('suicides_per_bomb')
    spb = float('inf') if (not bombs or spb is None) else float(spb)
    return (not gated,
            -round(min(norm, 1.0), 2),
            -round(min(placement, 1.0), 2),
            -round(min(crates / args.gate_crates, 1.0), 2),
            -round(min(bombs, args.gate_bombs), 2),
            spb,
            ratio,
            -float(p2.get('coins') or 0.0))


def bomb_propensity(c) -> float:
    return float((c.get('p2') or {}).get('bombs') or 0.0)


def select_checkpoint(candidates, args) -> dict:
    considered = list(candidates)
    if not considered:
        return {'best': None, 's0_filter_empty': False,
                'bomb_filter_empty': False, 'probe_rank_regret': None,
                'eligible': 0, 'considered': 0}
    eligible = [c for c in considered if s0_eligible(c['p0'], args)]
    empty = not eligible
    if empty:
        eligible = considered
    bombers = [c for c in eligible
               if bomb_propensity(c) >= args.e_gate_bombs]
    bomb_empty = not bombers
    population = eligible if bomb_empty else bombers
    best = min(population,
               key=lambda c: (probe_rank(c['p2'], c['p1'], c['p0'], args,
                                         c.get('fb2'), c.get('sb')),
                              c['episode']))

    def norm_of(c):
        value = (c.get('sb') or {}).get('sb_escape_norm')
        return None if value is None else float(value)

    norms = [n for n in (norm_of(c) for c in population) if n is not None]
    selected = norm_of(best)
    regret = (round(max(norms) - selected, 6)
              if (norms and selected is not None) else None)
    return {'best': best, 's0_filter_empty': empty,
            'bomb_filter_empty': bomb_empty,
            'probe_rank_regret': regret,
            'eligible': 0 if empty else len(eligible),
            'bombing_candidates': 0 if bomb_empty else len(bombers),
            'considered': len(considered)}


def seed_bimodality(s2_rows, final_s2, final_sb, buffer_rows, args) -> dict:
    bombs = float(final_s2.get('bombs') or 0.0)
    first = None
    for row in s2_rows:
        if float(row.get('bombs') or 0.0) >= args.e_gate_bombs:
            first = int(row['episode'])
            break
    last = buffer_rows[-1] if buffer_rows else {}

    def cell(key):
        value = last.get(key)
        return None if value is None else value

    return {
        'final_bombs_s2': bombs,
        'bombs_e1': bool(bombs >= args.e_gate_bombs),
        'bombs_any': bool(bombs > 0.0),
        'first_bomb_episode': first,
        'final_sb_bombs': (final_sb or {}).get('sb_bombs'),
        'final_sb_feasible': (final_sb or {}).get('sb_feasible'),
        'buffer_share_BOMB': cell('share_BOMB'),
        'buffer_mean_r_BOMB': cell('mean_r_BOMB'),
        'buffer_mean_target_BOMB': cell('mean_target_BOMB'),
        'buffer_mean_r_terminal': cell('mean_r_terminal'),
        'buffer_mean_destroyed_at_terminal': cell('mean_destroyed_at_terminal'),
        'buffer_effective_death_reward': cell('effective_death_reward'),
        'conj_place_support': cell('conj_place_support'),
        'conj_place_support_BOMB': cell('conj_place_support_BOMB'),
        'lstd_cond': cell('cond'),
        'why': '',
    }


def episodes_to_gate(probe_rows, column='gate'):
    for i in range(len(probe_rows) - 1):
        if probe_rows[i][column] and probe_rows[i + 1][column]:
            return probe_rows[i]['episode']
    return None


def forced_trials_for(args, n_trials: int, last: bool) -> int:
    if last and args.forced_trials_s2_final > 0:
        return args.forced_trials_s2_final
    if not last and args.forced_trials_s2 > 0:
        return args.forced_trials_s2
    return n_trials


def strip_raw_s2(fb) -> dict:
    if not fb or fb.get('stage') != 'S2':
        return fb
    out = dict(fb)
    out['rate_raw'] = None
    out['rate'] = None
    out['rate_raw_note'] = ('')
    return out


def forced_row(fb, episode, wall_s) -> dict:
    fb = strip_raw_s2(fb)
    return {
        'episode': episode, 'stage': fb['stage'], 'trials': fb['trials'],
        'unescapable': fb['unescapable'], 'feasible': fb['feasible'],
        'successes': fb['successes'],
        'feasible_successes': fb['feasible_successes'],
        'rate_raw': ('' if fb['rate_raw'] is None
                     else round(fb['rate_raw'], 4)),
        'rate_norm': round(fb['rate_norm'], 4),
        'skipped': fb['skipped'], 'target': fb['target'],
        'pass': int(fb['pass']),
        'containment_violations': fb['containment_violations'],
        'time_to_safety': json.dumps(fb['time_to_safety'], sort_keys=True),
        'wall_s': round(wall_s, 3),
    }


def bench(args):
    dim = dim_for(args.features)
    model = LinearQ(dim=dim)
    learner = make_learner(args.rule, model, args)
    world = make_worlds(args)['S2']
    rng = np.random.default_rng(0)
    run_training_episode(world, model, learner, rng, 1.0, args, 0,
                         stage='S2')  # warm up numba

    t0 = time.perf_counter()
    episodes = steps = 0
    while time.perf_counter() - t0 < args.bench:
        st = run_training_episode(world, model, learner, rng, 1.0, args,
                                  1 + episodes, stage='S2')
        episodes += 1
        steps += st['steps']
    dt = time.perf_counter() - t0

    out = {'seconds': dt, 'episodes': episodes, 'steps': steps,
           'steps_per_s': steps / dt, 'episodes_per_s': episodes / dt,
           'dim': dim, 'block_dim': N_ACTIONS * dim, 'rule': args.rule,
           'features': args.features,
           'cpu': platform.processor() or platform.machine(),
           'platform': platform.platform(),
           'threads': os.environ.get('OMP_NUM_THREADS', 'unset')}
    if isinstance(learner, LSTDQ):
        t1 = time.perf_counter()
        learner.solve()
        out['lstd_solve_s'] = time.perf_counter() - t1
        out['lstd_buffer_used'] = int(learner.size)
    return out


def train(args):
    dim = dim_for(args.features)
    model = LinearQ(dim=dim)
    if args.init_weights:
        model = LinearQ(np.load(args.init_weights), dim=dim)
    learner = make_learner(args.rule, model, args)
    worlds = make_worlds(args)
    rng = np.random.default_rng(10_000 + args.seed)
    stage_rng = np.random.default_rng(STAGE_SEED_BASE + args.seed)
    inject_rng = np.random.default_rng(INJECT_SEED_BASE + args.seed)
    waypoints = CURRICULA[args.curriculum]
    oracle = FeasibilityCache(args.escape_feasible, horizon=args.forced_horizon,
                              crate_density=args.crate_density,
                              coin_count=args.coin_count, max_steps=args.max_steps)
    oracle_s2 = FeasibilityCache(args.escape_feasible_s2,
                                 horizon=args.forced_horizon,
                                 crate_density=args.s2_crate_density,
                                 coin_count=args.s2_coin_count,
                                 max_steps=args.max_steps)

    log_rows = []
    s2_rows = []
    s1_rows = []
    s0_rows = []
    forced_rows = []
    forced_s2_rows = []
    margin_rows = []
    buffer_rows = []
    sb_rows = []
    last_fb = None
    last_fb2 = None
    candidates = []
    probe_at = None
    s1_length = float(args.max_steps)   # estimate of the next S1 episode's length
    inject_pending = False              # a planned injection that never fired
    t0 = time.perf_counter()

    for episode in range(args.episodes):
        u = float(stage_rng.random())
        probs = mixture3(episode, waypoints)
        stage = draw_stage(u, probs)
        world = worlds[stage]

        eps = epsilon(episode, args.eps_start, args.eps_end, args.eps_decay)
        inject = plan_injection(args.inject, args.inject_prob, stage, inject_rng,
                                s1_length, pending=inject_pending,
                                max_steps=args.max_steps,
                                inject_stage=args.inject_stage)
        st = run_training_episode(world, model, learner, rng, eps, args,
                                  board_seed(args.seed, episode),
                                  inject=inject, stage=stage)
        if stage == 'S1':
            s1_length = 0.5 * s1_length + 0.5 * st['steps']
            inject_pending = (st['planned_inject'] != 'none'
                              and st['bomb_events'] == 0)
        solves_before = getattr(learner, 'solves', 0)
        learner.episode_end_hook(episode)
        if getattr(learner, 'solves', 0) > solves_before:
            composition = getattr(learner, 'last_composition', None) or {}
            buffer_rows.append({'episode': episode + 1, 'solve': learner.solves,
                                **{k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in composition.items()},
                                'wall_s': round(time.perf_counter() - t0, 3)})
        crate_reward_paid = (args.crate_kappa * st['crates']
                             if args.crate_reward != 'none' else 0.0)
        log_rows.append({
            'episode': episode, 'stage': stage, 'steps': st['steps'],
            'coins': st['coins'], 'crates': st['crates'],
            'crate_reward': round(crate_reward_paid, 4), 'bombs': st['bombs'],
            'invalid': st['invalid'], 'idle': st['idle'], 'suicide': st['suicide'],
            'inject': st['inject'], 'bomb_events': st['bomb_events'],
            'return': round(st['ret'], 4), 'w_norm': round(model.norm(), 6),
            'eps': round(eps, 4), 'p_s0': round(probs[0], 4),
            'p_s1': round(probs[1], 4), 'p_s2': round(probs[2], 4),
            'alpha': args.alpha,
            'wall_s': round(time.perf_counter() - t0, 3),
        })

        last = episode == args.episodes - 1
        if (episode + 1) % args.forced_every == 0 or last:
            n_trials = args.forced_trials_final if last else args.forced_trials
            n_trials_s2 = forced_trials_for(args, n_trials, last)
            fb2 = forced_bomb_probe(model, args, n_trials_s2, oracle_s2, stage='S2')
            last_fb2 = fb2
            forced_s2_rows.append(forced_row(fb2, episode + 1,
                                             time.perf_counter() - t0))
            fb = forced_bomb_probe(model, args, n_trials, oracle, stage='S1')
            last_fb = fb
            forced_rows.append(forced_row(fb, episode + 1,
                                          time.perf_counter() - t0))
            if not args.quiet:
                print(f'  ep {episode + 1:5d}       forced-escape norm '
                      f'S2 {fb2["rate_norm"]:.3f} ({fb2["feasible_successes"]}/'
                      f'{fb2["feasible"]} feasible, {fb2["unescapable"]} declined) | '
                      f'S1 {fb["rate_norm"]:.3f} ({fb["feasible_successes"]}/'
                      f'{fb["feasible"]} feasible, {fb["unescapable"]} declined)',
                      flush=True)

        if (episode + 1) % args.probe_every == 0 or last:
            n1 = args.probe_episodes_final if last else args.probe_episodes
            p2 = probe_s2(model, args, n1)
            p1 = probe_s1(model, args, n1)
            p0 = probe_s0(model, args, n1)
            margin = bomb_margin_probe(model, args, n1)
            n_sb = args.sb_episodes_final if last else args.sb_episodes
            sb = self_bomb_stats(model, args, n_sb, stage='S2') if n_sb else None
            sb1 = (self_bomb_stats(model, args, n_sb, stage='S1')
                   if (n_sb and args.sb_stage_s1) else None)
            now = time.perf_counter() - t0
            if sb:
                sb_rows.append(self_bomb_row(sb, episode + 1, now))
            if sb1:
                sb_rows.append(self_bomb_row(sb1, episode + 1, now))
            margin_rows.append({
                'episode': episode + 1, 'states': margin['states'],
                'start_mean': round(margin['start_mean'], 6),
                'start_positive': round(margin['start_positive'], 4),
                'post_states': margin['post_states'],
                'post_mean': round(margin['post_mean'], 6),
                'post_positive': round(margin['post_positive'], 4),
                's2_start_mean': round(margin['s2_start_mean'], 6),
                's2_start_positive': round(margin['s2_start_positive'], 4),
                'wall_s': round(time.perf_counter() - t0, 3),
            })
            candidates.append({'episode': episode + 1, 'p2': dict(p2),
                               'p1': dict(p1), 'p0': dict(p0),
                               'fb2': dict(last_fb2) if last_fb2 else None,
                               'sb': dict(sb) if sb else None,
                               'w': model.w.copy()})
            if episode + 1 == args.probe_at:
                probe_at = {'episode': episode + 1, 's2': jsonable(p2),
                            's1': jsonable(p1), 's0': jsonable(p0),
                            'bomb_margin': jsonable(margin),
                            'self_bomb_s2': jsonable(sb) if sb else None,
                            'self_bomb_s1': jsonable(sb1) if sb1 else None,
                            'forced_bomb_s2': (jsonable(strip_raw_s2(last_fb2))
                                               if last_fb2 else None),
                            'forced_bomb_s1': jsonable(last_fb) if last_fb else None,
                            'e_gate': e_gate(p2, sb, args),
                            'gate_all': bool(gate_all(p2, p1, p0, args)),
                            'why': ''}
            gated = int(gate_all(p2, p1, p0, args))
            both = int(p1['gate'] and p0['gate'])
            s2_rows.append({
                'episode': episode + 1, 'episodes_probed': p2['episodes'],
                'crates': round(p2['crates'], 3), 'coins': round(p2['coins'], 3),
                'bombs': round(p2['bombs'], 4),
                'suicides': round(p2['suicides'], 4),
                'suicides_per_bomb': ('' if p2['suicides_per_bomb'] is None
                                      else round(p2['suicides_per_bomb'], 4)),
                'idle': round(p2['idle'], 2), 'steps': round(p2['steps'], 2),
                'invalid_per_step': round(p2['invalid_per_step'], 6),
                'survival': round(p2['survival'], 4),
                'bomb_margin_start': round(margin['s2_start_mean'], 6),
                'bomb_margin_positive': round(margin['s2_start_positive'], 4),
                'fb_rate_norm': (round(last_fb2['rate_norm'], 4)
                                 if last_fb2 else ''),
                'sb_escape_norm': ('' if not sb or sb['sb_escape_norm'] is None
                                   else round(sb['sb_escape_norm'], 4)),
                'sb_placement_ok': ('' if not sb or sb['sb_placement_ok'] is None
                                    else round(sb['sb_placement_ok'], 4)),
                'sb_feasible': (sb['sb_feasible'] if sb else ''),
                'sb_bombs': (sb['sb_bombs'] if sb else ''),
                'e_gate': int(e_gate(p2, sb, args)['pass']),
                'gate_s2': int(p2['gate']), 'gate_all': gated,
                'w_norm': round(model.norm(), 6),
                'wall_s': round(time.perf_counter() - t0, 3),
            })
            s1_rows.append({
                'episode': episode + 1, 'episodes_probed': p1['episodes'],
                'suicides': round(p1['suicides'], 4),
                'bombs': round(p1['bombs'], 4),
                'suicides_per_bomb': ('' if p1['suicides_per_bomb'] is None
                                      else round(p1['suicides_per_bomb'], 4)),
                'crates': round(p1['crates'], 3), 'coins': round(p1['coins'], 3),
                'idle': round(p1['idle'], 2), 'steps': round(p1['steps'], 2),
                'invalid_per_step': round(p1['invalid_per_step'], 6),
                'survival': round(p1['survival'], 4),
                'bomb_margin_start': round(margin['start_mean'], 6),
                'bomb_margin_positive': round(margin['start_positive'], 4),
                'gate_s1': int(p1['gate']), 'gate_both': both,
                'gate_all': gated,
                'w_norm': round(model.norm(), 6),
                'wall_s': round(time.perf_counter() - t0, 3),
            })
            s0_rows.append({
                'episode': episode + 1, 'episodes_probed': p0['episodes'],
                'ratio': round(p0['ratio'], 5) if np.isfinite(p0['ratio']) else '',
                'ratio_mean': round(p0['ratio_mean'], 5) if np.isfinite(p0['ratio_mean']) else '',
                'invalid_per_step': round(p0['invalid_per_step'], 6),
                'coins': round(p0['coins'], 3), 'steps': round(p0['steps'], 2),
                'suicides': round(p0['suicides'], 4), 'censored': p0['censored'],
                'gate_s0': int(p0['gate']), 'gate_both': both,
                'gate_all': gated,
                'w_norm': round(model.norm(), 6),
                'wall_s': round(time.perf_counter() - t0, 3),
            })
            if not args.quiet:
                print(f'  ep {episode + 1:5d} [{stage}]  S2: crates={p2["crates"]:.2f} '
                      f'coins={p2["coins"]:.2f} bombs={p2["bombs"]:.2f} '
                      f'suicides={p2["suicides"]:.3f} idle={p2["idle"]:.0f} '
                      f'gate={int(p2["gate"])} | S1: bombs={p1["bombs"]:.2f} '
                      f'suicides={p1["suicides"]:.3f} gate={int(p1["gate"])} | '
                      f'S0: ratio={p0["ratio"]:.3f} gate={int(p0["gate"])} | '
                      f'margin S2={margin["s2_start_mean"]:+.3f} '
                      f'S1={margin["start_mean"]:+.3f}',
                      flush=True)

    wall = time.perf_counter() - t0
    selection = select_checkpoint(candidates, args)
    best_c = selection['best']
    best_w = best_c['w'] if best_c is not None else None
    best_episode = best_c['episode'] if best_c is not None else None
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        model.save(args.out)
        if best_w is not None:
            LinearQ(best_w).save(best_weights_path(args.out))
    write_csv(args.log_csv, log_rows)
    write_csv(args.probe_s2_csv, s2_rows)
    write_csv(args.probe_s1_csv, s1_rows)
    write_csv(args.probe_s0_csv, s0_rows)
    write_csv(args.forced_csv, forced_rows)
    write_csv(args.forced_s2_csv, forced_s2_rows)
    write_csv(args.bomb_margin_csv, margin_rows)
    write_csv(args.buffer_composition_csv, buffer_rows)
    write_csv(args.self_bomb_csv, sb_rows)

    final_s2 = probe_s2(model, args, args.probe_episodes_final)
    final_s1 = probe_s1(model, args, args.probe_episodes_final)
    final_s0 = probe_s0(model, args, args.probe_episodes_final)
    final_fb = forced_bomb_probe(model, args, args.forced_trials_final, oracle,
                                 stage='S1')
    final_fb2 = forced_bomb_probe(model, args,
                                  forced_trials_for(args, args.forced_trials_final,
                                                    True),
                                  oracle_s2, stage='S2')
    final_margin = bomb_margin_probe(model, args, args.probe_episodes_final)
    final_sb = (self_bomb_stats(model, args, args.sb_episodes_final, stage='S2',
                                oracle_cap=args.sb_oracle_cap_final)
                if args.sb_episodes_final else None)
    final_sb1 = (self_bomb_stats(model, args, args.sb_episodes_final, stage='S1')
                 if (args.sb_episodes_final and args.sb_stage_s1) else None)
    if final_sb:
        sb_rows.append(self_bomb_row(final_sb, args.episodes,
                                     time.perf_counter() - t0))
    if final_sb1:
        sb_rows.append(self_bomb_row(final_sb1, args.episodes,
                                     time.perf_counter() - t0))
    write_csv(args.self_bomb_csv, sb_rows)
    final_e_gate = e_gate(final_s2, final_sb, args)
    if args.escape_feasible:
        oracle.save()
    if args.escape_feasible_s2:
        oracle_s2.save()
    injected_episodes = sum(1 for r in log_rows if r['inject'] != 'none')
    bomb_events = sum(r['bomb_events'] for r in log_rows)
    stage_counts = {tag: sum(1 for r in log_rows if r['stage'] == tag)
                    for tag in ('S0', 'S1', 'S2')}
    s1_episodes = stage_counts['S1']
    summary = {
        'variant': args.name, 'rule': args.rule, 'features': args.features,
        'dim': dim, 'seed': args.seed, 'episodes': args.episodes,
        'alpha': args.alpha, 'gamma': args.gamma, 'n_step': args.n_step,
        'r_invalid': args.r_invalid, 'r_death': args.r_death,
        'crate_reward': args.crate_reward, 'crate_kappa': args.crate_kappa,
        'crate_clawback': args.crate_clawback,
        'escape_reward': args.escape_reward, 'escape_rho': args.escape_rho,
        'curriculum': args.curriculum,
        'mix_waypoints': [list(w) for w in CURRICULA[args.curriculum]],
        'inject': args.inject, 'inject_prob': args.inject_prob,
        'inject_stage': args.inject_stage,
        'injected_episodes': injected_episodes,
        'bomb_events': bomb_events,
        'bomb_events_per_episode': bomb_events / args.episodes if args.episodes else 0.0,
        'bomb_events_per_s1_episode': (bomb_events / s1_episodes) if s1_episodes else 0.0,
        'stage_counts': stage_counts,
        'crates_trained': sum(r['crates'] for r in log_rows),
        'crate_reward_paid': round(sum(r['crate_reward'] for r in log_rows), 3),
        'gate': {'ratio': args.gate_ratio, 'invalid': args.gate_invalid,
                 'bombs': args.gate_bombs, 'suicides': args.gate_suicides,
                 'crates': args.gate_crates,
                 'forced_target': args.forced_target,
                 'forced_target_s2': args.forced_target_s2},
        'e_gate_thresholds': {'e1_bombs': args.e_gate_bombs,
                              'e2_suicides_per_bomb': args.e_gate_suicides_per_bomb,
                              'e3_sb_escape_norm': args.e_gate_escape_norm},
        'wall_s': wall, 'w_norm': model.norm(),
        'lstd_solves': getattr(learner, 'solves', 0),
        'lstd_solve_s': round(getattr(learner, 'solve_seconds', 0.0), 3),
        'lstd_cond': getattr(learner, 'last_cond', None),
        'episodes_to_gate_s2': episodes_to_gate(s2_rows, 'gate_s2'),
        'episodes_to_gate_s1': episodes_to_gate(s1_rows, 'gate_s1'),
        'episodes_to_gate_s0': episodes_to_gate(s0_rows, 'gate_s0'),
        'episodes_to_gate': episodes_to_gate(s2_rows, 'gate_all'),
        'final_s2': jsonable(final_s2),
        'final_s1': jsonable(final_s1),
        'final_s0': jsonable(final_s0),
        'final_forced_bomb': jsonable(final_fb),
        'final_forced_bomb_s2': jsonable(strip_raw_s2(final_fb2)),
        'final_bomb_margin': jsonable(final_margin),
        'final_self_bomb': jsonable(final_sb) if final_sb else None,
        'final_self_bomb_s1': jsonable(final_sb1) if final_sb1 else None,
        'final_e_gate': final_e_gate,
        'probe_at_4000': probe_at,
        'final_buffer_composition': (jsonable(buffer_rows[-1]) if buffer_rows
                                     else None),
        'escape_ceiling': oracle.summary(
            [FORCED_SEED_BASE + i for i in range(args.forced_trials_final)]),
        'escape_ceiling_s2': oracle_s2.summary(
            [FORCED_S2_SEED_BASE + i
             for i in range(forced_trials_for(args, args.forced_trials_final,
                                              True))]),
        'best': ({'episode': best_episode, 's2': jsonable(best_c['p2']),
                  's1': jsonable(best_c['p1']), 's0': jsonable(best_c['p0']),
                  'forced_bomb_s2': (jsonable(strip_raw_s2(best_c['fb2']))
                                     if best_c['fb2'] else None),
                  'self_bomb': jsonable(best_c['sb']) if best_c['sb'] else None,
                  'e_gate': e_gate(best_c['p2'], best_c['sb'], args),
                  'rank_key': [None if v == float('inf') else v
                               for v in probe_rank(best_c['p2'], best_c['p1'],
                                                   best_c['p0'], args,
                                                   best_c['fb2'], best_c['sb'])]}
                 if best_c else None),
        'bimodality': seed_bimodality(s2_rows, final_s2, final_sb, buffer_rows,
                                      args),
        's0_filter_empty': selection['s0_filter_empty'],
        'bomb_filter_empty': selection['bomb_filter_empty'],
        'probe_rank_regret': selection['probe_rank_regret'],
        'checkpoints_considered': selection['considered'],
        'checkpoints_s0_eligible': selection['eligible'],
        'checkpoints_bombing': selection.get('bombing_candidates', 0),
        'weights': args.out,
        'weights_best': best_weights_path(args.out) if (args.out and best_w is not None)
                        else None,
        'selection': 'lexicographic key with a hard filter on the S0 gate: gates, sb_escape_norm, sb_placement_ok, crates, bombs, suicides_per_bomb, s0_ratio, coins',
    }
    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, 'w') as fh:
            json.dump(summary, fh, indent=2)
    if args.probe_at_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.probe_at_json)),
                    exist_ok=True)
        with open(args.probe_at_json, 'w') as fh:
            json.dump(probe_at if probe_at else {
                'episode': args.probe_at, 'reached': False,
                'why': ''}, fh, indent=2)
    if args.weights_table:
        write_weights_table(args, model)
    if not args.quiet:
        sb_txt = 'n/a'
        if final_sb:
            norm = final_sb['sb_escape_norm']
            sb_txt = (f'{norm:.3f}' if norm is not None else 'NaN')
            sb_txt += (f' ({final_sb["sb_survived_feasible"]}/'
                       f'{final_sb["sb_feasible"]} feasible of '
                       f'{final_sb["sb_bombs"]} bombs)')
        print(f'[{args.name} seed {args.seed}] {args.episodes} episodes in {wall:.1f}s, '
              f'S2 crates={final_s2["crates"]:.2f} coins={final_s2["coins"]:.2f} '
              f'bombs={final_s2["bombs"]:.2f} suicides={final_s2["suicides"]:.3f} '
              f'gate={int(final_s2["gate"])}, '
              f'S1 bombs={final_s1["bombs"]:.2f} suicides={final_s1["suicides"]:.3f}, '
              f'S0 ratio={final_s0["ratio"]:.3f}, '
              f'sb_escape_norm={sb_txt}, E-gate={int(final_e_gate["pass"])}, '
              f'fb_rate_norm S2={final_fb2["rate_norm"]:.3f}/'
              f'S1={final_fb["rate_norm"]:.3f}, '
              f'bomb_margin S2={final_margin["s2_start_mean"]:+.3f}/'
              f'S1={final_margin["start_mean"]:+.3f}, '
              f'cond={summary["lstd_cond"]}, '
              f's0_filter_empty={int(selection["s0_filter_empty"])}, '
              f'bomb_filter_empty={int(selection["bomb_filter_empty"])}, '
              f'probe_rank_regret={selection["probe_rank_regret"]}, '
              f'episodes_to_gate={summary["episodes_to_gate"]}', flush=True)
    return summary


def jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and not np.isfinite(v):
            out[k] = None
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def write_weights_table(args, model):
    os.makedirs(os.path.dirname(os.path.abspath(args.weights_table)), exist_ok=True)
    with open(args.weights_table, 'w') as fh:
        fh.write(f'### {args.name} (rule={args.rule}, features={args.features}, '
                 f'd={model.dim}, r_death={args.r_death}, seed={args.seed})\n\n')
        fh.write(model.table(feature_names(args.features), ACTIONS))
        fh.write('\n')


def best_weights_path(out: str) -> str:
    root, ext = os.path.splitext(out)
    return f'{root}_best{ext}'


def write_csv(path, rows):
    if not path or not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    p = argparse.ArgumentParser(description='')
    p.add_argument('--name', default='lin_agent', help='')
    p.add_argument('--rule', default='lstd', choices=['q1', 'nstep', 'lstd'])
    p.add_argument('--features', default=DEFAULT_MODE, choices=list(FEATURE_MODES))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--episodes', type=int, default=2000)
    p.add_argument('--alpha', type=float, default=0.02)
    p.add_argument('--gamma', type=float, default=GAMMA)
    p.add_argument('--n-step', type=int, default=1)
    p.add_argument('--lstd-every', type=int, default=25)
    p.add_argument('--lstd-buffer', type=int, default=100_000)
    p.add_argument('--lstd-ridge', type=float, default=1e-3)
    p.add_argument('--eps-start', type=float, default=EPS_START)
    p.add_argument('--eps-end', type=float, default=EPS_END)
    p.add_argument('--eps-decay', type=int, default=EPS_DECAY_EPISODES)
    p.add_argument('--r-invalid', type=float, default=R_INVALID,
                   help='')
    p.add_argument('--r-death', type=float, default=R_DEATH,
                   help='')
    p.add_argument('--crate-reward', default=CRATE_REWARD_MODE,
                   choices=list(CRATE_REWARD_MODES),
                   help='')
    p.add_argument('--crate-kappa', type=float, default=CRATE_KAPPA,
                   help='')
    p.add_argument('--crate-clawback', default=CRATE_CLAWBACK,
                   choices=list(CRATE_CLAWBACK_MODES),
                   help='')
    p.add_argument('--escape-reward', default=ESCAPE_REWARD_MODE,
                   choices=list(ESCAPE_REWARD_MODES),
                   help='')
    p.add_argument('--escape-rho', type=float, default=0.0,
                   help='')
    p.add_argument('--escape-cap', type=int, default=ESCAPE_POTENTIAL_CAP,
                   help='')
    p.add_argument('--curriculum', default='mix', choices=sorted(CURRICULA),
                   help='')
    p.add_argument('--inject', default='none', choices=list(INJECT_MODES),
                   help='')
    p.add_argument('--inject-prob', type=float, default=INJECT_PROB,
                   help='')
    p.add_argument('--inject-stage', default='S1', choices=list(INJECT_STAGES),
                   help='')
    p.add_argument('--escape-feasible', default='',
                   help='')
    p.add_argument('--escape-feasible-s2', default='',
                   help='')
    p.add_argument('--crate-density', type=float, default=S1_CRATE_DENSITY,
                   help='')
    p.add_argument('--coin-count', type=int, default=S1_COIN_COUNT)
    p.add_argument('--s0-crate-density', type=float, default=S0_CRATE_DENSITY)
    p.add_argument('--s0-coin-count', type=int, default=S0_COIN_COUNT)
    p.add_argument('--s2-crate-density', type=float, default=S2_CRATE_DENSITY,
                   help='')
    p.add_argument('--s2-coin-count', type=int, default=S2_COIN_COUNT)
    p.add_argument('--max-steps', type=int, default=MAX_STEPS)
    p.add_argument('--probe-every', type=int, default=100)
    p.add_argument('--probe-episodes', type=int, default=10,
                   help='')
    p.add_argument('--probe-episodes-final', type=int, default=20)
    p.add_argument('--forced-every', type=int, default=200)
    p.add_argument('--forced-trials', type=int, default=20)
    p.add_argument('--forced-trials-final', type=int, default=500)
    p.add_argument('--forced-trials-s2', type=int, default=0,
                   help='')
    p.add_argument('--forced-trials-s2-final', type=int, default=0,
                   help='')
    p.add_argument('--forced-horizon', type=int, default=FORCED_BOMB_HORIZON)
    p.add_argument('--forced-target', type=float, default=FORCED_ESCAPE_TARGET,
                   help='')
    p.add_argument('--forced-target-s2', type=float,
                   default=FORCED_ESCAPE_TARGET_S2,
                   help='')
    p.add_argument('--gate-ratio', type=float, default=GATE_RATIO)
    p.add_argument('--gate-invalid', type=float, default=GATE_INVALID)
    p.add_argument('--gate-bombs', type=float, default=GATE_BOMBS_MIN)
    p.add_argument('--gate-crates', type=float, default=GATE_CRATES_MIN,
                   help='')
    p.add_argument('--gate-suicides', type=float, default=GATE_SUICIDES,
                   help='')
    p.add_argument('--e-gate-bombs', type=float, default=E_GATE_BOMBS_MIN,
                   help='')
    p.add_argument('--e-gate-suicides-per-bomb', type=float,
                   default=E_GATE_SUICIDES_PER_BOMB_FLOOR,
                   help='')
    p.add_argument('--e-gate-escape-norm', type=float,
                   default=E_GATE_ESCAPE_NORM_FLOOR,
                   help='')
    p.add_argument('--sb-episodes', type=int, default=10,
                   help='')
    p.add_argument('--sb-episodes-final', type=int, default=200,
                   help='')
    p.add_argument('--sb-stage-s1', type=int, default=1,
                   help='')
    p.add_argument('--sb-oracle-cap', type=int, default=ORACLE_CAP,
                   help='')
    p.add_argument('--sb-oracle-cap-final', type=int,
                   default=SB_ORACLE_CAP_FINAL,
                   help='')
    p.add_argument('--probe-at', type=int, default=0,
                   help='')
    p.add_argument('--init-weights', default='',
                   help='')
    p.add_argument('--out', default='', help='')
    p.add_argument('--log-csv', default='')
    p.add_argument('--probe-s2-csv', default='')
    p.add_argument('--probe-s1-csv', default='')
    p.add_argument('--probe-s0-csv', default='')
    p.add_argument('--forced-csv', default='')
    p.add_argument('--forced-s2-csv', default='')
    p.add_argument('--bomb-margin-csv', default='')
    p.add_argument('--buffer-composition-csv', default='')
    p.add_argument('--self-bomb-csv', default='')
    p.add_argument('--probe-at-json', default='')
    p.add_argument('--summary-json', default='')
    p.add_argument('--weights-table', default='', help='')
    p.add_argument('--bench', type=float, default=0.0,
                   help='')
    p.add_argument('--bench-json', default='')
    p.add_argument('--quiet', action='store_true')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.bench > 0:
        out = bench(args)
        text = json.dumps(out, indent=2)
        if args.bench_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.bench_json)), exist_ok=True)
            with open(args.bench_json, 'w') as fh:
                fh.write(text)
        print(text)
        return 0

    train(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
