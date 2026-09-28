
import numpy as np

import events as e

from .config import (
    ESCAPE_POTENTIAL_CAP,
    GAMMA,
    REWARD_KILL,
    VARIANT,
)
from .features import danger_map, game_state_to_arrays, safe_distance
from .train_lin import step_reward

CRATE_REWARD = VARIANT['crate_reward']

CRATE_KAPPA = float(VARIANT['crate_kappa'])

CRATE_CLAWBACK = VARIANT.get('crate_clawback', 'terminal')

ESCAPE_REWARD = VARIANT.get('escape_reward', 'none')

ESCAPE_RHO = float(VARIANT.get('escape_rho', 0.0))

R_DEATH = float(VARIANT['r_death'])

R_INVALID = float(VARIANT['r_invalid'])

EVENT_TO_INFO = {
    e.COIN_COLLECTED: 'coins',
    e.CRATE_DESTROYED: 'crates',
    e.INVALID_ACTION: 'invalid',
    e.GOT_KILLED: 'suicide',
}


def info_from_events(events: list, done: bool = False) -> dict:
    info = {'coins': 0, 'crates': 0, 'invalid': 0, 'suicide': 0, 'done': bool(done)}
    for ev in events:
        key = EVENT_TO_INFO.get(ev)
        if key is None:
            continue
        info[key] += 1
    info['invalid'] = int(bool(info['invalid']))
    info['suicide'] = int(bool(info['suicide']))
    return info


def coin_distance(game_state: dict) -> int:
    from .config import DIST_CAP
    from .features import bfs_coin_distance
    if game_state is None:
        return DIST_CAP
    field, coins, occupied, _expl, _bx, _by, _bt, x, y, _left = \
        game_state_to_arrays(game_state)
    return int(bfs_coin_distance(field, coins, occupied, x, y))


def safe_dist(game_state: dict, cap: int = ESCAPE_POTENTIAL_CAP) -> int:
    if game_state is None or ESCAPE_REWARD == 'none':
        return 0
    field, _coins, occupied, _expl, _bx, _by, _bt, x, y, _left = \
        game_state_to_arrays(game_state)
    return int(safe_distance(field, occupied, danger_map(game_state), x, y,
                             int(cap)))


def setup_training(self):
    self.episode_reward = 0.0
    self.episode_returns = []
    self.episode_events = {}
    self.destroyed = 0          # crates destroyed so far this round: Phi's state


def count_events(self, events: list):
    for ev in events:
        self.episode_events[ev] = self.episode_events.get(ev, 0) + 1


def reward_from_events(self, events: list) -> float:
    info = info_from_events(events)
    r = step_reward(info, 0, 0, GAMMA, R_INVALID, False, R_DEATH,
                    crate_mode=CRATE_REWARD, kappa=CRATE_KAPPA,
                    destroyed_before=0, destroyed_after=0,
                    escape_mode='none', rho=0.0, clawback=CRATE_CLAWBACK)
    r += REWARD_KILL * sum(1 for ev in events if ev == e.KILLED_OPPONENT)
    return r


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: list):
    info = info_from_events(events)
    destroyed_before = self.destroyed
    self.destroyed += info['crates']
    count_events(self, events)

    r = step_reward(info, coin_distance(old_game_state),
                    coin_distance(new_game_state), GAMMA, R_INVALID, False,
                    R_DEATH, crate_mode=CRATE_REWARD, kappa=CRATE_KAPPA,
                    destroyed_before=destroyed_before,
                    destroyed_after=self.destroyed,
                    escape_mode=ESCAPE_REWARD, rho=ESCAPE_RHO,
                    safe_before=safe_dist(old_game_state),
                    safe_after=safe_dist(new_game_state),
                    clawback=CRATE_CLAWBACK)
    r += REWARD_KILL * sum(1 for ev in events if ev == e.KILLED_OPPONENT)
    if old_game_state is None:
        r = 0.0                 # no transition yet: there is nothing to score
    self.episode_reward += r


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    info = info_from_events(events, done=True)
    destroyed_before = self.destroyed
    self.destroyed += info['crates']
    count_events(self, events)

    r = step_reward(info, coin_distance(last_game_state), 0, GAMMA, R_INVALID,
                    True, R_DEATH, crate_mode=CRATE_REWARD, kappa=CRATE_KAPPA,
                    destroyed_before=destroyed_before,
                    destroyed_after=self.destroyed,
                    escape_mode=ESCAPE_REWARD, rho=ESCAPE_RHO,
                    safe_before=safe_dist(last_game_state), safe_after=0,
                    clawback=CRATE_CLAWBACK)
    r += REWARD_KILL * sum(1 for ev in events if ev == e.KILLED_OPPONENT)
    self.episode_reward += r
    self.episode_returns.append(self.episode_reward)

    summary = ', '.join(f'{k}={v}' for k, v in sorted(self.episode_events.items()))
    self.logger.info(
        f'round {last_game_state["round"]}: score={last_game_state["self"][1]} '
        f'steps={last_game_state["step"]} return={self.episode_reward:.2f} '
        f'mean_return={np.mean(self.episode_returns):.2f} | '
        f'crate_reward={CRATE_REWARD}@{CRATE_KAPPA} '
        f'escape_reward={ESCAPE_REWARD}@{ESCAPE_RHO} | {summary}')

    self.episode_reward = 0.0
    self.episode_events = {}
    self.destroyed = 0
