
import hashlib
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Tuple

import numpy as np

import settings as s
from environment import BombeRLeWorld, WorldArgs
from main import world_controller


@dataclass
class Board:

    round: int
    field: np.ndarray
    coins: List[Tuple[int, int, bool]]
    starts: Dict[str, Tuple[int, int]]


@dataclass
class RoundRow:

    round: int
    agent: str
    code_name: str
    seat: int
    score: int
    coins: int
    kills: int
    suicides: int
    crates: int
    bombs: int
    moves: int
    invalid: int
    steps: int
    deaths_by_other: int
    survived: int
    round_steps: int
    think: List[float] = dc_field(default_factory=list)

    def as_dict(self, with_think=False) -> dict:
        out = {k: v for k, v in self.__dict__.items() if k != 'think'}
        if with_think and self.think:
            out['think_mean'] = float(np.mean(self.think))
            out['think_p95'] = float(np.percentile(self.think, 95))
            out['think_max'] = float(np.max(self.think))
        return out


@dataclass
class MatchResult:

    rows: List[RoundRow]
    boards: List[Board]
    config: dict
    arena_hashes: List[str] = dc_field(default_factory=list)

    def scores(self, name: str) -> List[float]:
        return [float(r.score) for r in sorted(self.for_agent(name),
                                               key=lambda r: r.round)]

    def for_agent(self, name: str) -> List[RoundRow]:
        return [r for r in self.rows if r.agent == name]

    @property
    def agent_names(self) -> List[str]:
        seen = {}
        for r in self.rows:
            seen.setdefault(r.agent, r.seat)
        return sorted(seen, key=seen.get)


class StatsWorld(BombeRLeWorld):

    def __init__(self, args, agents):
        super().__init__(args, agents)
        self.rows: List[RoundRow] = []
        self.boards: List[Board] = []
        self._think: Dict[Tuple[int, str], List[float]] = {}
        self._seat = {a.name: i for i, a in enumerate(self.agents)}
        for agent in self.agents:
            self._instrument(agent)

    def _instrument(self, agent):
        original = agent.wait_for_act

        def wrapped():
            action, think_time = original()
            self._think.setdefault((self.round, agent.name), []).append(think_time)
            return action, think_time

        agent.wait_for_act = wrapped

    def new_round(self):
        super().new_round()
        self.boards.append(Board(
            round=self.round,
            field=np.array(self.arena),
            coins=[(int(c.x), int(c.y), bool(c.collectable)) for c in self.coins],
            starts={a.name: (int(a.x), int(a.y)) for a in self.active_agents},
        ))

    def end_round(self):
        super().end_round()
        for agent in self.agents:
            st = agent.statistics
            suicides = int(st.get('suicides', 0))
            self.rows.append(RoundRow(
                round=self.round,
                agent=agent.name,
                code_name=agent.code_name,
                seat=self._seat[agent.name],
                score=int(agent.score),
                coins=int(st.get('coins', 0)),
                kills=int(st.get('kills', 0)),
                suicides=suicides,
                crates=int(st.get('crates', 0)),
                bombs=int(st.get('bombs', 0)),
                moves=int(st.get('moves', 0)),
                invalid=int(st.get('invalid', 0)),
                steps=int(st.get('steps', 0)),
                deaths_by_other=int(bool(agent.dead) and suicides == 0),
                survived=int(not agent.dead),
                round_steps=int(self.step),
                think=list(self._think.get((self.round, agent.name), [])),
            ))


def arena_digest(board: Board) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(board.field, dtype=np.int64).tobytes())
    for (x, y, collectable) in sorted(board.coins):
        h.update(bytes((x, y, int(collectable))))
    return h.hexdigest()[:16]


def assert_paired(a: MatchResult, b: MatchResult):
    if len(a.arena_hashes) != len(b.arena_hashes):
        raise ValueError(f'round counts differ: {len(a.arena_hashes)} vs '
                         f'{len(b.arena_hashes)}')
    bad = [k for k, (x, y) in enumerate(zip(a.arena_hashes, b.arena_hashes), 1)
           if x != y]
    if bad:
        raise ValueError(f'{len(bad)} of {len(a.arena_hashes)} arenas differ '
                         f'(first: round {bad[0]}); run both sides with paired=True')


def _release_log_handlers():
    for logger in list(logging.Logger.manager.loggerDict.values()):
        for handler in list(getattr(logger, 'handlers', [])):
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                handler.close()


WEIGHTS_ENV = 'BOMBERMAN_WEIGHTS'


@contextmanager
def weights_override(path: Optional[str]):
    if not path:
        yield ''
        return
    if not os.path.isfile(path):
        raise FileNotFoundError(f'weights not found: {path}')
    before = os.environ.get(WEIGHTS_ENV)
    os.environ[WEIGHTS_ENV] = os.path.abspath(path)
    try:
        yield os.environ[WEIGHTS_ENV]
    finally:
        if before is None:
            os.environ.pop(WEIGHTS_ENV, None)
        else:
            os.environ[WEIGHTS_ENV] = before


TTA8_ENV = 'BOMBERMAN_TTA8'


@contextmanager
def tta8_override(enabled: bool):
    if not enabled:
        yield False
        return
    before = os.environ.get(TTA8_ENV)
    os.environ[TTA8_ENV] = '1'
    try:
        yield True
    finally:
        if before is None:
            os.environ.pop(TTA8_ENV, None)
        else:
            os.environ[TTA8_ENV] = before


def make_args(scenario, seed, log_dir='logs', match_name=None,
              save_stats=False) -> WorldArgs:
    os.makedirs(log_dir, exist_ok=True)
    return WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False,
        continue_without_training=True, log_dir=log_dir, save_stats=save_stats,
        match_name=match_name, seed=seed, silence_errors=False, scenario=scenario)


def run_match(agents, n_rounds, scenario='classic', seed=42, train=0,
              match_name=None, log_dir='logs', save_stats=False,
              weights=None, paired=False, tta8=False) -> MatchResult:
    if not 1 <= len(agents) <= s.MAX_AGENTS:
        raise ValueError(f'need 1..{s.MAX_AGENTS} agents, got {len(agents)}')

    seats = [(a, i < train) for i, a in enumerate(agents)]
    rows: List[RoundRow] = []
    boards: List[Board] = []
    with weights_override(weights) as in_force, tta8_override(tta8):
        if not paired:
            args = make_args(scenario, seed, log_dir=log_dir, match_name=match_name,
                             save_stats=save_stats)
            world = StatsWorld(args, seats)
            world_controller(world, n_rounds, gui=None, every_step=False,
                             turn_based=False, make_video=False,
                             update_interval=args.update_interval)
            rows, boards = world.rows, world.boards
        else:
            for k in range(n_rounds):
                args = make_args(scenario, seed + k, log_dir=log_dir,
                                 match_name=match_name, save_stats=save_stats)
                world = StatsWorld(args, seats)
                world_controller(world, 1, gui=None, every_step=False,
                                 turn_based=False, make_video=False,
                                 update_interval=args.update_interval)
                for row_k in world.rows:
                    row_k.round = k + 1
                for board_k in world.boards:
                    board_k.round = k + 1
                rows.extend(world.rows)
                boards.extend(world.boards)
                _release_log_handlers()
    return MatchResult(rows=rows, boards=boards, config={
        'agents': list(agents), 'n_rounds': n_rounds, 'scenario': scenario,
        'seed': seed, 'train': train, 'match_name': match_name,
        'weights': in_force or None, 'paired': bool(paired), 'tta8': bool(tta8),
    }, arena_hashes=[arena_digest(b) for b in boards])


def think_times(result: MatchResult, agent: Optional[str] = None) -> np.ndarray:
    vals = [t for r in result.rows if agent in (None, r.agent) for t in r.think]
    return np.asarray(vals, dtype=float)
