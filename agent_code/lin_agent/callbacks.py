
import os

import numpy as np

from .config import (
    ACTIONS,
    COLS,
    ROWS,
    WEIGHTS_FILE,
    agent_path,
    dim_for,
    load_variant,
)
from .features import state_to_features
from .model import LinearQ


def setup(self):
    self.variant = load_variant()
    self.mode = self.variant['features']
    self.dim = dim_for(self.mode)

    try:
        self.eps = float(os.environ.get('BOMBERMAN_EPS', 0.0))
    except ValueError:
        self.eps = 0.0
    self.rng = np.random.default_rng()

    self.model = None
    path = agent_path(WEIGHTS_FILE)
    if os.path.isfile(path):
        try:
            self.model = LinearQ.load(path, dim=self.dim)
            self.logger.info(f'Loaded weights from {path} '
                             f'(variant={self.variant["name"]}, features={self.mode}, '
                             f'd={self.dim}, ||w||={self.model.norm():.3f})')
        except Exception as exc:  # noqa: BLE001 - never let a bad file crash the game
            self.logger.error(f'Could not load {path}: {exc!r}')
    else:
        self.logger.warning(
            '')

    _warm_up(self)


def _warm_up(self):
    field = np.zeros((COLS, ROWS), dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    field[3, 3] = 1
    explosion = np.zeros((COLS, ROWS))
    explosion[3, 1] = 1.0
    for bombs, expl in (([], np.zeros((COLS, ROWS))), ([((5, 1), 2)], explosion)):
        dummy = {
            'round': 1, 'step': 1,
            'field': field,
            'self': ('me', 0, True, (1, 1)),
            'others': [],
            'bombs': bombs,
            'coins': [(5, 5)],
            'explosion_map': expl,
            'user_input': None,
        }
        phi = state_to_features(dummy, self.mode)
        if self.model is not None:
            self.model.greedy(phi)


def act(self, game_state: dict) -> str:
    if game_state is None:
        return 'WAIT'

    phi = state_to_features(game_state, self.mode)

    if self.model is None or (self.eps > 0 and self.rng.random() < self.eps):
        free = [i for i in range(4) if phi[i] == 0.0]
        choices = free + [4] if free else [4]
        return ACTIONS[int(self.rng.choice(choices))]

    action = self.model.greedy(phi)
    self.logger.debug(f'step {game_state["step"]}: {ACTIONS[action]}')
    return ACTIONS[action]
