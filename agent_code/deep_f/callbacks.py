
import os

import numpy as np
import torch

from .config import (
    ACTIONS,
    CHANNELS,
    COLS,
    N_ACTIONS,
    ROWS,
    get_device,
    get_weights_path,
    load_variant,
    variant_settings,
)
from .features import (
    D4_ACTION,
    D4_COMPOSE,
    apply_d4,
    canonical_view,
    d4_position,
    state_to_features,
)
from .model import ActorCritic


def _float_env(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def setup(self):
    self.device = torch.device(get_device('cpu'))
    if self.device.type == 'cpu':
        torch.set_num_threads(int(os.environ.get('BOMBERMAN_PLAY_THREADS', 2)))

    self.temperature = _float_env('BOMBERMAN_TEMP', 0.0)
    self.eps = _float_env('BOMBERMAN_EPS', 0.0)
    self.rng = np.random.default_rng()

    self.settings = variant_settings(load_variant())
    self.canonical = bool(self.settings['canonical'])
    self.tta8 = _float_env('BOMBERMAN_TTA8', 0.0) > 0
    self.d4_scratch = np.zeros((CHANNELS, COLS, ROWS), dtype=np.uint8)
    self.logger.info(f"variant {self.settings} canonical={self.canonical} "
                     f"tta8={self.tta8}")

    weights = get_weights_path()
    self.model = None
    if os.path.isfile(weights):
        try:
            self.model, ckpt = ActorCritic.load(weights, self.device)
            self.logger.info(
                f"Loaded model from {weights} "
                f"(updates={ckpt.get('updates', '?')}, steps={ckpt.get('env_steps', '?')})")
        except Exception as exc:  # never let a bad checkpoint crash the game
            self.logger.error(f'Could not load {weights}: {exc!r}')
    else:
        self.logger.warning(
            f'{weights} not found -- playing randomly. Train first with: '
            f'python -m agent_code.<arm>.train_ppo')

    _warm_up(self)


def _warm_up(self):
    dummy = {
        'round': 1, 'step': 1,
        'field': np.zeros((COLS, ROWS), dtype=int),
        'self': ('me', 0, True, (1, 1)),
        'others': [],
        'bombs': [],
        'coins': [],
        'explosion_map': np.zeros((COLS, ROWS)),
        'user_input': None,
    }
    obs, mask, _ = state_to_features(dummy, self.canonical)
    if self.model is not None:
        with torch.no_grad():
            act(self, dummy)


def _forward(self, obs_batch, mask_batch):
    obs_t = torch.from_numpy(obs_batch).to(self.device)
    mask_t = torch.from_numpy(mask_batch).to(self.device).bool()
    logits, _ = self.model(obs_t, mask_t)
    return logits


def _action_probs(self, obs, mask, pos):
    views = tuple(range(8)) if self.tta8 else (0,)
    obs_batch = np.zeros((len(views), *obs.shape), dtype=np.uint8)
    mask_batch = np.zeros((len(views), N_ACTIONS), dtype=np.uint8)
    perms = []
    for i, v in enumerate(views):
        o, m = apply_d4(obs, mask, v)
        t = v
        if self.canonical:
            vx, vy = d4_position(v, pos[0], pos[1])
            g = canonical_view(o, m, vx, vy, self.d4_scratch)
            t = int(D4_COMPOSE[g][v])
        obs_batch[i] = o
        mask_batch[i] = m
        perms.append(D4_ACTION[t])
    with torch.no_grad():
        logits = _forward(self, obs_batch, mask_batch)
    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    real = np.zeros(N_ACTIONS, dtype=np.float64)
    for i, perm in enumerate(perms):
        real += probs[i][perm]
    real /= max(len(perms), 1)
    return real


def act(self, game_state: dict) -> str:
    if game_state is None:
        return 'WAIT'

    obs, mask, _ = state_to_features(game_state)

    if self.model is None:
        legal = np.flatnonzero(mask)
        return ACTIONS[int(self.rng.choice(legal))]

    probs = _action_probs(self, obs, mask, game_state['self'][3])

    if self.eps > 0 and self.rng.random() < self.eps:
        legal = np.flatnonzero(mask)
        action = int(self.rng.choice(legal))
    elif self.temperature > 0:
        tempered = np.power(np.clip(probs, 1e-12, None), 1.0 / self.temperature)
        tempered *= mask
        action = int(self.rng.choice(N_ACTIONS, p=tempered / tempered.sum()))
    else:
        action = int(np.argmax(np.where(mask, probs, -1.0)))

    self.logger.debug(f'step {game_state["step"]}: {ACTIONS[action]}')
    return ACTIONS[action]
