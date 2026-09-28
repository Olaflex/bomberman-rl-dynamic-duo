import numpy as np
import torch

from .features import ACTIONS, D4_ACTION, canonical_view, encode
from .model import load_policy


def setup(self):
    torch.set_num_threads(2)
    self.net = load_policy('model.pt')
    self.scratch = np.zeros((12, 17, 17), dtype=np.uint8)
    self.logger.info('policy loaded')
    warm_up = {
        'round': 1, 'step': 1,
        'field': np.zeros((17, 17), dtype=int),
        'self': ('me', 0, True, (1, 1)),
        'others': [], 'bombs': [], 'coins': [],
        'explosion_map': np.zeros((17, 17)),
        'user_input': None,
    }
    act(self, warm_up)


def act(self, game_state):
    if game_state is None:
        return 'WAIT'
    obs, legal = encode(game_state)
    view, mask = obs.copy(), legal.copy()
    x, y = game_state['self'][3]
    g = canonical_view(view, mask, x, y, self.scratch)
    with torch.no_grad():
        logits = self.net(torch.from_numpy(view[None]), torch.from_numpy(mask[None]).bool())
    probs = torch.softmax(logits, dim=-1).numpy()[0].astype(np.float64)
    probs = probs[D4_ACTION[g]]
    action = int(np.argmax(np.where(legal, probs, -1.0)))
    self.logger.debug(f"step {game_state['step']}: {ACTIONS[action]}")
    return ACTIONS[action]
