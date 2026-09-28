
import json
import os


COLS = 17

ROWS = 17

BOMB_POWER = 3

BOMB_TIMER = 4

EXPLOSION_TIMER = 2

MAX_STEPS = 400

N_AGENTS = 4

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

N_ACTIONS = len(ACTIONS)

A_UP, A_RIGHT, A_DOWN, A_LEFT, A_WAIT, A_BOMB = range(6)


CHANNELS = 12

LATEST_FILE = 'model_latest.pt'

MODEL_FILE = 'model.pt'


GATE_S1_BOMBS = 1.0

GATE_S1_SUICIDES = 0.02

GATE_S1_FORCED_SURVIVAL = 0.95

FORCED_BOMB_HORIZON = 8

WEIGHTS_ENV = 'BOMBERMAN_WEIGHTS'


def get_weights_path(default: str = MODEL_FILE) -> str:
    return os.environ.get(WEIGHTS_ENV, default).strip() or default


def get_device(default: str = 'cpu') -> str:
    return os.environ.get('BOMBERMAN_DEVICE', default).strip() or default


D2_ARMS = ('f_ref', 'f_mb8', 'f_hunt', 'f_opp25', 'f_pot', 'f_anneal',
           'f_long')

D2_RETIRED_ARMS = ('f_sym',)

VARIANT_FILE = 'variant.json'

D1_TOTAL_STEPS = 19_500_000

REWARD_ANNEAL_STEPS = 12_000_000

INJECT_ANNEAL_STEPS = 8_000_000

REWARD_ANNEAL_FRAC = REWARD_ANNEAL_STEPS / D1_TOTAL_STEPS

INJECT_ANNEAL_FRAC = INJECT_ANNEAL_STEPS / D1_TOTAL_STEPS


def anneal_steps(frac, total_steps, fallback=0):
    if frac is None or frac < 0:
        return int(fallback)
    return int(round(float(frac) * float(total_steps)))


def load_variant(path: str = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), VARIANT_FILE)
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def variant_settings(variant: dict) -> dict:
    base = dict(variant.get('base', {}))
    return {
        'rewards': base.get('rewards', 'd1'),
        'reward_anneal_frac': float(base.get('reward_anneal_frac', REWARD_ANNEAL_FRAC)),
        'inject_anneal_frac': float(base.get('inject_anneal_frac', INJECT_ANNEAL_FRAC)),
        'reward_anneal_steps': int(base.get('reward_anneal_steps', -1)),
        'inject_anneal_steps': int(base.get('inject_anneal_steps', -1)),
        'opp_frac': float(base.get('opp_frac', 0.0)),
        'inject_dose': float(base.get('inject_dose', 0.25)),
        'canonical': bool(base.get('canonical', True)),
        'bombopp': float(base.get('bombopp', 0.0)),
        'pot_opp': float(base.get('pot_opp', 0.0)),
        'minibatch': int(base.get('minibatch', 0)),
    }


def get_num_threads(default: int = 0) -> int:
    try:
        return int(os.environ.get('BOMBERMAN_THREADS', default))
    except ValueError:
        return default
