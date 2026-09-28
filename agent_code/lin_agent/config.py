
import json
import os


COLS = 17

ROWS = 17

BOMB_POWER = 3

BOMB_TIMER = 4

EXPLOSION_TIMER = 2

MAX_STEPS = 400

REWARD_COIN = 1

REWARD_KILL = 5


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

N_ACTIONS = len(ACTIONS)

A_UP, A_RIGHT, A_DOWN, A_LEFT, A_WAIT, A_BOMB = range(6)


D_NAV = 11

DANGER_SLOT = D_NAV

N_DANGER = 5

ROOM_SLOT = DANGER_SLOT + N_DANGER

N_ROOM = 4

CRATE_SLOT = ROOM_SLOT + N_ROOM

N_CRATE = 4

PLACE_SLOT = CRATE_SLOT + N_CRATE

N_PLACE = 2

CONJ_SLOT = PLACE_SLOT + N_PLACE

N_CONJ = 1

ESCAPE_SLOT = CONJ_SLOT + N_CONJ

N_ESCAPE = 1

CONJ_PLACE_SLOT = ESCAPE_SLOT + N_ESCAPE

N_CONJ_PLACE = 1

ESCAPE_DIST_DEPTH = BOMB_TIMER

ESCAPE_DIST_NORM = float(BOMB_TIMER)

CRATE_BLAST_NORM = 4.0

POST_BOMB_ROOM_DEPTH = BOMB_TIMER

POST_BOMB_ROOM_NORM = 9.0

FEATURE_NAMES_FULL = [
    'blocked_up', 'blocked_right', 'blocked_down', 'blocked_left',
    'coin_up', 'coin_right', 'coin_down', 'coin_left',
    'coin_closeness', 'bomb_available', 'bias',
    'danger_here', 'danger_up', 'danger_right', 'danger_down', 'danger_left',
    'room_up', 'room_right', 'room_down', 'room_left',
    'crate_up', 'crate_right', 'crate_down', 'crate_left',
    'crates_in_blast', 'post_bomb_room', 'conj_bomb_here', 'escape_dist',
    'conj_place',
]

FEATURE_MODES = ('nav', 'inblast1', 'binary', 'graded', 'room', 'crate',
                 'place', 'conj', 'esc', 'crates_only', 'conjesc')

MODE_SLOTS = {
    'nav': tuple(range(0, D_NAV)),
    'inblast1': tuple(range(0, D_NAV + 1)),
    'binary': tuple(range(0, D_NAV + N_DANGER)),
    'graded': tuple(range(0, D_NAV + N_DANGER)),
    'room': tuple(range(0, ROOM_SLOT + N_ROOM)),
    'crate': tuple(range(0, CRATE_SLOT + N_CRATE)),
    'place': tuple(range(0, PLACE_SLOT + N_PLACE)),
    'conj': tuple(range(0, CONJ_SLOT + N_CONJ)),
    'esc': tuple(range(0, PLACE_SLOT + N_PLACE)) + (ESCAPE_SLOT,),
    'crates_only': tuple(range(0, PLACE_SLOT + 1)),
    'conjesc': tuple(range(0, PLACE_SLOT + N_PLACE)) + (CONJ_PLACE_SLOT,),
}

MODE_DIM = {mode: len(slots) for mode, slots in MODE_SLOTS.items()}

DANGER_HORIZON = 4

ROOM_DEPTH = 3

ROOM_NORM = 9.0

DIST_CAP = 10

DEFAULT_MODE = 'graded'


def dim_for(mode: str) -> int:
    return MODE_DIM[mode]


def feature_names(mode: str) -> list:
    names = [FEATURE_NAMES_FULL[slot] for slot in MODE_SLOTS[mode]]
    if mode == 'inblast1':
        names[DANGER_SLOT] = 'in_blast_here'
    elif mode == 'binary':
        for i in range(N_DANGER):
            names[DANGER_SLOT + i] = names[DANGER_SLOT + i].replace('danger', 'in_blast')
    return names


S0_CRATE_DENSITY = 0.0

S0_COIN_COUNT = 50

S1_CRATE_DENSITY = 0.35

S1_COIN_COUNT = 9

S2_CRATE_DENSITY = 0.75

S2_COIN_COUNT = 50

S2_SCENARIO = 'loot-crate'

S1_SCENARIO = 'ladder-s1'


MIX_TARGET = 0.3

MIX_WAYPOINTS = ((0, 1.00, 0.00), (300, 0.15, 0.85), (800, 0.15, 0.15))

S2_ONLY_WAYPOINTS = ((0, 0.0, 0.0),)

CURRICULA = {'mix': MIX_WAYPOINTS, 's2only': S2_ONLY_WAYPOINTS}

STAGE_SEED_BASE = 77_000


GAMMA = 0.95

EPS_START = 1.0

EPS_END = 0.05

EPS_DECAY_EPISODES = 500

POTENTIAL_SCALE = 0.1

R_INVALID = -0.1

R_CRATE = 0.0

CRATE_REWARD_MODES = ('none', 'potential', 'event')

CRATE_REWARD_MODE = 'none'

CRATE_CLAWBACK_MODES = ('terminal', 'suppressed')

CRATE_CLAWBACK = 'terminal'

CRATE_KAPPA_LADDER = (0.0, 0.1, 0.4)

CRATE_KAPPA = 0.0

R_DEATH = -1.0

ESCAPE_REWARD_MODES = ('none', 'potential')

ESCAPE_REWARD_MODE = 'none'

ESCAPE_POTENTIAL_RHO = 0.4

ESCAPE_POTENTIAL_CAP = BOMB_TIMER

ESCAPE_POTENTIAL_NONNEGATIVE = True


GATE_RATIO = 1.3

GATE_INVALID = 0.005

GATE_BOMBS_MIN = 1.0

GATE_CRATES_MIN = 8.0

GATE_SUICIDES = 0.02

FORCED_BOMB_HORIZON = 8

SB_ORACLE_CAP_FINAL = 0

FORCED_S2_BLOCK_N = 2000


E_GATE_BOMBS_MIN = 1.0

E_GATE_SUICIDES_PER_BOMB_FLOOR = 0.10

E_GATE_ESCAPE_NORM_FLOOR = 0.75

E_GATE_ANCHOR_FACTOR = 0.90

FORCED_ESCAPE_TARGET = 0.90

FORCED_ESCAPE_TARGET_S2 = 0.90

FORCED_CEILING_L2 = 368 / 500


INJECT_STAGES = ('S1', 'S2')

INJECT_MODES = ('none', 'random', 'safe', 'seed')

INJECT_PROB = 0.25

INJECT_BOMB_TIMER = BOMB_TIMER - 1

INJECT_SEED_BASE = 55_000


WEIGHTS_FILE = 'weights.npy'

VARIANT_FILE = 'variant.json'


def agent_path(name: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


DEFAULT_VARIANT = {
    'name': 'lin_agent',
    'features': DEFAULT_MODE,
    'rule': 'lstd',
    'n_step': 1,
    'r_death': R_DEATH,
    'r_invalid': R_INVALID,
    'crate_reward': CRATE_REWARD_MODE,
    'crate_kappa': CRATE_KAPPA,
    'crate_clawback': CRATE_CLAWBACK,
    'escape_reward': ESCAPE_REWARD_MODE,
    'escape_rho': 0.0,
    'curriculum': 'mix',
    'inject': 'none',
    'inject_prob': 0.0,
    'inject_stage': 'S1',
}


def load_variant() -> dict:
    cfg = dict(DEFAULT_VARIANT)
    path = agent_path(VARIANT_FILE)
    if os.path.isfile(path):
        try:
            with open(path) as fh:
                cfg.update(json.load(fh))
        except Exception:  # noqa: BLE001 - never let configuration crash the agent
            pass
    if cfg.get('features') not in FEATURE_MODES:
        cfg['features'] = DEFAULT_VARIANT['features']
    return cfg


VARIANT = load_variant()

FEATURE_MODE = VARIANT['features']

D = dim_for(FEATURE_MODE)

FEATURE_NAMES = feature_names(FEATURE_MODE)
