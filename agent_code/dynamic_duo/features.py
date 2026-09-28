import numpy as np
from numba import njit

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
N_ACTIONS = 6
WAIT, BOMB = 4, 5
SIZE = 17
CENTRE = SIZE // 2
CHANNELS = 12
BOMB_POWER = 3
MAX_STEPS = 400

DX = np.array([0, 1, 0, -1], dtype=np.int64)
DY = np.array([-1, 0, 1, 0], dtype=np.int64)

D4_MAT = np.array([
    [1, 0, 0, 1], [0, -1, 1, 0], [-1, 0, 0, -1], [0, 1, -1, 0],
    [-1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, -1, -1, 0],
], dtype=np.int64)


def _symmetry_tables():
    mats = [m.reshape(2, 2) for m in D4_MAT]

    def index_of(mat):
        for i, m in enumerate(mats):
            if np.array_equal(m, mat):
                return i
        raise ValueError('not a symmetry of the square')

    inverse = np.array([index_of(m.T) for m in mats], dtype=np.int64)
    action = np.zeros((8, N_ACTIONS), dtype=np.int64)
    for g, m in enumerate(mats):
        for a in range(N_ACTIONS):
            if a >= 4:
                action[g, a] = a
                continue
            d = m @ np.array([DX[a], DY[a]], dtype=np.int64)
            action[g, a] = int(np.flatnonzero((DX == d[0]) & (DY == d[1]))[0])
    src_x = np.zeros((8, SIZE, SIZE), dtype=np.int64)
    src_y = np.zeros((8, SIZE, SIZE), dtype=np.int64)
    for g in range(8):
        a, b, c, d = D4_MAT[inverse[g]]
        for x in range(SIZE):
            for y in range(SIZE):
                u, v = x - CENTRE, y - CENTRE
                src_x[g, x, y] = CENTRE + a * u + b * v
                src_y[g, x, y] = CENTRE + c * u + d * v
    return inverse, action, src_x, src_y


D4_INV, D4_ACTION, D4_SRC_X, D4_SRC_Y = _symmetry_tables()


@njit(cache=True, inline='always')
def _rot_x(g, x, y):
    return CENTRE + D4_MAT[g, 0] * (x - CENTRE) + D4_MAT[g, 1] * (y - CENTRE)


@njit(cache=True, inline='always')
def _rot_y(g, x, y):
    return CENTRE + D4_MAT[g, 2] * (x - CENTRE) + D4_MAT[g, 3] * (y - CENTRE)


@njit(cache=True)
def pick_symmetry(obs, mx, my):
    cand = np.empty(8, dtype=np.int64)
    keep = np.zeros(8, dtype=np.uint8)
    k = 0
    for g in range(8):
        x = _rot_x(g, mx, my)
        y = _rot_y(g, mx, my)
        if x <= CENTRE and y <= CENTRE and x <= y:
            cand[k] = g
            keep[k] = 1
            k += 1
    if k == 1:
        return cand[0]
    alive = k
    for ci in range(2):
        c = 1 if ci == 0 else 7
        for x in range(SIZE):
            for y in range(SIZE):
                if alive <= 1:
                    break
                best = 256
                for i in range(k):
                    if keep[i] == 0:
                        continue
                    gi = D4_INV[cand[i]]
                    val = obs[c, _rot_x(gi, x, y), _rot_y(gi, x, y)]
                    if val < best:
                        best = val
                for i in range(k):
                    if keep[i] == 0:
                        continue
                    gi = D4_INV[cand[i]]
                    if obs[c, _rot_x(gi, x, y), _rot_y(gi, x, y)] != best:
                        keep[i] = 0
                        alive -= 1
            if alive <= 1:
                break
        if alive <= 1:
            break
    for i in range(k):
        if keep[i] != 0:
            return cand[i]
    return cand[0]


@njit(cache=True)
def canonical_view(obs, mask, mx, my, scratch):
    g = pick_symmetry(obs, mx, my)
    if g == 0:
        return 0
    for c in range(CHANNELS):
        for x in range(SIZE):
            for y in range(SIZE):
                scratch[c, x, y] = obs[c, D4_SRC_X[g, x, y], D4_SRC_Y[g, x, y]]
    for c in range(CHANNELS):
        for x in range(SIZE):
            for y in range(SIZE):
                obs[c, x, y] = scratch[c, x, y]
    old = np.empty(N_ACTIONS, dtype=np.uint8)
    for a in range(N_ACTIONS):
        old[a] = mask[a]
    for a in range(N_ACTIONS):
        mask[D4_ACTION[g, a]] = old[a]
    return g


@njit(cache=True, inline='always')
def _blocked(field, bombs, n_bombs, agents, n_agents, x, y):
    if field[x, y] != 0:
        return True
    for b in range(n_bombs):
        if bombs[b, 0] == x and bombs[b, 1] == y:
            return True
    for a in range(n_agents):
        if agents[a, 0] == x and agents[a, 1] == y:
            return True
    return False


@njit(cache=True)
def fill_planes(out, field, coins, bombs, n_bombs, expl, agents, n_agents, step):
    mx = agents[0, 0]
    my = agents[0, 1]
    for x in range(SIZE):
        for y in range(SIZE):
            f = field[x, y]
            if f == -1:
                out[0, x, y] = 255
            elif f == 1:
                out[1, x, y] = 255
            if coins[x, y] != 0:
                out[5, x, y] = 255
            if expl[x, y] >= 1:
                out[8, x, y] = 255
    out[2, mx, my] = 255
    for a in range(1, n_agents):
        if agents[a, 0] < 0:
            continue
        ox = agents[a, 0]
        oy = agents[a, 1]
        out[3, ox, oy] = 255
        if agents[a, 2] != 0:
            out[4, ox, oy] = 255
    for b in range(n_bombs):
        bx = bombs[b, 0]
        by = bombs[b, 1]
        t = bombs[b, 2]
        if bx < 0:
            continue
        urgency = np.uint8(int(255.0 * (4.0 - t) / 4.0))
        if urgency > out[6, bx, by]:
            out[6, bx, by] = urgency
        danger = np.uint8(int(255.0 * (5.0 - t) / 5.0))
        if danger > out[7, bx, by]:
            out[7, bx, by] = danger
        for d in range(4):
            for i in range(1, BOMB_POWER + 1):
                x = bx + DX[d] * i
                y = by + DY[d] * i
                if field[x, y] == -1:
                    break
                if danger > out[7, x, y]:
                    out[7, x, y] = danger
    has_bomb = np.uint8(255) if agents[0, 2] != 0 else np.uint8(0)
    progress = np.uint8(int(255.0 * min(1.0, step / MAX_STEPS)))
    for x in range(SIZE):
        for y in range(SIZE):
            out[9, x, y] = has_bomb
            out[10, x, y] = progress
            if not _blocked(field, bombs, n_bombs, agents, n_agents, x, y):
                out[11, x, y] = 255


@njit(cache=True)
def fill_mask(mask, field, bombs, n_bombs, agents, n_agents):
    mx = agents[0, 0]
    my = agents[0, 1]
    for d in range(4):
        if _blocked(field, bombs, n_bombs, agents, n_agents, mx + DX[d], my + DY[d]):
            mask[d] = 0
        else:
            mask[d] = 1
    mask[WAIT] = 1
    mask[BOMB] = 1 if agents[0, 2] != 0 else 0


def encode(game_state):
    field = np.ascontiguousarray(game_state['field'], dtype=np.int8)
    coins = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for x, y in game_state['coins']:
        coins[x, y] = 1
    bomb_list = game_state['bombs']
    bombs = np.full((max(len(bomb_list), 1), 3), -1, dtype=np.int64)
    for i, ((x, y), t) in enumerate(bomb_list):
        bombs[i] = (x, y, t)
    expl = np.ascontiguousarray(game_state['explosion_map'], dtype=np.int8)
    _, _, has_bomb, (sx, sy) = game_state['self']
    others = game_state['others']
    agents = np.full((1 + len(others), 3), -1, dtype=np.int64)
    agents[0] = (sx, sy, 1 if has_bomb else 0)
    for i, (_, _, bomb_left, (x, y)) in enumerate(others):
        agents[i + 1] = (x, y, 1 if bomb_left else 0)
    obs = np.zeros((CHANNELS, SIZE, SIZE), dtype=np.uint8)
    fill_planes(obs, field, coins, bombs, len(bomb_list), expl, agents, len(agents), game_state['step'])
    mask = np.zeros(N_ACTIONS, dtype=np.uint8)
    fill_mask(mask, field, bombs, len(bomb_list), agents, len(agents))
    return obs, mask
