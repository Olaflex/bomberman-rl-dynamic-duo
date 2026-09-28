
import numpy as np
from numba import njit

from .config import A_BOMB, A_WAIT, BOMB_POWER, CHANNELS, COLS, MAX_STEPS, N_ACTIONS, ROWS

DX = np.array([0, 1, 0, -1], dtype=np.int64)

DY = np.array([-1, 0, 1, 0], dtype=np.int64)


CENTRE = COLS // 2

D4_MAT = np.array([
    [1, 0, 0, 1],      # 0  identity
    [0, -1, 1, 0],     # 1  rotate 90
    [-1, 0, 0, -1],    # 2  rotate 180
    [0, 1, -1, 0],     # 3  rotate 270
    [-1, 0, 0, 1],     # 4  mirror across the vertical axis
    [1, 0, 0, -1],     # 5  mirror across the horizontal axis
    [0, 1, 1, 0],      # 6  transpose (mirror across the main diagonal)
    [0, -1, -1, 0],    # 7  anti-transpose
], dtype=np.int64)


def _d4_tables():
    mats = [np.array([[m[0], m[1]], [m[2], m[3]]], dtype=np.int64) for m in D4_MAT]

    def index_of(mat):
        for i, m in enumerate(mats):
            if np.array_equal(m, mat):
                return i
        raise AssertionError('D4_MAT is not closed under composition')

    inv = np.array([index_of(m.T) for m in mats], dtype=np.int64)
    compose = np.array([[index_of(mats[h] @ mats[g]) for g in range(8)]
                        for h in range(8)], dtype=np.int64)

    action = np.zeros((8, N_ACTIONS), dtype=np.int64)
    for g, m in enumerate(mats):
        for a in range(N_ACTIONS):
            if a >= 4:                       # WAIT and BOMB are fixed points
                action[g, a] = a
                continue
            d = m @ np.array([DX[a], DY[a]], dtype=np.int64)
            action[g, a] = int(np.flatnonzero((DX == d[0]) & (DY == d[1]))[0])
    action_inv = np.zeros_like(action)
    for g in range(8):
        action_inv[g, action[g]] = np.arange(N_ACTIONS)

    src_x = np.zeros((8, COLS, ROWS), dtype=np.int64)
    src_y = np.zeros((8, COLS, ROWS), dtype=np.int64)
    for g in range(8):
        a, b, c, d = D4_MAT[inv[g]]
        for x in range(COLS):
            for y in range(ROWS):
                u, v = x - CENTRE, y - CENTRE
                src_x[g, x, y] = CENTRE + a * u + b * v
                src_y[g, x, y] = CENTRE + c * u + d * v
    return inv, action, action_inv, compose, src_x, src_y


D4_INV, D4_ACTION, D4_ACTION_INV, D4_COMPOSE, D4_SRC_X, D4_SRC_Y = _d4_tables()


@njit(cache=True, inline='always')
def _d4_x(g, x, y):
    return CENTRE + D4_MAT[g, 0] * (x - CENTRE) + D4_MAT[g, 1] * (y - CENTRE)


@njit(cache=True, inline='always')
def _d4_y(g, x, y):
    return CENTRE + D4_MAT[g, 2] * (x - CENTRE) + D4_MAT[g, 3] * (y - CENTRE)


@njit(cache=True)
def pick_d4(out, mx, my):
    cand = np.empty(8, dtype=np.int64)
    keep = np.zeros(8, dtype=np.uint8)
    k = 0
    for g in range(8):
        x = _d4_x(g, mx, my)
        y = _d4_y(g, mx, my)
        if x <= CENTRE and y <= CENTRE and x <= y:
            cand[k] = g
            keep[k] = 1
            k += 1
    if k == 1:
        return cand[0]

    alive = k
    for ci in range(2):
        c = 1 if ci == 0 else 7          # crate plane, then danger plane
        for x in range(COLS):
            for y in range(ROWS):
                if alive <= 1:
                    break
                best = 256
                for i in range(k):
                    if keep[i] == 0:
                        continue
                    gi = D4_INV[cand[i]]
                    val = out[c, _d4_x(gi, x, y), _d4_y(gi, x, y)]
                    if val < best:
                        best = val
                for i in range(k):
                    if keep[i] == 0:
                        continue
                    gi = D4_INV[cand[i]]
                    if out[c, _d4_x(gi, x, y), _d4_y(gi, x, y)] != best:
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
def canonical_view(out, mask, mx, my, scratch):
    g = pick_d4(out, mx, my)
    if g == 0:
        return 0
    for c in range(CHANNELS):
        for x in range(COLS):
            for y in range(ROWS):
                scratch[c, x, y] = out[c, D4_SRC_X[g, x, y], D4_SRC_Y[g, x, y]]
    for c in range(CHANNELS):
        for x in range(COLS):
            for y in range(ROWS):
                out[c, x, y] = scratch[c, x, y]
    old = np.empty(N_ACTIONS, dtype=np.uint8)
    for a in range(N_ACTIONS):
        old[a] = mask[a]
    for a in range(N_ACTIONS):
        mask[D4_ACTION[g, a]] = old[a]
    return g


def d4_position(g, x, y):
    a, b, c, d = D4_MAT[g]
    u, v = x - CENTRE, y - CENTRE
    return int(CENTRE + a * u + b * v), int(CENTRE + c * u + d * v)


def apply_d4(obs, mask, g):
    if g == 0:
        return obs.copy(), mask.copy()
    new_obs = np.ascontiguousarray(obs[:, D4_SRC_X[g], D4_SRC_Y[g]])
    new_mask = np.zeros_like(mask)
    new_mask[D4_ACTION[g]] = mask
    return new_obs, new_mask


@njit(cache=True, inline='always')
def _blocked(field, bombs, n_bombs, agents, n_agents, x, y):
    if field[x, y] != 0:
        return True
    for b in range(n_bombs):
        if bombs[b, 0] == x and bombs[b, 1] == y:
            return True
    for a in range(n_agents):  # noqa: SIM110 - numba compiles the loop, not any()
        if agents[a, 0] == x and agents[a, 1] == y:
            return True
    return False


@njit(cache=True)
def write_obs(out, field, coins, bombs, n_bombs, expl, agents, n_agents, me, step):
    for c in range(CHANNELS):
        for x in range(COLS):
            for y in range(ROWS):
                out[c, x, y] = 0
    mx = agents[me, 0]
    my = agents[me, 1]

    for x in range(COLS):
        for y in range(ROWS):
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
    for a in range(n_agents):
        if a == me or agents[a, 0] < 0:
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

    own_bomb = np.uint8(255) if agents[me, 2] != 0 else np.uint8(0)
    v = np.uint8(int(255.0 * min(1.0, step / MAX_STEPS)))
    for x in range(COLS):
        for y in range(ROWS):
            out[9, x, y] = own_bomb
            out[10, x, y] = v
            if not _blocked(field, bombs, n_bombs, agents, n_agents, x, y):
                out[11, x, y] = 255


@njit(cache=True)
def write_action_mask(mask, field, bombs, n_bombs, agents, n_agents, me):
    mx = agents[me, 0]
    my = agents[me, 1]
    for d in range(4):
        if _blocked(field, bombs, n_bombs, agents, n_agents, mx + DX[d], my + DY[d]):
            mask[d] = 0
        else:
            mask[d] = 1
    mask[A_WAIT] = 1
    mask[A_BOMB] = 1 if agents[me, 2] != 0 else 0


@njit(cache=True)
def target_distance(field, coins, x0, y0):
    dist = np.full((COLS, ROWS), -1, dtype=np.int64)
    queue = np.empty((COLS * ROWS, 2), dtype=np.int64)
    head = 0
    tail = 0
    queue[tail, 0] = x0
    queue[tail, 1] = y0
    tail += 1
    dist[x0, y0] = 0
    best_crate = 30
    while head < tail:
        x = queue[head, 0]
        y = queue[head, 1]
        head += 1
        d = dist[x, y]
        if coins[x, y] != 0:
            return d
        for k in range(4):
            nx = x + DX[k]
            ny = y + DY[k]
            if dist[nx, ny] != -1:
                continue
            f = field[nx, ny]
            if f == 1:
                if d + 1 < best_crate:
                    best_crate = d + 1
                dist[nx, ny] = d + 1
                continue
            if f != 0:
                continue
            dist[nx, ny] = d + 1
            if d + 1 < 30:
                queue[tail, 0] = nx
                queue[tail, 1] = ny
                tail += 1
    return best_crate


def game_state_to_arrays(game_state: dict):
    field = np.ascontiguousarray(game_state['field'], dtype=np.int8)

    coins = np.zeros((COLS, ROWS), dtype=np.uint8)
    for (x, y) in game_state['coins']:
        coins[x, y] = 1

    bomb_list = game_state['bombs']
    bombs = np.full((max(len(bomb_list), 1), 3), -1, dtype=np.int64)
    for i, ((x, y), t) in enumerate(bomb_list):
        bombs[i] = (x, y, t)
    n_bombs = len(bomb_list)

    expl = np.ascontiguousarray(game_state['explosion_map'], dtype=np.int8)

    _, _, own_bomb, (sx, sy) = game_state['self']
    others = game_state['others']
    agents = np.full((1 + len(others), 3), -1, dtype=np.int64)
    agents[0] = (sx, sy, 1 if own_bomb else 0)
    for i, (_, _, bomb_left, (x, y)) in enumerate(others):
        agents[i + 1] = (x, y, 1 if bomb_left else 0)

    return field, coins, bombs, n_bombs, expl, agents, len(others) + 1


def state_to_features(game_state: dict, canonical: bool = False):
    if game_state is None:
        return None, None, 0
    field, coins, bombs, n_bombs, expl, agents, n_agents = game_state_to_arrays(game_state)
    obs = np.zeros((CHANNELS, COLS, ROWS), dtype=np.uint8)
    write_obs(obs, field, coins, bombs, n_bombs, expl, agents, n_agents, 0,
              game_state['step'])
    mask = np.zeros(N_ACTIONS, dtype=np.uint8)
    write_action_mask(mask, field, bombs, n_bombs, agents, n_agents, 0)
    if not canonical:
        return obs, mask, 0
    scratch = np.zeros((CHANNELS, COLS, ROWS), dtype=np.uint8)
    g = canonical_view(obs, mask, int(agents[0, 0]), int(agents[0, 1]), scratch)
    return obs, mask, int(g)
