
import numpy as np
from numba import njit

from .config import (
    COLS,
    CONJ_PLACE_SLOT,
    CONJ_SLOT,
    CRATE_BLAST_NORM,
    CRATE_SLOT,
    DANGER_HORIZON,
    DANGER_SLOT,
    DIST_CAP,
    D_NAV,
    ESCAPE_DIST_DEPTH,
    ESCAPE_DIST_NORM,
    ESCAPE_SLOT,
    FEATURE_MODES,
    MODE_DIM,
    MODE_SLOTS,
    N_CRATE,
    N_DANGER,
    PLACE_SLOT,
    POST_BOMB_ROOM_DEPTH,
    POST_BOMB_ROOM_NORM,
    ROOM_DEPTH,
    ROOM_NORM,
    ROOM_SLOT,
    ROWS,
    dim_for,
)

DX = np.array([0, 1, 0, -1], dtype=np.int64)

DY = np.array([-1, 0, 1, 0], dtype=np.int64)

MODE_ID = {name: i for i, name in enumerate(FEATURE_MODES)}

CONJ_PLACE_OUT_INDEX = PLACE_SLOT + 2

ESCAPE_OUT_INDEX = PLACE_SLOT + 2

WALL = -1

CRATE = 1


@njit(cache=True, inline='always')
def _blocked(field, occupied, x, y):
    return field[x, y] != 0 or occupied[x, y] != 0


@njit(cache=True)
def bfs_coin(field, coins, occupied, x0, y0):
    if coins[x0, y0] != 0:
        return 0, -1

    dist = np.full((COLS, ROWS), -1, dtype=np.int64)
    origin = np.full((COLS, ROWS), -1, dtype=np.int64)
    qx = np.empty(COLS * ROWS, dtype=np.int64)
    qy = np.empty(COLS * ROWS, dtype=np.int64)
    head = 0
    tail = 0
    dist[x0, y0] = 0
    qx[tail] = x0
    qy[tail] = y0
    tail += 1

    while head < tail:
        x = qx[head]
        y = qy[head]
        head += 1
        d = dist[x, y]
        for k in range(4):
            nx = x + DX[k]
            ny = y + DY[k]
            if dist[nx, ny] != -1:
                continue
            if _blocked(field, occupied, nx, ny):
                continue
            dist[nx, ny] = d + 1
            origin[nx, ny] = k if d == 0 else origin[x, y]
            if coins[nx, ny] != 0:
                return d + 1, origin[nx, ny]
            qx[tail] = nx
            qy[tail] = ny
            tail += 1
    return -1, -1


@njit(cache=True)
def write_danger_map(dmap, field, expl, bomb_x, bomb_y, bomb_t, n_bombs):
    for x in range(COLS):
        for y in range(ROWS):
            dmap[x, y] = 1.0 if expl[x, y] > 0.0 else 0.0

    for i in range(n_bombs):
        tau = bomb_t[i]
        if tau < 0:
            tau = 0
        if tau > DANGER_HORIZON:
            tau = DANGER_HORIZON
        value = (DANGER_HORIZON + 1 - tau) / (DANGER_HORIZON + 1)
        bx = bomb_x[i]
        by = bomb_y[i]
        if value > dmap[bx, by]:
            dmap[bx, by] = value
        for k in range(4):
            for r in range(1, 4):  # BOMB_POWER, spelled out for numba
                nx = bx + r * DX[k]
                ny = by + r * DY[k]
                if field[nx, ny] == WALL:
                    break
                if value > dmap[nx, ny]:
                    dmap[nx, ny] = value


@njit(cache=True)
def room_size(field, occupied, x0, y0, k):
    nx = x0 + DX[k]
    ny = y0 + DY[k]
    if _blocked(field, occupied, nx, ny):
        return 0.0

    seen = np.zeros((COLS, ROWS), dtype=np.uint8)
    seen[x0, y0] = 1          # never walk back through ourselves
    seen[nx, ny] = 1
    qx = np.empty(COLS * ROWS, dtype=np.int64)
    qy = np.empty(COLS * ROWS, dtype=np.int64)
    qd = np.empty(COLS * ROWS, dtype=np.int64)
    qx[0] = nx
    qy[0] = ny
    qd[0] = 0
    head = 0
    tail = 1
    count = 1

    while head < tail:
        x = qx[head]
        y = qy[head]
        d = qd[head]
        head += 1
        if d >= ROOM_DEPTH:
            continue
        for j in range(4):
            ax = x + DX[j]
            ay = y + DY[j]
            if seen[ax, ay] != 0:
                continue
            if _blocked(field, occupied, ax, ay):
                continue
            seen[ax, ay] = 1
            count += 1
            qx[tail] = ax
            qy[tail] = ay
            qd[tail] = d + 1
            tail += 1

    value = count / ROOM_NORM
    if value > 1.0:
        value = 1.0
    return value


@njit(cache=True)
def crates_in_blast(field, x0, y0):
    n = 0
    for k in range(4):
        for r in range(1, 4):   # BOMB_POWER, spelled out for numba
            nx = x0 + r * DX[k]
            ny = y0 + r * DY[k]
            if field[nx, ny] == WALL:
                break
            if field[nx, ny] == CRATE:
                n += 1
    return n


@njit(cache=True)
def post_bomb_room(field, occupied, x0, y0):
    blast = np.zeros((COLS, ROWS), dtype=np.uint8)
    blast[x0, y0] = 1
    for k in range(4):
        for r in range(1, 4):   # BOMB_POWER, spelled out for numba
            bx = x0 + r * DX[k]
            by = y0 + r * DY[k]
            if field[bx, by] == WALL:
                break
            blast[bx, by] = 1

    seen = np.zeros((COLS, ROWS), dtype=np.uint8)
    seen[x0, y0] = 1
    qx = np.empty(COLS * ROWS, dtype=np.int64)
    qy = np.empty(COLS * ROWS, dtype=np.int64)
    qd = np.empty(COLS * ROWS, dtype=np.int64)
    qx[0] = x0
    qy[0] = y0
    qd[0] = 0
    head = 0
    tail = 1
    count = 0

    while head < tail:
        x = qx[head]
        y = qy[head]
        d = qd[head]
        head += 1
        if d >= POST_BOMB_ROOM_DEPTH:
            continue
        for j in range(4):
            ax = x + DX[j]
            ay = y + DY[j]
            if seen[ax, ay] != 0:
                continue
            if _blocked(field, occupied, ax, ay):
                continue
            seen[ax, ay] = 1
            if blast[ax, ay] == 0:
                count += 1
            qx[tail] = ax
            qy[tail] = ay
            qd[tail] = d + 1
            tail += 1

    value = count / POST_BOMB_ROOM_NORM
    if value > 1.0:
        value = 1.0
    return value


@njit(cache=True)
def escape_dist(field, occupied, x0, y0):
    blast = np.zeros((COLS, ROWS), dtype=np.uint8)
    blast[x0, y0] = 1
    for k in range(4):
        for r in range(1, 4):   # BOMB_POWER, spelled out for numba
            bx = x0 + r * DX[k]
            by = y0 + r * DY[k]
            if field[bx, by] == WALL:
                break
            blast[bx, by] = 1

    seen = np.zeros((COLS, ROWS), dtype=np.uint8)
    seen[x0, y0] = 1
    qx = np.empty(COLS * ROWS, dtype=np.int64)
    qy = np.empty(COLS * ROWS, dtype=np.int64)
    qd = np.empty(COLS * ROWS, dtype=np.int64)
    qx[0] = x0
    qy[0] = y0
    qd[0] = 0
    head = 0
    tail = 1
    steps = ESCAPE_DIST_DEPTH

    while head < tail:
        x = qx[head]
        y = qy[head]
        d = qd[head]
        head += 1
        if d >= ESCAPE_DIST_DEPTH:
            continue
        for j in range(4):
            ax = x + DX[j]
            ay = y + DY[j]
            if seen[ax, ay] != 0:
                continue
            if _blocked(field, occupied, ax, ay):
                continue
            seen[ax, ay] = 1
            if blast[ax, ay] == 0:
                steps = d + 1
                head = tail          # stop the search: BFS found the nearest
                break
            qx[tail] = ax
            qy[tail] = ay
            qd[tail] = d + 1
            tail += 1

    if steps > ESCAPE_DIST_DEPTH:
        steps = ESCAPE_DIST_DEPTH
    return 1.0 - steps / ESCAPE_DIST_NORM


@njit(cache=True)
def safe_distance(field, occupied, dmap, x0, y0, cap):
    if dmap[x0, y0] <= 0.0:
        return 0

    seen = np.zeros((COLS, ROWS), dtype=np.uint8)
    seen[x0, y0] = 1
    qx = np.empty(COLS * ROWS, dtype=np.int64)
    qy = np.empty(COLS * ROWS, dtype=np.int64)
    qd = np.empty(COLS * ROWS, dtype=np.int64)
    qx[0] = x0
    qy[0] = y0
    qd[0] = 0
    head = 0
    tail = 1

    while head < tail:
        x = qx[head]
        y = qy[head]
        d = qd[head]
        head += 1
        if d >= cap:
            continue
        for j in range(4):
            ax = x + DX[j]
            ay = y + DY[j]
            if seen[ax, ay] != 0:
                continue
            if _blocked(field, occupied, ax, ay):
                continue
            seen[ax, ay] = 1
            if dmap[ax, ay] <= 0.0:
                return d + 1
            qx[tail] = ax
            qy[tail] = ay
            qd[tail] = d + 1
            tail += 1
    return cap


@njit(cache=True)
def write_phi(out, field, coins, occupied, expl, bomb_x, bomb_y, bomb_t, n_bombs,
              x0, y0, bombs_left, mode_id):
    n = out.shape[0]
    for i in range(n):
        out[i] = 0.0

    for k in range(4):
        if _blocked(field, occupied, x0 + DX[k], y0 + DY[k]):
            out[k] = 1.0

    d, direction = bfs_coin(field, coins, occupied, x0, y0)
    if d >= 0 and direction >= 0:
        out[4 + direction] = 1.0
    if d < 0:
        dc = DIST_CAP
    else:
        dc = d if d < DIST_CAP else DIST_CAP
        out[8] = 1.0 - dc / DIST_CAP

    if bombs_left:
        out[9] = 1.0
    out[10] = 1.0

    if mode_id == 0:      # nav: no danger group at all
        return dc

    dmap = np.zeros((COLS, ROWS))
    write_danger_map(dmap, field, expl, bomb_x, bomb_y, bomb_t, n_bombs)

    if mode_id == 1:      # inblast1: one binary flag, own tile only
        if dmap[x0, y0] > 0.0:
            out[DANGER_SLOT] = 1.0
        return dc

    here = dmap[x0, y0]
    out[DANGER_SLOT] = 1.0 if (mode_id == 2 and here > 0.0) else here
    for k in range(4):
        value = dmap[x0 + DX[k], y0 + DY[k]]
        if mode_id == 2:
            value = 1.0 if value > 0.0 else 0.0
        out[DANGER_SLOT + 1 + k] = value

    has_room = mode_id >= 4
    has_crate = mode_id >= 5
    has_blast = mode_id >= 6
    has_post_bomb_room = mode_id >= 6 and mode_id != 9
    has_conj = mode_id == 7
    has_escape = mode_id == 8
    has_conj_place = mode_id == 10

    if has_room:          # room and up: four bounded depth-3 BFS runs
        for k in range(4):
            out[ROOM_SLOT + k] = room_size(field, occupied, x0, y0, k)

    if has_crate:         # crate and up: four array reads, no search at all
        for k in range(4):
            if field[x0 + DX[k], y0 + DY[k]] == CRATE:
                out[CRATE_SLOT + k] = 1.0

    if has_blast:         # O(12) reads for crates_in_blast
        value = crates_in_blast(field, x0, y0) / CRATE_BLAST_NORM
        out[PLACE_SLOT] = value if value < 1.0 else 1.0

    if has_post_bomb_room:    # one bounded depth-4 BFS
        out[PLACE_SLOT + 1] = post_bomb_room(field, occupied, x0, y0)

    if has_conj:          # conj: one hand-chosen product of slots 9, 24 and 25
        if out[PLACE_SLOT + 1] > 0.0:
            out[CONJ_SLOT] = out[9] * out[PLACE_SLOT]

    if has_escape:        # esc: a second bounded depth-4 BFS, at vector index 26
        out[ESCAPE_OUT_INDEX] = escape_dist(field, occupied, x0, y0)

    if has_conj_place:
        if out[9] > 0.0 and out[PLACE_SLOT] > 0.0:
            out[CONJ_PLACE_OUT_INDEX] = (out[9] * out[PLACE_SLOT]
                                         * escape_dist(field, occupied, x0, y0))
    return dc


@njit(cache=True)
def bfs_coin_distance(field, coins, occupied, x0, y0):
    d, _ = bfs_coin(field, coins, occupied, x0, y0)
    if d < 0 or d > DIST_CAP:
        return DIST_CAP
    return d


def empty_bombs():
    z = np.zeros(0, dtype=np.int64)
    return z, z.copy(), z.copy()


def bomb_arrays(bombs):
    n = len(bombs)
    bx = np.empty(n, dtype=np.int64)
    by = np.empty(n, dtype=np.int64)
    bt = np.empty(n, dtype=np.int64)
    for i, (x, y, t) in enumerate(bombs):
        bx[i] = x
        by[i] = y
        bt[i] = t
    return bx, by, bt


def game_state_to_arrays(game_state: dict):
    field = np.ascontiguousarray(game_state['field'], dtype=np.int8)

    coins = np.zeros((COLS, ROWS), dtype=np.uint8)
    for (x, y) in game_state['coins']:
        coins[x, y] = 1

    occupied = np.zeros((COLS, ROWS), dtype=np.uint8)
    bombs = []
    for ((x, y), t) in game_state['bombs']:
        occupied[x, y] = 1
        bombs.append((int(x), int(y), int(t)))
    for (_n, _s, _b, (x, y)) in game_state['others']:
        occupied[x, y] = 1

    expl = np.ascontiguousarray(game_state['explosion_map'], dtype=np.float64)
    bomb_x, bomb_y, bomb_t = bomb_arrays(bombs)

    _, _, bombs_left, (sx, sy) = game_state['self']
    return (field, coins, occupied, expl, bomb_x, bomb_y, bomb_t, sx, sy,
            bool(bombs_left))


def state_to_features(game_state: dict, mode: str) -> np.ndarray:
    if game_state is None:
        return None
    (field, coins, occupied, expl, bomb_x, bomb_y, bomb_t, x, y,
     bombs_left) = game_state_to_arrays(game_state)
    phi = np.zeros(dim_for(mode), dtype=np.float64)
    write_phi(phi, field, coins, occupied, expl, bomb_x, bomb_y, bomb_t,
              len(bomb_t), x, y, bombs_left, MODE_ID[mode])
    return phi


def danger_map(game_state: dict) -> np.ndarray:
    (field, _coins, _occ, expl, bomb_x, bomb_y, bomb_t, _x, _y,
     _bl) = game_state_to_arrays(game_state)
    out = np.zeros((COLS, ROWS))
    write_danger_map(out, field, expl, bomb_x, bomb_y, bomb_t, len(bomb_t))
    return out


def slot_names(mode: str) -> list:
    from .config import feature_names
    return feature_names(mode)


__all__ = ['DX', 'DY', 'MODE_ID', 'MODE_DIM', 'MODE_SLOTS', 'D_NAV', 'N_DANGER',
           'DANGER_SLOT', 'ROOM_SLOT', 'CRATE_SLOT', 'N_CRATE', 'PLACE_SLOT',
           'CONJ_SLOT', 'ESCAPE_SLOT', 'ESCAPE_OUT_INDEX',
           'bfs_coin', 'bfs_coin_distance', 'write_danger_map', 'write_phi',
           'room_size', 'crates_in_blast', 'post_bomb_room', 'escape_dist',
           'safe_distance', 'state_to_features',
           'game_state_to_arrays', 'bomb_arrays', 'empty_bombs', 'danger_map',
           'slot_names']
