
import numpy as np
from numba import njit, prange

from .config import (
    A_BOMB,
    A_WAIT,
    BOMB_POWER,
    BOMB_TIMER,
    CHANNELS,
    COLS,
    MAX_STEPS,
    N_ACTIONS,
    N_AGENTS,
    REWARD_ANNEAL_STEPS,
    ROWS,
)
from .features import (
    DX,
    DY,
    canonical_view,
    target_distance,
    write_action_mask,
    write_obs,
)

EXPL_LIFE = 4

DANGEROUS = 3

(R_COIN, R_KILL, R_SELFKILL, R_KILLED, R_CRATE, R_INVALID, R_STEP, R_SURVIVE,
 R_SHAPE, R_BOMBOPP, R_POTOPP) = range(11)

DEFAULT_REWARDS = np.array([
    1.0,    # coin collected
    2.0,    # killed an opponent
    -2.0,   # blew itself up
    -2.0,   # got blown up by someone else
    0.05,   # crate destroyed
    -0.02,  # invalid action
    -0.003,  # per step (mild urgency)
    0.5,    # alive when the round ends
    0.02,   # potential-based shaping towards coins/crates
    0.0,
    0.0,
], dtype=np.float64)

D1_REWARDS_START = np.array([
    1.0,     # coin collected -- the game's own value
    5.0,
    -1.0,    # blew itself up -- kept equal to being killed, see below
    -1.0,    # got blown up by someone else
    0.1,     # crate destroyed -- scaffolding, annealed to 0
    -0.02,
    -0.003,  # per step -- scaffolding, annealed to 0
    0.0,     # alive when the round ends -- REMOVED, not annealed
    0.02,    # potential-based shaping -- scaffolding, annealed to 0
    0.0,
    0.0,
], dtype=np.float64)

D1_REWARDS_FINAL = np.array([
    1.0, 5.0, -1.0, -1.0, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=np.float64)


def d1_rewards_at(env_steps, anneal_steps=REWARD_ANNEAL_STEPS, bombopp=0.0,
                  pot_opp=0.0):
    if anneal_steps <= 0:
        out = D1_REWARDS_FINAL.copy()
        frac = 1.0
    else:
        frac = min(1.0, max(0.0, float(env_steps) / float(anneal_steps)))
        out = D1_REWARDS_START + frac * (D1_REWARDS_FINAL - D1_REWARDS_START)
    out = np.asarray(out, dtype=np.float64).copy()
    out[R_BOMBOPP] = float(bombopp) * (1.0 - frac)
    out[R_POTOPP] = float(pot_opp)
    return out

N_REWARDS = 11

(ST_SCORE, ST_COINS, ST_KILLS, ST_SUICIDE, ST_CRATES, ST_STEPS, ST_INVALID,
 ST_DEATH_OTHER, ST_BOMBS, ST_SURVIVED, ST_BOMBOPP) = range(11)

N_STATS = 11

SC_DENSITY, SC_COINS, SC_ACTIVE, SC_OPP = range(4)

N_STAGE_COLS = 4

OPP_POLICY, OPP_PEACEFUL, OPP_COIN, OPP_MIXED, OPP_RULE = range(5)

COIN_FREE = 2

OPP_POT_CAP = 10

START_X = np.array([1, 1, COLS - 2, COLS - 2], dtype=np.int64)

START_Y = np.array([1, ROWS - 2, 1, ROWS - 2], dtype=np.int64)


@njit(cache=True)
def _seed_kernel(seed):
    np.random.seed(seed)


@njit(cache=True)
def _draw_stage(stage_weights):
    n = stage_weights.shape[0]
    if n <= 1:
        return 0
    total = 0.0
    for k in range(n):
        total += stage_weights[k]
    if total <= 0.0:
        return n - 1
    u = np.random.random() * total
    acc = 0.0
    for k in range(n):
        acc += stage_weights[k]
        if u < acc:
            return k
    return n - 1


@njit(cache=True)
def _reset_one(n, field, coins, agx, agy, alive, bleft, bx, by, bt, el, emask,
               stepc, stage_table, stage_weights, cfg_density, cfg_coins,
               cfg_active, cfg_opp, stage_id):
    sid = _draw_stage(stage_weights)
    stage_id[n] = sid
    crate_density = stage_table[sid, SC_DENSITY]
    coin_count = int(stage_table[sid, SC_COINS])
    n_active = int(stage_table[sid, SC_ACTIVE])
    cfg_density[n] = crate_density
    cfg_coins[n] = coin_count
    cfg_active[n] = n_active
    cfg_opp[n] = int(stage_table[sid, SC_OPP])

    for x in range(COLS):
        for y in range(ROWS):
            if x == 0 or y == 0 or x == COLS - 1 or y == ROWS - 1 or (x + 1) * (y + 1) % 2 == 1:
                field[n, x, y] = -1
            elif np.random.random() < crate_density:
                field[n, x, y] = 1
            else:
                field[n, x, y] = 0
            coins[n, x, y] = 0

    for k in range(4):
        x = START_X[k]
        y = START_Y[k]
        for d in range(5):
            xx = x + (DX[d - 1] if d > 0 else 0)
            yy = y + (DY[d - 1] if d > 0 else 0)
            if field[n, xx, yy] == 1:
                field[n, xx, yy] = 0

    crate_pos = np.empty((COLS * ROWS, 2), dtype=np.int64)
    free_pos = np.empty((COLS * ROWS, 2), dtype=np.int64)
    n_crate = 0
    n_free = 0
    for x in range(COLS):
        for y in range(ROWS):
            if field[n, x, y] == 1:
                crate_pos[n_crate, 0] = x
                crate_pos[n_crate, 1] = y
                n_crate += 1
            elif field[n, x, y] == 0:
                free_pos[n_free, 0] = x
                free_pos[n_free, 1] = y
                n_free += 1
    for i in range(n_crate - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        for c in range(2):
            tmp = crate_pos[i, c]
            crate_pos[i, c] = crate_pos[j, c]
            crate_pos[j, c] = tmp
    for i in range(n_free - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        for c in range(2):
            tmp = free_pos[i, c]
            free_pos[i, c] = free_pos[j, c]
            free_pos[j, c] = tmp

    placed = 0
    for i in range(n_crate):
        if placed >= coin_count:
            break
        coins[n, crate_pos[i, 0], crate_pos[i, 1]] = 1  # hidden under a crate
        placed += 1
    for i in range(n_free):
        if placed >= coin_count:
            break
        coins[n, free_pos[i, 0], free_pos[i, 1]] = 2  # lying around
        placed += 1

    order = np.arange(4)
    for i in range(3, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = order[i]
        order[i] = order[j]
        order[j] = tmp
    for a in range(N_AGENTS):
        agx[n, a] = START_X[order[a]]
        agy[n, a] = START_Y[order[a]]
        alive[n, a] = 1 if a < n_active else 0
        bleft[n, a] = 1
        bx[n, a] = -1
        by[n, a] = -1
        bt[n, a] = -1
        el[n, a] = 0
        for x in range(COLS):
            for y in range(ROWS):
                emask[n, a, x, y] = 0
    stepc[n] = 0


@njit(cache=True)
def _collect_view(n, field, coins, agx, agy, alive, bleft, bx, by, bt, el, emask,
                  view_coins, view_bombs, view_expl, view_agents):
    n_bombs = 0
    for x in range(COLS):
        for y in range(ROWS):
            view_coins[x, y] = 1 if coins[n, x, y] == 2 else 0
            view_expl[x, y] = 0
    for a in range(N_AGENTS):
        if bt[n, a] >= 0:
            view_bombs[n_bombs, 0] = bx[n, a]
            view_bombs[n_bombs, 1] = by[n, a]
            view_bombs[n_bombs, 2] = bt[n, a]
            n_bombs += 1
        if el[n, a] >= DANGEROUS:
            v = el[n, a] - DANGEROUS
            for x in range(COLS):
                for y in range(ROWS):
                    if emask[n, a, x, y] != 0 and v > view_expl[x, y]:
                        view_expl[x, y] = v
        if alive[n, a] != 0:
            view_agents[a, 0] = agx[n, a]
            view_agents[a, 1] = agy[n, a]
            view_agents[a, 2] = bleft[n, a]
        else:
            view_agents[a, 0] = -1
            view_agents[a, 1] = -1
            view_agents[a, 2] = 0
    return n_bombs


@njit(cache=True, parallel=True)
def vec_observe(field, coins, agx, agy, alive, bleft, bx, by, bt, el, emask,
                stepc, obs, masks, canonical, gsel):
    n_envs = field.shape[0]
    for n in prange(n_envs):
        view_coins = np.zeros((COLS, ROWS), dtype=np.uint8)
        view_expl = np.zeros((COLS, ROWS), dtype=np.int8)
        view_bombs = np.full((N_AGENTS, 3), -1, dtype=np.int64)
        view_agents = np.full((N_AGENTS, 3), -1, dtype=np.int64)
        scratch = np.zeros((CHANNELS, COLS, ROWS), dtype=np.uint8)
        n_bombs = _collect_view(n, field, coins, agx, agy, alive, bleft, bx, by,
                                bt, el, emask, view_coins, view_bombs,
                                view_expl, view_agents)
        for a in range(N_AGENTS):
            gsel[n, a] = 0
            if alive[n, a] == 0:
                for c in range(CHANNELS):
                    for x in range(COLS):
                        for y in range(ROWS):
                            obs[n, a, c, x, y] = 0
                for k in range(N_ACTIONS):
                    masks[n, a, k] = 0
                masks[n, a, A_WAIT] = 1
                continue
            write_obs(obs[n, a], field[n], view_coins, view_bombs, n_bombs,
                      view_expl, view_agents, N_AGENTS, a, stepc[n])
            write_action_mask(masks[n, a], field[n], view_bombs, n_bombs,
                              view_agents, N_AGENTS, a)
            if canonical != 0:
                gsel[n, a] = canonical_view(obs[n, a], masks[n, a], agx[n, a],
                                            agy[n, a], scratch)


@njit(cache=True)
def _potential(n, field, coins, agx, agy, alive, phi, shape_coef):
    view_coins = np.zeros((COLS, ROWS), dtype=np.uint8)
    for x in range(COLS):
        for y in range(ROWS):
            view_coins[x, y] = 1 if coins[n, x, y] == 2 else 0
    for a in range(N_AGENTS):
        if alive[n, a] == 0:
            phi[n, a] = 0.0
        else:
            d = target_distance(field[n], view_coins, agx[n, a], agy[n, a])
            phi[n, a] = -shape_coef * d


@njit(cache=True)
def _opponent_in_blast(n, field, agx, agy, alive, ox, oy, a):
    for o in range(N_AGENTS):
        if o == a or alive[n, o] == 0:
            continue
        if agx[n, o] == ox and agy[n, o] == oy:
            return True
    for d in range(4):
        for i in range(1, BOMB_POWER + 1):
            x = ox + DX[d] * i
            y = oy + DY[d] * i
            if field[n, x, y] == -1:
                break
            for o in range(N_AGENTS):
                if o == a or alive[n, o] == 0:
                    continue
                if agx[n, o] == x and agy[n, o] == y:
                    return True
    return False


@njit(cache=True)
def _potential_opp(n, field, agx, agy, alive, phi_opp, coef):
    for a in range(N_AGENTS):
        if alive[n, a] == 0:
            phi_opp[n, a] = 0.0
            continue
        dist = np.full((COLS, ROWS), -1, dtype=np.int64)
        qx = np.empty(COLS * ROWS, dtype=np.int64)
        qy = np.empty(COLS * ROWS, dtype=np.int64)
        head = 0
        tail = 0
        qx[tail] = agx[n, a]
        qy[tail] = agy[n, a]
        tail += 1
        dist[agx[n, a], agy[n, a]] = 0
        best = OPP_POT_CAP
        while head < tail:
            x = qx[head]
            y = qy[head]
            head += 1
            d0 = dist[x, y]
            if d0 >= OPP_POT_CAP:
                continue
            for k in range(4):
                nx = x + DX[k]
                ny = y + DY[k]
                if dist[nx, ny] >= 0 or field[n, nx, ny] != 0:
                    continue
                dist[nx, ny] = d0 + 1
                for o in range(N_AGENTS):
                    if o == a or alive[n, o] == 0:
                        continue
                    if agx[n, o] == nx and agy[n, o] == ny and d0 + 1 < best:
                        best = d0 + 1
                qx[tail] = nx
                qy[tail] = ny
                tail += 1
        phi_opp[n, a] = -coef * best


@njit(cache=True)
def _danger_grid(n, field, bx, by, bt, el, emask, out):
    for x in range(COLS):
        for y in range(ROWS):
            out[x, y] = 0
    for a in range(N_AGENTS):
        if el[n, a] >= DANGEROUS:
            for x in range(COLS):
                for y in range(ROWS):
                    if emask[n, a, x, y] != 0:
                        out[x, y] = 1
        if bt[n, a] >= 0:
            ox = bx[n, a]
            oy = by[n, a]
            out[ox, oy] = 1
            for d in range(4):
                for i in range(1, BOMB_POWER + 1):
                    x = ox + DX[d] * i
                    y = oy + DY[d] * i
                    if field[n, x, y] == -1:
                        break
                    out[x, y] = 1


@njit(cache=True)
def _tile_walkable(n, field, agx, agy, alive, bx, by, bt, x, y):
    if field[n, x, y] != 0:
        return False
    for b in range(N_AGENTS):
        if bt[n, b] >= 0 and bx[n, b] == x and by[n, b] == y:
            return False
    for b in range(N_AGENTS):
        if alive[n, b] != 0 and agx[n, b] == x and agy[n, b] == y:
            return False
    return True


@njit(cache=True)
def _peaceful_action(n, field, agx, agy, alive, bx, by, bt, a):
    choices = np.empty(4, dtype=np.int64)
    k = 0
    for d in range(4):
        if _tile_walkable(n, field, agx, agy, alive, bx, by, bt,
                          agx[n, a] + DX[d], agy[n, a] + DY[d]):
            choices[k] = d
            k += 1
    if k == 0:
        return A_WAIT
    return choices[np.random.randint(0, k)]


@njit(cache=True)
def _coin_action(n, field, coins, agx, agy, alive, bx, by, bt, danger, a):
    x0 = agx[n, a]
    y0 = agy[n, a]
    fleeing = danger[x0, y0] != 0

    first = np.full((COLS, ROWS), -1, dtype=np.int64)
    seen = np.zeros((COLS, ROWS), dtype=np.uint8)
    queue = np.empty((COLS * ROWS, 2), dtype=np.int64)
    head = 0
    tail = 0
    queue[tail, 0] = x0
    queue[tail, 1] = y0
    tail += 1
    seen[x0, y0] = 1

    while head < tail:
        x = queue[head, 0]
        y = queue[head, 1]
        head += 1
        for d in range(4):
            nx = x + DX[d]
            ny = y + DY[d]
            if seen[nx, ny] != 0:
                continue
            if x == x0 and y == y0:
                if not _tile_walkable(n, field, agx, agy, alive, bx, by, bt, nx, ny):
                    continue
            elif field[n, nx, ny] != 0:
                continue
            if not fleeing and danger[nx, ny] != 0:
                continue
            seen[nx, ny] = 1
            first[nx, ny] = d if (x == x0 and y == y0) else first[x, y]
            if fleeing:
                if danger[nx, ny] == 0:
                    return first[nx, ny]
            elif coins[n, nx, ny] == COIN_FREE:
                return first[nx, ny]
            queue[tail, 0] = nx
            queue[tail, 1] = ny
            tail += 1
    return A_WAIT


@njit(cache=True)
def _rule_action(n, field, coins, agx, agy, alive, bleft, bx, by, bt, el, emask,
                 danger, a):
    x0 = agx[n, a]
    y0 = agy[n, a]

    if danger[x0, y0] != 0:
        return _coin_action(n, field, coins, agx, agy, alive, bx, by, bt, danger, a)

    if bleft[n, a] != 0:
        useful = False
        for d in range(4):
            nx = x0 + DX[d]
            ny = y0 + DY[d]
            if field[n, nx, ny] == 1:
                useful = True
            for b in range(N_AGENTS):
                if b != a and alive[n, b] != 0 and agx[n, b] == nx and agy[n, b] == ny:
                    useful = True
        if useful:
            scratch_danger = np.zeros((COLS, ROWS), dtype=np.uint8)
            scratch_dist = np.empty((COLS, ROWS), dtype=np.int64)
            escape = _bomb_escape_one(n, field, coins, agx, agy, alive, bx, by, bt,
                                      el, emask, a, scratch_danger, scratch_dist)
            if escape != NOT_SAFE and escape <= BOMB_TIMER:
                return A_BOMB

    first = np.full((COLS, ROWS), -1, dtype=np.int64)
    seen = np.zeros((COLS, ROWS), dtype=np.uint8)
    queue = np.empty((COLS * ROWS, 2), dtype=np.int64)
    head = 0
    tail = 0
    queue[tail, 0] = x0
    queue[tail, 1] = y0
    tail += 1
    seen[x0, y0] = 1
    crate_move = -1

    while head < tail:
        x = queue[head, 0]
        y = queue[head, 1]
        head += 1
        at_start = x == x0 and y == y0
        for d in range(4):
            nx = x + DX[d]
            ny = y + DY[d]
            if seen[nx, ny] != 0:
                continue
            if field[n, nx, ny] == 1:
                seen[nx, ny] = 1
                if crate_move < 0:
                    crate_move = A_WAIT if at_start else first[x, y]
                continue
            if at_start:
                if not _tile_walkable(n, field, agx, agy, alive, bx, by, bt, nx, ny):
                    continue
            elif field[n, nx, ny] != 0:
                continue
            if danger[nx, ny] != 0:
                continue
            seen[nx, ny] = 1
            first[nx, ny] = d if at_start else first[x, y]
            if coins[n, nx, ny] == COIN_FREE:
                return first[nx, ny]
            queue[tail, 0] = nx
            queue[tail, 1] = ny
            tail += 1

    if crate_move >= 0:
        return crate_move
    return A_WAIT


@njit(cache=True)
def _step_one(n, field, coins, agx, agy, alive, bleft, bx, by, bt, el, emask,
              stepc, actions, rew, agent_done, stats, rcfg, order_in, opp):
    stepc[n] += 1

    act_eff = np.empty(N_AGENTS, dtype=np.int64)
    for a in range(N_AGENTS):
        act_eff[a] = actions[n, a]
    if opp != OPP_POLICY:
        danger = np.zeros((COLS, ROWS), dtype=np.uint8)
        _danger_grid(n, field, bx, by, bt, el, emask, danger)
        for a in range(1, N_AGENTS):
            if alive[n, a] == 0:
                continue
            kind = opp
            if opp == OPP_MIXED:
                kind = OPP_PEACEFUL if a == 1 else OPP_COIN
            if kind == OPP_PEACEFUL:
                act_eff[a] = _peaceful_action(n, field, agx, agy, alive, bx, by, bt, a)
            elif kind == OPP_RULE:
                act_eff[a] = _rule_action(n, field, coins, agx, agy, alive, bleft,
                                          bx, by, bt, el, emask, danger, a)
            else:
                act_eff[a] = _coin_action(n, field, coins, agx, agy, alive, bx, by,
                                          bt, danger, a)

    order = np.arange(N_AGENTS)
    if order_in[n, 0] >= 0:          # order forced (used by selfcheck.py)
        for i in range(N_AGENTS):
            order[i] = order_in[n, i]
    else:
        for i in range(N_AGENTS - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            tmp = order[i]
            order[i] = order[j]
            order[j] = tmp

    for oi in range(N_AGENTS):
        a = order[oi]
        if alive[n, a] == 0:
            continue
        act = act_eff[a]
        rew[n, a] += rcfg[R_STEP]
        if act < 4:
            tx = agx[n, a] + DX[act]
            ty = agy[n, a] + DY[act]
            free = field[n, tx, ty] == 0
            if free:
                for b in range(N_AGENTS):
                    if bt[n, b] >= 0 and bx[n, b] == tx and by[n, b] == ty:
                        free = False
                        break
            if free:
                for b in range(N_AGENTS):
                    if alive[n, b] != 0 and agx[n, b] == tx and agy[n, b] == ty:
                        free = False
                        break
            if free:
                agx[n, a] = tx
                agy[n, a] = ty
            else:
                rew[n, a] += rcfg[R_INVALID]
                stats[n, a, ST_INVALID] += 1
        elif act == A_BOMB:
            if bleft[n, a] != 0:
                bx[n, a] = agx[n, a]
                by[n, a] = agy[n, a]
                bt[n, a] = BOMB_TIMER
                bleft[n, a] = 0
                stats[n, a, ST_BOMBS] += 1
                if _opponent_in_blast(n, field, agx, agy, alive,
                                      agx[n, a], agy[n, a], a):
                    stats[n, a, ST_BOMBOPP] += 1
                    rew[n, a] += rcfg[R_BOMBOPP]
            else:
                rew[n, a] += rcfg[R_INVALID]
                stats[n, a, ST_INVALID] += 1

    for a in range(N_AGENTS):
        if alive[n, a] == 0:
            continue
        if coins[n, agx[n, a], agy[n, a]] == 2:
            coins[n, agx[n, a], agy[n, a]] = 0
            rew[n, a] += rcfg[R_COIN]
            stats[n, a, ST_SCORE] += 1
            stats[n, a, ST_COINS] += 1

    for a in range(N_AGENTS):
        if el[n, a] > 0:
            el[n, a] -= 1
            if el[n, a] == DANGEROUS - 1:
                bleft[n, a] = 1
            if el[n, a] == 0:
                for x in range(COLS):
                    for y in range(ROWS):
                        emask[n, a, x, y] = 0

    for a in range(N_AGENTS):
        if bt[n, a] < 0:
            continue
        if bt[n, a] == 0:
            ox = bx[n, a]
            oy = by[n, a]
            emask[n, a, ox, oy] = 1
            for d in range(4):
                for i in range(1, BOMB_POWER + 1):
                    x = ox + DX[d] * i
                    y = oy + DY[d] * i
                    if field[n, x, y] == -1:
                        break
                    emask[n, a, x, y] = 1
            for x in range(COLS):
                for y in range(ROWS):
                    if emask[n, a, x, y] != 0 and field[n, x, y] == 1:
                        field[n, x, y] = 0
                        rew[n, a] += rcfg[R_CRATE]
                        stats[n, a, ST_CRATES] += 1
                        if coins[n, x, y] == 1:
                            coins[n, x, y] = 2
            el[n, a] = EXPL_LIFE
            bt[n, a] = -1
            bx[n, a] = -1
            by[n, a] = -1
        else:
            bt[n, a] -= 1

    hit = np.zeros(N_AGENTS, dtype=np.uint8)
    self_hit = np.zeros(N_AGENTS, dtype=np.uint8)
    for a in range(N_AGENTS):
        if alive[n, a] == 0:
            continue
        for o in range(N_AGENTS):
            if el[n, o] >= DANGEROUS and emask[n, o, agx[n, a], agy[n, a]] != 0:
                hit[a] = 1
                if o == a:
                    self_hit[a] = 1
                    rew[n, a] += rcfg[R_SELFKILL]
                    stats[n, a, ST_SUICIDE] += 1
                else:
                    rew[n, o] += rcfg[R_KILL]
                    stats[n, o, ST_SCORE] += 5
                    stats[n, o, ST_KILLS] += 1
    for a in range(N_AGENTS):
        if hit[a] != 0:
            alive[n, a] = 0
            rew[n, a] += rcfg[R_KILLED]
            agent_done[n, a] = 1
            if self_hit[a] == 0:
                stats[n, a, ST_DEATH_OTHER] = 1

    n_alive = 0
    for a in range(N_AGENTS):
        n_alive += alive[n, a]
    if n_alive == 0:
        return 1
    if n_alive == 1:
        crates = 0
        loose_coins = 0
        for x in range(COLS):
            for y in range(ROWS):
                if field[n, x, y] == 1:
                    crates += 1
                if coins[n, x, y] == 2:
                    loose_coins += 1
        pending = 0
        for a in range(N_AGENTS):
            if bt[n, a] >= 0 or el[n, a] > 0:
                pending = 1
        if crates == 0 and loose_coins == 0 and pending == 0:
            return 1
    if stepc[n] >= MAX_STEPS:
        return 1
    return 0


@njit(cache=True, parallel=True)
def vec_step(field, coins, agx, agy, alive, bleft, bx, by, bt, el, emask, stepc,
             actions, rew, agent_done, env_done, stats, last_stats, phi, phi_opp,
             stage_table, stage_weights, cfg_density, cfg_coins, cfg_active,
             cfg_opp, stage_id, last_stage, rcfg, gamma, order_in, autoreset):
    n_envs = field.shape[0]
    for n in prange(n_envs):
        for a in range(N_AGENTS):
            rew[n, a] = 0.0
            agent_done[n, a] = 0

        rc = rcfg[stage_id[n]]
        done = _step_one(n, field, coins, agx, agy, alive, bleft, bx, by, bt,
                         el, emask, stepc, actions, rew, agent_done, stats, rc,
                         order_in, cfg_opp[n])

        if rc[R_SHAPE] > 0.0:
            old = np.empty(N_AGENTS, dtype=np.float64)
            for a in range(N_AGENTS):
                old[a] = phi[n, a]
            _potential(n, field, coins, agx, agy, alive, phi, rc[R_SHAPE])
            for a in range(N_AGENTS):
                if agent_done[n, a] != 0 or done != 0:
                    rew[n, a] += -old[a]        # terminal potential is 0
                elif alive[n, a] != 0:
                    rew[n, a] += gamma * phi[n, a] - old[a]

        if rc[R_POTOPP] > 0.0:
            old_o = np.empty(N_AGENTS, dtype=np.float64)
            for a in range(N_AGENTS):
                old_o[a] = phi_opp[n, a]
            _potential_opp(n, field, agx, agy, alive, phi_opp, rc[R_POTOPP])
            for a in range(N_AGENTS):
                if agent_done[n, a] != 0 or done != 0:
                    rew[n, a] += -old_o[a]      # terminal potential is 0
                elif alive[n, a] != 0:
                    rew[n, a] += gamma * phi_opp[n, a] - old_o[a]

        env_done[n] = done
        if done != 0 and autoreset != 0:
            for a in range(N_AGENTS):
                if alive[n, a] != 0:
                    rew[n, a] += rc[R_SURVIVE]
                agent_done[n, a] = 1
                stats[n, a, ST_STEPS] = stepc[n]
                stats[n, a, ST_SURVIVED] = alive[n, a]
                for k in range(N_STATS):
                    last_stats[n, a, k] = stats[n, a, k]
                    stats[n, a, k] = 0
            last_stage[n] = stage_id[n]
            _reset_one(n, field, coins, agx, agy, alive, bleft, bx, by, bt, el,
                       emask, stepc, stage_table, stage_weights, cfg_density,
                       cfg_coins, cfg_active, cfg_opp, stage_id)
            _potential(n, field, coins, agx, agy, alive, phi,
                       max(rcfg[stage_id[n], R_SHAPE], 1e-12))
            if rcfg[stage_id[n], R_POTOPP] > 0.0:
                _potential_opp(n, field, agx, agy, alive, phi_opp,
                               rcfg[stage_id[n], R_POTOPP])


@njit(cache=True, parallel=True)
def vec_reset(field, coins, agx, agy, alive, bleft, bx, by, bt, el, emask,
              stepc, phi, phi_opp, stage_table, stage_weights, cfg_density,
              cfg_coins, cfg_active, cfg_opp, stage_id, rcfg):
    n_envs = field.shape[0]
    for n in prange(n_envs):
        _reset_one(n, field, coins, agx, agy, alive, bleft, bx, by, bt, el,
                   emask, stepc, stage_table, stage_weights, cfg_density,
                   cfg_coins, cfg_active, cfg_opp, stage_id)
        _potential(n, field, coins, agx, agy, alive, phi,
                   max(rcfg[stage_id[n], R_SHAPE], 1e-12))
        if rcfg[stage_id[n], R_POTOPP] > 0.0:
            _potential_opp(n, field, agx, agy, alive, phi_opp,
                           rcfg[stage_id[n], R_POTOPP])


@njit(cache=True)
def greedy_coin_tour(field, coins, x0, y0):
    left = np.zeros((COLS, ROWS), dtype=np.uint8)
    remaining = 0
    for x in range(COLS):
        for y in range(ROWS):
            if coins[x, y] == COIN_FREE:
                left[x, y] = 1
                remaining += 1

    total = 0
    collected = 0
    cx = x0
    cy = y0
    dist = np.empty((COLS, ROWS), dtype=np.int64)
    queue = np.empty((COLS * ROWS, 2), dtype=np.int64)
    while remaining > 0:
        if left[cx, cy] != 0:          # standing on one
            left[cx, cy] = 0
            remaining -= 1
            collected += 1
            continue
        for x in range(COLS):
            for y in range(ROWS):
                dist[x, y] = -1
        head = 0
        tail = 0
        queue[tail, 0] = cx
        queue[tail, 1] = cy
        tail += 1
        dist[cx, cy] = 0
        best = -1
        bestx = -1
        besty = -1
        while head < tail:
            x = queue[head, 0]
            y = queue[head, 1]
            head += 1
            if left[x, y] != 0:
                best = dist[x, y]
                bestx = x
                besty = y
                break
            for d in range(4):
                nx = x + DX[d]
                ny = y + DY[d]
                if dist[nx, ny] != -1 or field[nx, ny] != 0:
                    continue
                dist[nx, ny] = dist[x, y] + 1
                queue[tail, 0] = nx
                queue[tail, 1] = ny
                tail += 1
        if best < 0:                   # nothing reachable any more
            break
        total += best
        cx = bestx
        cy = besty
        left[cx, cy] = 0
        remaining -= 1
        collected += 1
    return total, collected


@njit(cache=True, parallel=True)
def vec_reference_tour(field, coins, agx, agy, out_steps, out_coins):
    n_envs = field.shape[0]
    for n in prange(n_envs):
        steps, got = greedy_coin_tour(field[n], coins[n], agx[n, 0], agy[n, 0])
        out_steps[n] = steps
        out_coins[n] = got


NOT_SAFE = -1

NO_ESCAPE = 99


@njit(cache=True)
def _bomb_escape_one(n, field, coins, agx, agy, alive, bx, by, bt, el, emask, a,
                     danger, dist):
    if alive[n, a] == 0:
        return NOT_SAFE
    ax = agx[n, a]
    ay = agy[n, a]
    _danger_grid(n, field, bx, by, bt, el, emask, danger)
    if danger[ax, ay] != 0:
        return NOT_SAFE

    danger[ax, ay] = 1
    for d in range(4):
        for i in range(1, BOMB_POWER + 1):
            x = ax + DX[d] * i
            y = ay + DY[d] * i
            if field[n, x, y] == -1:
                break
            danger[x, y] = 1

    for x in range(COLS):
        for y in range(ROWS):
            dist[x, y] = -1
    qx = np.empty(COLS * ROWS, dtype=np.int64)
    qy = np.empty(COLS * ROWS, dtype=np.int64)
    qx[0] = ax
    qy[0] = ay
    dist[ax, ay] = 0
    head = 0
    tail = 1
    while head < tail:
        x = qx[head]
        y = qy[head]
        head += 1
        if danger[x, y] == 0:
            return dist[x, y]
        for d in range(4):
            nx = x + DX[d]
            ny = y + DY[d]
            if dist[nx, ny] >= 0:
                continue
            if not _tile_walkable(n, field, agx, agy, alive, bx, by, bt, nx, ny):
                continue
            dist[nx, ny] = dist[x, y] + 1
            qx[tail] = nx
            qy[tail] = ny
            tail += 1
    return NO_ESCAPE


@njit(cache=True, parallel=True)
def vec_bomb_escape_seat0(field, coins, agx, agy, alive, bx, by, bt, el, emask, out):
    n_envs = field.shape[0]
    for n in prange(n_envs):
        danger = np.zeros((COLS, ROWS), dtype=np.uint8)
        dist = np.empty((COLS, ROWS), dtype=np.int64)
        out[n] = _bomb_escape_one(n, field, coins, agx, agy, alive, bx, by,
                                  bt, el, emask, 0, danger, dist)


@njit(cache=True, parallel=True)
def vec_bomb_escape(field, coins, agx, agy, alive, bx, by, bt, el, emask, out):
    n_envs = field.shape[0]
    for n in prange(n_envs):
        danger = np.zeros((COLS, ROWS), dtype=np.uint8)
        dist = np.empty((COLS, ROWS), dtype=np.int64)
        for a in range(N_AGENTS):
            out[n, a] = _bomb_escape_one(n, field, coins, agx, agy, alive, bx, by,
                                         bt, el, emask, a, danger, dist)


def _pad_rewards(rewards, n_stages):
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim == 1:
        r = np.tile(r, (n_stages, 1))
    if r.ndim != 2 or r.shape[0] != n_stages or r.shape[1] > N_REWARDS:
        raise ValueError(f'rewards must be ({n_stages}, <= {N_REWARDS})')
    out = np.zeros((n_stages, N_REWARDS), dtype=np.float64)
    out[:, :r.shape[1]] = r
    return out


class VecBomberman:

    def __init__(self, n_envs, stage_table=None, stage_weights=None,
                 crate_density=0.75, coin_count=9, seed=0,
                 rewards=None, gamma=0.99, canonical=False):
        self.n_envs = n_envs
        self.canonical = bool(canonical)
        np.random.seed(seed)
        _seed_kernel(seed)
        self.gamma = gamma

        if stage_table is None:
            stage_table = np.array([[crate_density, coin_count, N_AGENTS, OPP_POLICY]],
                                   dtype=np.float64)
        self.stage_table = np.ascontiguousarray(stage_table, dtype=np.float64)
        if self.stage_table.ndim != 2 or self.stage_table.shape[1] != N_STAGE_COLS:
            raise ValueError(f'stage_table must be (n_stages, {N_STAGE_COLS})')
        self.n_stages = self.stage_table.shape[0]
        self.stage_weights = np.zeros(self.n_stages, dtype=np.float64)
        if stage_weights is None:
            self.stage_weights[-1] = 1.0
        else:
            self.set_stage_weights(stage_weights)

        rewards = DEFAULT_REWARDS if rewards is None else rewards
        self.rewards = np.ascontiguousarray(_pad_rewards(rewards, self.n_stages))

        self.field = np.zeros((n_envs, COLS, ROWS), dtype=np.int8)
        self.coins = np.zeros((n_envs, COLS, ROWS), dtype=np.int8)
        self.agx = np.zeros((n_envs, N_AGENTS), dtype=np.int64)
        self.agy = np.zeros((n_envs, N_AGENTS), dtype=np.int64)
        self.alive = np.zeros((n_envs, N_AGENTS), dtype=np.uint8)
        self.bleft = np.zeros((n_envs, N_AGENTS), dtype=np.uint8)
        self.bx = np.zeros((n_envs, N_AGENTS), dtype=np.int64)
        self.by = np.zeros((n_envs, N_AGENTS), dtype=np.int64)
        self.bt = np.zeros((n_envs, N_AGENTS), dtype=np.int64)
        self.el = np.zeros((n_envs, N_AGENTS), dtype=np.int64)
        self.emask = np.zeros((n_envs, N_AGENTS, COLS, ROWS), dtype=np.uint8)
        self.stepc = np.zeros(n_envs, dtype=np.int64)
        self.phi = np.zeros((n_envs, N_AGENTS), dtype=np.float64)
        self.phi_opp = np.zeros((n_envs, N_AGENTS), dtype=np.float64)

        self.obs = np.zeros((n_envs, N_AGENTS, CHANNELS, COLS, ROWS), dtype=np.uint8)
        self.masks = np.zeros((n_envs, N_AGENTS, N_ACTIONS), dtype=np.uint8)
        self.gsel = np.zeros((n_envs, N_AGENTS), dtype=np.int64)
        self.rew = np.zeros((n_envs, N_AGENTS), dtype=np.float64)
        self.agent_done = np.zeros((n_envs, N_AGENTS), dtype=np.uint8)
        self.env_done = np.zeros(n_envs, dtype=np.uint8)
        self.stats = np.zeros((n_envs, N_AGENTS, N_STATS), dtype=np.int64)
        self.last_stats = np.zeros((n_envs, N_AGENTS, N_STATS), dtype=np.int64)

        self.crate_density = np.full(n_envs, self.stage_table[-1, SC_DENSITY],
                                     dtype=np.float64)
        self.coin_count = np.full(n_envs, int(self.stage_table[-1, SC_COINS]),
                                  dtype=np.int64)
        self.n_active = np.full(n_envs, int(self.stage_table[-1, SC_ACTIVE]),
                                dtype=np.int64)
        self.opp_kind = np.full(n_envs, int(self.stage_table[-1, SC_OPP]), dtype=np.int64)
        self.stage_id = np.full(n_envs, self.n_stages - 1, dtype=np.int64)
        self.last_stage = np.zeros(n_envs, dtype=np.int64)
        self.order_in = np.full((n_envs, N_AGENTS), -1, dtype=np.int64)

    def set_stage_weights(self, weights):
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (self.n_stages,):
            raise ValueError(f'stage_weights must have length {self.n_stages}')
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError('stage_weights must be non-negative and not all zero')
        self.stage_weights[:] = w / w.sum()

    def set_rewards(self, rewards):
        self.rewards[:] = _pad_rewards(rewards, self.n_stages)

    def set_scenario(self, crate_density, coin_count):
        if self.n_stages != 1:
            raise ValueError('set_scenario is only defined for a single-stage table; '
                             'use set_stage_weights / the stage table instead')
        self.stage_table[0, SC_DENSITY] = crate_density
        self.stage_table[0, SC_COINS] = coin_count

    def reference_tour(self):
        steps = np.zeros(self.n_envs, dtype=np.int64)
        got = np.zeros(self.n_envs, dtype=np.int64)
        vec_reference_tour(self.field, self.coins, self.agx, self.agy, steps, got)
        return steps, got

    def bomb_escape_distance(self, seat=None):
        if seat is None:
            out = np.zeros((self.n_envs, N_AGENTS), dtype=np.int64)
            vec_bomb_escape(self.field, self.coins, self.agx, self.agy, self.alive,
                            self.bx, self.by, self.bt, self.el, self.emask, out)
            return out
        if seat != 0:
            raise ValueError('only seat 0 has a fast path; pass seat=None')
        out = np.zeros(self.n_envs, dtype=np.int64)
        vec_bomb_escape_seat0(self.field, self.coins, self.agx, self.agy, self.alive,
                              self.bx, self.by, self.bt, self.el, self.emask, out)
        return out

    def reset(self, stagger=True):
        vec_reset(self.field, self.coins, self.agx, self.agy, self.alive,
                  self.bleft, self.bx, self.by, self.bt, self.el, self.emask,
                  self.stepc, self.phi, self.phi_opp, self.stage_table,
                  self.stage_weights, self.crate_density, self.coin_count,
                  self.n_active, self.opp_kind, self.stage_id, self.rewards)
        if stagger:
            self.stepc[:] = np.random.randint(0, MAX_STEPS, self.n_envs)
        self._observe()
        return self.obs, self.masks, self.alive.copy()

    def _observe(self):
        vec_observe(self.field, self.coins, self.agx, self.agy, self.alive,
                    self.bleft, self.bx, self.by, self.bt, self.el, self.emask,
                    self.stepc, self.obs, self.masks,
                    1 if self.canonical else 0, self.gsel)

    def step(self, actions, autoreset=True):
        alive_before = self.alive.copy()
        vec_step(self.field, self.coins, self.agx, self.agy, self.alive,
                 self.bleft, self.bx, self.by, self.bt, self.el, self.emask,
                 self.stepc, actions, self.rew, self.agent_done, self.env_done,
                 self.stats, self.last_stats, self.phi, self.phi_opp,
                 self.stage_table,
                 self.stage_weights, self.crate_density, self.coin_count,
                 self.n_active, self.opp_kind, self.stage_id, self.last_stage,
                 self.rewards, self.gamma, self.order_in, 1 if autoreset else 0)
        self._observe()
        return (self.obs, self.masks, alive_before, self.rew, self.agent_done,
                self.env_done)

    def load_round(self, n, field, coins_xy_collectable, agent_positions):
        self.field[n] = field.astype(np.int8)
        self.coins[n] = 0
        for (x, y), collectable in coins_xy_collectable:
            self.coins[n, x, y] = 2 if collectable else 1
        for a in range(N_AGENTS):
            if a < len(agent_positions):
                self.agx[n, a], self.agy[n, a] = agent_positions[a]
                self.alive[n, a] = 1
            else:
                self.agx[n, a] = self.agy[n, a] = 1
                self.alive[n, a] = 0
            self.bleft[n, a] = 1
            self.bx[n, a] = self.by[n, a] = self.bt[n, a] = -1
            self.el[n, a] = 0
            self.emask[n, a] = 0
        self.stepc[n] = 0
        self.stats[n] = 0
        self.phi[n] = 0.0
        self.phi_opp[n] = 0.0

    def explosion_map(self, n):
        out = np.zeros((COLS, ROWS), dtype=np.int64)
        for a in range(N_AGENTS):
            if self.el[n, a] >= DANGEROUS:
                out = np.maximum(out, self.emask[n, a] * (self.el[n, a] - DANGEROUS))
        return out

    def bomb_list(self, n):
        return sorted((int(self.bx[n, a]), int(self.by[n, a]), int(self.bt[n, a]))
                      for a in range(N_AGENTS) if self.bt[n, a] >= 0)

    def finished_episode_stats(self):
        idx = np.nonzero(self.env_done)[0]
        return self.last_stats[idx]

    def finished_episode_stages(self):
        idx = np.nonzero(self.env_done)[0]
        return self.last_stage[idx]
