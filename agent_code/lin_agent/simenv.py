
import numpy as np

from .config import (
    A_BOMB,
    A_DOWN,
    A_LEFT,
    A_RIGHT,
    A_UP,
    A_WAIT,
    BOMB_POWER,
    BOMB_TIMER,
    COLS,
    DEFAULT_MODE,
    EXPLOSION_TIMER,
    INJECT_BOMB_TIMER,
    MAX_STEPS,
    REWARD_COIN,
    ROWS,
    dim_for,
)
from .features import (
    DX,
    DY,
    MODE_ID,
    bfs_coin,
    bfs_coin_distance,
    bomb_arrays,
    safe_distance,
    write_danger_map,
    write_phi,
)

WALL = -1
FREE = 0
CRATE = 1

START_POSITIONS = [(1, 1), (1, ROWS - 2), (COLS - 2, 1), (COLS - 2, ROWS - 2)]

COIN_NONE = 0
COIN_HIDDEN = 1
COIN_FREE = 2


def build_arena(seed, crate_density, coin_count, n_agents=1):
    rng = np.random.default_rng(seed)
    arena = np.zeros((COLS, ROWS), int)
    arena[rng.random((COLS, ROWS)) < crate_density] = CRATE

    arena[:1, :] = WALL
    arena[-1:, :] = WALL
    arena[:, :1] = WALL
    arena[:, -1:] = WALL
    for x in range(COLS):
        for y in range(ROWS):
            if (x + 1) * (y + 1) % 2 == 1:
                arena[x, y] = WALL

    for (x, y) in START_POSITIONS:
        for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if arena[xx, yy] == CRATE:
                arena[xx, yy] = FREE

    all_positions = np.stack(np.meshgrid(np.arange(COLS), np.arange(ROWS), indexing='ij'), -1)
    crate_positions = rng.permutation(all_positions[arena == CRATE])
    free_positions = rng.permutation(all_positions[arena == FREE])
    coin_positions = np.concatenate([crate_positions, free_positions], 0)[:coin_count]

    coins = np.zeros((COLS, ROWS), dtype=np.uint8)
    for x, y in coin_positions:
        coins[x, y] = COIN_FREE if arena[x, y] == FREE else COIN_HIDDEN

    starts = [tuple(int(v) for v in p) for p in rng.permutation(START_POSITIONS)[:n_agents]]
    return arena.astype(np.int8), coins, starts


class SoloWorld:

    def __init__(self, crate_density=0.0, coin_count=50, max_steps=MAX_STEPS,
                 feature_mode=DEFAULT_MODE):
        self.crate_density = crate_density
        self.coin_count = coin_count
        self.max_steps = max_steps
        self.feature_mode = feature_mode
        self.mode_id = MODE_ID[feature_mode]
        self._phi = np.zeros(dim_for(feature_mode), dtype=np.float64)
        self._occupied = np.zeros((COLS, ROWS), dtype=np.uint8)
        self._danger = np.zeros((COLS, ROWS), dtype=np.float64)
        self.field = None


    def reset(self, seed, inject_bomb=False):
        field, coins, starts = build_arena(seed, self.crate_density, self.coin_count, 1)
        return self.load_round(field, coins, starts[0], inject_bomb=inject_bomb)

    def load_round(self, field, coins, start, inject_bomb=False):
        self.field = np.ascontiguousarray(field, dtype=np.int8).copy()
        self.coins = np.ascontiguousarray(coins, dtype=np.uint8).copy()
        self.x, self.y = int(start[0]), int(start[1])
        self.bombs = []          # [x, y, timer]
        self.explosions = []     # [stage, timer, [(x, y), ...]]
        self.bombs_left = True
        self.dead = False
        self.score = 0
        self.step = 0
        self.running = True
        self.stats = {'coins': 0, 'invalid': 0, 'bombs': 0, 'crates': 0,
                      'suicide': 0, 'waited': 0, 'moves': 0, 'seeded_bombs': 0}
        if inject_bomb:
            self.bombs.append([self.x, self.y, INJECT_BOMB_TIMER])
            self.bombs_left = False
            self.stats['seeded_bombs'] = 1
        return self.features()


    def _occupancy(self):
        self._occupied[:] = 0
        for (bx, by, _t) in self.bombs:
            self._occupied[bx, by] = 1
        return self._occupied

    def _collectable(self):
        return (self.coins == COIN_FREE).astype(np.uint8)

    def _bomb_arrays(self):
        return bomb_arrays([(int(b[0]), int(b[1]), int(b[2])) for b in self.bombs])

    def features(self):
        bx, by, bt = self._bomb_arrays()
        write_phi(self._phi, self.field, self._collectable(), self._occupancy(),
                  self.explosion_map(), bx, by, bt, len(bt), self.x, self.y,
                  self.bombs_left, self.mode_id)
        return self._phi.copy()

    def features_for(self, mode):
        out = np.zeros(dim_for(mode), dtype=np.float64)
        bx, by, bt = self._bomb_arrays()
        write_phi(out, self.field, self._collectable(), self._occupancy(),
                  self.explosion_map(), bx, by, bt, len(bt), self.x, self.y,
                  self.bombs_left, MODE_ID[mode])
        return out

    def danger(self):
        bx, by, bt = self._bomb_arrays()
        write_danger_map(self._danger, self.field, self.explosion_map(),
                         bx, by, bt, len(bt))
        return self._danger

    def safe_here(self) -> bool:
        return float(self.danger()[self.x, self.y]) == 0.0

    def safe_distance(self, cap):
        return int(safe_distance(self.field, self._occupancy(), self.danger(),
                                 self.x, self.y, int(cap)))

    def coin_distance(self):
        return bfs_coin_distance(self.field, self._collectable(), self._occupancy(),
                                 self.x, self.y)

    def greedy_step(self):
        _d, direction = bfs_coin(self.field, self._collectable(), self._occupancy(),
                                 self.x, self.y)
        if direction < 0:
            return A_WAIT
        return int(direction)

    def explosion_map(self):
        m = np.zeros((COLS, ROWS))
        for (stage, timer, coords) in self.explosions:
            if stage == 0:
                for (x, y) in coords:
                    m[x, y] = max(m[x, y], timer - 1)
        return m

    def bomb_list(self):
        return sorted((int(b[0]), int(b[1]), int(b[2])) for b in self.bombs)


    def tile_is_free(self, x, y):
        if self.field[x, y] != FREE:
            return False
        return all(not (bx == x and by == y) for (bx, by, _t) in self.bombs)

    def _blast_coords(self, x, y):
        coords = [(x, y)]
        for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for i in range(1, BOMB_POWER + 1):
                nx, ny = x + i * dx, y + i * dy
                if self.field[nx, ny] == WALL:
                    break
                coords.append((nx, ny))
        return coords

    def _perform_action(self, action):
        if action == A_UP and self.tile_is_free(self.x, self.y - 1):
            self.y -= 1
            self.stats['moves'] += 1
        elif action == A_DOWN and self.tile_is_free(self.x, self.y + 1):
            self.y += 1
            self.stats['moves'] += 1
        elif action == A_LEFT and self.tile_is_free(self.x - 1, self.y):
            self.x -= 1
            self.stats['moves'] += 1
        elif action == A_RIGHT and self.tile_is_free(self.x + 1, self.y):
            self.x += 1
            self.stats['moves'] += 1
        elif action == A_BOMB and self.bombs_left:
            self.bombs.append([self.x, self.y, BOMB_TIMER])
            self.bombs_left = False
            self.stats['bombs'] += 1
        elif action == A_WAIT:
            self.stats['waited'] += 1
        else:
            self.stats['invalid'] += 1

    def _collect_coins(self):
        if self.coins[self.x, self.y] == COIN_FREE:
            self.coins[self.x, self.y] = COIN_NONE
            self.score += REWARD_COIN
            self.stats['coins'] += 1
            return 1
        return 0

    def _update_explosions(self):
        remaining = []
        for exp in self.explosions:
            exp[1] -= 1
            if exp[1] <= 0:
                exp[0] += 1
                if exp[0] == 1:
                    exp[1] = 2  # len(Explosion.ASSETS[1]); smoke, harmless
                    self.bombs_left = True
                else:
                    exp[0] = None  # IndexError branch of Explosion.next_stage
            if exp[0] is not None:
                remaining.append(exp)
        self.explosions = remaining

    def _update_bombs(self):
        crates = 0
        still_live = []
        for bomb in self.bombs:
            if bomb[2] <= 0:
                coords = self._blast_coords(bomb[0], bomb[1])
                for (x, y) in coords:
                    if self.field[x, y] == CRATE:
                        self.field[x, y] = FREE
                        crates += 1
                        if self.coins[x, y] == COIN_HIDDEN:
                            self.coins[x, y] = COIN_FREE
                self.explosions.append([0, EXPLOSION_TIMER, coords])
            else:
                bomb[2] -= 1
                still_live.append(bomb)
        self.bombs = still_live
        self.stats['crates'] += crates
        return crates

    def _evaluate_explosions(self):
        for (stage, _timer, coords) in self.explosions:
            if stage == 0 and (self.x, self.y) in coords:
                self.dead = True
                self.stats['suicide'] += 1
                return True
        return False

    def _time_to_stop(self):
        if self.dead:
            return True
        if ((self.field == CRATE).sum() == 0
                and not (self.coins == COIN_FREE).any()
                and len(self.bombs) + len(self.explosions) == 0):
            return True
        return self.step >= self.max_steps

    def step_action(self, action):
        if not self.running:
            raise RuntimeError('step_action called after the round ended')

        self.step += 1
        invalid_before = self.stats['invalid']
        self._perform_action(action)
        invalid = self.stats['invalid'] > invalid_before

        coins = self._collect_coins()
        self._update_explosions()
        crates = self._update_bombs()
        died = self._evaluate_explosions()

        if self._time_to_stop():
            self.running = False

        phi = np.zeros(self._phi.shape[0], dtype=np.float64) if self.dead \
            else self.features()
        return phi, {'coins': coins, 'crates': crates, 'invalid': invalid,
                     'suicide': died, 'done': not self.running}
