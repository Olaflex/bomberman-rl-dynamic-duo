
from collections import deque
from typing import Optional

import numpy as np

DX = (0, 1, 0, -1)

DY = (-1, 0, 1, 0)


def first_step_to_nearest(field, coins, x0, y0):
    if coins[x0, y0]:
        return 0, -1
    cols, rows = field.shape
    dist = np.full((cols, rows), -1, dtype=np.int64)
    origin = np.full((cols, rows), -1, dtype=np.int64)
    dist[x0, y0] = 0
    queue = deque([(x0, y0)])
    while queue:
        x, y = queue.popleft()
        d = dist[x, y]
        for k in range(4):
            nx, ny = x + DX[k], y + DY[k]
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            if dist[nx, ny] != -1 or field[nx, ny] != 0:
                continue
            dist[nx, ny] = d + 1
            origin[nx, ny] = k if d == 0 else origin[x, y]
            if coins[nx, ny]:
                return d + 1, int(origin[nx, ny])
            queue.append((nx, ny))
    return -1, -1


def gbr_play(field, coin_positions, start, max_steps=400) -> dict:
    field = np.asarray(field)
    coins = np.zeros(field.shape, dtype=bool)
    for (x, y) in coin_positions:
        coins[x, y] = True

    x, y = int(start[0]), int(start[1])
    steps = 0
    collected = 0
    coin_steps = []
    if coins[x, y]:       # the framework collects by position, after the step
        pass
    while steps < max_steps and coins.any():
        _d, direction = first_step_to_nearest(field, coins, x, y)
        if direction >= 0:
            x += DX[direction]
            y += DY[direction]
        steps += 1
        if coins[x, y]:
            coins[x, y] = False
            collected += 1
            coin_steps.append(steps)
        elif direction < 0:
            break          # nothing reachable and nothing underfoot: stuck
    return {'steps': steps, 'coins': collected, 'coin_steps': coin_steps}


def gbr_for_board(board, agent_name: str, max_steps=400) -> Optional[dict]:
    if (np.asarray(board.field) == 1).any():
        return None
    if agent_name not in board.starts:
        return None
    coins = [(x, y) for (x, y, collectable) in board.coins if collectable]
    return gbr_play(board.field, coins, board.starts[agent_name], max_steps)


def prefix_steps(ref: dict, k: int) -> Optional[float]:
    if ref is None or k <= 0 or k > len(ref['coin_steps']):
        return None
    return float(ref['coin_steps'][k - 1])
