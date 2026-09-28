
import numpy as np

from .config import D, N_ACTIONS


class LinearQ:

    def __init__(self, w=None, dim=None):
        if w is None:
            self.w = np.zeros((N_ACTIONS, D if dim is None else dim), dtype=np.float64)
            return
        w = np.asarray(w, dtype=np.float64)
        want_dim = w.shape[1] if (dim is None and w.ndim == 2) else dim
        if w.ndim != 2 or w.shape != (N_ACTIONS, want_dim):
            raise ValueError(f'weights must have shape {(N_ACTIONS, want_dim)}, '
                             f'got {w.shape}')
        self.w = w.copy()

    @property
    def dim(self) -> int:
        return int(self.w.shape[1])

    def q(self, phi: np.ndarray) -> np.ndarray:
        return self.w @ phi

    def greedy(self, phi: np.ndarray) -> int:
        return int(np.argmax(self.w @ phi))

    def norm(self) -> float:
        return float(np.linalg.norm(self.w))

    def save(self, path: str):
        np.save(path, self.w)

    @classmethod
    def load(cls, path: str, dim=None) -> 'LinearQ':
        return cls(np.load(path), dim=dim)

    def table(self, feature_names, action_names) -> str:
        head = '| action | ' + ' | '.join(feature_names) + ' |'
        rule = '| --- | ' + ' | '.join(['---'] * len(feature_names)) + ' |'
        rows = [
            '| ' + a + ' | ' + ' | '.join(f'{v:+.3f}' for v in self.w[i]) + ' |'
            for i, a in enumerate(action_names)
        ]
        return '\n'.join([head, rule, *rows])
