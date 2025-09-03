
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Callable, Any, Dict

@dataclass
class AlgoConfig:
    N: int
    T_iter: int

class Algorithm:
    name: str = "BASE"
    def __init__(self, cfg: AlgoConfig, fitness_fn: Callable[[np.ndarray], float],
                 lb: np.ndarray, ub: np.ndarray, rng: np.random.Generator):
        self.cfg, self.fitness_fn, self.lb, self.ub, self.rng = cfg, fitness_fn, lb, ub, rng
    def initialize(self) -> np.ndarray:
        return self.lb + (self.ub - self.lb) * self.rng.random((self.cfg.N, self.lb.size))
    def evolve(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def run(self) -> Dict[str, Any]:
        X = self.initialize()
        fitness = np.apply_along_axis(self.fitness_fn, 1, X)
        bi = int(np.argmin(fitness))
        Xb, fb = X[bi].copy(), float(fitness[bi])
        for t in range(1, self.cfg.T_iter+1):
            X = self.evolve(X)
            fitness = np.apply_along_axis(self.fitness_fn, 1, X)
            i = int(np.argmin(fitness))
            if fitness[i] < fb:
                fb, Xb = float(fitness[i]), X[i].copy()
        return {"X_best": Xb, "f_best": fb}
