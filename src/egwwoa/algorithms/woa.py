
import numpy as np
from .base import Algorithm

class WOA(Algorithm):
    name = "WOA"
    def __init__(self, *a, b_spiral: float = 1.0, **k):
        super().__init__(*a, **k); self.b = b_spiral
    def evolve(self, X: np.ndarray) -> np.ndarray:
        fit = np.apply_along_axis(self.fitness_fn, 1, X)
        best = X[int(np.argmin(fit))]
        l = self.rng.uniform(-1,1,size=(X.shape[0],1))
        Dp = np.abs(best - X)
        Xn = Dp * np.exp(self.b*l) * np.cos(2*np.pi*l) + best
        return np.clip(Xn, self.lb, self.ub)
