
import numpy as np
from .base import Algorithm
from .gwo import GWO
from .woa import WOA

class GW_WOA_Base(Algorithm):
    name = "GW--WOA (base)"
    def __init__(self, *a, b_spiral: float = 1.0, **k):
        super().__init__(*a, **k)
        self._gwo = GWO(*a, **k); self._woa = WOA(*a, b_spiral=b_spiral, **k)
    def evolve(self, X: np.ndarray) -> np.ndarray:
        p = self.rng.random(X.shape[0])[:,None]
        Xg = self._gwo.evolve(X)
        Xw = self._woa.evolve(X)
        Xn = np.where(p<0.5, Xg, Xw)
        return np.clip(Xn, self.lb, self.ub)
