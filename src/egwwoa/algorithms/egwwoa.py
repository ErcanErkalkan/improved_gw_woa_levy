
import numpy as np
from .base import Algorithm
from .base_gwwoa import GW_WOA_Base
from ..operators.levy import levy_flight
from ..operators.chaos import logistic_map_reseed
from ..operators.schedules import omega_cosine

class EGW_WOA(Algorithm):
    name = "EGW-WOA"
    def __init__(self, *a, p_levy:float=0.25, beta_L:float=1.5, b_spiral:float=1.0,
                 I_stall:int=10, phi:float=0.2, omega_max:float=1.0, omega_min:float=0.1, **k):
        super().__init__(*a, **k)
        self.hybrid = GW_WOA_Base(*a, b_spiral=b_spiral, **k)
        self.p_levy, self.beta_L, self.I_stall, self.phi = p_levy, beta_L, I_stall, phi
        self.omega_max, self.omega_min = omega_max, omega_min
        self._stall, self._best = 0, np.inf
    def evolve(self, X: np.ndarray, t:int|None=None) -> np.ndarray:
        Xn = self.hybrid.evolve(X)
        N, D = Xn.shape
        mask = (self.rng.random(N) < self.p_levy)
        if mask.any():
            w = omega_cosine(t or 1, self.cfg.T_iter, self.omega_max, self.omega_min)
            Xn[mask] += levy_flight(int(mask.sum()), D, self.beta_L, self.rng) * w
        return np.clip(Xn, self.lb, self.ub)
    def run(self):
        X = self.initialize()
        fit = np.apply_along_axis(self.fitness_fn, 1, X)
        bi = int(np.argmin(fit))
        Xb, fb = X[bi].copy(), float(fit[bi]); self._best = fb
        for t in range(1, self.cfg.T_iter+1):
            X = self.evolve(X, t=t)
            fit = np.apply_along_axis(self.fitness_fn, 1, X)
            i = int(np.argmin(fit))
            if fit[i] + 1e-12 < fb:
                fb, Xb, self._stall = float(fit[i]), X[i].copy(), 0
            else:
                self._stall += 1
            if self._stall >= self.I_stall:
                k = max(1, int(np.ceil(self.phi * X.shape[0])))
                worst = np.argsort(fit)[-k:]
                X[worst] = logistic_map_reseed(k, X.shape[1], self.lb, self.ub, self.rng)
                fit[worst] = np.apply_along_axis(self.fitness_fn, 1, X[worst])
                self._stall = 0
        return {"X_best": Xb, "f_best": fb}
