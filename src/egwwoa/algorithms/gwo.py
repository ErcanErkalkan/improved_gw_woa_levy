
import numpy as np
from .base import Algorithm

class GWO(Algorithm):
    name = "GWO"
    def evolve(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape
        r1 = self.rng.random((N, D))
        r2 = self.rng.random((N, D))
        a = 2.0 * (1 - (1 / max(1,self.cfg.T_iter)))
        A = 2*a*r1 - a
        C = 2*r2
        fit = np.apply_along_axis(self.fitness_fn, 1, X)
        idx = np.argsort(fit)[:3]
        Xa, Xb, Xd = X[idx[0]], X[idx[1]], X[idx[2]]
        Da = np.abs(C*Xa - X); Db = np.abs(C*Xb - X); Dd = np.abs(C*Xd - X)
        X1 = Xa - A*Da; X2 = Xb - A*Db; X3 = Xd - A*Dd
        Xn = (X1+X2+X3)/3.0
        return np.clip(Xn, self.lb, self.ub)
